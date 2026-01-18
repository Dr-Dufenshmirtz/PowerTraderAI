"""
pt_thinker.py - ApolloTrader AI Prediction Engine

Description:
This module implements the AI "Thinker" used by ApolloTrader. It contains
helpers and long-running per-coin stepping logic that fetches market data
(KuCoin and Robinhood), generates predictions, and exposes utilities used
by the GUI and runner processes. This file intentionally performs some
top-level initialization (for example, the KuCoin `Market` client) to
preserve the original runtime behavior.

Primary Repository: https://github.com/Dr-Dufenshmirtz/ApolloTrader
Primary Author: Dr Dufenshmirtz

Original Project: https://github.com/garagesteve1155/PowerTrader_AI
Original Author: Stephen Hughes (garagesteve1155)

Notes on AI behavior and trading rules (informational only):

- Start trade signal:
	The AI's Thinker sends a start signal for a coin when the current ask
	price drops below at least three of the AI's predicted low prices for
	that coin. The AI predicts the active candle's high/low for each
	timeframe from 1 hour up to 1 week; those predicted lows/highs are
	the blue/orange horizontal lines shown on the charts.

- DCA rules:
	DCA (Dollar Cost Averaging) decisions use either the AI's current
	price level tied to the number of DCA buys already executed for the
	trade (e.g., the 4th blue line after trade start), or a hardcoded
	drawdown percentage for the current level — whichever triggers first.
	A maximum of 2 DCA orders are allowed within a rolling 24-hour
	window to prevent over-allocating into prolonged downtrends.

- Sell rules:
	The bot uses a trailing profit margin to capture upside. The margin is
	set to 5% if no DCA occurred on the trade, or 2.5% if any DCA occurred.
	The trailing gap is 0.5% (the amount the price must exceed the margin
	before the trailing line moves up). When the price drops below the
	trailing margin after rising, the bot sells to capture gains.

- Training summary:
	Training iterates over the historical candles for a coin on multiple
	timeframes and records each observed pattern along with the next
	candle's outcome. The trainer generates predictions by kernel-weighted
	averaging of the closest memory-pattern matches, where closer matches
	get exponentially higher weight. This per-timeframe predicted candle
	yields the high/low lines used by the Thinker.

- Pattern Matching:
	Uses relative threshold matching (percentage of pattern magnitude) for
	consistent behavior across different price levels and market conditions.
	Thresholds are volatility-based (5.0× average volatility) and use a
	volatility-adaptive baseline (0.1× volatility, typically ~0.4%) for
	near-zero patterns. Kernel weighting uses exponential distance penalty
	(e^(-diff/threshold)) scaled by threshold enforcement setting.

- Threshold Enforcement (configurable in training_settings.json):
	• Tight (1×): Strong distance penalty, only uses very similar patterns
	• Balanced (2×): Moderate penalty, default setting
	• Loose (5×): Weak penalty, considers wider range of patterns
	• None (10×): Minimal penalty, nearly uniform weighting

- Prediction Candles:
	Generates future-timestamped prediction candles showing AI's expected
	price action for the next interval. Accumulates predictions across all
	timeframes and writes to prediction_candles.json for Hub visualization.
	Displays as white candle extending beyond current price action.
"""

import os
import sys
import time
from datetime import datetime
import random
import calendar
import linecache
import traceback
import base64
import hashlib
import hmac
import json
import uuid
import logging
import math
import threading
from contextlib import contextmanager

# Ensure clean console output (avoid encoding issues)
try:
	sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
	sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
except Exception:
	pass  # Not all systems support reconfigure; gracefully degrade

# third-party
import requests
from kucoin.client import Market
import psutil
from nacl.signing import SigningKey

# instantiate KuCoin market client (kept at top-level like original)
market = Market(url='https://api.kucoin.com')

# ============================================================================
# THINKER ALGORITHM CONSTANTS
# ============================================================================

# Volatility and default values
DEFAULT_VOLATILITY = 4.0  # Default volatility fallback (typical crypto high-low range %)
DEFAULT_DISTANCE = 0.5  # Default distance value for calculations

# Trading thresholds and margins
MIN_PROFIT_MARGIN = 0.25  # Minimum profit margin % for trades (also used as default/fallback)
GAP_ENFORCEMENT_MIN = 0.25  # Minimum gap % between neural levels
GAP_INCREMENT = 0.25  # Gap modifier increment % when enforcing level separation

# Pattern matching parameters
BASE_MAX_CUTOFF_MULTIPLIER = 1.25  # Base multiplier for pattern exclusion (× threshold × enforcement)
BASE_PERFECT_MATCH_MULTIPLIER = 1.0  # Base multiplier for fallback trigger (× threshold × enforcement)
BASE_KERNEL_BANDWIDTH = 2.5  # Base kernel bandwidth in threshold-widths (× enforcement, lower = stronger distance penalty)

# API rate limiting
KUCOIN_API_RATE_LIMIT = 0.15  # Seconds between KuCoin API calls (150ms, ~6.7 req/s)
ROBINHOOD_API_RATE_LIMIT = 0.5  # Seconds between Robinhood API calls

# Percentage conversion
PERCENT_MULTIPLIER = 100  # Multiplier to convert decimal to percentage

# Windows DPAPI decryption
def _decrypt_with_dpapi(encrypted_data: bytes) -> str:
	"""Decrypt bytes using Windows DPAPI.
	Can only be decrypted by the same Windows user account that encrypted it."""
	import ctypes
	from ctypes import wintypes
	
	class DATA_BLOB(ctypes.Structure):
		_fields_ = [('cbData', wintypes.DWORD), ('pbData', ctypes.POINTER(ctypes.c_char))]
	
	blob_in = DATA_BLOB(len(encrypted_data), ctypes.cast(ctypes.c_char_p(encrypted_data), ctypes.POINTER(ctypes.c_char)))
	blob_out = DATA_BLOB()
	
	if ctypes.windll.crypt32.CryptUnprotectData(
		ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
	):
		decrypted = ctypes.string_at(blob_out.pbData, blob_out.cbData)
		ctypes.windll.kernel32.LocalFree(blob_out.pbData)
		return decrypted.decode('utf-8')
	else:
		raise RuntimeError("Failed to decrypt data with Windows DPAPI")

# Helper function to clean training file strings by removing quotes, brackets, and commas
def _clean_training_string(text: str) -> str:
	"""Remove quotes, brackets, commas from training file data."""
	for char in ("'", ',', '"', ']', '['):
		text = text.replace(char, '')
	return text

# Robinhood market-data API for current ask prices, same source as trader.
# Uses GET /api/v1/crypto/marketdata/best_bid_ask/?symbol=BTC-USD
# Returns result["ask_inclusive_of_buy_spread"]
ROBINHOOD_BASE_URL = "https://trading.robinhood.com"

_RH_MD = None  # lazy-init so import doesn't explode if creds missing

class RobinhoodMarketData:
	def __init__(self, api_key: str, base64_private_key: str, base_url: str = ROBINHOOD_BASE_URL, timeout: int = 10):
		self.api_key = (api_key or "").strip()
		self.base_url = (base_url or "").rstrip("/")
		self.timeout = timeout

		if not self.api_key:
			raise RuntimeError("Robinhood API key is empty (rh_key.enc).")

		try:
			raw_private = base64.b64decode((base64_private_key or "").strip())
			self.private_key = SigningKey(raw_private)
		except Exception as e:
			raise RuntimeError(f"Failed to decode Robinhood private key (rh_secret.enc): {e}")

		self.session = requests.Session()

	def _get_current_timestamp(self) -> int:
		return int(time.time())

	def _get_authorization_header(self, method: str, path: str, body: str, timestamp: int) -> dict:
		# matches the trader's signing format
		method = method.upper()
		body = body or ""
		# Construct the exact message that the trader expects to sign.
		# This order and concatenation MUST match the server/trader signing
		# implementation so signatures validate correctly.
		message_to_sign = f"{self.api_key}{timestamp}{path}{method}{body}"
		signed = self.private_key.sign(message_to_sign.encode("utf-8"))
		signature_b64 = base64.b64encode(signed.signature).decode("utf-8")

		return {
			"x-api-key": self.api_key,
			"x-timestamp": str(timestamp),
			"x-signature": signature_b64,
			"Content-Type": "application/json",
		}

	def make_api_request(self, method: str, path: str, body: str = "") -> dict:
		url = f"{self.base_url}{path}"
		ts = self._get_current_timestamp()
		headers = self._get_authorization_header(method, path, body, ts)

		# Perform a signed HTTP request. Exceptions are propagated to callers
		# so higher-level logic can retry or handle failures as appropriate.
		resp = self.session.request(method=method.upper(), url=url, headers=headers, data=body or None, timeout=self.timeout)
		if resp.status_code >= 400:
			raise RuntimeError(f"Robinhood HTTP {resp.status_code}: {resp.text}")
		return resp.json()

	def get_current_ask(self, symbol: str) -> float:
		symbol = (symbol or "").strip().upper()
		path = f"/api/v1/crypto/marketdata/best_bid_ask/?symbol={symbol}"
		data = self.make_api_request("GET", path)

		is_valid, price, error_msg = _validate_robinhood_response(data, symbol)
		if not is_valid:
			raise RuntimeError(f"Robinhood API validation failed: {error_msg}")
		
		return price

def _load_and_validate_thinker_credentials(base_dir: str) -> tuple:
	"""Load and validate Robinhood API credentials for Thinker with strict format checking.
	
	Args:
		base_dir: Directory containing the credential files
	
	Returns:
		tuple: (api_key: str, base64_private_key: str)
	
	Raises:
		FileNotFoundError: If credential files don't exist
		ValueError: If credentials are invalid format
		RuntimeError: If decryption fails
	
	Validation checks:
		- Files exist and are readable
		- Decryption succeeds
		- API key length >= 20 characters
		- Private key is valid base64 (32 bytes decoded)
		- Private key can create valid SigningKey
	"""
	key_path = os.path.join(base_dir, "rh_key.enc")
	secret_path = os.path.join(base_dir, "rh_secret.enc")
	
	# Step 1: Check file existence
	if not os.path.isfile(key_path):
		raise FileNotFoundError(
			f"API key file not found: {key_path}\n"
			"Open the Hub and go to Settings → Robinhood API → Setup / Update."
		)
	
	if not os.path.isfile(secret_path):
		raise FileNotFoundError(
			f"Private key file not found: {secret_path}\n"
			"Open the Hub and go to Settings → Robinhood API → Setup / Update."
		)
	
	# Step 2: Decrypt credentials
	try:
		with open(key_path, "rb") as f:
			encrypted_key = f.read()
			if not encrypted_key:
				raise ValueError(f"File '{key_path}' is empty")
			api_key = _decrypt_with_dpapi(encrypted_key).strip()
	except Exception as e:
		raise RuntimeError(f"Failed to decrypt API key: {e}")
	
	try:
		with open(secret_path, "rb") as f:
			encrypted_secret = f.read()
			if not encrypted_secret:
				raise ValueError(f"File '{secret_path}' is empty")
			base64_private_key = _decrypt_with_dpapi(encrypted_secret).strip()
	except Exception as e:
		raise RuntimeError(f"Failed to decrypt private key: {e}")
	
	# Step 3: Validate API key format
	if not api_key or len(api_key) == 0:
		raise ValueError("API key is empty after decryption")
	
	if len(api_key) < 20:
		raise ValueError(f"API key too short ({len(api_key)} chars, minimum 20 required)")
	
	if len(api_key) > 200:
		raise ValueError(f"API key too long ({len(api_key)} chars, maximum 200)")
	
	if api_key != api_key.strip():
		raise ValueError("API key contains leading or trailing whitespace")
	
	# Check for null bytes or control characters
	if '\x00' in api_key:
		raise ValueError("API key contains null bytes (corrupted)")
	
	# Check for newlines or tabs that could break API calls
	if '\n' in api_key or '\r' in api_key or '\t' in api_key:
		raise ValueError("API key contains newline or tab characters (corrupted)")
	
	# Validate character set (alphanumeric, hyphens, underscores)
	import string
	allowed_chars = string.ascii_letters + string.digits + '-_'
	invalid_chars = [c for c in api_key if c not in allowed_chars]
	if invalid_chars:
		sample = ''.join(invalid_chars[:5])
		raise ValueError(f"API key contains invalid characters: {repr(sample)}")
	
	# Step 4: Validate private key format
	if not base64_private_key or len(base64_private_key) == 0:
		raise ValueError("Private key is empty after decryption")
	
	# Check for null bytes or control characters
	if '\x00' in base64_private_key:
		raise ValueError("Private key contains null bytes (corrupted)")
	
	# Check for newlines/whitespace that could break base64 decoding
	if '\n' in base64_private_key or '\r' in base64_private_key or '\t' in base64_private_key:
		raise ValueError("Private key contains newline or tab characters (corrupted)")
	
	# Validate base64 character set
	import string
	base64_chars = string.ascii_letters + string.digits + '+/='
	invalid_chars = [c for c in base64_private_key if c not in base64_chars]
	if invalid_chars:
		sample = ''.join(set(invalid_chars[:5]))
		raise ValueError(f"Private key contains invalid base64 characters: {repr(sample)}")
	
	# Attempt to decode base64 with validation
	try:
		import base64 as b64
		decoded_key = b64.b64decode(base64_private_key, validate=True)
	except Exception as e:
		raise ValueError(f"Private key is not valid base64: {e}")
	
	if len(decoded_key) != 32:
		raise ValueError(f"Private key must be 32 bytes for Ed25519, got {len(decoded_key)} bytes")
	
	# Verify decoded bytes are not all zeros (invalid key)
	if all(b == 0 for b in decoded_key):
		raise ValueError("Private key decodes to all zeros (invalid Ed25519 key)")
	
	# Step 5: Test SigningKey creation
	try:
		from nacl.signing import SigningKey
		test_key = SigningKey(decoded_key)
	except Exception as e:
		raise ValueError(f"Private key cannot create valid SigningKey: {e}")
	
	return api_key, base64_private_key

def robinhood_current_ask(symbol: str) -> float:
    """
    Returns Robinhood current BUY price (ask_inclusive_of_buy_spread) for symbols like 'BTC-USD'.
    Reads encrypted creds from rh_key.enc and rh_secret.enc in the same folder as this script.
    
    Credentials are validated on first use:
        - File existence
        - Decryption success
        - Format validation (API key length, base64 private key)
        - Ed25519 key compatibility
    """
    global _RH_MD
    if _RH_MD is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load and validate credentials with comprehensive checking
        try:
            api_key, priv_b64 = _load_and_validate_thinker_credentials(base_dir)
            debug_print("[DEBUG] THINKER: API credentials loaded and validated successfully")
        except FileNotFoundError as e:
            print(f"\n{'='*60}", flush=True)
            print(f"❌ [Thinker] API Credentials Not Found", flush=True)
            print(f"{'='*60}", flush=True)
            print(f"{e}", flush=True)
            print(f"{'='*60}\n", flush=True)
            raise
        except ValueError as e:
            print(f"\n{'='*60}", flush=True)
            print(f"❌ [Thinker] Invalid API Credential Format", flush=True)
            print(f"{'='*60}", flush=True)
            print(f"{e}", flush=True)
            print(f"\nYour credential files may be corrupted.", flush=True)
            print(f"Re-generate via Hub: Settings → Robinhood API → Setup / Update", flush=True)
            print(f"{'='*60}\n", flush=True)
            raise
        except RuntimeError as e:
            print(f"\n{'='*60}", flush=True)
            print(f"❌ [Thinker] API Credential Decryption Failed", flush=True)
            print(f"{'='*60}", flush=True)
            print(f"{e}", flush=True)
            print(f"\nFiles may be from different Windows user or machine.", flush=True)
            print(f"Re-generate via Hub: Settings → Robinhood API → Setup / Update", flush=True)
            print(f"{'='*60}\n", flush=True)
            raise

        _RH_MD = RobinhoodMarketData(api_key=api_key, base64_private_key=priv_b64)

    return _RH_MD.get_current_ask(symbol)

def restart_program():
	"""Restarts the current program (no CLI args; uses hardcoded COIN_SYMBOLS)."""
	try:
		os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)])
	except Exception as e:
		print(f'Error during program restart: {e}', flush=True)

def _timeframe_to_seconds(timeframe: str) -> int:
	"""Convert timeframe string to seconds for timestamp calculation."""
	timeframe_map = {
		'1hour': 3600,
		'2hour': 7200,
		'4hour': 14400,
		'8hour': 28800,
		'12hour': 43200,
		'1day': 86400,
		'1week': 604800
	}
	return timeframe_map.get(timeframe, 3600)  # Default to 1 hour if unknown

# Utility: PrintException prints a helpful one-line context for exceptions
# by locating the source file/line and printing the offending line. This
# mirrors a similar helper in other scripts and is useful when debugging
# runtime errors during long-running loops.

def PrintException():
	"""Extract and log detailed exception information for debugging.
	
	Walks the traceback to the innermost frame (actual error location) and logs
	the exact line of code, line number, and error message. Logs to both console
	and debug_thinker.log file for persistent error tracking. Always logs exceptions
	regardless of debug mode since exceptions indicate serious issues.
	"""
	exc_type, exc_obj, tb = sys.exc_info()

	# Walk to the innermost frame (where the error actually happened)
	while tb and tb.tb_next:
		tb = tb.tb_next

	f = tb.tb_frame
	lineno = tb.tb_lineno
	filename = f.f_code.co_filename

	linecache.checkcache(filename)
	line = linecache.getline(filename, lineno, f.f_globals)
	# Truncate exception message for console if too long (e.g., HTML error pages)
	exc_msg_full = str(exc_obj)
	exc_msg_console = exc_msg_full
	if len(exc_msg_full) > 500:
		exc_msg_console = exc_msg_full[:500] + f"... ({len(exc_msg_full) - 500} more chars)"
	msg_console = 'EXCEPTION IN (LINE {} "{}"): {}'.format(lineno, line.strip(), exc_msg_console)
	msg_full = 'EXCEPTION IN (LINE {} "{}"): {}'.format(lineno, line.strip(), exc_msg_full)
	print(msg_console, flush=True)
	# Always log exceptions to file (even without debug mode) with full error message
	try:
		import datetime
		timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		with open("debug_thinker.log", "a", encoding="utf-8") as f:
			f.write(f"[{timestamp}] {msg_full}\n")
	except Exception:
		pass

# Legacy state flags preserved for historical compatibility (may be unused in current codebase)
restarted = 'no'
short_started = 'no'
long_started = 'no'
minute = 0
last_minute = 0

# GUI settings for coins list loaded from gui_settings.json with hot-reload capability
_GUI_SETTINGS_PATH = os.environ.get("POWERTRADER_GUI_SETTINGS") or os.path.join(
	os.path.dirname(os.path.abspath(__file__)),
	"gui_settings.json"
)

_gui_settings_cache = {
	"mtime": None,
	"coins": ['BTC', 'ETH', 'XRP', 'BNB', 'DOGE'],  # fallback defaults
	"debug_mode": False,
}

def _is_debug_mode() -> bool:
	"""Check if debug mode is enabled in gui_settings.json"""
	try:
		if not os.path.isfile(_GUI_SETTINGS_PATH):
			return _gui_settings_cache["debug_mode"]
		with open(_GUI_SETTINGS_PATH, "r", encoding="utf-8") as f:
			data = json.load(f) or {}
		debug = data.get("debug_mode", False)
		_gui_settings_cache["debug_mode"] = bool(debug)
		return bool(debug)
	except Exception:
		return _gui_settings_cache["debug_mode"]

def debug_print(msg: str):
	"""Print debug message only if debug mode is enabled, also log to file"""
	if _is_debug_mode():
		print(msg, flush=True)
		# Also write to debug log file
		try:
			import datetime
			timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			with open("debug_thinker.log", "a", encoding="utf-8") as f:
				f.write(f"[{timestamp}] {msg}\n")
		except Exception:
			pass  # Don't let logging errors break the thinker

def _get_sleep_timing(key: str) -> float:
	"""Get sleep timing from gui_settings.json with fallback to cache defaults"""
	try:
		if os.path.isfile(_GUI_SETTINGS_PATH):
			with open(_GUI_SETTINGS_PATH, "r", encoding="utf-8") as f:
				data = json.load(f) or {}
			value = data.get(key)
			if value is not None:
				return float(value)
	except Exception:
		pass
	return _gui_settings_cache.get(key, 1.0)


def _is_valid_coin_symbol(symbol: str) -> bool:
	"""Validate that a coin symbol is safe and well-formed.
	
	Ensures coin symbols only contain uppercase letters (A-Z) and are reasonable length.
	This prevents path traversal attacks and ensures compatibility with file system operations.
	
	Args:
		symbol: The coin symbol to validate (e.g., "BTC", "ETH")
	
	Returns:
		True if valid, False otherwise
	
	Valid examples: BTC, ETH, DOGE, USDT
	Invalid examples: BTC/USDT, ../BTC, BTC123, btc, VERYLONGCOINNAME
	"""
	if not symbol or not isinstance(symbol, str):
		return False
	
	# Must be 2-10 uppercase letters only
	if len(symbol) < 2 or len(symbol) > 10:
		return False
	
	# Only uppercase letters allowed
	if not symbol.isupper() or not symbol.isalpha():
		return False
	
	return True


def handle_network_error(operation: str, error: Exception):
	"""Handle fatal network errors by logging details and exiting gracefully.
	
	Called when network operations (KuCoin API, Robinhood market data) fail persistently
	after retries. Provides user-friendly error messaging with troubleshooting guidance.
	Exits the process since continued operation without network access would generate
	misleading predictions and potentially dangerous trading signals.
	
	Args:
		operation: Description of the failed operation for logging (e.g., "KuCoin candle fetch")
		error: The exception that triggered the failure
	"""
	print(f"\n{'='*60}", flush=True)
	print(f"❌ NETWORK ERROR: {operation} failed", flush=True)
	print(f"Error: {type(error).__name__}: {str(error)[:200]}", flush=True)
	print(f"The process will exit. Please check:", flush=True)
	print(f"  1. Your internet connection", flush=True)
	print(f"  2. API service status (KuCoin/Robinhood)", flush=True)
	print(f"  3. Enable debug_mode in gui_settings.json for more details", flush=True)
	print(f"{'='*60}\n", flush=True)
	time.sleep(_get_sleep_timing("sleep_startup_error"))
	sys.exit(1)

def _load_gui_coins() -> list:
	"""
	Load the list of coins to track from gui_settings.json with hot-reload capability.
	
	Reads the coins list from GUI settings and caches it by file modification time.
	Subsequent calls only re-read the file if it changed, making frequent polling
	efficient for hot-reload support. Returns uppercased symbols ("BTC", "ETH", etc.)
	for consistency with other modules.
	
	Hot-reload allows adding/removing coins from the GUI while the Thinker is running
	without requiring a process restart. The Thinker's main loop calls this periodically
	and uses _sync_coins_from_settings() to start/stop tracking coins dynamically.
	
	Returns:
		List of uppercased coin symbols to track (e.g., ['BTC', 'ETH', 'XRP'])
	"""
	debug_print("[DEBUG] _load_gui_coins() called")
	debug_print(f"[DEBUG] _GUI_SETTINGS_PATH = {_GUI_SETTINGS_PATH}")
	try:
		if not os.path.isfile(_GUI_SETTINGS_PATH):
			debug_print(f"[DEBUG] gui_settings.json not found, returning cached: {_gui_settings_cache['coins']}")
			return list(_gui_settings_cache["coins"])

		mtime = os.path.getmtime(_GUI_SETTINGS_PATH)
		if _gui_settings_cache["mtime"] == mtime:
			return list(_gui_settings_cache["coins"])

		debug_print(f"[DEBUG] Reading gui_settings.json")
		with open(_GUI_SETTINGS_PATH, "r", encoding="utf-8") as f:
			data = json.load(f) or {}
		debug_print(f"[DEBUG] JSON loaded successfully")

		# Handle new three-category structure or legacy list format
		coins_raw = data.get("coins", None)
		debug_print(f"[DEBUG] coins_raw type: {type(coins_raw)}, value: {coins_raw}")
		if isinstance(coins_raw, dict):
			# New format: combine all three categories
			all_coins = (
				coins_raw.get("active", []) + 
				coins_raw.get("accumulate", []) + 
				coins_raw.get("liquidate", [])
			)
			coins = [str(c).strip().upper() for c in all_coins if str(c).strip()]
		elif isinstance(coins_raw, list):
			# Legacy format: list of coins
			coins = [str(c).strip().upper() for c in coins_raw if str(c).strip()]
		else:
			# Fallback
			coins = list(_gui_settings_cache["coins"])
		
		if not coins:
			coins = list(_gui_settings_cache["coins"])

		# Remove any non-alphanumeric characters (including hidden Unicode)
		coins = [''.join(ch for ch in coin if ch.isalnum()) for coin in coins]
		coins = [c for c in coins if c]  # Remove any empty strings
		
		# Validate coin symbols for safety (prevent path traversal, ensure proper format)
		valid_coins = []
		debug_print(f"[DEBUG] Validating {len(coins)} coins: {coins}")
		for coin in coins:
			if _is_valid_coin_symbol(coin):
				valid_coins.append(coin)
				debug_print(f"[DEBUG] Coin {coin} passed validation")
			else:
				print(f"[SECURITY] Rejected invalid coin symbol: {repr(coin)}", flush=True)
				debug_print(f"[SECURITY] Rejected invalid coin symbol: {repr(coin)}")
		
		debug_print(f"[DEBUG] After validation: {len(valid_coins)} valid coins")
		if not valid_coins:
			print("[SECURITY] No valid coins after validation, using cached defaults", flush=True)
			debug_print("[SECURITY] No valid coins after validation, using cached defaults")
			valid_coins = list(_gui_settings_cache["coins"])

		_gui_settings_cache["mtime"] = mtime
		_gui_settings_cache["coins"] = valid_coins
		debug_print(f"[DEBUG] _load_gui_coins() returning: {valid_coins}")
		return list(valid_coins)
	except Exception as e:
		debug_print(f"[DEBUG] _load_gui_coins() exception: {e}")
		return list(_gui_settings_cache["coins"])

# Initial coin list (will be kept live via _sync_coins_from_settings())
COIN_SYMBOLS = _load_gui_coins()
CURRENT_COINS = list(COIN_SYMBOLS)
print(f"Loaded {len(CURRENT_COINS)} coins: {COIN_SYMBOLS}", flush=True)
if not CURRENT_COINS:
	print("WARNING: No coins loaded! Using fallback defaults", flush=True)
	CURRENT_COINS = ['BTC', 'ETH', 'SOL']
	COIN_SYMBOLS = list(CURRENT_COINS)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def coin_folder(sym: str) -> str:
	sym = sym.upper()
	# All coins use their own subfolder
	return os.path.join(BASE_DIR, sym)

class _CoinState:
	"""Container for managing the list of active coins without global mutation."""
	def __init__(self):
		self.coins = []
	
	def set_coins(self, coins_list):
		"""Update the coins list."""
		self.coins = coins_list
	
	def get_coins(self):
		"""Get the current coins list."""
		return self.coins
	
	def add_coin(self, coin):
		"""Add a coin if not already present."""
		if coin not in self.coins:
			self.coins.append(coin)


# Initialize coin_state after _CoinState class definition
coin_state = _CoinState()
coin_state.set_coins(CURRENT_COINS)


@contextmanager
def coin_directory(sym: str):
	"""
	Context manager for safely changing to a coin's directory.
	Ensures CWD is always restored even if an exception occurs.
	Creates the directory if it doesn't exist.
	"""
	debug_print(f"[DEBUG] coin_directory({sym}) - Entry")
	old_dir = os.getcwd()
	debug_print(f"[DEBUG] coin_directory({sym}) - Current dir: {old_dir}")
	try:
		folder = coin_folder(sym)
		debug_print(f"[DEBUG] coin_directory({sym}) - Target folder: {folder}")
		# Ensure folder exists before attempting to cd
		os.makedirs(folder, exist_ok=True)
		debug_print(f"[DEBUG] coin_directory({sym}) - About to chdir")
		os.chdir(folder)
		debug_print(f"[DEBUG] coin_directory({sym}) - Changed to {folder}, yielding")
		yield
		debug_print(f"[DEBUG] coin_directory({sym}) - Returning from yield")
	finally:
		debug_print(f"[DEBUG] coin_directory({sym}) - Finally block, restoring dir to {old_dir}")
		os.chdir(old_dir)


# Training freshness gate (mirrors pt_hub.py) - checks if coin training is current
_TRAINING_STALE_SECONDS = 14 * 24 * 60 * 60  # 14 days

def _is_coin_actively_training(sym: str) -> bool:
	"""Check if a coin is currently in TRAINING state (trainer process is running).
	
	Returns True if trainer_status.json shows state=TRAINING, False otherwise.
	Used to suppress incomplete training warnings for coins actively being trained.
	"""
	try:
		folder = coin_folder(sym)
		status_path = os.path.join(folder, "trainer_status.json")
		if os.path.isfile(status_path):
			with open(status_path, "r", encoding="utf-8") as f:
				status_data = json.load(f)
			if isinstance(status_data, dict):
				state = str(status_data.get("state", "")).upper()
				return state == "TRAINING"
	except Exception:
		pass
	return False

def _coin_is_trained(sym: str) -> bool:
	"""
	Training freshness gate:

	pt_trainer.py writes `trainer_last_training_time.txt` in the coin folder
	when training starts. If that file is missing OR older than 14 days, we treat
	the coin as NOT TRAINED.
	
	Also checks trainer_status.json - if state is TRAINING or ERROR, the coin
	is not considered trained yet (matches pt_hub.py behavior).

	This is intentionally the same logic as pt_hub.py so runner behavior matches
	what the GUI shows.
	"""

	try:
		folder = coin_folder(sym)
		
		# Check trainer_status.json first - if trainer is currently running or errored, not ready
		status_path = os.path.join(folder, "trainer_status.json")
		if os.path.isfile(status_path):
			try:
				with open(status_path, "r", encoding="utf-8") as f:
					status_data = json.load(f)
				if isinstance(status_data, dict):
					state = str(status_data.get("state", "")).upper()
					if state in ("TRAINING", "ERROR"):
						debug_print(f"[DEBUG] {sym}: Trainer status is {state}, not ready")
						return False
			except Exception:
				pass  # If status file is corrupt, continue to timestamp check
		
		# Check timestamp freshness
		stamp_path = os.path.join(folder, "trainer_last_training_time.txt")
		if not os.path.isfile(stamp_path):
			return False
		with open(stamp_path, "r", encoding="utf-8") as f:
			raw = (f.read() or "").strip()
		
		try:
			ts = float(raw) if raw else 0.0
		except ValueError:
			debug_print(f"[DEBUG] {sym}: trainer_last_training_time.txt contains invalid data: '{raw[:50]}'")
			return False
		
		if ts <= 0:
			return False
		
		# Timestamp is fresh - now validate that actual training files exist
		# This prevents processing a coin where training was interrupted or files deleted
		is_valid, missing_tfs = _validate_coin_training(sym)
		if not is_valid:
			debug_print(f"[DEBUG] {sym}: Training files incomplete despite fresh timestamp. Missing: {', '.join(missing_tfs)}")
			return False
		
		return (time.time() - ts) <= _TRAINING_STALE_SECONDS
	except OSError as e:
		debug_print(f"[DEBUG] {sym}: Failed to read trainer_last_training_time.txt: {e}")
		return False
	except Exception as e:
		debug_print(f"[DEBUG] {sym}: Unexpected error checking training status: {type(e).__name__}: {e}")
		return False

# GUI hub "runner ready" gate file read by gui_hub.py Autopilot toggle
HUB_DIR = os.environ.get("POWERTRADER_HUB_DIR") or os.path.join(BASE_DIR, "hub_data")
try:
	os.makedirs(HUB_DIR, exist_ok=True)
except Exception:
	pass

RUNNER_READY_PATH = os.path.join(HUB_DIR, "runner_ready.json")

class FileOperationTimeout(Exception):
	"""Raised when a file operation exceeds timeout."""
	pass

def _run_with_timeout(func, args: tuple, kwargs: dict, timeout_seconds: float):
	"""Run a function with timeout protection using threading (Windows-compatible).
	
	Args:
		func: Function to execute
		args: Positional arguments for function
		kwargs: Keyword arguments for function
		timeout_seconds: Maximum execution time in seconds
	
	Returns:
		Result from function if successful
	
	Raises:
		FileOperationTimeout: If function exceeds timeout
		Exception: Any exception raised by the function
	"""
	result = [None]
	exception = [None]
	
	def wrapper():
		try:
			result[0] = func(*args, **kwargs)
		except Exception as e:
			exception[0] = e
	
	thread = threading.Thread(target=wrapper, daemon=True)
	thread.start()
	thread.join(timeout_seconds)
	
	if thread.is_alive():
		# Thread still running after timeout
		raise FileOperationTimeout(f"File operation exceeded {timeout_seconds}s timeout")
	
	if exception[0]:
		raise exception[0]
	
	return result[0]

def _atomic_write_json_safe(path: str, data: dict, timeout: float = 5.0, max_retries: int = 3) -> None:
	"""Atomic write for JSON files with timeout and retry protection.
	
	Args:
		path: Target file path
		data: Dictionary to write as JSON
		timeout: Maximum seconds per write attempt (default 5.0)
		max_retries: Maximum retry attempts on timeout/error (default 3)
	
	Raises:
		FileOperationTimeout: If all retries exceed timeout
		Exception: If all retries fail with other errors
	"""
	for attempt in range(max_retries):
		try:
			# Use unique temp file name to avoid conflicts
			tmp = f"{path}.tmp.{os.getpid()}.{int(time.time() * 1000)}"
			
			def write_op():
				with open(tmp, "w", encoding="utf-8") as f:
					json.dump(data, f, indent=2)
				os.replace(tmp, path)
			
			_run_with_timeout(write_op, (), {}, timeout)
			return  # Success
			
		except FileOperationTimeout:
			debug_print(f"[WARNING] THINKER: JSON write timeout on attempt {attempt + 1}/{max_retries}: {path}")
			if attempt == max_retries - 1:
				debug_print(f"[ERROR] THINKER: All JSON write attempts timed out for: {path}")
				raise
			time.sleep(0.5 * (attempt + 1))  # Exponential backoff
			
		except Exception as e:
			debug_print(f"[WARNING] THINKER: JSON write error on attempt {attempt + 1}/{max_retries}: {e}")
			if attempt == max_retries - 1:
				debug_print(f"[ERROR] THINKER: All JSON write attempts failed for: {path}")
				raise
			time.sleep(0.5 * (attempt + 1))
		finally:
			# Clean up temp file if it exists
			try:
				if os.path.exists(tmp):
					os.remove(tmp)
			except:
				pass

def _atomic_write_text_safe(path: str, content: str, timeout: float = 5.0, max_retries: int = 3) -> None:
	"""Atomic write for text files with timeout and retry protection.
	
	Args:
		path: Target file path
		content: String content to write
		timeout: Maximum seconds per write attempt (default 5.0)
		max_retries: Maximum retry attempts on timeout/error (default 3)
	
	Raises:
		FileOperationTimeout: If all retries exceed timeout
		Exception: If all retries fail with other errors
	"""
	for attempt in range(max_retries):
		try:
			# Use unique temp file name to avoid conflicts
			tmp = f"{path}.tmp.{os.getpid()}.{int(time.time() * 1000)}"
			
			def write_op():
				with open(tmp, "w", encoding="utf-8") as f:
					f.write(content)
				os.replace(tmp, path)
			
			_run_with_timeout(write_op, (), {}, timeout)
			return  # Success
			
		except FileOperationTimeout:
			debug_print(f"[WARNING] THINKER: Text write timeout on attempt {attempt + 1}/{max_retries}: {path}")
			if attempt == max_retries - 1:
				debug_print(f"[ERROR] THINKER: All text write attempts timed out for: {path}")
				raise
			time.sleep(0.5 * (attempt + 1))  # Exponential backoff
			
		except Exception as e:
			debug_print(f"[WARNING] THINKER: Text write error on attempt {attempt + 1}/{max_retries}: {e}")
			if attempt == max_retries - 1:
				debug_print(f"[ERROR] THINKER: All text write attempts failed for: {path}")
				raise
			time.sleep(0.5 * (attempt + 1))
		finally:
			# Clean up temp file if it exists
			try:
				if os.path.exists(tmp):
					os.remove(tmp)
			except:
				pass

# Legacy atomic write functions (kept for backward compatibility, no timeout protection)
def _atomic_write_json(path: str, data: dict) -> None:
	try:
		tmp = path + ".tmp"
		with open(tmp, "w", encoding="utf-8") as f:
			json.dump(data, f, indent=2)
		os.replace(tmp, path)
	except Exception:
		pass

def _atomic_write_text(path: str, content: str) -> None:
	"""Atomic write for text files to prevent corruption on crash."""
	try:
		tmp = path + ".tmp"
		with open(tmp, "w", encoding="utf-8") as f:
			f.write(content)
		os.replace(tmp, path)
	except Exception:
		pass

def _write_priority_coin(coin: str, distance_pct: float, reason: str) -> None:
	"""Write priority coin information to hub_data/priority_coin.txt for auto-switch feature."""
	try:
		priority_path = os.path.join(HUB_DIR, "priority_coin.txt")
		data = {
			"coin": coin,
			"distance_pct": distance_pct,
			"reason": reason,
			"timestamp": time.time()
		}
		_atomic_write_json(priority_path, data)
	except Exception:
		pass  # Don't crash thinker if priority file write fails

def _write_runner_ready(ready: bool, stage: str, ready_coins=None, total_coins: int = 0) -> None:
	obj = {
		"timestamp": time.time(),
		"ready": bool(ready),
		"stage": stage,
		"ready_coins": ready_coins or [],
		"total_coins": int(total_coins or 0),
	}
	_atomic_write_json(RUNNER_READY_PATH, obj)

def _prune_removed_coins(valid_coins: list) -> None:
	"""Remove state for coins no longer in the active list to prevent memory bloat."""
	valid_set = set(valid_coins)
	
	# Prune global state dictionaries
	for coin in list(states.keys()):
		if coin not in valid_set:
			states.pop(coin, None)
	
	for coin in list(display_cache.keys()):
		if coin not in valid_set:
			display_cache.pop(coin, None)
	
	for coin in list(summary_cache.keys()):
		if coin not in valid_set:
			summary_cache.pop(coin, None)
	
	for coin in list(_last_printed_bounds_version.keys()):
		if coin not in valid_set:
			_last_printed_bounds_version.pop(coin, None)
	
	for coin in list(_last_written_bounds.keys()):
		if coin not in valid_set:
			_last_written_bounds.pop(coin, None)
	
	# Clear training data cache for removed coins
	for cache_key in list(_training_data_cache.keys()):
		if cache_key[0] not in valid_set:  # cache_key is (coin, timeframe)
			_training_data_cache.pop(cache_key, None)
	
	# Sets are already cleaned during coin sync, but ensure consistency
	_ready_coins.intersection_update(valid_set)
	_startup_messages_shown.intersection_update(valid_set)

def _safe_float_convert(value, field_name: str = "value", default: float = 0.0) -> float:
	"""Safely convert a value to float with validation and fallback."""
	try:
		result = float(value)
		# Check for NaN (NaN != NaN is True by IEEE 754 standard)
		if result != result:
			debug_print(f"[VALIDATION] Invalid float conversion for {field_name}: {value} (NaN detected)")
			return default
		return result
	except (ValueError, TypeError) as e:
		debug_print(f"[VALIDATION] Failed to convert {field_name} to float: {value} - {e}")
		return default

def _validate_kline_response(history: str, coin: str, timeframe: str, min_candles: int = 2) -> tuple:
	"""Validate KuCoin kline API response structure and content.
	
	Returns: (is_valid: bool, history_list: list, error_msg: str)
	"""
	if not history or not isinstance(history, str):
		return False, [], f"Empty or invalid response type for {coin} {timeframe}"
	
	try:
		history_list = history.split("], [")
		if len(history_list) < min_candles:
			return False, [], f"Insufficient candles ({len(history_list)} < {min_candles}) for {coin} {timeframe}"
		
		# Validate first candle structure to catch malformed data early
		if history_list:
			test_candle = str(history_list[0]).replace('"', '').replace("'", "").replace('[', '').split(", ")
			if len(test_candle) < 3:
				return False, [], f"Malformed candle data for {coin} {timeframe} (expected 3+ fields, got {len(test_candle)})"
		
		return True, history_list, ""
	except Exception as e:
		return False, [], f"Exception parsing kline data for {coin} {timeframe}: {e}"

def _validate_robinhood_response(data: dict, symbol: str) -> tuple:
	"""Validate Robinhood API response structure.
	
	Returns: (is_valid: bool, price: float, error_msg: str)
	"""
	try:
		if not data or not isinstance(data, dict):
			return False, 0.0, f"Invalid response type for {symbol}"
		
		if "results" not in data or not data["results"]:
			return False, 0.0, f"Missing or empty 'results' field for {symbol}"
		
		result = data["results"][0]
		if "ask_inclusive_of_buy_spread" not in result:
			return False, 0.0, f"Missing 'ask_inclusive_of_buy_spread' field for {symbol}"
		
		price = _safe_float_convert(result["ask_inclusive_of_buy_spread"], f"{symbol} ask price", 0.0)
		if price <= 0.0:
			return False, 0.0, f"Invalid price ({price}) for {symbol}"
		
		return True, price, ""
	except Exception as e:
		return False, 0.0, f"Exception validating Robinhood response for {symbol}: {e}"

# Performance optimization: String cleaning translation table (faster than multiple replace() calls)
_CLEAN_CHARS = str.maketrans('', '', '\'"[]')

def _clean_candle_string(s: str) -> list:
	"""Fast candle string parsing using translate() instead of multiple replace() calls."""
	return s.translate(_CLEAN_CHARS).split(", ")

def _parse_candle_time(history_list: list, index: int = 1) -> float:
	"""Extract timestamp from candle data with optimized string parsing."""
	try:
		return float(_clean_candle_string(str(history_list[index]))[0])
	except (IndexError, ValueError):
		return 0.0

# Cache for training data to avoid repeated file reads (cleared on trainer updates)
_training_data_cache = {}  # Key: (coin, timeframe, pattern_size), Value: (mtime, memories, weights, weights_high, weights_low, threshold)

def _load_training_data_cached(coin: str, timeframe: str, folder: str, pattern_size: int) -> tuple:
	"""Load training data with file modification time caching to avoid redundant reads.
	
	Args:
		coin: Coin symbol (e.g., 'BTC')
		timeframe: Timeframe (e.g., '1hour')
		folder: Folder path where memory files are stored
		pattern_size: Pattern size to load (e.g., 2, 3, 4, 5)
	
	Returns: (memories_list, weight_list, high_weight_list, low_weight_list, perfect_threshold)
	Returns (None, None, None, None, None) if files missing or invalid.
	"""
	cache_key = (coin, timeframe, pattern_size)
	
	debug_print(f"[DEBUG] _load_training_data_cached({coin}, {timeframe}, p{pattern_size}): folder={folder}")
	
	# Build file paths with size-specific naming
	memories_file = os.path.join(folder, f'memories_{timeframe}_p{pattern_size}.dat')
	weights_file = os.path.join(folder, f'memory_weights_{timeframe}_p{pattern_size}.dat')
	weights_high_file = os.path.join(folder, f'memory_weights_high_{timeframe}_p{pattern_size}.dat')
	weights_low_file = os.path.join(folder, f'memory_weights_low_{timeframe}_p{pattern_size}.dat')
	threshold_file = os.path.join(folder, f'neural_perfect_threshold_{timeframe}_p{pattern_size}.dat')
	
	# Check if all files exist - debug which files are missing
	all_files = [memories_file, weights_file, weights_high_file, weights_low_file, threshold_file]
	missing_files = [f for f in all_files if not os.path.isfile(f)]
	if missing_files:
		debug_print(f"[DEBUG] {coin} {timeframe} p{pattern_size}: Missing files: {[os.path.basename(f) for f in missing_files]}")
		debug_print(f"[DEBUG] {coin} {timeframe} p{pattern_size}: Search folder: {folder}")
		return None, None, None, None, None, None
	
	try:
		# Get latest modification time across all files
		mtimes = [os.path.getmtime(f) for f in [memories_file, weights_file, weights_high_file, weights_low_file, threshold_file]]
		latest_mtime = max(mtimes)
		
		# Check cache validity
		if cache_key in _training_data_cache:
			cached_mtime, cached_data = _training_data_cache[cache_key]
			if cached_mtime >= latest_mtime:
				return cached_data
		
		# Load from disk (cache miss or stale)
		# memories split by '~', weights split by ' '
		with open(memories_file, 'r') as f:
			memory_list = f.read().translate(str.maketrans('', '', '\'"[],')).split('~')
		
		with open(weights_file, 'r') as f:
			weight_list = f.read().translate(str.maketrans('', '', '\'"[],')).split(' ')
		
		with open(weights_high_file, 'r') as f:
			high_weight_list = f.read().translate(str.maketrans('', '', '\'"[],')).split(' ')
		
		with open(weights_low_file, 'r') as f:
			low_weight_list = f.read().translate(str.maketrans('', '', '\'"[],')).split(' ')
		
		with open(threshold_file, 'r') as f:
			perfect_threshold = _safe_float_convert(f.read(), f"{coin}_{timeframe}_threshold", DEFAULT_THRESHOLD_FALLBACK)
		
		# Load volatility (default if file missing)
		volatility_file = os.path.join(folder, f"volatility_{timeframe}_p{pattern_size}.dat")
		volatility = DEFAULT_VOLATILITY
		if os.path.exists(volatility_file):
			with open(volatility_file, 'r') as f:
				volatility = _safe_float_convert(f.read(), f"{coin}_{timeframe}_volatility", DEFAULT_VOLATILITY)
		
		# Cache the data
		cached_data = (memory_list, weight_list, high_weight_list, low_weight_list, perfect_threshold, volatility)
		_training_data_cache[cache_key] = (latest_mtime, cached_data)
		
		return cached_data
	except Exception as e:
		debug_print(f"[DEBUG] {coin} {timeframe} p{pattern_size}: Error loading training data: {type(e).__name__}: {e}")
		return None, None, None, None, None, None

# Rate limiter to prevent API overload and respect provider limits
class APIRateLimiter:
	"""Rate limiter for API calls to prevent overload and respect provider limits.
	
	Tracks last call time for each API endpoint and enforces minimum intervals.
	KuCoin: ~10 requests/sec per IP (we use 0.15s = ~6.7 req/s for safety margin)
	Robinhood: Conservative rate (0.5s between calls to avoid aggressive limiting)
	"""
	def __init__(self):
		self._last_call_times = {}  # Key: api_name, Value: timestamp
		self._min_intervals = {
			'kucoin': KUCOIN_API_RATE_LIMIT,      # 150ms between KuCoin calls (~6.7 req/s)
			'robinhood': ROBINHOOD_API_RATE_LIMIT,    # 500ms between Robinhood calls (conservative)
		}
	
	def wait_if_needed(self, api_name: str) -> float:
		"""Wait if needed to respect rate limit. Returns seconds waited."""
		if api_name not in self._min_intervals:
			return 0.0
		
		min_interval = self._min_intervals[api_name]
		last_call = self._last_call_times.get(api_name, 0.0)
		now = time.time()
		elapsed = now - last_call
		
		if elapsed < min_interval:
			wait_time = min_interval - elapsed
			time.sleep(wait_time)
			self._last_call_times[api_name] = time.time()
			return wait_time
		else:
			self._last_call_times[api_name] = now
			return 0.0
	
	def record_call(self, api_name: str):
		"""Record an API call timestamp without waiting (for manual rate limit control)."""
		self._last_call_times[api_name] = time.time()

# Global rate limiter instance
_rate_limiter = APIRateLimiter()

# Ensure folders exist for the current configured coins so signal files can be written
debug_print(f"[DEBUG] Creating folders for coins: {CURRENT_COINS}")
for _sym in CURRENT_COINS:
	debug_print(f"[DEBUG] Creating folder for: {_sym}")
	os.makedirs(coin_folder(_sym), exist_ok=True)
debug_print(f"[DEBUG] Folder creation complete")

# Required timeframes that the Thinker uses for multi-timeframe predictions and neural level signals.
# All seven timeframes MUST be trained before the Thinker can produce valid trading signals. The
# Thinker generates predicted high/low for each timeframe, then combines them to determine trade
# entry signals (when price drops below 3+ predicted lows) and DCA levels (neural levels 4-7).
# Missing any timeframe causes the Thinker to block signal generation for that coin.
REQUIRED_THINKER_TIMEFRAMES = [
	'1hour', '2hour', '4hour', '8hour', '12hour', '1day', '1week'
]

distance = DEFAULT_DISTANCE
tf_choices = REQUIRED_THINKER_TIMEFRAMES

# Load pattern_size from training_settings.json
try:
	training_settings_path = os.path.join(os.path.dirname(__file__), "training_settings.json")
	with open(training_settings_path, 'r') as f:
		training_settings = json.load(f)
		PATTERN_SIZE = training_settings["pattern_size"]
		ENABLE_PATTERN_FALLBACK = training_settings.get("enable_pattern_fallback", True)
		min_threshold = training_settings.get("min_threshold", 5.0)
		max_threshold = training_settings.get("max_threshold", 25.0)
		DEFAULT_THRESHOLD_FALLBACK = (min_threshold + max_threshold) / 2.0  # Midpoint of user's threshold range
		
		# Load threshold enforcement setting and calculate effective multipliers
		threshold_enforcement = training_settings.get("threshold_enforcement", "Balanced")
		enforcement_multipliers = {"Tight": 1, "Balanced": 2, "Loose": 5, "None": 10}
		enforcement_value = enforcement_multipliers.get(threshold_enforcement, 2)  # Default to Balanced (2)
		
		# Calculate effective multipliers (base × enforcement)
		MAX_CUTOFF_MULTIPLIER = BASE_MAX_CUTOFF_MULTIPLIER * enforcement_value
		PERFECT_MATCH_MULTIPLIER = BASE_PERFECT_MATCH_MULTIPLIER * enforcement_value
		KERNEL_BANDWIDTH = BASE_KERNEL_BANDWIDTH * enforcement_value
		
		print(f"Loaded pattern_size={PATTERN_SIZE}, enable_pattern_fallback={ENABLE_PATTERN_FALLBACK}, threshold_enforcement={threshold_enforcement} (cutoff={MAX_CUTOFF_MULTIPLIER:.2f}x, perfect={PERFECT_MATCH_MULTIPLIER:.2f}x, kernel={KERNEL_BANDWIDTH:.2f}x) from training_settings.json", flush=True)
		debug_print(f"[DEBUG] Training settings loaded successfully")
except Exception as e:
	# Print error immediately (before any other initialization) so Hub can capture it
	print(f"FATAL ERROR: Failed to load training_settings.json: {e}", flush=True)
	print(f"Current working directory: {os.getcwd()}", flush=True)
	print(f"Script location: {os.path.dirname(__file__)}", flush=True)
	print(f"Expected path: {training_settings_path}", flush=True)
	sys.exit(1)

# Load trading config for signal thresholds
_TRADING_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "trading_settings.json")
_trading_config_cache = {"mtime": 0, "config": {"entry_signals": {"long_signal_min": 4, "short_signal_max": 0}}}

def _load_trading_config() -> dict:
	"""Load trading_settings.json with mtime caching"""
	try:
		if not os.path.isfile(_TRADING_SETTINGS_PATH):
			return dict(_trading_config_cache["config"])
		mtime = os.path.getmtime(_TRADING_SETTINGS_PATH)
		if _trading_config_cache["mtime"] == mtime:
			return dict(_trading_config_cache["config"])
		with open(_TRADING_SETTINGS_PATH, "r", encoding="utf-8") as f:
			data = json.load(f) or {}
		config = dict(_trading_config_cache["config"])
		for key in data:
			if isinstance(data[key], dict) and isinstance(config.get(key), dict):
				config[key].update(data[key])
			else:
				config[key] = data[key]
		_trading_config_cache["mtime"] = mtime
		_trading_config_cache["config"] = config
		return dict(config)
	except Exception:
		return dict(_trading_config_cache["config"])

def new_coin_state():
	"""Create initial state dictionary for a newly tracked coin.
	
	Initializes all per-timeframe arrays with placeholder values (0.0 for lows, inf for highs)
	and sets up tracking fields for prediction updates, message generation, and readiness gating.
	
	The bounds_version counter tracks when we've updated from placeholder values to real predictions.
	This prevents the Thinker from signaling "ready" to the Hub until actual predictions replace
	the startup placeholders. Only after bounds_version increments (real data loaded) does the
	Thinker start producing LONG/SHORT/WITHIN messages for the trader.
	
	Returns:
		Dict with per-timeframe arrays and state tracking fields
	"""
	return {
		# Predicted price levels: low_bound = predicted lows (blue lines), high_bound = predicted highs (orange lines)
		'low_bound_prices': [0.0] * len(tf_choices),
		'high_bound_prices': [float('inf')] * len(tf_choices),

		'tf_times': [],  # Target update times for each timeframe
		'tf_choice_index': 0,  # Current timeframe being processed in the step loop

		'tf_update': ['yes'] * len(tf_choices),
		'messages': ['none'] * len(tf_choices),
		'last_messages': ['none'] * len(tf_choices),
		'margins': [MIN_PROFIT_MARGIN] * len(tf_choices),

		'high_tf_prices': [float('inf')] * len(tf_choices),
		'low_tf_prices': [0.0] * len(tf_choices),

		'tf_sides': ['none'] * len(tf_choices),
		'messaged': ['no'] * len(tf_choices),
		'updated': [0] * len(tf_choices),
		'perfects': ['active'] * len(tf_choices),
		'match_qualities': [0.0] * len(tf_choices),  # Match quality percentage (200 = perfect, 100 = threshold, <50 = weak)
		'training_issues': [0] * len(tf_choices),

		# readiness gating (no placeholder-number checks; this is process-based)
		'bounds_version': 0,
		'last_display_bounds_version': -1,

		# Prediction candles for Hub chart visualization (accumulates across timeframes)
		'prediction_candles': {},

	}

states = {}

display_cache = {sym: f"[{sym}] Starting...\n[{sym}] Initializing predictions for all timeframes" for sym in CURRENT_COINS}
summary_cache = {}  # Stores compact summary data for each coin
_last_printed_bounds_version = {}  # Track per-coin to avoid printing duplicate displays
_last_written_bounds = {}  # Track last written bounds to avoid redundant file writes

def _validate_coin_training(coin: str) -> tuple:
	"""Validate that a coin has all required timeframe data files present and non-empty.
	
	Checks for existence of memories, weights (base, high, low), and threshold files for each
	required timeframe. All five files must exist and be at least 10 bytes (non-empty) for the
	timeframe to be considered trained. This validation runs before the Thinker attempts to load
	predictions, preventing crashes from missing or corrupted training data.
	
	Called by both the Thinker (to check training freshness) and the Hub (to show training
	status in the GUI). Missing files indicate the Trainer hasn't run or training was interrupted.
	
	Args:
		coin: Coin symbol (e.g., "BTC")
		
	Returns:
		Tuple of (is_valid: bool, missing_timeframes: list). is_valid is True only if ALL
		required timeframes are present. missing_timeframes lists any incomplete timeframes.
	"""
	missing = []
	folder = coin_folder(coin)
	
	debug_print(f"[DEBUG] THINKER: Validating {coin} training in folder: {folder}")
	debug_print(f"[DEBUG] THINKER: Current working directory: {os.getcwd()}")
	
	if not os.path.isdir(folder):
		debug_print(f"[DEBUG] THINKER: Folder does not exist: {folder}")
		return (False, REQUIRED_THINKER_TIMEFRAMES)
	
	# Check for required pattern sizes based on user setting
	# If fallback enabled: at least ONE size from 2→PATTERN_SIZE must exist (cascade 5→4→3→2→INACTIVE)
	# If fallback disabled: only PATTERN_SIZE must exist (legacy single-size mode)
	for tf in REQUIRED_THINKER_TIMEFRAMES:
		found_any_size = False
		
		# Determine which pattern sizes to validate based on user setting
		if ENABLE_PATTERN_FALLBACK:
			pattern_sizes_to_check = range(2, PATTERN_SIZE + 1)  # Check all sizes (cascading fallback mode)
		else:
			pattern_sizes_to_check = [PATTERN_SIZE]  # Only check configured size (legacy mode)
		
		for pattern_size in pattern_sizes_to_check:
			memory_file = os.path.join(folder, f"memories_{tf}_p{pattern_size}.dat")
			weight_file = os.path.join(folder, f"memory_weights_{tf}_p{pattern_size}.dat")
			weight_high_file = os.path.join(folder, f"memory_weights_high_{tf}_p{pattern_size}.dat")
			weight_low_file = os.path.join(folder, f"memory_weights_low_{tf}_p{pattern_size}.dat")
			threshold_file = os.path.join(folder, f"neural_perfect_threshold_{tf}_p{pattern_size}.dat")
			
			# Check if all required files exist and are non-empty for this size
			# Threshold files only contain a single number (can be 4 bytes), so check for 1+ bytes
			# Memory/weight files need at least 10 bytes
			files_to_check = [
				(memory_file, 10),
				(weight_file, 10),
				(weight_high_file, 10),
				(weight_low_file, 10),
				(threshold_file, 1)
			]
			
			size_complete = True
			for fpath, min_size in files_to_check:
				if not os.path.isfile(fpath):
					debug_print(f"[DEBUG] THINKER: {coin} {tf} size {pattern_size} - File not found: {fpath}")
					size_complete = False
					break
				try:
					fsize = os.path.getsize(fpath)
					if fsize < min_size:
						debug_print(f"[DEBUG] THINKER: {coin} {tf} size {pattern_size} - File too small ({fsize} bytes, need {min_size}): {fpath}")
						size_complete = False
						break
				except Exception as e:
					debug_print(f"[DEBUG] THINKER: {coin} {tf} size {pattern_size} - Error checking file size: {fpath}, error: {e}")
					size_complete = False
					break
			
			if size_complete:
				found_any_size = True
				debug_print(f"[DEBUG] THINKER: {coin} {tf} size {pattern_size} - Complete")
		
		# If no pattern sizes exist for this timeframe, mark as missing
		if not found_any_size:
			debug_print(f"[DEBUG] THINKER: {coin} {tf} - No complete pattern sizes found (need at least one size from 2-{PATTERN_SIZE})")
			if tf not in missing:
				missing.append(tf)
	
	if len(missing) == 0:
		debug_print(f"[DEBUG] THINKER: {coin} validation passed - all timeframes complete")
	else:
		debug_print(f"[DEBUG] THINKER: {coin} validation failed - missing: {', '.join(missing)}")
	
	return (len(missing) == 0, missing)

# Track which coins have produced REAL predicted levels (not placeholder 0.0 / inf)
_ready_coins = set()

# Track which coins have already shown their "Starting..." message (prevent repetitive output)
_startup_messages_shown = set()

# Readiness detection: the runner is considered "READY" only when it's producing real prediction
# messages (WITHIN/LONG/SHORT) instead of placeholder messages (INACTIVE/Starting). This ensures
# the Hub doesn't enable Autopilot until predictions are based on actual trained data, not startup
# placeholders. Prevents premature trading signals that could execute on invalid prediction data.
def _is_printing_real_predictions(messages) -> bool:
	"""Check if message list contains real prediction output (not placeholders).
	
	Examines the current message array for each timeframe and returns True if ANY timeframe
	is producing real prediction messages (WITHIN/LONG/SHORT). These messages indicate the
	Thinker has loaded trained data and is generating actionable predictions. INACTIVE messages
	mean the timeframe hasn't produced predictions yet (still initializing or training incomplete).
	
	Used by the readiness gate to determine when to signal the Hub that trading can begin.
	
	Args:
		messages: List of message strings from state['messages']
		
	Returns:
		True if any message is a real prediction type, False if all are placeholders
	"""
	try:
		for m in (messages or []):
			if not isinstance(m, str):
				continue
			# These are the only message types produced once predictions are being used in output.
			# (INACTIVE means it's still not printing real prediction output for that timeframe.)
			if m.startswith("WITHIN") or m.startswith("LONG") or m.startswith("SHORT"):
				return True
		return False
	except Exception:
		return False

def _sync_coins_from_settings():
	"""
	Hot-reload coin list from gui_settings.json and dynamically start/stop coin tracking.
	
	Called periodically by the main runner loop to detect coin list changes without requiring
	a process restart. Enables adding new coins (creates their folder, initializes state, and
	starts generating predictions) or removing coins (stops processing them but preserves their
	data files for potential re-addition later).
	
	Adding a coin:
		1. Creates coin subfolder (BTC/, ETH/, etc.) if it doesn't exist
		2. Initializes state dict via new_coin_state()
		3. Adds to active processing in the step loop
		4. Starts generating predictions on next cycle (assuming training data exists)
	
	Removing a coin:
		1. Removes from active state dict (stops processing)
		2. Leaves training data files intact on disk
		3. Stops writing signal files for that coin
		4. Can be re-added later without retraining
	"""
	global CURRENT_COINS
	global COIN_SYMBOLS
	
	new_list = _load_gui_coins()
	if new_list == CURRENT_COINS:
		return

	old_list = list(CURRENT_COINS)
	added = [c for c in new_list if c not in old_list]
	removed = [c for c in old_list if c not in new_list]

	# Handle removed coins: stop stepping + clear UI cache entries
	for sym in removed:
		try:
			_ready_coins.discard(sym)
		except Exception:
			pass  # Best-effort cleanup; OK if already removed
		try:
			_startup_messages_shown.discard(sym)
		except Exception:
			pass  # Best-effort cleanup; OK if already removed
		try:
			display_cache.pop(sym, None)
		except Exception:
			pass  # Best-effort cleanup; OK if not in cache
		try:
			# Clean up state dict to prevent memory leak
			states.pop(sym, None)
		except Exception:
			pass  # Best-effort cleanup; OK if not in states

	# Handle added coins: create folder + init state + show in UI output
	for sym in added:
		try:
			os.makedirs(coin_folder(sym), exist_ok=True)
		except Exception:
			pass  # Best-effort; folder may already exist
		try:
			display_cache[sym] = f"[{sym}] Starting...\n[{sym}] Initializing predictions for all timeframes"
		except Exception:
			pass  # Best-effort UI update; non-critical
		try:
			# init_coin uses coin_directory() context manager, so CWD is auto-restored
			init_coin(sym)
		except Exception:
			pass  # Coin will be skipped this cycle; init on next pass

	coin_state.set_coins( list(new_list))
	CURRENT_COINS = coin_state.get_coins()
	COIN_SYMBOLS = list(CURRENT_COINS)
	
	# Prune memory for any coins no longer in the active list
	_prune_removed_coins(COIN_SYMBOLS)

_write_runner_ready(False, stage="starting", ready_coins=[], total_coins=len(CURRENT_COINS))

def init_coin(sym: str):
	debug_print(f"[DEBUG] init_coin({sym}) - Entry point reached")
	try:
		debug_print(f"[DEBUG] init_coin({sym}) - About to enter coin_directory context")
	except Exception as e:
		print(f"[ERROR] Failed to print debug message: {e}", flush=True)
	with coin_directory(sym):
		debug_print(f"[DEBUG] Changed to coin directory for {sym}")

		# per-coin "version" + on/off files (no collisions between coins)
		_atomic_write_text('alerts_version.dat', '5/3/2022/9am')
		_atomic_write_text('futures_long_onoff.dat', 'OFF')
		_atomic_write_text('futures_short_onoff.dat', 'OFF')

		st = new_coin_state()

		coin = sym + '-USDT'
		debug_print(f"[DEBUG] Fetching history for {coin}, starting with timeframe index 0")
		ind = 0
		tf_times_local = []
		while True:
			history_list = []
			init_retry_count = 0
			init_max_retries = 10
			while True:
				try:
					# Fetch klines for `coin` at the configured timeframe index.
					# The KuCoin client sometimes returns nested lists without a
					# trailing comma; replace() calls normalize the string so the
					# subsequent split() yields consistent elements.
					debug_print(f"[DEBUG] Fetching kline: coin={coin}, tf={tf_choices[ind]}, attempt {init_retry_count + 1}")
					_rate_limiter.wait_if_needed('kucoin')
					debug_print(f"[DEBUG] About to call market.get_kline()")
					history = str(market.get_kline(coin, tf_choices[ind])).replace(']]', '], ').replace('[[', '[')
					debug_print(f"[DEBUG] Received kline data, validating...")
					is_valid, history_list, error_msg = _validate_kline_response(history, coin, tf_choices[ind], 2)
					if not is_valid:
						print(f"[VALIDATION ERROR] {error_msg}", flush=True)
						debug_print(f"[VALIDATION] {error_msg}")
						raise ValueError(error_msg)
					debug_print(f"[DEBUG] Kline validation passed for {coin} {tf_choices[ind]}")
					break
				except Exception as e:
					init_retry_count += 1
					print(f"[ERROR] Exception during kline fetch: {type(e).__name__}: {str(e)[:200]}", flush=True)
					if init_retry_count >= init_max_retries:
						debug_print(f"[DEBUG] {sym}: init_coin failed after {init_max_retries} retries on timeframe {tf_choices[ind]}")
						print(f"WARNING: Failed to initialize {sym} after {init_max_retries} retries - skipping this coin", flush=True)
						print(f"Last error: {type(e).__name__}: {str(e)}", flush=True)
						raise  # Re-raise to be caught by outer try/except
					debug_print(f"[DEBUG] Retry {init_retry_count}/{init_max_retries}, sleeping {_get_sleep_timing('sleep_api_retry')}s")
					time.sleep(_get_sleep_timing("sleep_api_retry"))
					if 'Requests' in str(e):
						pass
					else:
						PrintException()
					continue

			ind += 1
			the_time = _parse_candle_time(history_list, 1)

			tf_times_local.append(the_time)
			if len(tf_times_local) >= len(tf_choices):
				break

		st['tf_times'] = tf_times_local
		states[sym] = st

# init all coins once (from GUI settings)
print(f"\nInitializing {len(CURRENT_COINS)} coins: {CURRENT_COINS}", flush=True)
for _sym in CURRENT_COINS:
	try:
		print(f"Initializing coin: {_sym}", flush=True)
		init_coin(_sym)
		print(f"Successfully initialized: {_sym}", flush=True)
	except Exception as e:
		print(f"WARNING: Failed to initialize {_sym}: {e}", flush=True)
		print(f"Continuing with remaining coins...", flush=True)

# restore CWD to base after init
os.chdir(BASE_DIR)

wallet_addr_list = []
wallet_addr_users = []
total_long = 0
total_short = 0
last_hour = 565457457357

cc_index = 0
tf_choice = []
prices = []
starts = []
long_start_prices = []
short_start_prices = []
buy_coins = []
cc_update = 'yes'
wr_update = 'yes'

def find_purple_area(lines):
    """
    Given a list of (price, color) pairs (color is 'orange' or 'blue'),
    returns (purple_bottom, purple_top) if a purple area exists,
    else (None, None).
    """
    oranges = sorted([price for price, color in lines if color == 'orange'], reverse=True)
    blues   = sorted([price for price, color in lines if color == 'blue'])
    if not oranges or not blues:
        return (None, None)
    purple_bottom = None
    purple_top = None
    all_levels = sorted(set(oranges + blues + [float('-inf'), float('inf')]), reverse=True)
    for i in range(len(all_levels) - 1):
        top = all_levels[i]
        bottom = all_levels[i+1]
        oranges_below = [o for o in oranges if o < bottom]
        blues_above = [b for b in blues if b > top]
        has_orange_below = any(o < top for o in oranges)
        has_blue_above = any(b > bottom for b in blues)
        if has_orange_below and has_blue_above:
            if purple_bottom is None or bottom < purple_bottom:
                purple_bottom = bottom
            if purple_top is None or top > purple_top:
                purple_top = top
    if purple_bottom is not None and purple_top is not None and purple_top > purple_bottom:
        return (purple_bottom, purple_top)
    return (None, None)
def step_coin(sym: str):
	"""Execute one prediction cycle for a single coin across all timeframes.
	
	This is the core prediction engine called periodically for each tracked coin. Each cycle:
	1. Validates training freshness (skips if training > 14 days old)
	2. Fetches latest candle data from KuCoin for each timeframe
	3. Loads trained memories/weights from disk
	4. Generates weighted-average predictions for current candle high/low
	5. Writes predicted levels to signal files (low_bound_prices.dat, high_bound_prices.dat)
	6. Generates trading signals based on predicted levels vs current price
	7. Writes long/short DCA signals for trader to consume
	
	Training freshness gate: Prevents trading on stale predictions by blocking signal generation
	when training data is >14 days old. Forces retraining before allowing new trades to start.
	Sets all signals to OFF and profit margins to baseline to safely hold existing positions.
	
	Timeframe rotation: Steps through all 7 timeframes in sequence, updating one per cycle to
	distribute API calls over time and avoid rate limits. Full update cycle completes every
	7 iterations (one per timeframe).
	
	Error handling: Retries failed API calls with exponential backoff. After max retries, calls
	handle_network_error() which logs the issue and exits (trading requires reliable data).
	
	Args:
		sym: Coin symbol (e.g., "BTC") - function changes to coin folder internally
	"""
	debug_print(f"[DEBUG] {sym}: step_coin() called")
	
	# Load state before training check so we can persist it properly
	st = states[sym]
	
	# Training freshness gate - check before processing predictions. If GUI would show
	# NOT TRAINED (missing or stale trainer_last_training_time.txt), skip this coin so
	# no new trades can start until it is trained again.
	if not _coin_is_trained(sym):
		with coin_directory(sym):
			try:
				# Prevent new trades (and DCA) by forcing signals to 0 and keeping PM at baseline.
				# Use timeout-protected writes to prevent hangs on disk issues
				_atomic_write_text_safe('futures_long_profit_margin.dat', str(MIN_PROFIT_MARGIN), timeout=3.0)
				_atomic_write_text_safe('futures_short_profit_margin.dat', str(MIN_PROFIT_MARGIN), timeout=3.0)
				_atomic_write_text_safe('long_dca_signal.dat', '0', timeout=3.0)
				_atomic_write_text_safe('short_dca_signal.dat', '0', timeout=3.0)
			except FileOperationTimeout as e:
				debug_print(f"[ERROR] THINKER: Timeout writing untrained status for {sym}: {e}")
			except Exception as e:
				debug_print(f"[ERROR] THINKER: Failed writing untrained status for {sym}: {e}")
		try:
			display_cache[sym] = (
				f"[{sym}] NOT TRAINED / OUTDATED\n"
				f"[{sym}] WARNING: Training data is missing or stale (>14 days old)\n"
				f"[{sym}] Please run the trainer before starting predictions"
			)
		except Exception:
			pass
		try:
			_ready_coins.discard(sym)
			all_ready = len(_ready_coins) > 0  # At least one coin trained
			_write_runner_ready(
				all_ready,
				stage=("real_predictions" if all_ready else "training_required"),
				ready_coins=sorted(list(_ready_coins)),
				total_coins=len(CURRENT_COINS),
			)
		except Exception:
			pass  # Best-effort readiness update; non-critical
		
		# Persist state even when untrained to maintain consistency
		states[sym] = st
		return

	# Process predictions for trained coins
	with coin_directory(sym):
		coin = sym + '-USDT'

		# ensure new readiness-version keys exist even if restarting from an older state dict
		if 'bounds_version' not in st:
			st['bounds_version'] = 0
		if 'last_display_bounds_version' not in st:
			st['last_display_bounds_version'] = -1
		tf_times = st['tf_times']
		tf_choice_index = st['tf_choice_index']

		# Copy arrays from state to prevent in-place modifications from accumulating
		tf_update = st['tf_update']
		messages = st['messages']
		last_messages = st['last_messages']
		margins = st['margins']

		# Load bound prices from state and pad immediately to ensure correct length
		low_bound_prices = st.get('low_bound_prices', [0.0] * len(tf_choices))
		high_bound_prices = st.get('high_bound_prices', [float('inf')] * len(tf_choices))
		
		# Pad/trim bound prices to match tf_choices length right away
		if len(low_bound_prices) < len(tf_choices):
			low_bound_prices.extend([0.0] * (len(tf_choices) - len(low_bound_prices)))
		elif len(low_bound_prices) > len(tf_choices):
			del low_bound_prices[len(tf_choices):]
		
		if len(high_bound_prices) < len(tf_choices):
			high_bound_prices.extend([float('inf')] * (len(tf_choices) - len(high_bound_prices)))
		elif len(high_bound_prices) > len(tf_choices):
			del high_bound_prices[len(tf_choices):]

		high_tf_prices = st['high_tf_prices']
		low_tf_prices = st['low_tf_prices']
		tf_sides = st['tf_sides']
		messaged = st['messaged']
		updated = st['updated']
		perfects = st['perfects']
		# Copy arrays from state to prevent in-place modifications
		match_qualities = st.get('match_qualities', [100.0] * len(tf_choices)) if st.get('match_qualities') else [100.0] * len(tf_choices)
		training_issues = st.get('training_issues', [0] * len(tf_choices)) if st.get('training_issues') else [0] * len(tf_choices)
		# keep training_issues aligned to tf_choices
		if len(training_issues) < len(tf_choices):
			training_issues.extend([0] * (len(tf_choices) - len(training_issues)))
		elif len(training_issues) > len(tf_choices):
			del training_issues[len(tf_choices):]
		
		pattern_sizes_used = st.get('pattern_sizes_used', [0] * len(tf_choices)) if st.get('pattern_sizes_used') else [0] * len(tf_choices)
		# keep pattern_sizes_used aligned to tf_choices
		if len(pattern_sizes_used) < len(tf_choices):
			pattern_sizes_used.extend([0] * (len(tf_choices) - len(pattern_sizes_used)))
		elif len(pattern_sizes_used) > len(tf_choices):
			del pattern_sizes_used[len(tf_choices):]
		
		# Load prediction candles dictionary from state (persists across timeframe cycles)
		if 'prediction_candles' not in st:
			st['prediction_candles'] = {}
		prediction_candles = st['prediction_candles']

		# Fetch current pattern (multiple candles for pattern matching)
		debug_print(f"[DEBUG] {sym}: Fetching market data for timeframe {tf_choices[tf_choice_index]} (pattern_size={PATTERN_SIZE} → {PATTERN_SIZE-1} pct_changes)...")
		kucoin_api_retry_count = 0
		kucoin_empty_data_count = 0
		kucoin_max_retries = 5
		while True:
			history_list = []
			while True:
				try:
					_rate_limiter.wait_if_needed('kucoin')
					history = str(market.get_kline(coin, tf_choices[tf_choice_index])).replace(']]', '], ').replace('[[', '[')
					is_valid, history_list, error_msg = _validate_kline_response(history, coin, tf_choices[tf_choice_index], PATTERN_SIZE)
					if not is_valid:
						debug_print(f"[VALIDATION] {error_msg}")
						raise ValueError(error_msg)
					break
				except Exception as e:
					kucoin_api_retry_count += 1
					debug_print(f"[DEBUG] {sym}: KuCoin API error - {type(e).__name__}: {str(e)[:100]} - retrying in 3.5s...")
					if kucoin_api_retry_count >= kucoin_max_retries:
						handle_network_error(f"KuCoin market data fetch for {sym} ({tf_choices[tf_choice_index]})", e)
					time.sleep(_get_sleep_timing("sleep_api_retry"))
					if 'Requests' in str(e):
						pass
					else:
						pass
					continue
			# KuCoin can occasionally return an empty/short kline response.
			# Need at least PATTERN_SIZE candles for pattern matching
			if len(history_list) < PATTERN_SIZE:
				kucoin_empty_data_count += 1
				if kucoin_empty_data_count >= kucoin_max_retries:
					debug_print(f"[DEBUG] {sym}: KuCoin returned insufficient data after {kucoin_max_retries} attempts (need {PATTERN_SIZE} candles to build pattern)")
					# Skip this cycle rather than crashing
					st['tf_choice_index'] = tf_choice_index
					states[sym] = st
					return
				time.sleep(_get_sleep_timing("sleep_data_validation_retry"))
				continue
			
			# Fetch last PATTERN_SIZE candles and convert to percentage changes
			# For pattern_size=5, we need 5 candles to create 4 percentage changes (close-to-close)
			# The trainer creates patterns with (pattern_size - 1) percentage changes
			# Pattern: [candle0_close, candle1_close, ...] → [pct_change1, pct_change2, ...]
			current_pattern = []
			temp_close_prices = []
			try:
				# First, collect all close prices
				for i in range(PATTERN_SIZE):
					working_candle = _clean_candle_string(str(history_list[i]))
					if len(working_candle) < 3:
						raise ValueError(f"Candle {i} has insufficient fields: {len(working_candle)}")
					closePrice = _safe_float_convert(working_candle[2], f"candle[{i}].close", 0.0)
					if closePrice == 0:
						raise ValueError(f"Zero closePrice at candle index {i}")
					temp_close_prices.append(closePrice)
				
				# Convert to close-to-close percentage changes (matching trainer logic)
				for i in range(1, len(temp_close_prices)):
					if temp_close_prices[i-1] != 0:
						pct_change = 100 * ((temp_close_prices[i] - temp_close_prices[i-1]) / temp_close_prices[i-1])
					else:
						pct_change = 0.0
					current_pattern.append(pct_change)
				break
			except Exception as e:
				debug_print(f"[DEBUG] {sym}: Failed to parse candle data: {e}")
				continue

		debug_print(f"[DEBUG] {sym}: Market data fetched successfully. Current pattern ({PATTERN_SIZE-1} percentage changes): {[f'{v:.4f}%' for v in current_pattern]}")

		# Cascading fallback: Try pattern sizes from PATTERN_SIZE down to 2
		# Only mark INACTIVE if no size produces matches
		folder = os.getcwd()  # Already in coin directory via coin_directory() context
		memory_list = None
		weight_list = None
		high_weight_list = None
		low_weight_list = None
		perfect_threshold = None
		matched_pattern_size = None
		
		# Determine which pattern sizes to try based on user setting
		if ENABLE_PATTERN_FALLBACK:
			pattern_sizes_to_try = range(PATTERN_SIZE, 1, -1)  # PATTERN_SIZE → 2 (cascading fallback)
		else:
			pattern_sizes_to_try = [PATTERN_SIZE]  # Only try configured size (legacy mode)
		
		for try_size in pattern_sizes_to_try:
			debug_print(f"[DEBUG] {sym}: Attempting pattern matching with size {try_size} for {tf_choices[tf_choice_index]}...")
			
			# Load training data for this pattern size
			memory_list, weight_list, high_weight_list, low_weight_list, perfect_threshold, volatility = _load_training_data_cached(
				sym, tf_choices[tf_choice_index], folder, try_size
			)
			
			if memory_list is None:
				debug_print(f"[DEBUG] {sym}: Training files missing for size {try_size}, trying next smaller size...")
				continue  # Try next smaller size - skip pattern processing
			
			# Found training data for this size - will process it to check quality
			matched_pattern_size = try_size  # Set for this iteration's pattern processing
			any_perfect = 'no'  # Initialize before try block in case of early failure
			final_moves = 0.0
			high_final_moves = 0.0
			low_final_moves = 0.0
			debug_print(f"[DEBUG] {sym}: Found training data for size {try_size}, checking pattern quality...")
			# Don't break yet - need to process patterns and check any_perfect
		
			try:
				# If we can read/parse training files, this timeframe is NOT a training-file issue.
				training_issues[tf_choice_index] = 0

				# Extract pattern subset matching the loaded pattern size
				# If we loaded size 3, use last 3 candles (2 pct changes)
				# If we loaded size 4, use last 4 candles (3 pct changes), etc.
				pattern_length = matched_pattern_size - 1  # Size 3 → 2 values, size 4 → 3 values
				current_pattern_subset = current_pattern[-(pattern_length):]  # Take last N values
				
				debug_print(f"[DEBUG] {sym}: Using pattern size {matched_pattern_size} ({pattern_length} pct changes) for matching")
				debug_print(f"[DEBUG] {sym}: Pattern subset: {[f'{v:.4f}%' for v in current_pattern_subset]}")

				mem_ind = 0
				diffs_list = []
				perfect_dexs = []
				perfect_diffs = []
				moves = []
				move_weights = []
				unweighted = []
				high_unweighted = []
				low_unweighted = []
				high_moves = []
				low_moves = []
				
				# Track pattern matching statistics for diagnostics
				patterns_checked = 0
				patterns_skipped = 0
				patterns_matched = 0
				closest_diff = float('inf')

				debug_print(f"[DEBUG] {sym}: Processing {len(memory_list)} memory patterns at size {matched_pattern_size}...")
				while mem_ind < len(memory_list):
					# Validate training data format before parsing
					parts = memory_list[mem_ind].split('{}')
					if len(parts) < 3:
						# Corrupted/incomplete training data - skip this pattern
						debug_print(f"[DEBUG] {sym}: Skipping corrupted memory pattern at index {mem_ind} (insufficient parts: {len(parts)})")
						patterns_skipped += 1
						mem_ind += 1
						continue
					
					try:
						memory_pattern = _clean_training_string(parts[0]).split('|')
						if not memory_pattern:
							# Empty pattern after cleaning - skip
							patterns_skipped += 1
							mem_ind += 1
							continue
					except Exception:
						debug_print(f"[DEBUG] {sym}: Failed to parse memory pattern at index {mem_ind}")
						patterns_skipped += 1
						mem_ind += 1
						continue
					
					# For multi-candle patterns, compare ALL candles in the pattern
					# memory_pattern contains (matched_pattern_size - 1) values, same as current_pattern_subset
					# Validate memory pattern has correct length
					if len(memory_pattern) < pattern_length:
						debug_print(f"[DEBUG] {sym}: Memory pattern too short ({len(memory_pattern)} < {pattern_length}) at index {mem_ind}")
						patterns_skipped += 1
						mem_ind += 1
						continue
					
					# Calculate average difference across all candles in the pattern
					# Relative percentage difference (threshold applies as % of pattern value)
					# Example: memory=5%, current=6%, threshold=15% → diff = |6-5|/5 * 100 = 20% > 15% (no match)
					# Example: memory=5%, current=5.5%, threshold=15% → diff = |5.5-5|/5 * 100 = 10% < 15% (match)
					# This scales precision: small moves get tight absolute tolerances, large moves get proportional tolerances
					total_diff = 0.0
					valid_comparisons = 0
					for i in range(pattern_length):
						try:
							current_val = current_pattern_subset[i]
							memory_val = float(memory_pattern[i])
							# Relative difference as percentage of pattern value
							# Use volatility-adaptive baseline (scaled by market conditions)
							# Scales noise floor with market: high volatility → higher baseline, low volatility → tighter baseline
							baseline = max(abs(memory_val), 0.1 * volatility)
							diff = (abs(current_val - memory_val) / baseline) * PERCENT_MULTIPLIER
							total_diff += diff
							valid_comparisons += 1
						except Exception:
							continue
					
					if valid_comparisons == 0:
						patterns_skipped += 1
						mem_ind += 1
						continue
					
					diff_avg = total_diff / valid_comparisons
					patterns_checked += 1

					# Parse high_diff and low_diff for all patterns (needed for later use)
					# Use pre-validated parts from split (already verified len >= 3)
					try:
						high_diff = float(_clean_training_string(parts[1])) / PERCENT_MULTIPLIER
						low_diff = float(_clean_training_string(parts[2])) / PERCENT_MULTIPLIER
					except (ValueError, IndexError) as e:
						# Failed to parse high/low diffs - skip this pattern
						debug_print(f"[DEBUG] {sym}: Failed to parse high/low diffs at index {mem_ind}: {e}")
						patterns_skipped += 1
						mem_ind += 1
						continue

					# Accept patterns within PERFECT_MATCH_MULTIPLIER × threshold as "good enough"
					# This determines whether to use this pattern size or fallback to smaller size
					if diff_avg <= (perfect_threshold * PERFECT_MATCH_MULTIPLIER):
						any_perfect = 'yes'
						patterns_matched += 1

					# Validate weight list indices before accessing
					if mem_ind >= len(weight_list) or mem_ind >= len(high_weight_list) or mem_ind >= len(low_weight_list):
						debug_print(f"[DEBUG] {sym}: Weight list length mismatch at index {mem_ind} (memory: {len(memory_list)}, weight: {len(weight_list)}, high: {len(high_weight_list)}, low: {len(low_weight_list)}) - skipping pattern")
						patterns_skipped += 1
						mem_ind += 1
						continue

					# Track closest match for diagnostics (even if later rejected by directional filter)
					# This helps identify if directional filtering is too aggressive
					if diff_avg < closest_diff:
						closest_diff = diff_avg

					# Maximum cutoff: Only use patterns within MAX_CUTOFF_MULTIPLIER × threshold for quality matching
					# This ensures prediction quality while preventing wildly dissimilar patterns.
					# Example: threshold=15%, cutoff=75% - patterns beyond 75% similarity are excluded
					max_cutoff = perfect_threshold * MAX_CUTOFF_MULTIPLIER
					if diff_avg > max_cutoff:
						# Pattern too dissimilar - skip it
						patterns_skipped += 1
						mem_ind += 1
						continue

					unweighted.append(float(memory_pattern[len(memory_pattern) - 1]))
					move_weights.append(float(weight_list[mem_ind]))
					high_unweighted.append(high_diff)
					low_unweighted.append(low_diff)

					if float(weight_list[mem_ind]) != 0.0:
						moves.append(float(memory_pattern[len(memory_pattern) - 1]) * float(weight_list[mem_ind]))

					if float(high_weight_list[mem_ind]) != 0.0:
						high_moves.append(high_diff * float(high_weight_list[mem_ind]))

					if float(low_weight_list[mem_ind]) != 0.0:
						low_moves.append(low_diff * float(low_weight_list[mem_ind]))

					perfect_dexs.append(mem_ind)
					perfect_diffs.append(diff_avg)

					diffs_list.append(diff_avg)
					mem_ind += 1

				# After processing all memory patterns, calculate quality and set active/inactive status
				# Calculate match quality: 100% at threshold, >100% for better matches, logarithmic decay to 0%
				# Quality formula: 200 / (1 + (closest_diff / perfect_threshold))
				# With relative thresholds: 200% at 0% diff (perfect), 100% at threshold, 50% at 3x threshold, 20% at 9x threshold
				if closest_diff < float('inf'):
					match_quality = 200 / (1 + (closest_diff / perfect_threshold))
				else:
					match_quality = 0.0  # Only 0 if no patterns exist at all
				match_qualities[tf_choice_index] = match_quality
				
				if any_perfect == 'no':
					debug_print(f"[DEBUG] {sym} {tf_choices[tf_choice_index]} (size {matched_pattern_size}): No patterns matched threshold {perfect_threshold:.2f}% relative tolerance. Current pattern: {[f'{v:.2f}%' for v in current_pattern_subset]}, Checked: {patterns_checked}, Skipped: {patterns_skipped}, Closest: {closest_diff:.2f}% relative (Quality: {match_quality:.0f}%)")
					# ALWAYS generate predictions from closest patterns (even if they don't meet threshold)
					if moves and high_moves and low_moves:
						# Kernel-weighted average (closer matches weighted higher, normalized by threshold)
						kernel_weights = [math.exp(-d / (perfect_threshold * KERNEL_BANDWIDTH)) for d in diffs_list]
						total_weight = sum(kernel_weights)
						final_moves = sum(m * w for m, w in zip(moves, kernel_weights)) / total_weight
						high_final_moves = sum(h * w for h, w in zip(high_moves, kernel_weights)) / total_weight
						low_final_moves = sum(l * w for l, w in zip(low_moves, kernel_weights)) / total_weight
						perfects[tf_choice_index] = 'active'  # Mark active even if match quality is low
						pattern_sizes_used[tf_choice_index] = matched_pattern_size if matched_pattern_size is not None else 0
						debug_print(f"[DEBUG] {sym} {tf_choices[tf_choice_index]}: Using pattern size {matched_pattern_size} (fallback successful)")
					else:
						final_moves = 0.0
						high_final_moves = 0.0
						low_final_moves = 0.0
						perfects[tf_choice_index] = 'inactive'
						pattern_sizes_used[tf_choice_index] = 0
						debug_print(f"[DEBUG] {sym} {tf_choices[tf_choice_index]}: No valid patterns at any size - marking INACTIVE")
				elif moves and high_moves and low_moves:
					# Only calculate averages if lists are non-empty
					debug_print(f"[DEBUG] {sym} {tf_choices[tf_choice_index]} (size {matched_pattern_size}): Matched {patterns_matched} patterns. Checked: {patterns_checked}, Skipped: {patterns_skipped} (Quality: {match_quality:.0f}%)")
					# Kernel-weighted average (closer matches weighted higher, normalized by threshold)
					kernel_weights = [math.exp(-d / (perfect_threshold * KERNEL_BANDWIDTH)) for d in diffs_list]
					total_weight = sum(kernel_weights)
					final_moves = sum(m * w for m, w in zip(moves, kernel_weights)) / total_weight
					high_final_moves = sum(h * w for h, w in zip(high_moves, kernel_weights)) / total_weight
					low_final_moves = sum(l * w for l, w in zip(low_moves, kernel_weights)) / total_weight
					perfects[tf_choice_index] = 'active'
					pattern_sizes_used[tf_choice_index] = matched_pattern_size if matched_pattern_size is not None else 0
				else:
					final_moves = 0.0
					high_final_moves = 0.0
					low_final_moves = 0.0
					perfects[tf_choice_index] = 'inactive'
					pattern_sizes_used[tf_choice_index] = 0
					debug_print(f"[DEBUG] {sym} {tf_choices[tf_choice_index]}: No valid weighted predictions - marking INACTIVE")

			except Exception:
				PrintException()
				training_issues[tf_choice_index] = 1
				final_moves = 0.0
				high_final_moves = 0.0
				low_final_moves = 0.0
				perfects[tf_choice_index] = 'inactive'
				pattern_sizes_used[tf_choice_index] = 0

			# After processing patterns for this size, check if quality is good enough
			if any_perfect == 'yes':
				# Found good matches - use this size
				matched_pattern_size = try_size
				debug_print(f"[DEBUG] {sym} {tf_choices[tf_choice_index]}: Size {try_size} has good matches, using this size")
				break
			else:
				# No good matches at this size
				if try_size == 2:
					# Last size also failed - mark as inactive
					debug_print(f"[DEBUG] {sym} {tf_choices[tf_choice_index]}: Size {try_size} (last size) has no good matches - marking INACTIVE")
					perfects[tf_choice_index] = 'inactive'
					pattern_sizes_used[tf_choice_index] = 0
					final_moves = 0.0
					high_final_moves = 0.0
					low_final_moves = 0.0
					matched_pattern_size = None
					break
				else:
					# Try next smaller size
					debug_print(f"[DEBUG] {sym} {tf_choices[tf_choice_index]}: Size {try_size} has no good matches, trying smaller size...")
					continue
		
		# If no training data found for any size, mark as training issue (will be checked after loop)
		if memory_list is None:
			debug_print(f"[DEBUG] {sym}: No training files found for any pattern size (tried {PATTERN_SIZE} down to 2) for {tf_choices[tf_choice_index]}")
			training_issues[tf_choice_index] = 1
			st['training_issues'] = training_issues
			st['tf_choice_index'] = (tf_choice_index + 1) % len(tf_choices)
			states[sym] = st
			return

		# If all pattern sizes failed quality check (even size 2), skip prediction calculation
		if matched_pattern_size is None:
			debug_print(f"[DEBUG] {sym}: All pattern sizes failed quality check for {tf_choices[tf_choice_index]} - skipping prediction")
			st['tf_choice_index'] = (tf_choice_index + 1) % len(tf_choices)
			states[sym] = st
			return

		# Compute new high/low predictions
		# For price predictions, we need the most recent (last) candle's close price
		last_candle_data = _clean_candle_string(str(history_list[PATTERN_SIZE-1]))
		if len(last_candle_data) < 3:
			debug_print(f"[VALIDATION] {sym}: Last candle has insufficient fields: {len(last_candle_data)}")
			raise ValueError("Insufficient candle data fields")
		last_open = _safe_float_convert(last_candle_data[1], "last_candle.open", 0.0)
		last_close = _safe_float_convert(last_candle_data[2], "last_candle.close", 0.0)
		
		try:
			c_diff = final_moves / PERCENT_MULTIPLIER
			high_diff = high_final_moves
			low_diff = low_final_moves

			# Use the close price of the most recent candle as the starting point
			start_price = last_close
			high_new_price = start_price + (start_price * high_diff)
			low_new_price = start_price + (start_price * low_diff)
			
			debug_print(
				f"[DEBUG] {sym} {tf_choices[tf_choice_index]}: "
				f"start_price={start_price:.2f}, final_moves={final_moves:.2f}%, "
				f"high_diff={high_diff:.4f} ({high_diff*PERCENT_MULTIPLIER:.2f}%), low_diff={low_diff:.4f} ({low_diff*PERCENT_MULTIPLIER:.2f}%) | "
				f"high_new_price={high_new_price:.2f}, low_new_price={low_new_price:.2f}"
			)
		except Exception:
			start_price = last_close
			high_new_price = start_price
			low_new_price = start_price

		if perfects[tf_choice_index] == 'inactive':
			high_tf_prices[tf_choice_index] = start_price
			low_tf_prices[tf_choice_index] = start_price
		else:
			high_tf_prices[tf_choice_index] = high_new_price
			low_tf_prices[tf_choice_index] = low_new_price

		# Calculate prediction candle for Hub chart visualization (only for active timeframes)
		if perfects[tf_choice_index] != 'inactive':
			from datetime import datetime, timedelta
			current_tf = tf_choices[tf_choice_index]
			tf_seconds = _timeframe_to_seconds(current_tf)
			future_timestamp = datetime.now() + timedelta(seconds=tf_seconds)
			
			# Build prediction candle: open=current close, high/low from predictions, close from final_moves
			pred_close = start_price * (1 + (final_moves / PERCENT_MULTIPLIER))
			
			# Ensure high/low properly encompass open and close (valid candle structure)
			pred_high = max(start_price, pred_close, high_tf_prices[tf_choice_index])
			pred_low = min(start_price, pred_close, low_tf_prices[tf_choice_index])
			
			# Debug output to verify prediction candle values
			debug_print(
				f"[DEBUG] {sym} {current_tf} Prediction Candle: "
				f"open={start_price:.2f}, close={pred_close:.2f}, "
				f"high={pred_high:.2f}, low={pred_low:.2f} | "
				f"final_moves={final_moves:.2f}%, high_final_moves={high_final_moves:.4f}, low_final_moves={low_final_moves:.4f}"
			)
			
			prediction_candles[current_tf] = {
				'timestamp': future_timestamp.isoformat(),
				'open': round(start_price, 2),
				'high': round(pred_high, 2),
				'low': round(pred_low, 2),
				'close': round(pred_close, 2)
			}

		# Advance tf index; if full sweep complete, compute signals
		tf_choice_index += 1

		if tf_choice_index >= len(tf_choices):
			debug_print(f"[DEBUG] {sym}: Full timeframe cycle complete. Generating messages...")
			tf_choice_index = 0

			# reset tf_update for this coin (but DO NOT block-wait; just detect updates and return)
			tf_update = ['no'] * len(tf_choices)

			# get current price ONCE per coin — use Robinhood's current ASK (same as rhcb trader buy price)
			# Validate symbol format before calling API
			if not sym or not sym.strip():
				debug_print(f"[DEBUG] {sym}: Invalid empty symbol, skipping price fetch")
				st['tf_choice_index'] = 0
				states[sym] = st
				return
			
			# Ensure symbol is alphanumeric only (no special chars that could break API)
			clean_sym = ''.join(ch for ch in sym if ch.isalnum())
			if clean_sym != sym:
				debug_print(f"[DEBUG] {sym}: Symbol contains invalid characters, using cleaned version: {clean_sym}")
			
			rh_symbol = f"{clean_sym}-USD"
			debug_print(f"[DEBUG] {sym}: Fetching current price from Robinhood for {rh_symbol}...")
			retry_count = 0
			max_retries = 5
			while True:
				try:
					_rate_limiter.wait_if_needed('robinhood')
					current = robinhood_current_ask(rh_symbol)
					debug_print(f"[DEBUG] {sym}: Current price: ${current:.2f}")
					break
				except Exception as e:
					retry_count += 1
					debug_print(f"[DEBUG] {sym}: Robinhood API error (attempt {retry_count}/{max_retries}): {type(e).__name__}: {str(e)[:100]}")
					if retry_count >= max_retries:
						handle_network_error(f"Robinhood price fetch for {sym}", e)
				time.sleep(_get_sleep_timing("sleep_robinhood_retry"))
			# IMPORTANT: messages printed below use the bounds currently in state.
			# We only allow "ready" once messages are generated using a non-startup bounds_version.
			bounds_version_used_for_messages = st.get('bounds_version', 0)

			# Hard guarantee that all timeframe arrays stay length==len(tf_choices) with fallback placeholders
			def _pad_to_len(lst, n, fill):
				if lst is None:
					lst = []
				if len(lst) < n:
					lst.extend([fill] * (n - len(lst)))
				elif len(lst) > n:
					del lst[n:]
				return lst

			n_tfs = len(tf_choices)

			# bounds: already padded at lines 730-738, no need to pad again
			# low_bound_prices and high_bound_prices are guaranteed correct length

			# predicted prices: keep equal when missing so it never triggers LONG/SHORT
			high_tf_prices = _pad_to_len(high_tf_prices, n_tfs, current)
			low_tf_prices = _pad_to_len(low_tf_prices, n_tfs, current)

			# status arrays
			perfects = _pad_to_len(perfects, n_tfs, 'inactive')
			training_issues = _pad_to_len(training_issues, n_tfs, 0)
			messages = _pad_to_len(messages, n_tfs, 'none')

			tf_sides = _pad_to_len(tf_sides, n_tfs, 'none')
			messaged = _pad_to_len(messaged, n_tfs, 'no')
			margins = _pad_to_len(margins, n_tfs, 0.0)
			updated = _pad_to_len(updated, n_tfs, 0)

			# per-timeframe message logic (same decisions as before)
			inder = 0
			while inder < len(tf_choices):
				# Save old message for comparison to detect changes
				old_message = messages[inder]
				
				# update the_time snapshot (same as before)
				while True:

					try:
						_rate_limiter.wait_if_needed('kucoin')
						history = str(market.get_kline(coin, tf_choices[inder])).replace(']]', '], ').replace('[[', '[')
						is_valid, history_list, error_msg = _validate_kline_response(history, coin, tf_choices[inder], 2)
						if not is_valid:
							debug_print(f"[VALIDATION] {error_msg}")
							raise ValueError(error_msg)
						break
					except Exception as e:
						time.sleep(_get_sleep_timing("sleep_api_retry"))
						if 'Requests' in str(e):
							pass
						else:
							PrintException()
						continue
				
				the_time = _parse_candle_time(history_list, 1)

				# (original comparisons)
				if current > high_bound_prices[inder] and high_tf_prices[inder] != low_tf_prices[inder]:
					size_info = f" [size {pattern_sizes_used[inder]}]" if pattern_sizes_used[inder] > 0 else ""
					message = 'SHORT on ' + tf_choices[inder] + ' timeframe' + size_info + '.\t' + format(((high_bound_prices[inder] - current) / abs(current)) * PERCENT_MULTIPLIER, '.8f') + '\tHigh: ' + str(high_bound_prices[inder])
					if messaged[inder] != 'yes':
						messaged[inder] = 'yes'
					margins[inder] = ((high_tf_prices[inder] - current) / abs(current)) * PERCENT_MULTIPLIER

					# Compare with old message to detect changes
					if 'SHORT' in old_message:
						messages[inder] = message
						updated[inder] = 0
					else:
						messages[inder] = message
						updated[inder] = 1

					tf_sides[inder] = 'short'

				elif current < low_bound_prices[inder] and high_tf_prices[inder] != low_tf_prices[inder]:
					size_info = f" [size {pattern_sizes_used[inder]}]" if pattern_sizes_used[inder] > 0 else ""
					message = 'LONG on ' + tf_choices[inder] + ' timeframe' + size_info + '.\t' + format(((low_bound_prices[inder] - current) / abs(current)) * PERCENT_MULTIPLIER, '.8f') + '\tLow: ' + str(low_bound_prices[inder])
					if messaged[inder] != 'yes':
						messaged[inder] = 'yes'

					margins[inder] = ((low_tf_prices[inder] - current) / abs(current)) * PERCENT_MULTIPLIER

					tf_sides[inder] = 'long'

					# Compare with old message to detect changes
					if 'LONG' in old_message:
						messages[inder] = message
						updated[inder] = 0
					else:
						messages[inder] = message
						updated[inder] = 1

				else:
					if perfects[inder] == 'inactive':
						if training_issues[inder] == 1:
							message = 'INACTIVE (training data issue) on ' + tf_choices[inder] + ' timeframe.\tLow: ' + str(low_bound_prices[inder]) + '\tHigh: ' + str(high_bound_prices[inder])
						else:
							message = 'INACTIVE on ' + tf_choices[inder] + ' timeframe.\tLow: ' + str(low_bound_prices[inder]) + '\tHigh: ' + str(high_bound_prices[inder])
					else:
						size_info = f" [size {pattern_sizes_used[inder]}]" if pattern_sizes_used[inder] > 0 else ""
						message = 'WITHIN on ' + tf_choices[inder] + ' timeframe' + size_info + '.\tLow: ' + str(low_bound_prices[inder]) + '\tHigh: ' + str(high_bound_prices[inder])

					margins[inder] = 0.0

					# Compare with OLD message to detect changes
					if message == old_message:
						messages[inder] = message
						updated[inder] = 0
					else:
						messages[inder] = message
						updated[inder] = 1

					tf_sides[inder] = 'none'

					messaged[inder] = 'no'

				inder += 1

			# rebuild bounds (same math as before)
			price_list_index = 0
			low_bound_prices = []
			high_bound_prices = []
			while True:
				pred_low_val = low_tf_prices[price_list_index]
				pred_high_val = high_tf_prices[price_list_index]
				new_low_price = pred_low_val - (pred_low_val * (distance / PERCENT_MULTIPLIER))
				new_high_price = pred_high_val + (pred_high_val * (distance / PERCENT_MULTIPLIER))
				if perfects[price_list_index] != 'inactive':
					low_bound_prices.append(new_low_price)
					high_bound_prices.append(new_high_price)
				else:
					low_bound_prices.append(0.0)
					high_bound_prices.append(float('inf'))

				price_list_index += 1
				if price_list_index >= len(high_tf_prices):
					break

			# Sort with original indices to handle duplicate values correctly
			# Create tuples of (value, original_index) to preserve position information
			low_with_indices = [(low_bound_prices[i], i) for i in range(len(low_bound_prices))]
			high_with_indices = [(high_bound_prices[i], i) for i in range(len(high_bound_prices))]
			
			# Sort by value (reversed for low, normal for high)
			low_with_indices.sort(key=lambda x: x[0], reverse=True)
			high_with_indices.sort(key=lambda x: x[0])
			
			# Extract sorted values and index mappings
			new_low_bound_prices = [x[0] for x in low_with_indices]
			new_high_bound_prices = [x[0] for x in high_with_indices]
			og_low_index_list = [x[1] for x in low_with_indices]  # Maps sorted position -> original position
			og_high_index_list = [x[1] for x in high_with_indices]

			og_index = 0
			gap_modifier = 0.0
			# Only adjust gaps if we have at least 2 elements (need og_index and og_index+1)
			if len(new_low_bound_prices) > 1 and len(new_high_bound_prices) > 1:
				gap_adjustment_iterations = 0
				max_gap_iterations = 1000  # Prevent infinite loops with pathological data
				while True:
					gap_adjustment_iterations += 1
					if gap_adjustment_iterations > max_gap_iterations:
						debug_print(f"[DEBUG] {sym}: Gap adjustment exceeded {max_gap_iterations} iterations, breaking")
						break
					
					if new_low_bound_prices[og_index] == 0.0 or new_low_bound_prices[og_index + 1] == 0.0 or new_high_bound_prices[og_index] == float('inf') or new_high_bound_prices[og_index + 1] == float('inf'):
						pass
					else:
						# Calculate percentage differences with zero-denominator protection
						low_avg = (new_low_bound_prices[og_index] + new_low_bound_prices[og_index + 1]) / 2
						if abs(low_avg) < 1e-10:
							low_perc_diff = 0.0
						else:
							try:
								low_perc_diff = (abs(new_low_bound_prices[og_index] - new_low_bound_prices[og_index + 1]) / low_avg) * PERCENT_MULTIPLIER
							except Exception:
								low_perc_diff = 0.0
						
						high_avg = (new_high_bound_prices[og_index] + new_high_bound_prices[og_index + 1]) / 2
						if abs(high_avg) < 1e-10:
							high_perc_diff = 0.0
						else:
							try:
								high_perc_diff = (abs(new_high_bound_prices[og_index] - new_high_bound_prices[og_index + 1]) / high_avg) * PERCENT_MULTIPLIER
							except Exception:
								high_perc_diff = 0.0

						if low_perc_diff < GAP_ENFORCEMENT_MIN + gap_modifier or new_low_bound_prices[og_index + 1] > new_low_bound_prices[og_index]:
							new_price = new_low_bound_prices[og_index + 1] - (new_low_bound_prices[og_index + 1] * 0.0005)
							new_low_bound_prices[og_index + 1] = new_price
							continue

						if high_perc_diff < GAP_ENFORCEMENT_MIN + gap_modifier or new_high_bound_prices[og_index + 1] < new_high_bound_prices[og_index]:
							new_price = new_high_bound_prices[og_index + 1] + (new_high_bound_prices[og_index + 1] * 0.0005)
							new_high_bound_prices[og_index + 1] = new_price
							continue

					og_index += 1
					gap_modifier += GAP_INCREMENT
					if og_index >= len(new_low_bound_prices) - 1:
						break

			# Rebuild arrays in original timeframe order using reverse mapping
			# og_low_index_list[sorted_pos] = original_pos, so we need to reverse this
			low_bound_prices = [0.0] * len(new_low_bound_prices)
			high_bound_prices = [float('inf')] * len(new_high_bound_prices)
			
			for sorted_pos in range(len(og_low_index_list)):
				original_pos = og_low_index_list[sorted_pos]
				low_bound_prices[original_pos] = new_low_bound_prices[sorted_pos]
			
			for sorted_pos in range(len(og_high_index_list)):
				original_pos = og_high_index_list[sorted_pos]
				high_bound_prices[original_pos] = new_high_bound_prices[sorted_pos]

			# bump bounds_version now that we've computed a new set of prediction bounds
			st['bounds_version'] = bounds_version_used_for_messages + 1

			# Write boundaries with timeframe labels for chart display
			# Format: "1h:93524.5 2h:92100.0 4h:91200.0 ..."
			tf_labels = ['1h', '2h', '4h', '8h', '12h', '1d', '1w']
			low_labeled = []
			high_labeled = []
			for sorted_pos in range(len(new_low_bound_prices)):
				original_pos = og_low_index_list[sorted_pos]
				if original_pos < len(tf_labels):
					tf_label = tf_labels[original_pos]
					low_labeled.append(f"{tf_label}:{new_low_bound_prices[sorted_pos]}")
				
			for sorted_pos in range(len(new_high_bound_prices)):
				original_pos = og_high_index_list[sorted_pos]
				if original_pos < len(tf_labels):
					tf_label = tf_labels[original_pos]
					high_labeled.append(f"{tf_label}:{new_high_bound_prices[sorted_pos]}")
				
			# Only write HTML files if bounds changed (reduces disk I/O ~99%)
			low_content = ' '.join(low_labeled)
			high_content = ' '.join(high_labeled)
			last_bounds = _last_written_bounds.get(sym, {})
			if last_bounds.get('low') != low_content or last_bounds.get('high') != high_content:
				try:
					# Use timeout-protected writes for critical trading data
					_atomic_write_text_safe('low_bound_prices.dat', low_content, timeout=3.0)
					_atomic_write_text_safe('high_bound_prices.dat', high_content, timeout=3.0)
					_last_written_bounds[sym] = {'low': low_content, 'high': high_content}
				except FileOperationTimeout as e:
					debug_print(f"[ERROR] THINKER: Timeout writing price bounds for {sym}: {e}")
					# Don't update cache so we retry next cycle
				except Exception as e:
					debug_print(f"[ERROR] THINKER: Failed writing price bounds for {sym}: {e}")

			# Write prediction candles for Hub chart visualization
			if prediction_candles:
				try:
					_atomic_write_json_safe('prediction_candles.json', prediction_candles, timeout=3.0)
				except FileOperationTimeout as e:
					debug_print(f"[ERROR] THINKER: Timeout writing prediction candles for {sym}: {e}")
				except Exception as e:
					debug_print(f"[ERROR] THINKER: Failed writing prediction candles for {sym}: {e}")
			
			# Save prediction candles back to state and reset for next cycle
			st['prediction_candles'] = {}

			# cache display text for this coin (main loop prints everything on one screen)
			try:
				# Count signal consensus
				actives = perfects.count('active')
				inactives = perfects.count('inactive')
				withins = sum(1 for msg in messages if 'WITHIN' in msg)
				longs_count = tf_sides.count('long')
				shorts_count = tf_sides.count('short')
					
				# Load trading config to get signal thresholds (match trader logic)
				trading_cfg = _load_trading_config()
				long_min = max(1, min(7, int(trading_cfg.get("entry_signals", {}).get("long_signal_min", 4))))
				# For SHORT display, use same threshold as LONG for consistency
				# (Trader uses short_signal_max as a filter, not an entry trigger)
					
				# Determine overall signal (use config thresholds so display matches trader behavior)
				if longs_count >= long_min:
					signal_status = "LONG"
					signal_indicator = "[v]"
				elif shorts_count >= long_min:
					signal_status = "SHORT"
					signal_indicator = "[^]"
				elif withins > 0:
					signal_status = "WITHIN"
					signal_indicator = "[=]"
				else:
					signal_status = "INACTIVE"
					signal_indicator = "[o]"
					
				# Format current price with proper decimals
				price_str = f"${current:,.2f}" if current < 1000 else f"${current:,.0f}"
					
				# Add DCA signal info
				active_margins = [m for m in margins if m != 0]
				if len(active_margins) > 0:
					pm_value = sum(active_margins) / len(active_margins)
					pm_display = f"{pm_value:+.2f}%"
				else:
					pm_display = "-----"

				# Build output lines
				lines = []
				current_time = datetime.now().strftime("%H:%M:%S")
				lines.append(f"\n[{sym}] Current Price at {current_time}: {price_str}")
				lines.append(f"[{sym}] Signal: {signal_indicator} {signal_status} (L/S/W:{longs_count}/{shorts_count}/{withins})   Profit Margin: {pm_display}")
				lines.append(f"[{sym}] Pattern Matching: {actives}/7 active, {inactives}/7 inactive")
					
				# Build timeframe rows
				for i, tf in enumerate(tf_choices):
					# Get status
					if 'LONG' in messages[i]:
						status = "LONG"
					elif 'SHORT' in messages[i]:
						status = "SHORT"
					elif 'WITHIN' in messages[i]:
						status = "HODL"
					elif 'INACTIVE' in messages[i]:
						status = "INACTIVE"
					else:
						status = "STARTING"
					
					# Get pattern size info for this timeframe
					pattern_size_used = pattern_sizes_used[i] if i < len(pattern_sizes_used) else 0
					size_indicator = f"{pattern_size_used}p" if pattern_size_used > 0 else "0p"
						
					# For inactive timeframes, show status with debug info
					if status == "INACTIVE":
						# Get match quality and training issue info for troubleshooting
						quality = match_qualities[i] if i < len(match_qualities) else 0.0
						training_issue = training_issues[i] if i < len(training_issues) else 0
						
						# Build debug info
						if training_issue == 1:
							reason = "training data missing"
						elif pattern_size_used == 0:
							reason = "no pattern files"
						elif quality == 0.0:
							reason = "no matches found"
						else:
							reason = f"quality {quality:.0f}% too low"
						
						lines.append(f"[{sym}]  {tf:>6s}: {status} {size_indicator} ({reason})")
						continue
						
					# Calculate distance to boundaries as percentage
					low_bound = low_bound_prices[i]
					high_bound = high_bound_prices[i]
						
					if low_bound != 0.0 and high_bound != float('inf'):
						low_dist = ((current - low_bound) / current) * PERCENT_MULTIPLIER
						high_dist = ((high_bound - current) / current) * PERCENT_MULTIPLIER
						bounds_text = f"Low:{low_dist:+5.1f}%   High:{high_dist:+5.1f}%"
					elif low_bound != 0.0:
						low_dist = ((current - low_bound) / current) * PERCENT_MULTIPLIER
						bounds_text = f"Low:{low_dist:+5.1f}%   High:    --"
					elif high_bound != float('inf'):
						high_dist = ((high_bound - current) / current) * PERCENT_MULTIPLIER
						bounds_text = f"Low: ------   High:{high_dist:+5.1f}%"
					else:
						bounds_text = "No boundaries"
						
					# Add match quality indicator (ASCII only for Hub display compatibility)
					quality = match_qualities[i] if i < len(match_qualities) else 100.0
					if quality >= 100:
						quality_icon = f"Signal: {size_indicator} {quality:.0f}% MINT"  # Excellent match
					elif quality >= 67:
						quality_icon = f"Signal: {size_indicator}  {quality:.0f}% GOOD"  # Good match
					elif quality >= 33:
						quality_icon = f"Signal: {size_indicator}  {quality:.0f}% EDGE"  # Marginal match
					else:
						quality_icon = f"Signal: {size_indicator}  {quality:.0f}% WEAK"  # Weak match
						
					lines.append(f"[{sym}]  {tf:>6s}: {status:6s} {bounds_text}   {quality_icon}")
								
				# Combine all parts
				display_cache[sym] = '\n'.join(lines)
				
				# Mark this bounds_version as needing display update
				st['last_display_bounds_version'] = st.get('bounds_version', 0)
					
				# Calculate summary data for this coin
				# Load trading config to get required signal thresholds
				trading_cfg = _load_trading_config()
				long_signal_min = max(1, min(7, int(trading_cfg.get("entry_signals", {}).get("long_signal_min", 4))))
					
				# Find Nth low boundary below current (Next Long trigger)
				# Need long_signal_min timeframes showing LONG, so find the Nth-highest low_bound below current
				valid_lows = [low_bound_prices[i] for i in range(len(low_bound_prices)) if low_bound_prices[i] != 0.0 and low_bound_prices[i] < current]
				if len(valid_lows) >= long_signal_min:
					# Sort descending (highest to lowest), take the Nth one
					valid_lows.sort(reverse=True)
					long_trigger_price = valid_lows[long_signal_min - 1]
					next_long_pct = ((long_trigger_price - current) / current) * PERCENT_MULTIPLIER
				else:
					next_long_pct = None
					
				# Find Nth high boundary above current (Next Short trigger)
				# Need long_signal_min timeframes showing SHORT, so find the Nth-lowest high_bound above current
				valid_highs = [high_bound_prices[i] for i in range(len(high_bound_prices)) if high_bound_prices[i] != float('inf') and high_bound_prices[i] > current]
				if len(valid_highs) >= long_signal_min:
					# Sort ascending (lowest to highest), take the Nth one
					valid_highs.sort()
					short_trigger_price = valid_highs[long_signal_min - 1]
					next_short_pct = ((short_trigger_price - current) / current) * PERCENT_MULTIPLIER
				else:
					next_short_pct = None
				
				summary_cache[sym] = {
					'price': current,
					'price_str': price_str,
					'signal': signal_status,
					'next_long_pct': next_long_pct,
					'next_short_pct': next_short_pct,
					'pm_display': pm_display
				}

				# Only consider this coin "ready" once we've already rebuilt bounds at least once
				# AND we're now printing messages generated from those rebuilt bounds.
				if (st['last_display_bounds_version'] >= 1) and _is_printing_real_predictions(messages):
					_ready_coins.add(sym)
				else:
					_ready_coins.discard(sym)

				all_ready = len(_ready_coins) > 0  # At least one coin trained
				_write_runner_ready(
					all_ready,
					stage=("real_predictions" if all_ready else "warming_up"),
					ready_coins=sorted(list(_ready_coins)),
					total_coins=len(COIN_SYMBOLS),
				)

			except Exception:
				PrintException()

			# write PM + DCA signals (same as before)
			try:
				longs = tf_sides.count('long')
				shorts = tf_sides.count('short')

				# long pm
				current_pms = [m for m in margins if m != 0]
				try:
					if len(current_pms) > 0:
						pm = sum(current_pms) / len(current_pms)
						if pm < MIN_PROFIT_MARGIN:
							pm = MIN_PROFIT_MARGIN
					else:
						pm = MIN_PROFIT_MARGIN
				except Exception:
					pm = MIN_PROFIT_MARGIN

				# Use timeout-protected writes for critical trading signals
				try:
					_atomic_write_text_safe('futures_long_profit_margin.dat', str(pm), timeout=3.0)
					_atomic_write_text_safe('long_dca_signal.dat', str(longs), timeout=3.0)
				except FileOperationTimeout as e:
					debug_print(f"[ERROR] THINKER: Timeout writing long signals for {sym}: {e}")
				except Exception as e:
					debug_print(f"[ERROR] THINKER: Failed writing long signals for {sym}: {e}")

				# short pm
				current_pms = [m for m in margins if m != 0]
				try:
					if len(current_pms) > 0:
						pm = sum(current_pms) / len(current_pms)
						if pm < MIN_PROFIT_MARGIN:
							pm = MIN_PROFIT_MARGIN
					else:
						pm = MIN_PROFIT_MARGIN
				except Exception:
					pm = MIN_PROFIT_MARGIN

				# Use timeout-protected writes for critical trading signals
				try:
					_atomic_write_text_safe('futures_short_profit_margin.dat', str(abs(pm)), timeout=3.0)
					_atomic_write_text_safe('short_dca_signal.dat', str(shorts), timeout=3.0)
				except FileOperationTimeout as e:
					debug_print(f"[ERROR] THINKER: Timeout writing short signals for {sym}: {e}")
				except Exception as e:
					debug_print(f"[ERROR] THINKER: Failed writing short signals for {sym}: {e}")
				
				# Write inactive count for visual indication in hub
				_atomic_write_text('inactive_count.dat', str(inactives))

			except Exception:
				PrintException()

			# NON-BLOCKING candle update check (single pass)
			tf_update_index = 0
			while tf_update_index < len(tf_update):
				while True:
					try:
						_rate_limiter.wait_if_needed('kucoin')
						history = str(market.get_kline(coin, tf_choices[tf_update_index])).replace(']]', '], ').replace('[[', '[')
						is_valid, history_list, error_msg = _validate_kline_response(history, coin, tf_choices[tf_update_index], 2)
						if not is_valid:
							debug_print(f"[VALIDATION] {error_msg}")
							raise ValueError(error_msg)
						break
					except Exception as e:
						time.sleep(_get_sleep_timing("sleep_api_retry"))
						if 'Requests' in str(e):
							pass
						else:
							PrintException()
						continue
				
				the_time = _parse_candle_time(history_list, 1)

				if the_time != tf_times[tf_update_index]:
					tf_update[tf_update_index] = 'yes'
					tf_times[tf_update_index] = the_time

				tf_update_index += 1

		# Save state back
		debug_print(f"[DEBUG] {sym}: Saving state - tf_choice_index={tf_choice_index}")
		# Only persist state if at least one timeframe has valid predictions (not all timeframes marked as having training issues)
		all_timeframes_failed = all(training_issues[i] == 1 for i in range(len(tf_choices)))
	
		if all_timeframes_failed:
			debug_print(f"[DEBUG] {sym}: All timeframes failed - not persisting invalid state")
			# Keep tf_choice_index to resume from correct position next cycle
			st['tf_choice_index'] = tf_choice_index
			states[sym] = st
			return
		
		st['low_bound_prices'] = low_bound_prices
		st['high_bound_prices'] = high_bound_prices
		st['tf_times'] = tf_times
		st['tf_choice_index'] = tf_choice_index

		# persist readiness gating fields
		st['bounds_version'] = st.get('bounds_version', 0)
		st['last_display_bounds_version'] = st.get('last_display_bounds_version', -1)

		st['tf_update'] = tf_update
		st['messages'] = messages
		st['last_messages'] = last_messages
		st['margins'] = margins

		st['high_tf_prices'] = high_tf_prices
		st['low_tf_prices'] = low_tf_prices
		st['updated'] = updated
		st['perfects'] = perfects
		st['tf_sides'] = tf_sides
		st['messaged'] = messaged
		st['match_qualities'] = match_qualities
		st['training_issues'] = training_issues
		st['pattern_sizes_used'] = pattern_sizes_used
_startup_message_shown = False
_first_run = True
_last_training_warning_time = 0.0  # Throttle training warnings to once per 60 seconds

print("Starting main loop...", flush=True)

try:
	while True:
		# Hot-reload coins from GUI settings while running
		_sync_coins_from_settings()
		
		# Capture coin list after sync to ensure new coins are processed immediately
		coins_this_iteration = list(CURRENT_COINS)

		for _sym in coins_this_iteration:
			step_coin(_sym)

		# NOTE: Skip screen clearing when running as subprocess (Hub captures output to text widget)
		# os.system('cls' if os.name == 'nt' else 'clear')

		# Print header with system status
		from datetime import datetime
		timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		ready_count = len(_ready_coins)
		total_count = len(coins_this_iteration)

		# Validate training status and display persistent warning if incomplete
		# Skip coins actively being trained and throttle warning to once per 60 seconds
		training_warnings = []
		for _sym in coins_this_iteration:
			# Skip coins that are currently training (not finished yet)
			if _is_coin_actively_training(_sym):
				continue
			
			is_valid, missing_tfs = _validate_coin_training(_sym)
			if not is_valid:
				training_warnings.append(f"{_sym}: Missing {', '.join(missing_tfs)}")

		# Throttle warning to once per 60 seconds to avoid log spam
		current_time = time.time()
		time_since_last_warning = current_time - _last_training_warning_time
		
		if training_warnings and time_since_last_warning >= 60.0:
			print("\nWARNING: INCOMPLETE TRAINING DETECTED", flush=True)
			for warning in training_warnings:
				print(f"  {warning}", flush=True)
			print(f"Required timeframes: {', '.join(REQUIRED_THINKER_TIMEFRAMES)}", flush=True)
			print("Predictions may be inaccurate until training is complete.\n", flush=True)
			_last_training_warning_time = current_time
			_startup_message_shown = False  # Reset if warnings appear
		elif not training_warnings:
			# Show startup message only on first successful validation
			if _first_run and not _startup_message_shown:
				_startup_message_shown = True

		_first_run = False

		# Print all coins' display output (use cached display from step_coin which includes symbol header)
		# Only print when bounds_version changes to avoid repetitive output during timeframe cycling
		any_coin_updated = False
		for _sym in coins_this_iteration:
			st = states.get(_sym)
			if not st:
				continue
			
			current_bounds_version = st.get('bounds_version', 0)
			last_printed_version = _last_printed_bounds_version.get(_sym, -1)
			
			# Only print if bounds_version has changed (new predictions ready) or first startup
			if current_bounds_version != last_printed_version:
				_last_printed_bounds_version[_sym] = current_bounds_version
				
				output = display_cache.get(_sym, f"[{_sym}] No data yet")
				
				# Skip startup message if already shown for this coin (prevents repetitive "Starting..." spam)
				if "Starting..." in output and "Initializing predictions" in output:
					if _sym in _startup_messages_shown:
						continue  # Already shown this startup message, skip it
					else:
						_startup_messages_shown.add(_sym)  # Mark as shown
				
				# Strip BOM, zero-width spaces, and other invisible Unicode
				output = output.replace('\ufeff', '').replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
				# Convert to ASCII only
				output = output.encode('ascii', 'ignore').decode('ascii')
				# Remove any remaining control characters except newline and tab
				output = ''.join(ch for ch in output if ch.isprintable() or ch in '\n\t')
				print(output, flush=True)
				any_coin_updated = True  # Mark that we printed something
		
		# Print summary table only if at least one coin was updated this iteration
		if any_coin_updated:
			valid_summaries = [summary_cache.get(_sym) for _sym in coins_this_iteration if summary_cache.get(_sym) and 'price_str' in summary_cache.get(_sym)]
			
			if valid_summaries:
				print("", flush=True)
				current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
				print(f"--- Thinking Summary {current_time} ---", flush=True)
				for _sym in coins_this_iteration:
					summary = summary_cache.get(_sym)
					if summary and 'price_str' in summary:
						# Format distances to boundaries
						if summary['next_long_pct'] is not None:
							long_str = f"{summary['next_long_pct']:+.1f}%"
						else:
							long_str = "-----"
						
						if summary['next_short_pct'] is not None:
							short_str = f"{summary['next_short_pct']:+.1f}%"
						else:
							short_str = "-----"
						
						# Dynamic spacing: 2 spaces for 3-letter tickers, 1 for 4-letter, 0 for 5-letter
						spacing = " " * max(0, 4 - len(_sym))
						print(f"[{_sym}]{spacing}{summary['price_str']:>8s}  {summary['signal']:6s}  Next Buy: {long_str}  Next Sell: {short_str}  PM: {summary['pm_display']}", flush=True)
		
		# Priority detection for auto-switch feature
		# Find the coin closest to a trigger (buy or exit), with tiebreakers
		try:
			trading_cfg = _load_trading_config()
			if trading_cfg.get("auto_switch", {}).get("enabled", False):
				priority_candidates = []
				
				for _sym in coins_this_iteration:
					summary = summary_cache.get(_sym)
					if not summary:
						continue
					
					# Check distance to buy trigger (next_long_pct is negative, distance is abs value)
					if summary['next_long_pct'] is not None:
						coin_distance = abs(summary['next_long_pct'])
						priority_candidates.append({
							'coin': _sym,
							'distance': coin_distance,
							'reason': 'buy',
							'price': summary['price']
						})
					
					# Check distance to exit trigger (next_short_pct is positive, distance is abs value)
					if summary['next_short_pct'] is not None:
						coin_distance = abs(summary['next_short_pct'])
						priority_candidates.append({
							'coin': _sym,
							'distance': coin_distance,
							'reason': 'exit',
							'price': summary['price']
						})
				
				if priority_candidates:
					# Sort by:
					# 1. Closest distance (ascending)
					# 2. Largest dollar value (descending) - using price as proxy
					# 3. Coin order in list (ascending index)
					priority_candidates.sort(key=lambda x: (
						x['distance'],
						-x['price'],
						coins_this_iteration.index(x['coin']) if x['coin'] in coins_this_iteration else 999
					))
					
					winner = priority_candidates[0]
					_write_priority_coin(winner['coin'], winner['distance'], winner['reason'])
		except Exception:
			pass  # Don't crash thinker if priority detection fails
		
		# small sleep so you don't peg CPU when running many coins
		time.sleep(_get_sleep_timing("sleep_main_loop"))

except Exception:
	PrintException()


