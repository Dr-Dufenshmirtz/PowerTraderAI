"""
pt_trader.py - ApolloTrader Trading Engine

Description:
This module implements the live trading component of ApolloTrader. It
wraps broker interactions (Robinhood), per-coin state management, DCA and
trading decision logic, and local GUI persistence (trade history, PnL
ledger, status). The `CryptoAPITrading` class encapsulates primary
trading workflows and safety checks used by the GUI.

Primary Repository: https://github.com/Dr-Dufenshmirtz/ApolloTrader
Primary Author: Dr Dufenshmirtz

Original Project: https://github.com/garagesteve1155/PowerTrader_AI
Original Author: Stephen Hughes (garagesteve1155)

Key behavioral notes (informational only):

- Coin Categories:
    Trades are governed by coin category (active/accumulate/liquidate) from gui_settings.json:
    * Active Trading: Full buy/sell automation, counts toward max_concurrent_positions limit
    * Accumulation: Buy-only (entry + DCA allowed), sells only on stop-loss, unlimited positions
    * Liquidation: Sell-only (no entry/DCA), auto-removes from list at dust threshold, unlimited positions
    Non-listed coins are completely ignored (safety feature).

- Entry signal (long positions):
    Start trades when long_signal >= entry_signals.long_signal_min (default 4)
    AND short_signal <= entry_signals.short_signal_max (default 0).
    Blocked for liquidation coins (sell-only).
    
    When multiple coins qualify simultaneously, opportunity scoring determines buy
    priority: coins are scored by long_signal strength (higher = better), and the
    highest-scoring coin is selected. Ties are broken by original coin list order.

- Exit signal (MTF confirmation):
    Trailing profit margin triggers sell when price crosses below trailing line,
    BUT requires multi-timeframe confirmation: short_signal >= exit_signals.short_signal_min
    (default 4) AND long_signal <= exit_signals.long_signal_max (default 0).
    Blocked for accumulation coins (buy-only), except stop-loss.
    Stop-loss (-40%) ALWAYS executes for all categories, bypassing MTF check for safety.

- DCA rules:
    DCA uses the AI's per-level price lines or a hardcoded drawdown % for
    the current DCA level — whichever triggers first. Max 2 DCA buys in a
    rolling 24-hour window to limit rapid overexposure.

- Trailing profit margin:
    Target: 5% if no DCA occurred, 3% if DCA triggered. Gap: 0.5%.
"""

import os
import sys
import time
import datetime
import math
import json
import uuid
import base64
import traceback
from typing import Any, Dict, Optional

# Ensure clean console output
try:
	sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
	sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
except:
	pass

# third-party
import requests
from nacl.signing import SigningKey

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
import colorama
from colorama import Fore, Style
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

# GUI hub outputs directory for trader status and trade history files
HUB_DATA_DIR = os.environ.get("POWERTRADER_HUB_DIR", os.path.join(os.path.dirname(__file__), "hub_data"))
os.makedirs(HUB_DATA_DIR, exist_ok=True)

# ============================================================================
# TRADER ALGORITHM CONSTANTS
# ============================================================================

# Profit margin and DCA settings
DEFAULT_TRAILING_GAP_PCT = 0.5  # Trailing profit margin gap percentage
DEFAULT_DCA_WINDOW_HOURS = 24  # Rolling DCA window in hours

# Position sizing
DEFAULT_MIN_ALLOCATION_USD = 0.5  # Minimum allocation in USD per coin

# Timing and execution
DEFAULT_MAIN_LOOP_DELAY = 0.5  # Main loop delay in seconds

# Percentage conversion
PERCENT_MULTIPLIER = 100  # Multiplier to convert decimal to percentage

# Debug mode support - reads from gui_settings.json to enable verbose logging
_GUI_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "gui_settings.json")
_debug_mode_cache = {"enabled": False}
_simulation_mode_cache = {"enabled": False}

def _is_debug_mode() -> bool:
	"""Check if debug mode is enabled in gui_settings.json"""
	try:
		if not os.path.isfile(_GUI_SETTINGS_PATH):
			return _debug_mode_cache["enabled"]
		with open(_GUI_SETTINGS_PATH, "r", encoding="utf-8") as f:
			data = json.load(f) or {}
		debug = data.get("debug_mode", False)
		_debug_mode_cache["enabled"] = bool(debug)
		return bool(debug)
	except Exception:
		return _debug_mode_cache["enabled"]

def _is_simulation_mode() -> bool:
	"""Check if simulation mode is enabled in gui_settings.json"""
	try:
		if not os.path.isfile(_GUI_SETTINGS_PATH):
			return _simulation_mode_cache["enabled"]
		with open(_GUI_SETTINGS_PATH, "r", encoding="utf-8") as f:
			data = json.load(f) or {}
		sim = data.get("simulation_mode", False)
		_simulation_mode_cache["enabled"] = bool(sim)
		return bool(sim)
	except Exception:
		return _simulation_mode_cache["enabled"]

# Circuit breaker for network errors - prevents process exit on transient API failures
_circuit_breaker = {
	"is_open": False,
	"consecutive_errors": 0,
	"last_error_time": None,
	"max_errors_before_open": 3,
	"cooldown_seconds": 60
}

def debug_print(msg: str):
	"""Print debug message only if debug mode is enabled, also log to file"""
	if _is_debug_mode():
		print(msg)
		# Also write to debug log file
		try:
			import datetime
			timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			with open("debug_trader.log", "a", encoding="utf-8") as f:
				f.write(f"[{timestamp}] {msg}\n")
		except Exception:
			pass  # Don't let logging errors break the trader

def sim_prefix() -> str:
	"""Return [SIM] prefix if simulation mode is enabled"""
	return "[SIM] " if _is_simulation_mode() else ""

def handle_network_error(operation: str, error: Exception):
	"""Set circuit breaker state instead of exiting on network errors"""
	_circuit_breaker["consecutive_errors"] += 1
	_circuit_breaker["last_error_time"] = time.time()
	
	print(f"\n{'='*60}")
	print(f"❌ NETWORK ERROR #{_circuit_breaker['consecutive_errors']}: {operation} failed")
	print(f"Error: {type(error).__name__}: {str(error)[:200]}")
	
	if _circuit_breaker["consecutive_errors"] >= _circuit_breaker["max_errors_before_open"]:
		if not _circuit_breaker["is_open"]:
			_circuit_breaker["is_open"] = True
			print(f"⚠ CIRCUIT BREAKER ACTIVATED (after {_circuit_breaker['consecutive_errors']} consecutive errors)")
			print(f"Trading PAUSED for {_circuit_breaker['cooldown_seconds']}s, monitoring continues")
			print(f"Will auto-recover when API connectivity restored")
		else:
			print(f"⚠ Circuit breaker still OPEN, trading paused")
	else:
		print(f"Retrying... ({_circuit_breaker['max_errors_before_open'] - _circuit_breaker['consecutive_errors']} attempts left before circuit opens)")
	
	print(f"Please check:")
	print(f"  1. Your internet connection")
	print(f"  2. API service status (Robinhood)")
	print(f"  3. IP whitelist settings for Robinhood API")
	print(f"  4. Enable debug_mode in gui_settings.json for more details")
	print(f"{'='*60}\n")
	
	debug_print(f"[DEBUG] TRADER: Circuit breaker state: {_circuit_breaker}")

def reset_circuit_breaker():
	"""Reset circuit breaker after successful API calls"""
	if _circuit_breaker["is_open"] or _circuit_breaker["consecutive_errors"] > 0:
		was_open = _circuit_breaker["is_open"]
		_circuit_breaker["is_open"] = False
		_circuit_breaker["consecutive_errors"] = 0
		_circuit_breaker["last_error_time"] = None
		if was_open:
			print(f"\n{'='*60}")
			print(f"CIRCUIT BREAKER CLOSED - API connectivity restored")
			print(f"Resuming normal trading operations")
			print(f"{'='*60}\n")
		debug_print(f"[DEBUG] TRADER: Circuit breaker reset after successful API call")

TRADER_STATUS_PATH = os.path.join(HUB_DATA_DIR, "trader_status.json")
TRADE_HISTORY_PATH = os.path.join(HUB_DATA_DIR, "trade_history.jsonl")
PNL_LEDGER_PATH = os.path.join(HUB_DATA_DIR, "pnl_ledger.json")
ACCOUNT_VALUE_HISTORY_PATH = os.path.join(HUB_DATA_DIR, "account_value_history.jsonl")

# Initialize colorama
colorama.init(autoreset=True)

# GUI settings for coins list and main neural directory path
_GUI_SETTINGS_PATH = os.environ.get("POWERTRADER_GUI_SETTINGS") or os.path.join(
	os.path.dirname(os.path.abspath(__file__)),
	"gui_settings.json"
)

_gui_settings_cache = {
	"mtime": None,
	"coins": ['ETH', 'SOL', 'XRP', 'LINK', 'BTC', 'DOGE'],  # fallback defaults (all categories combined)
	"main_neural_dir": None,
	"coin_categories": {"ETH": "active", "SOL": "active", "XRP": "active", "LINK": "active", "BTC": "accumulate", "DOGE": "liquidate"},
}

def _load_gui_settings() -> dict:
	"""
	Reads gui_settings.json and returns a dict with:
	- coins: uppercased list (all three categories combined)
	- main_neural_dir: string (may be None)
	- coin_categories: dict mapping coin to category ("active", "accumulate", "liquidate")
	Caches by mtime so it is cheap to call frequently.
	"""
	try:
		if not os.path.isfile(_GUI_SETTINGS_PATH):
			return dict(_gui_settings_cache)

		mtime = os.path.getmtime(_GUI_SETTINGS_PATH)
		if _gui_settings_cache["mtime"] == mtime:
			return dict(_gui_settings_cache)

		with open(_GUI_SETTINGS_PATH, "r", encoding="utf-8") as f:
			data = json.load(f) or {}

		main_neural_dir = data.get("main_neural_dir", "")
		# Handle new three-category structure or legacy list format
		coins_raw = data.get("coins", None)
		if isinstance(coins_raw, dict):
			# New format: combine all three categories
			all_coins = (
				coins_raw.get("active", []) + 
				coins_raw.get("accumulate", []) + 
				coins_raw.get("liquidate", [])
			)
			coins = [str(c).strip().upper() for c in all_coins if str(c).strip()]
			
			# Build category mapping
			coin_categories = {}
			for c in coins_raw.get("active", []):
				if c.strip():
					coin_categories[c.upper().strip()] = "active"
			for c in coins_raw.get("accumulate", []):
				if c.strip():
					coin_categories[c.upper().strip()] = "accumulate"
			for c in coins_raw.get("liquidate", []):
				if c.strip():
					coin_categories[c.upper().strip()] = "liquidate"
		elif isinstance(coins_raw, list):
			# Legacy format: all coins are active
			coins = [str(c).strip().upper() for c in coins_raw if str(c).strip()]
			coin_categories = {c: "active" for c in coins}
		else:
			# Fallback
			coins = list(_gui_settings_cache["coins"])
			coin_categories = _gui_settings_cache.get("coin_categories", {})
		
		if not coins:
			coins = list(_gui_settings_cache["coins"])
			coin_categories = _gui_settings_cache.get("coin_categories", {})
			main_neural_dir = main_neural_dir.strip() or None
		else:
			main_neural_dir = None

		_gui_settings_cache["mtime"] = mtime
		_gui_settings_cache["coins"] = coins
		_gui_settings_cache["main_neural_dir"] = main_neural_dir
		_gui_settings_cache["coin_categories"] = coin_categories

		return {
			"mtime": mtime,
			"coins": list(coins),
			"main_neural_dir": main_neural_dir,
			"coin_categories": dict(coin_categories),
		}
	except Exception:
		return dict(_gui_settings_cache)

def _build_base_paths(main_dir_in: str, coins_in: list) -> dict:
	"""
	All coins use their own subfolders: <main_dir>/<SYM>
	"""
	out = {}
	try:
		for sym in coins_in:
			sym = str(sym).strip().upper()
			if not sym:
				continue
			sub = os.path.join(main_dir_in, sym)
			if os.path.isdir(sub):
				out[sym] = sub
	except Exception:
		pass
	return out

# Live globals (will be refreshed inside manage_trades())
crypto_symbols = ['ETH', 'SOL', 'XRP', 'LINK', 'BTC', 'DOGE']
coin_categories = {"ETH": "active", "SOL": "active", "XRP": "active", "LINK": "active", "BTC": "accumulate", "DOGE": "liquidate"}

# Default main_dir behavior if settings are missing
main_dir = os.getcwd()
base_paths = {}

_last_settings_mtime = None

def _refresh_paths_and_symbols():
	"""
	Hot-reload coins + main_neural_dir while trader is running.
	Updates globals: crypto_symbols, main_dir, base_paths, coin_categories
	"""
	global crypto_symbols, main_dir, base_paths, coin_categories, _last_settings_mtime

	s = _load_gui_settings()
	mtime = s.get("mtime", None)

	# If settings file doesn't exist, keep current defaults
	if mtime is None:
		return

	if _last_settings_mtime == mtime:
		return

	_last_settings_mtime = mtime

	coins = s.get("coins") or list(crypto_symbols)
	mndir = s.get("main_neural_dir") or main_dir
	categories = s.get("coin_categories", {})

	# Keep it safe if folder isn't real on this machine
	if not os.path.isdir(mndir):
		mndir = os.getcwd()

	crypto_symbols = list(coins)
	coin_categories = dict(categories)
	main_dir = mndir
	base_paths = _build_base_paths(main_dir, crypto_symbols)

def _load_and_validate_api_credentials() -> tuple:
	"""Load and validate Robinhood API credentials with strict format checking.
	
	Returns:
		tuple: (api_key: str, base64_private_key: str)
	
	Raises:
		FileNotFoundError: If credential files don't exist
		ValueError: If credentials are invalid format
		RuntimeError: If decryption fails
	
	Validation checks:
		- Files exist and are readable
		- Decryption succeeds
		- API key length >= 20 characters (minimum for valid Robinhood key)
		- API key has no leading/trailing whitespace
		- Private key is valid base64
		- Private key decodes to exactly 32 bytes (Ed25519 requirement)
	"""
	key_file = 'rh_key.enc'
	secret_file = 'rh_secret.enc'
	
	# Step 1: Check file existence first
	if not os.path.exists(key_file):
		raise FileNotFoundError(
			f"API key file '{key_file}' not found.\n"
			"Open the GUI and go to Settings → Robinhood API → Setup / Update."
		)
	
	if not os.path.exists(secret_file):
		raise FileNotFoundError(
			f"Private key file '{secret_file}' not found.\n"
			"Open the GUI and go to Settings → Robinhood API → Setup / Update."
		)
	
	# Step 2: Decrypt credentials
	try:
		with open(key_file, 'rb') as f:
			encrypted_key = f.read()
			if not encrypted_key:
				raise ValueError(f"File '{key_file}' is empty")
			api_key = _decrypt_with_dpapi(encrypted_key).strip()
	except Exception as e:
		raise RuntimeError(f"Failed to decrypt API key from '{key_file}': {e}")
	
	try:
		with open(secret_file, 'rb') as f:
			encrypted_secret = f.read()
			if not encrypted_secret:
				raise ValueError(f"File '{secret_file}' is empty")
			base64_private_key = _decrypt_with_dpapi(encrypted_secret).strip()
	except Exception as e:
		raise RuntimeError(f"Failed to decrypt private key from '{secret_file}': {e}")
	
	# Step 3: Validate API key format
	if not api_key or len(api_key) == 0:
		raise ValueError("API key is empty after decryption")
	
	if len(api_key) < 20:
		raise ValueError(f"API key too short ({len(api_key)} chars, minimum 20 required)")
	
	if len(api_key) > 200:
		raise ValueError(f"API key too long ({len(api_key)} chars, maximum 200)")
	
	# Check for whitespace that would break authentication
	if api_key != api_key.strip():
		raise ValueError("API key contains leading or trailing whitespace")
	
	# Check for null bytes or control characters
	if '\x00' in api_key:
		raise ValueError("API key contains null bytes (corrupted)")
	
	# Check for newlines or tabs that could break API calls
	if '\n' in api_key or '\r' in api_key or '\t' in api_key:
		raise ValueError("API key contains newline or tab characters (corrupted)")
	
	# Validate character set (alphanumeric, hyphens, underscores are typical)
	import string
	allowed_chars = string.ascii_letters + string.digits + '-_'
	invalid_chars = [c for c in api_key if c not in allowed_chars]
	if invalid_chars:
		# Show first few invalid characters for debugging
		sample = ''.join(invalid_chars[:5])
		raise ValueError(f"API key contains invalid characters: {repr(sample)}")
	
	# Step 4: Validate private key format (must be valid base64)
	if not base64_private_key or len(base64_private_key) == 0:
		raise ValueError("Private key is empty after decryption")
	
	# Check for null bytes or control characters
	if '\x00' in base64_private_key:
		raise ValueError("Private key contains null bytes (corrupted)")
	
	# Check for newlines/whitespace that could break base64 decoding
	if '\n' in base64_private_key or '\r' in base64_private_key or '\t' in base64_private_key:
		raise ValueError("Private key contains newline or tab characters (corrupted)")
	
	# Validate base64 character set (A-Z, a-z, 0-9, +, /, =)
	import string
	base64_chars = string.ascii_letters + string.digits + '+/='
	invalid_chars = [c for c in base64_private_key if c not in base64_chars]
	if invalid_chars:
		sample = ''.join(set(invalid_chars[:5]))
		raise ValueError(f"Private key contains invalid base64 characters: {repr(sample)}")
	
	# Attempt to decode base64
	try:
		decoded_key = base64.b64decode(base64_private_key, validate=True)
	except Exception as e:
		raise ValueError(f"Private key is not valid base64 encoding: {e}")
	
	if len(decoded_key) != 32:
		raise ValueError(
			f"Private key must decode to exactly 32 bytes for Ed25519, got {len(decoded_key)} bytes"
		)
	
	# Verify decoded bytes are not all zeros (would be invalid key)
	if all(b == 0 for b in decoded_key):
		raise ValueError("Private key decodes to all zeros (invalid Ed25519 key)")
	
	# Step 5: Test that private key can create a SigningKey (final validation)
	try:
		from nacl.signing import SigningKey
		test_key = SigningKey(decoded_key)
		# If we get here, key is valid
	except Exception as e:
		raise ValueError(f"Private key cannot create valid SigningKey: {e}")
	
	return api_key, base64_private_key

#API STUFF - Read and validate encrypted keys
API_KEY = ""
BASE64_PRIVATE_KEY = ""

# Load and validate API credentials with comprehensive format checking
try:
    API_KEY, BASE64_PRIVATE_KEY = _load_and_validate_api_credentials()
    print(f"✅ API credentials loaded and validated successfully")
except FileNotFoundError as e:
    print(f"\n{'='*60}")
    print(f"❌ API CREDENTIAL ERROR: Files Not Found")
    print(f"{'='*60}")
    print(f"{e}")
    print(f"\nOpen the GUI and go to:")
    print(f"  Settings → Robinhood API → Setup / Update")
    print(f"{'='*60}\n")
    raise SystemExit(1)
except ValueError as e:
    print(f"\n{'='*60}")
    print(f"❌ API CREDENTIAL ERROR: Invalid Format")
    print(f"{'='*60}")
    print(f"{e}")
    print(f"\nYour credential files may be corrupted.")
    print(f"To fix:")
    print(f"  1. Open the GUI")
    print(f"  2. Go to Settings → Robinhood API → Setup / Update")
    print(f"  3. Re-generate your credentials")
    print(f"{'='*60}\n")
    raise SystemExit(1)
except RuntimeError as e:
    print(f"\n{'='*60}")
    print(f"❌ API CREDENTIAL ERROR: Decryption Failed")
    print(f"{'='*60}")
    print(f"{e}")
    print(f"\nThis can happen if:")
    print(f"  - Files were encrypted by a different Windows user")
    print(f"  - Files were copied from another machine")
    print(f"  - Windows user profile was recreated")
    print(f"\nTo fix, re-generate credentials on THIS machine:")
    print(f"  Settings → Robinhood API → Setup / Update")
    print(f"{'='*60}\n")
    raise SystemExit(1)
except Exception as e:
    print(f"\n{'='*60}")
    print(f"❌ API CREDENTIAL ERROR: Unexpected Error")
    print(f"{'='*60}")
    print(f"{e}")
    print(f"\nPlease report this error and re-generate credentials:")
    print(f"  Settings → Robinhood API → Setup / Update")
    print(f"{'='*60}\n")
    raise SystemExit(1)

# Note: the script intentionally exits if Robinhood credentials are missing.
# This keeps the trader from starting in an unauthenticated state where
# accidental live orders might be attempted. The GUI flow provides a
# setup wizard to generate and persist these credentials.

# Trading configuration loaded from trading_settings.json
_TRADING_SETTINGS_PATH = os.environ.get("POWERTRADER_TRADING_SETTINGS") or os.path.join(
	os.path.dirname(os.path.abspath(__file__)),
	"trading_settings.json"
)

_trading_settings_cache = {
	"mtime": None,
	"config": {
		"dca": {
			"levels": [-2.5, -5.0, -10.0, -20.0],
			"max_buys_per_window": 2,
			"window_hours": DEFAULT_DCA_WINDOW_HOURS,
			"position_multiplier": 2.0
		},
		"profit_margin": {
			"trailing_gap_pct": DEFAULT_TRAILING_GAP_PCT,
			"target_no_dca_pct": 5.0,
			"target_with_dca_pct": 3.0,
			"stop_loss_pct": -40.0
		},
		"entry_signals": {
			"long_signal_min": 4,
			"short_signal_max": 0
		},
		"exit_signals": {
			"short_signal_min": 4,
			"long_signal_max": 0
		},
		"position_sizing": {
			"initial_allocation_pct": 0.01,
			"min_allocation_usd": DEFAULT_MIN_ALLOCATION_USD,
			"max_concurrent_positions": 3
		},
		"timing": {
			"main_loop_delay_seconds": DEFAULT_MAIN_LOOP_DELAY,
			"post_trade_delay_seconds": 30
		}
	}
}

def _load_trading_config() -> dict:
	"""
	Reads trading_settings.json and returns config dict.
	Caches by mtime so it is cheap to call frequently.
	"""
	try:
		if not os.path.isfile(_TRADING_SETTINGS_PATH):
			return dict(_trading_settings_cache["config"])

		mtime = os.path.getmtime(_TRADING_SETTINGS_PATH)
		if _trading_settings_cache["mtime"] == mtime:
			return dict(_trading_settings_cache["config"])

		with open(_TRADING_SETTINGS_PATH, "r", encoding="utf-8") as f:
			data = json.load(f) or {}

		# Merge with defaults to ensure all keys exist
		config = dict(_trading_settings_cache["config"])
		for key in data:
			if isinstance(data[key], dict) and isinstance(config.get(key), dict):
				config[key].update(data[key])
			else:
				config[key] = data[key]

		_trading_settings_cache["mtime"] = mtime
		_trading_settings_cache["config"] = config
		return dict(config)
	except Exception:
		return dict(_trading_settings_cache["config"])

class CryptoAPITrading:
    def __init__(self):
        """Initialize the trading engine with API credentials, configuration, and state management.
        
        Sets up Robinhood API integration, loads DCA and profit margin settings from
        trading_settings.json, initializes cost basis tracking, and seeds the rolling 24-hour
        DCA window from trade history. All per-coin state (DCA levels, trailing PM, timestamps)
        is maintained in memory and persists across trading cycles.
        """
        # Keep a copy of the folder map so we can locate per-coin data files (memories, signals, etc.)
        # This mirrors the path resolution used in pt_trainer.py and pt_thinker.py
        self.path_map = dict(base_paths)
        
        # Store coin categories for buy/sell filtering
        self.coin_categories = dict(coin_categories)

        # Initialize Robinhood API credentials from encrypted key files
        self.api_key = API_KEY
        private_key_seed = base64.b64decode(BASE64_PRIVATE_KEY)
        self.private_key = SigningKey(private_key_seed)
        self.base_url = "https://trading.robinhood.com"

        # Load trading configuration (DCA levels, profit margins, timing) from trading_settings.json
        # This hot-reloads during execution so settings changes take effect without restart
        trading_cfg = _load_trading_config()

        # DCA state tracking: tracks which DCA stages have been triggered for each coin in the current trade
        # Stages are tracked by index (0, 1, 2, ...) rather than percentage values for cleaner neural/hardcoded logic
        self.dca_levels_triggered = {}  # { "BTC": [0, 1, 2], "ETH": [0], ... }
        self.dca_levels = trading_cfg.get("dca", {}).get("levels", [-2.5, -5.0, -10.0, -20.0])

        # Trailing profit margin per-coin state: each coin maintains isolated trailing stop state.
        # The trailing line starts at the PM target (5% no DCA, 2.5% with DCA) and follows price
        # upward by the trail gap percentage (default 0.5%). A sell triggers when price crosses
        # from above the trailing line to below it, capturing gains while allowing upside.
        # Structure: { "BTC": {"active": bool, "line": float, "peak": float, "was_above": bool}, ... }
        self.trailing_pm = {}
        self.trailing_gap_pct = trading_cfg.get("profit_margin", {}).get("trailing_gap_pct", DEFAULT_TRAILING_GAP_PCT)
        self.pm_start_pct_no_dca = trading_cfg.get("profit_margin", {}).get("target_no_dca_pct", 5.0)
        self.pm_start_pct_with_dca = trading_cfg.get("profit_margin", {}).get("target_with_dca_pct", 3.0)
        self.stop_loss_pct = trading_cfg.get("profit_margin", {}).get("stop_loss_pct", -40.0)

        # Calculate initial cost basis for all holdings by examining recent buy order execution prices
        # Uses LIFO reconstruction to match current quantities with most recent purchase prices
        self.cost_basis = self.calculate_cost_basis()
        
        # Initialize DCA tracking by counting buy orders since the last sell for each coin
        # Preserves state across restarts by reading actual order history from Robinhood API
        self.initialize_dca_levels()

        # GUI hub persistence: load cumulative realized profit/loss from local JSON ledger
        # Updated on every sell to track overall trading performance across all coins
        self._pnl_ledger = self._load_pnl_ledger()

        # Price caching: stores last successful bid/ask for each symbol to handle transient API failures
        # gracefully. If price fetch fails, uses cached values to prevent account value from spiking to $0
        # in the GUI, which would corrupt account history charts and cause false alarms.
        self._last_good_bid_ask = {}  # { "BTC-USD": {"ask": float, "bid": float, "ts": timestamp}, ... }

        # Account snapshot caching: stores last complete account state to prevent partial data from
        # corrupting GUI displays. Only updates when all required data (holdings, prices, buying power)
        # is successfully fetched. Falls back to cached snapshot if any component fails.
        self._last_good_account_snapshot = {
            "total_account_value": None,
            "buying_power": None,
            "holdings_sell_value": None,
            "holdings_buy_value": None,
            "percent_in_trade": None,
        }

        # DCA rate-limiting: enforces maximum DCA buys per rolling time window per trade to prevent
        # over-allocating capital during prolonged downtrends. The window resets when a trade closes
        # (sell order executes). Default: max 2 DCA buys per 24 hours per coin.
        self.max_dca_buys_per_window = trading_cfg.get("dca", {}).get("max_buys_per_window", 2)
        self.dca_window_seconds = trading_cfg.get("dca", {}).get("window_hours", DEFAULT_DCA_WINDOW_HOURS) * 60 * 60
        
        # DCA timestamp tracking: maintains lists of DCA buy timestamps for rolling window enforcement
        # Only includes buys after the most recent sell (current trade boundary) and within the window
        self._dca_buy_ts = {}         # { "BTC": [1234567890.0, 1234567900.0, ...], ... }
        self._dca_last_sell_ts = {}   # { "BTC": 1234567880.0, ... }
        
        # Seed DCA window from trade history file so limits persist across restarts
        # Reads trade_history.jsonl and reconstructs DCA buy timestamps for current trades
        self._seed_dca_window_from_history()

        # Cycle-level DCA tracking: prevents double-triggering the same DCA stage within a single
        # manage_trades() iteration when both neural and hardcoded conditions are met simultaneously
        self._dca_triggered_this_cycle = {}  # { "BTC": {0, 1, 2}, "ETH": {0}, ... }
        
        # Heartbeat tracking: periodic status messages to show trader is alive
        self._last_heartbeat = 0
        self._last_status_message = None  # Track last status to avoid duplicate prints

    def _atomic_write_json(self, path: str, data: dict) -> None:
        """Write JSON data atomically to prevent corruption from interrupted writes.
        
        Uses temp file + os.replace() pattern to ensure the target file is never in a
        partially-written state. Critical for status files read by the GUI hub during
        active trading. Silently fails on error to prevent trade execution from crashing.
        """
        try:
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)  # Atomic on Windows/Unix
        except Exception:
            pass  # Non-critical write, continue trading

    def _append_jsonl(self, path: str, obj: dict) -> None:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj) + "\n")
        except Exception:
            pass

    def _load_pnl_ledger(self) -> dict:
        try:
            if os.path.isfile(PNL_LEDGER_PATH):
                with open(PNL_LEDGER_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {"total_realized_profit_usd": 0.0, "last_updated_ts": time.time()}

    def _save_pnl_ledger(self) -> None:
        try:
            self._pnl_ledger["last_updated_ts"] = time.time()
            self._atomic_write_json(PNL_LEDGER_PATH, self._pnl_ledger)
        except Exception:
            pass

    @staticmethod
    def _extract_execution_price(order_response: Any) -> Optional[float]:
        """
        Extract actual fill price from Robinhood order response for accurate profit calculations.
        
        Robinhood market orders can execute across multiple fills at different prices. This method
        computes the weighted average effective_price from all execution records to get the true
        cost/proceeds per unit. Used instead of estimated market prices to ensure PnL tracking
        reflects actual execution prices, not bid/ask at order placement time.
        
        Returns:
            Weighted average fill price, or None if order hasn't executed yet or response is invalid
        """
        try:
            if not isinstance(order_response, dict):
                return None
            
            executions = order_response.get("executions", [])
            if not executions:
                return None
            
            total_cost = 0.0
            total_qty = 0.0
            
            for execution in executions:
                qty = float(execution.get("quantity", 0))
                price = float(execution.get("effective_price", 0))
                if qty > 0 and price > 0:
                    total_cost += qty * price
                    total_qty += qty
            
            if total_qty > 0:
                return total_cost / total_qty
            return None
        except Exception:
            return None

    def _record_trade(
        self,
        side: str,
        symbol: str,
        qty: float,
        price: Optional[float] = None,
        avg_cost_basis: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        tag: Optional[str] = None,
        order_id: Optional[str] = None,
    ) -> None:
        """
        Minimal local ledger for GUI:
        - append trade_history.jsonl
        - update pnl_ledger.json on sells (using actual execution price * qty)
        - store the exact PnL% at the moment for DCA buys / sells (for GUI trade history)
        """
        ts = time.time()
        realized = None
        is_simulation = (tag and "SIM_" in str(tag))
        
        # Compute realized PnL for sells when we have both price and avg cost.
        # Skip PnL accumulation for simulated trades to prevent fake profits
        # Guarded in a try/except to avoid one bad record from crashing the
        # trading process; failures here simply skip PnL aggregation.
        if side.lower() == "sell" and price is not None and avg_cost_basis is not None and not is_simulation:
            try:
                realized = (float(price) - float(avg_cost_basis)) * float(qty)
                self._pnl_ledger["total_realized_profit_usd"] = float(self._pnl_ledger.get("total_realized_profit_usd", 0.0)) + float(realized)
            except Exception:
                realized = None

        entry = {
            "ts": ts,
            "side": side,
            "tag": tag,
            "symbol": symbol,
            "qty": qty,
            "price": price,
            "avg_cost_basis": avg_cost_basis,
            "pnl_pct": pnl_pct,
            "realized_profit_usd": realized,
            "order_id": order_id,
        }
        self._append_jsonl(TRADE_HISTORY_PATH, entry)
        if realized is not None:
            self._save_pnl_ledger()

    def _write_trader_status(self, status: dict) -> None:
        self._atomic_write_json(TRADER_STATUS_PATH, status)

    @staticmethod
    def _get_current_timestamp() -> int:
        return int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp())

    @staticmethod
    def _fmt_price(price: float) -> str:
        """
        Dynamic decimal formatting by magnitude:
        - >= 1.0   -> 2 decimals (BTC/ETH/etc won't show 8 decimals)
        - <  1.0   -> enough decimals to show meaningful digits (based on first non-zero),
                     then trim trailing zeros.
        """
        try:
            p = float(price)
        except Exception:
            return "N/A"

        # Treat exact zero as a special-case for cleaner formatting.
        if p == 0:
            return "0"

        ap = abs(p)

        if ap >= 1.0:
            decimals = 2
        else:
            # Example:
            # 0.5      -> decimals ~ 4 (prints "0.5" after trimming zeros)
            # 0.05     -> 5
            # 0.005    -> 6
            # 0.000012 -> 8
            decimals = int(-math.floor(math.log10(ap))) + 3
            decimals = max(2, min(12, decimals))

        s = f"{p:.{decimals}f}"

        # Trim useless trailing zeros for cleaner output (0.5000 -> 0.5)
        if "." in s:
            s = s.rstrip("0").rstrip(".")

        return s

    @staticmethod
    def _read_validated_signal(
        file_path: str, 
        signal_name: str,
        min_value: int = 0,
        max_value: int = 10,
        default_value: int = 0
    ) -> int:
        """Read and validate a neural signal from file with range checking.
        
        Args:
            file_path: Path to signal file
            signal_name: Name for logging (e.g., "long_dca_signal")
            min_value: Minimum valid signal value (default 0)
            max_value: Maximum valid signal value (default 10 for 7 timeframes + buffer)
            default_value: Value to return on error or invalid range (default 0)
        
        Returns:
            Validated signal value or default_value if invalid
        
        Validation checks:
            - File exists and is readable
            - Content is parseable as integer
            - Value is within [min_value, max_value] range
            - Handles corrupt/malformed files gracefully
        """
        try:
            if not os.path.isfile(file_path):
                debug_print(f"[DEBUG] TRADER: Signal file not found: {file_path}, using default {default_value}")
                return default_value
            
            with open(file_path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            
            if not raw:
                debug_print(f"[DEBUG] TRADER: Empty signal file: {file_path}, using default {default_value}")
                return default_value
            
            # Parse as integer (handle float strings like "4.0")
            try:
                signal = int(float(raw))
            except ValueError as e:
                debug_print(f"[WARNING] TRADER: Invalid {signal_name} format '{raw}' in {file_path}: {e}, using default {default_value}")
                return default_value
            
            # Validate range
            if not (min_value <= signal <= max_value):
                debug_print(f"[WARNING] TRADER: {signal_name} value {signal} outside valid range [{min_value}, {max_value}] in {file_path}, using default {default_value}")
                return default_value
            
            return signal
            
        except Exception as e:
            debug_print(f"[ERROR] TRADER: Failed to read {signal_name} from {file_path}: {e}, using default {default_value}")
            return default_value

    @staticmethod
    def _read_long_dca_signal(symbol: str) -> int:
        """
        Reads long_dca_signal.dat from the per-coin folder (same folder rules as trader.py).

        Used for:
        - Start gate: start trades when signal >= entry_signals.long_signal_min (default 4)
        - DCA assist: stages offset from entry (e.g., if entry=4: stage 0 at L4, stage 1 at L5)
        
        Returns:
            Validated signal value (0-10 range) or 0 if invalid/missing
        """
        sym = str(symbol).upper().strip()
        folder = base_paths.get(sym, os.path.join(main_dir, sym))
        path = os.path.join(folder, "long_dca_signal.dat")
        return CryptoAPITrading._read_validated_signal(
            path, 
            f"{sym}_long_dca_signal",
            min_value=0,
            max_value=10,
            default_value=0
        )

    @staticmethod
    def _read_short_dca_signal(symbol: str) -> int:
        """
        Reads short_dca_signal.dat from the per-coin folder (same folder rules as trader.py).

        Used for:
        - Start gate: start trades when signal <= entry_signals.short_signal_max (default 0)
        - DCA assist: levels map to trader DCA stages for short positions
        
        Returns:
            Validated signal value (0-10 range) or 0 if invalid/missing
        """
        sym = str(symbol).upper().strip()
        folder = base_paths.get(sym, os.path.join(main_dir, sym))
        path = os.path.join(folder, "short_dca_signal.dat")
        return CryptoAPITrading._read_validated_signal(
            path, 
            f"{sym}_short_dca_signal",
            min_value=0,
            max_value=10,
            default_value=0
        )

    @staticmethod
    def _read_long_price_levels(symbol: str) -> list:
        """
        Reads low_bound_prices.dat from the per-coin folder and returns a list of LONG (blue) price levels.
        Returned ordering is highest->lowest so:
          N1 = 1st blue line (top)
          ...
          N7 = 7th blue line (bottom)
        """
        sym = str(symbol).upper().strip()
        folder = base_paths.get(sym, main_dir if sym == "BTC" else os.path.join(main_dir, sym))
        path = os.path.join(folder, "low_bound_prices.dat")
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = (f.read() or "").strip()
            if not raw:
                return []

            # Normalize common formats: python-list, comma-separated, newline-separated
            raw = raw.strip().strip("[]()")
            raw = raw.replace(",", " ").replace(";", " ").replace("|", " ")
            raw = raw.replace("\n", " ").replace("\t", " ")
            parts = [p for p in raw.split() if p]

            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except Exception:
                    continue

            # De-dupe, then sort high->low for stable N1..N7 mapping
            out = []
            seen = set()
            for v in vals:
                rounded_price = round(float(v), 12)
                if rounded_price in seen:
                    continue
                seen.add(rounded_price)
                out.append(float(v))
            out.sort(reverse=True)
            return out
        except Exception:
            return []

    @staticmethod
    def _read_short_price_levels(symbol: str) -> list:
        """
        Reads high_bound_prices.dat from the per-coin folder and returns a list of SHORT (orange) price levels.
        Returned ordering is lowest->highest so:
          N1 = 1st orange line (bottom)
          ...
          N7 = 7th orange line (top)
        """
        sym = str(symbol).upper().strip()
        folder = base_paths.get(sym, main_dir if sym == "BTC" else os.path.join(main_dir, sym))
        path = os.path.join(folder, "high_bound_prices.dat")
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = (f.read() or "").strip()
            if not raw:
                return []

            # Normalize common formats
            raw = raw.strip().strip("[]()")
            raw = raw.replace(",", " ").replace(";", " ").replace("|", " ")
            raw = raw.replace("\n", " ").replace("\t", " ")
            parts = [p for p in raw.split() if p]

            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except Exception:
                    continue

            # De-dupe, then sort low->high for stable N1..N7 mapping
            out = []
            seen = set()
            for v in vals:
                rounded_price = round(float(v), 12)
                if rounded_price in seen:
                    continue
                seen.add(rounded_price)
                out.append(float(v))
            out.sort()  # ascending order
            return out
        except Exception:
            return []

    def initialize_dca_levels(self):
        """
        Initialize DCA tracking by reconstructing current trade state from order history.
        
        For each held coin, determines which DCA stages have been triggered in the current
        trade by counting buy orders since the last sell. The first buy after a sell (or
        the first ever buy) is considered the trade entry (neural signal >= long_signal_min,
        default 4). Each additional buy represents one triggered DCA stage (stage 0, 1, 2, ...).
        
        State preservation: If order history is temporarily unavailable but we're still
        holding the position, preserves existing DCA state rather than resetting to empty.
        This prevents false re-triggers when the API has transient issues. Only resets to
        [] when we confirm the position is fully closed (quantity == 0).
        
        Called at startup and after any trade completes to synchronize in-memory state with
        actual execution history. Critical for preventing duplicate DCA triggers across restarts.
        """
        holdings = self.get_holdings()
        if not holdings or "results" not in holdings:
            print("No holdings found, skipping DCA initialization")
            return

        for holding in holdings.get("results", []):
            symbol = holding["asset_code"]
            
            # Check if we actually have a position before modifying DCA state
            try:
                holding_qty = float(holding.get("total_quantity", 0))
            except (ValueError, TypeError):
                holding_qty = 0.0
            
            debug_print(f"[DEBUG] TRADER: Initialize DCA for {symbol}, holding_qty={holding_qty}")

            full_symbol = f"{symbol}-USD"
            orders = self.get_orders(full_symbol)

            if not orders or "results" not in orders:
                debug_print(f"[DEBUG] TRADER: No orders found for {full_symbol}, preserving DCA state")
                # Preserve existing state if no orders data available
                if symbol not in self.dca_levels_triggered:
                    self.dca_levels_triggered[symbol] = []
                continue

            # Filter for filled buy and sell orders
            filled_orders = [
                order for order in orders["results"]
                if order["state"] == "filled" and order["side"] in ["buy", "sell"]
            ]

            if not filled_orders:
                debug_print(f"[DEBUG] TRADER: No filled orders for {full_symbol}, preserving DCA state")
                # Preserve existing state if no filled orders
                if symbol not in self.dca_levels_triggered:
                    self.dca_levels_triggered[symbol] = []
                continue

            # Sort orders by creation time in ascending order (oldest first)
            filled_orders.sort(key=lambda x: x["created_at"])

            # Find the timestamp of the most recent sell order
            most_recent_sell_time = None
            for order in reversed(filled_orders):
                if order["side"] == "sell":
                    most_recent_sell_time = order["created_at"]
                    break

            # Determine the cutoff time for buy orders
            if most_recent_sell_time:
                # Find all buy orders after the most recent sell
                relevant_buy_orders = [
                    order for order in filled_orders
                    if order["side"] == "buy" and order["created_at"] > most_recent_sell_time
                ]
                if not relevant_buy_orders:
                    # Only reset if we have no holding (position closed after sell)
                    if holding_qty <= 0.0:
                        debug_print(f"[DEBUG] TRADER: No holding for {symbol}, resetting DCA to []")
                        self.dca_levels_triggered[symbol] = []
                    else:
                        debug_print(f"[DEBUG] TRADER: Have holding {holding_qty} but no buy orders after sell, preserving DCA state for {symbol}")
                        # Preserve existing state - might be mid-transaction or data sync issue
                        if symbol not in self.dca_levels_triggered:
                            self.dca_levels_triggered[symbol] = []
                    continue
                debug_print(f"[DEBUG] TRADER: Most recent sell for {full_symbol} at {most_recent_sell_time}")
            else:
                # If no sell orders, consider all buy orders
                relevant_buy_orders = [
                    order for order in filled_orders
                    if order["side"] == "buy"
                ]
                if not relevant_buy_orders:
                    # Only reset if we have no holding
                    if holding_qty <= 0.0:
                        debug_print(f"[DEBUG] TRADER: No holding and no buy orders for {symbol}, resetting DCA to []")
                        self.dca_levels_triggered[symbol] = []
                    else:
                        debug_print(f"[DEBUG] TRADER: Have holding {holding_qty} but no buy orders found, preserving DCA state for {symbol}")
                        if symbol not in self.dca_levels_triggered:
                            self.dca_levels_triggered[symbol] = []
                    continue
                debug_print(f"[DEBUG] TRADER: No sell orders found for {full_symbol}. Considering all buy orders.")

            # Ensure buy orders are sorted by creation time ascending
            relevant_buy_orders.sort(key=lambda x: x["created_at"])

            # Identify the first buy order in the relevant list
            first_buy_order = relevant_buy_orders[0]
            first_buy_time = first_buy_order["created_at"]

            # Count the number of buy orders after the first buy
            buy_orders_after_first = [
                order for order in relevant_buy_orders
                if order["created_at"] > first_buy_time
            ]

            triggered_levels_count = len(buy_orders_after_first)

            # Track DCA by stage index (0, 1, 2, ...) rather than % values.
            # This makes neural-vs-hardcoded clean, and allows repeating the -50% stage indefinitely.
            self.dca_levels_triggered[symbol] = list(range(triggered_levels_count))
            debug_print(f"[DEBUG] TRADER: Initialized DCA stages for {symbol}: {triggered_levels_count} (holding_qty={holding_qty})")
            print(f"[{symbol}] Initialized DCA stages: {triggered_levels_count}")

    def _seed_dca_window_from_history(self) -> None:
        """
        Reconstruct DCA buy timestamp tracking from trade_history.jsonl for rolling window enforcement.
        
        Reads the local trade history file (written by _record_trade) and extracts all DCA buy
        timestamps for each coin, filtering to only buys that occurred after the most recent
        sell (the current trade boundary). This ensures the 24-hour DCA limit persists across
        trader restarts - we don't reset the count just because the process restarted.
        
        Also validates trade history data integrity: logs warnings for corrupted entries (invalid
        JSON, missing fields, malformed timestamps) to help diagnose trade_history.jsonl corruption.
        Skipped entries don't affect trading but indicate potential file corruption that should be
        investigated.
        
        Called once during __init__() to seed the rolling window state before trading begins.
        """
        now_ts = time.time()
        cutoff = now_ts - float(getattr(self, "dca_window_seconds", 86400))

        self._dca_buy_ts = {}
        self._dca_last_sell_ts = {}

        if not os.path.isfile(TRADE_HISTORY_PATH):
            debug_print("[DEBUG] TRADER: No trade history file found for DCA window seeding")
            return

        # Track validation stats for logging
        total_lines = 0
        skipped_empty = 0
        skipped_invalid_json = 0
        skipped_missing_fields = 0
        skipped_invalid_timestamp = 0
        processed_sells = 0
        processed_dca_buys = 0

        try:
            with open(TRADE_HISTORY_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    total_lines += 1
                    line = (line or "").strip()
                    if not line:
                        skipped_empty += 1
                        continue

                    try:
                        obj = json.loads(line)
                    except Exception as e:
                        skipped_invalid_json += 1
                        debug_print(f"[DEBUG] TRADER: Invalid JSON in trade history line {total_lines}: {e}")
                        continue

                    ts = obj.get("ts", None)
                    side = str(obj.get("side", "")).lower()
                    tag = obj.get("tag", None)
                    sym_full = str(obj.get("symbol", "")).upper().strip()
                    base = sym_full.split("-")[0].strip() if sym_full else ""
                    if not base:
                        skipped_missing_fields += 1
                        debug_print(f"[DEBUG] TRADER: Missing symbol in trade history line {total_lines}")
                        continue

                    try:
                        ts_f = float(ts)
                    except Exception:
                        skipped_invalid_timestamp += 1
                        debug_print(f"[DEBUG] TRADER: Invalid timestamp in trade history line {total_lines}: {ts}")
                        continue

                    if side == "sell":
                        prev = float(self._dca_last_sell_ts.get(base, 0.0) or 0.0)
                        if ts_f > prev:
                            self._dca_last_sell_ts[base] = ts_f
                            processed_sells += 1

                    elif side == "buy" and tag == "DCA":
                        self._dca_buy_ts.setdefault(base, []).append(ts_f)
                        processed_dca_buys += 1

        except Exception as e:
            msg = f"WARNING: Error reading trade history file: {e}"
            print(msg)
            debug_print(f"[DEBUG] TRADER: Trade history read error: {e}")
            # Log to file
            try:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open("debug_trader.log", "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] {msg}\n")
            except Exception:
                pass
            return

        # Log validation summary
        debug_print(f"[DEBUG] TRADER: DCA window seeding complete:")
        debug_print(f"  Total lines: {total_lines}")
        debug_print(f"  Processed: {processed_dca_buys} DCA buys, {processed_sells} sells")
        skipped_total = skipped_empty + skipped_invalid_json + skipped_missing_fields + skipped_invalid_timestamp
        if skipped_total > 0:
            print(f"⚠ WARNING: Skipped {skipped_total} invalid entries in trade history:")
            if skipped_empty > 0:
                debug_print(f"  - {skipped_empty} empty lines")
            if skipped_invalid_json > 0:
                print(f"  - {skipped_invalid_json} invalid JSON entries")
                debug_print(f"    (Check trade_history.jsonl for corruption)")
            if skipped_missing_fields > 0:
                print(f"  - {skipped_missing_fields} entries with missing fields")
            if skipped_invalid_timestamp > 0:
                print(f"  - {skipped_invalid_timestamp} entries with invalid timestamps")

        # Keep only DCA buys after the last sell (current trade) and within rolling 24h
        for base, ts_list in list(self._dca_buy_ts.items()):
            last_sell = float(self._dca_last_sell_ts.get(base, 0.0) or 0.0)
            kept = [t for t in ts_list if (t > last_sell) and (t >= cutoff)]
            kept.sort()
            self._dca_buy_ts[base] = kept

    def _dca_window_count(self, base_symbol: str, now_ts: Optional[float] = None) -> int:
        """
        Count DCA buys within the rolling 24-hour window for the current trade.
        
        The rolling window is scoped to the current trade: only counts DCAs that occurred
        after the most recent sell. This ensures each trade gets its own fresh DCA allowance
        rather than being penalized for DCA buys from previous trades. The window slides
        forward with time, dropping buys older than 24 hours automatically.
        
        Called before each potential DCA trigger to check if we've hit the limit (default 2
        buys per 24h). Prevents excessive averaging down during sustained price crashes by
        rate-limiting capital allocation.
        
        Args:
            base_symbol: Coin symbol without pair suffix (e.g., "BTC" not "BTC-USD")
            now_ts: Optional timestamp for testing; defaults to current time
            
        Returns:
            Number of DCA buys in the rolling window for this trade (0 to max_dca_buys_per_window)
        """
        base = str(base_symbol).upper().strip()
        if not base:
            return 0

        now = float(now_ts if now_ts is not None else time.time())
        cutoff = now - float(getattr(self, "dca_window_seconds", 86400))
        last_sell = float(self._dca_last_sell_ts.get(base, 0.0) or 0.0)

        ts_list = list(self._dca_buy_ts.get(base, []) or [])
        ts_list = [t for t in ts_list if (t > last_sell) and (t >= cutoff)]
        self._dca_buy_ts[base] = ts_list
        return len(ts_list)

    def _note_dca_buy(self, base_symbol: str, ts: Optional[float] = None) -> None:
        base = str(base_symbol).upper().strip()
        if not base:
            return
        t = float(ts if ts is not None else time.time())
        self._dca_buy_ts.setdefault(base, []).append(t)
        self._dca_window_count(base, now_ts=t)  # prune in-place

    def _reset_dca_window_for_trade(self, base_symbol: str, sold: bool = False, ts: Optional[float] = None) -> None:
        base = str(base_symbol).upper().strip()
        if not base:
            return
        if sold:
            self._dca_last_sell_ts[base] = float(ts if ts is not None else time.time())
        self._dca_buy_ts[base] = []

    def make_api_request(self, method: str, path: str, body: Optional[str] = "") -> Any:
        """
        Execute HTTP request to Robinhood API with authentication and retry logic.
        
        Handles API authentication via HMAC-signed headers, implements automatic retry with
        exponential backoff for transient failures (network timeouts, server errors), and
        triggers the circuit breaker on persistent failures to temporarily halt trading.
        
        Non-retryable errors (401/403 auth failures) are returned immediately without retries
        since they indicate configuration issues that won't resolve automatically.
        
        Args:
            method: HTTP method ("GET" or "POST")
            path: API endpoint path (e.g., "/api/v1/crypto/trading/orders/")
            body: JSON body for POST requests (empty string for GET)
            
        Returns:
            Parsed JSON response dict, or None if all retries exhausted
        """
        timestamp = self._get_current_timestamp()
        headers = self.get_authorization_header(method, path, body, timestamp)
        url = self.base_url + path

        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if method == "GET":
                    response = requests.get(url, headers=headers, timeout=15)
                elif method == "POST":
                    response = requests.post(url, headers=headers, json=json.loads(body), timeout=15)

                response.raise_for_status()
                return response.json()
            except requests.HTTPError as http_err:
                try:
                    # Parse and return the JSON error response
                    error_response = response.json()
                    # Check if it's an auth/IP error that won't recover with retries
                    if response.status_code in [401, 403]:
                        return error_response
                    retry_count += 1
                    if retry_count >= max_retries:
                        handle_network_error(f"Robinhood API {method} {path}", http_err)
                    time.sleep(2)
                except Exception:
                    retry_count += 1
                    if retry_count >= max_retries:
                        handle_network_error(f"Robinhood API {method} {path}", http_err)
                    time.sleep(2)
            except requests.Timeout as timeout_err:
                retry_count += 1
                debug_print(f"[DEBUG] TRADER: API timeout (attempt {retry_count}/{max_retries}): {path}")
                if retry_count >= max_retries:
                    handle_network_error(f"Robinhood API {method} {path} (timeout)", timeout_err)
                time.sleep(2)
            except Exception as e:
                retry_count += 1
                debug_print(f"[DEBUG] TRADER: API error (attempt {retry_count}/{max_retries}): {type(e).__name__}")
                if retry_count >= max_retries:
                    handle_network_error(f"Robinhood API {method} {path}", e)
                time.sleep(2)
        
        return None

    def get_authorization_header(
            self, method: str, path: str, body: str, timestamp: int
    ) -> Dict[str, str]:
        message_to_sign = f"{self.api_key}{timestamp}{path}{method}{body}"
        signed = self.private_key.sign(message_to_sign.encode("utf-8"))

        return {
            "x-api-key": self.api_key,
            "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
            "x-timestamp": str(timestamp),
        }

    def get_account(self) -> Any:
        path = "/api/v1/crypto/trading/accounts/"
        return self.make_api_request("GET", path)

    def get_holdings(self) -> Any:
        path = "/api/v1/crypto/trading/holdings/"
        return self.make_api_request("GET", path)

    def get_trading_pairs(self) -> Any:
        path = "/api/v1/crypto/trading/trading_pairs/"
        response = self.make_api_request("GET", path)

        if not response or "results" not in response:
            return []

        trading_pairs = response.get("results", [])
        if not trading_pairs:
            return []

        return trading_pairs

    def get_orders(self, symbol: str) -> Any:
        path = f"/api/v1/crypto/trading/orders/?symbol={symbol}"
        return self.make_api_request("GET", path)

    def calculate_cost_basis(self):
        """Calculate weighted average cost basis for all currently held assets.
        
        Uses LIFO (Last In First Out) reconstruction: works backward from most recent buy orders
        to match the current held quantity. This reflects the actual cost of the specific units
        we still hold after any partial sells. Each buy order's execution records provide the
        actual fill prices (not estimates) for precise cost basis calculation.
        
        Called at startup to initialize cost basis, then maintained incrementally via
        recalculate_single_cost_basis() after each trade for performance.
        
        Returns:
            dict: { "BTC": avg_cost_per_unit, "ETH": avg_cost_per_unit, ... }
        """
        holdings = self.get_holdings()
        if not holdings or "results" not in holdings:
            return {}

        # Build set of assets we currently hold and their quantities
        active_assets = {holding["asset_code"] for holding in holdings.get("results", [])}
        current_quantities = {
            holding["asset_code"]: float(holding["total_quantity"])
            for holding in holdings.get("results", [])
        }

        cost_basis = {}

        for asset_code in active_assets:
            orders = self.get_orders(f"{asset_code}-USD")
            if not orders or "results" not in orders:
                continue

            # Get all filled buy orders sorted oldest to newest (will process in reverse for LIFO)
            buy_orders = [
                order for order in orders["results"]
                if order["side"] == "buy" and order["state"] == "filled"
            ]
            buy_orders.sort(key=lambda x: x["created_at"])

            remaining_quantity = current_quantities[asset_code]
            total_cost = 0.0

            # Work backwards from newest orders since we need the most recent buys
            # to match the current holdings (LIFO for reconstruction)
            for order in reversed(buy_orders):
                for execution in order.get("executions", []):
                    quantity = float(execution["quantity"])
                    price = float(execution["effective_price"])

                    if remaining_quantity <= 0:
                        break

                    # Use only the portion of the quantity needed to match the current holdings
                    if quantity > remaining_quantity:
                        total_cost += remaining_quantity * price
                        remaining_quantity = 0
                    else:
                        total_cost += quantity * price
                        remaining_quantity -= quantity

                if remaining_quantity <= 0:
                    break

            if current_quantities[asset_code] > 0:
                cost_basis[asset_code] = total_cost / current_quantities[asset_code]
            else:
                cost_basis[asset_code] = 0.0

        return cost_basis

    def recalculate_single_cost_basis(self, symbol: str) -> Optional[float]:
        """
        Recalculate cost basis for a single coin immediately after a buy or sell executes.
        
        This incremental update is much faster than recalculating all holdings and ensures
        that subsequent DCA trigger checks and trailing profit margin calculations in the
        same trading cycle use the updated cost basis. Uses the same LIFO reconstruction
        as calculate_cost_basis() but scoped to one symbol.
        
        Critical for accuracy: if we buy and the price drops to a DCA level in the same
        cycle, we need the new blended cost basis to correctly calculate the DCA trigger
        percentage. Similarly for trailing PM: the profit margin line must reflect the
        updated cost after DCA buys.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC-USD")
            
        Returns:
            New cost basis per unit, or None if position is closed or calculation fails
        """
        try:
            asset_code = symbol.replace("-USD", "").strip()
            debug_print(f"[DEBUG] TRADER: Recalculating cost basis for {asset_code}...")
            
            holdings = self.get_holdings()
            if not holdings or "results" not in holdings:
                debug_print(f"[DEBUG] TRADER: No holdings found for {asset_code}")
                return None

            # Find current quantity for this asset
            current_quantity = 0.0
            for holding in holdings.get("results", []):
                if holding["asset_code"] == asset_code:
                    current_quantity = float(holding["total_quantity"])
                    break

            if current_quantity <= 0:
                # No longer holding this asset
                if asset_code in self.cost_basis:
                    del self.cost_basis[asset_code]
                debug_print(f"[DEBUG] TRADER: Removed cost basis for {asset_code} (position closed)")
                return None

            # Get recent buy orders
            orders = self.get_orders(symbol)
            if not orders or "results" not in orders:
                debug_print(f"[DEBUG] TRADER: No orders found for {symbol}")
                return None

            buy_orders = [
                order for order in orders["results"]
                if order["side"] == "buy" and order["state"] == "filled"
            ]
            # Sort oldest to newest for proper FIFO accounting
            buy_orders.sort(key=lambda x: x["created_at"])

            remaining_quantity = current_quantity
            total_cost = 0.0

            # Work backwards from newest orders since we need the most recent buys
            # to match the current holdings (LIFO for reconstruction)
            for order in reversed(buy_orders):
                for execution in order.get("executions", []):
                    quantity = float(execution["quantity"])
                    price = float(execution["effective_price"])

                    if remaining_quantity <= 0:
                        break

                    if quantity > remaining_quantity:
                        total_cost += remaining_quantity * price
                        remaining_quantity = 0
                    else:
                        total_cost += quantity * price
                        remaining_quantity -= quantity

                if remaining_quantity <= 0:
                    break

            if current_quantity > 0:
                new_basis = total_cost / current_quantity
                self.cost_basis[asset_code] = new_basis
                debug_print(f"[DEBUG] TRADER: Updated cost basis for {asset_code}: ${new_basis:.2f}")
                return new_basis
            else:
                return None

        except Exception as e:
            debug_print(f"[DEBUG] TRADER: Failed to recalculate cost basis for {symbol}: {e}")
            return None

    def get_price(self, symbols: list) -> Dict[str, float]:
        buy_prices = {}
        sell_prices = {}
        valid_symbols = []

        for symbol in symbols:
            if symbol == "USDC-USD":
                continue

            path = f"/api/v1/crypto/marketdata/best_bid_ask/?symbol={symbol}"
            response = self.make_api_request("GET", path)

            if response and "results" in response:
                result = response["results"][0]
                ask = float(result["ask_inclusive_of_buy_spread"])
                bid = float(result["bid_inclusive_of_sell_spread"])

                buy_prices[symbol] = ask
                sell_prices[symbol] = bid
                valid_symbols.append(symbol)

                # Update cache for transient failures later
                try:
                    self._last_good_bid_ask[symbol] = {"ask": ask, "bid": bid, "ts": time.time()}
                except Exception:
                    pass
            else:
                # Fallback to cached bid/ask so account value never drops due to a transient miss
                cached = None
                try:
                    cached = self._last_good_bid_ask.get(symbol)
                except Exception:
                    cached = None

                if cached:
                    ask = float(cached.get("ask", 0.0) or 0.0)
                    bid = float(cached.get("bid", 0.0) or 0.0)
                    if ask > 0.0 and bid > 0.0:
                        buy_prices[symbol] = ask
                        sell_prices[symbol] = bid
                        valid_symbols.append(symbol)

        return buy_prices, sell_prices, valid_symbols

    def place_buy_order(
        self,
        client_order_id: str,
        side: str,
        order_type: str,
        symbol: str,
        amount_in_usd: float,
        avg_cost_basis: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        tag: Optional[str] = None,
    ) -> Any:
        """
        Execute a market buy order for a specified USD amount of a cryptocurrency.
        
        Converts USD amount to crypto quantity at current ask price, handles Robinhood's
        precision requirements via retry logic, and records the trade with actual execution
        price (not estimated) for accurate cost basis and PnL tracking. In simulation mode,
        records a fake trade without hitting the API.
        
        Precision handling: Robinhood requires specific decimal precision per asset. If the
        initial order fails due to "too much precision", extracts the required precision from
        the error response and retries with rounded quantity. Retries up to 5 times with
        price refreshes after attempt 2 to avoid stale prices during volatility.
        
        Args:
            client_order_id: Unique order ID for idempotency
            side: Must be "buy"
            order_type: Must be "market"
            symbol: Trading pair (e.g., "BTC-USD")
            amount_in_usd: Dollar amount to spend (e.g., 100.00)
            avg_cost_basis: Current cost basis for PnL calculation at order time
            pnl_pct: Pre-calculated PnL percentage (for DCA orders)
            tag: Order classification ("ENTRY", "DCA", etc.) for trade history
            
        Returns:
            Robinhood API response dict on success, None on failure
        """
        debug_print(f"[DEBUG] TRADER: Placing BUY order for {symbol}: ${amount_in_usd:.2f} USD")
        # Fetch the current price of the asset
        current_buy_prices, current_sell_prices, valid_symbols = self.get_price([symbol])
        current_price = current_buy_prices[symbol]
        asset_quantity = amount_in_usd / current_price

        # Simulation mode: skip actual trade execution
        if _is_simulation_mode():
            rounded_quantity = round(asset_quantity, 8)
            fake_response = {
                "id": f"sim_{uuid.uuid4()}",
                "state": "filled",
                "side": "buy",
                "symbol": symbol
            }
            self._record_trade(
                side="buy",
                symbol=symbol,
                qty=float(rounded_quantity),
                price=float(current_price),
                avg_cost_basis=float(avg_cost_basis) if avg_cost_basis is not None else None,
                pnl_pct=float(pnl_pct) if pnl_pct is not None else None,
                tag=f"SIM_{tag}" if tag else "SIM_BUY",
                order_id=fake_response["id"],
            )
            debug_print(f"[DEBUG] TRADER: [SIM] Buy order simulated, updating cost basis...")
            time.sleep(2)
            self.recalculate_single_cost_basis(symbol)
            return fake_response

        max_retries = 5
        retries = 0

        while retries < max_retries:
            retries += 1
            
            # Refresh price if we've retried more than 2 times to avoid stale prices
            if retries > 2:
                debug_print(f"[DEBUG] TRADER: Retry {retries} - refreshing price for {symbol}")
                try:
                    fresh_buy_prices, fresh_sell_prices, fresh_valid = self.get_price([symbol])
                    if symbol in fresh_buy_prices and fresh_buy_prices[symbol] > 0:
                        current_price = fresh_buy_prices[symbol]
                        asset_quantity = amount_in_usd / current_price
                        debug_print(f"[DEBUG] TRADER: Updated price to ${current_price:.2f}, quantity to {asset_quantity:.8f}")
                    else:
                        debug_print(f"[DEBUG] TRADER: Failed to refresh price, using previous: ${current_price:.2f}")
                except Exception as e:
                    debug_print(f"[DEBUG] TRADER: Price refresh error: {e}")
            
            try:
                # Default precision to 8 decimals initially
                rounded_quantity = round(asset_quantity, 8)

                body = {
                    "client_order_id": client_order_id,
                    "side": side,
                    "type": order_type,
                    "symbol": symbol,
                    "market_order_config": {
                        "asset_quantity": f"{rounded_quantity:.8f}"  # Start with 8 decimal places
                    }
                }

                path = "/api/v1/crypto/trading/orders/"
                response = self.make_api_request("POST", path, json.dumps(body))
                if response and "errors" not in response:
                    # Extract actual execution price from response, fallback to estimated if not available
                    actual_price = self._extract_execution_price(response)
                    fill_price = actual_price if actual_price is not None else current_price
                    
                    try:
                        order_id = response.get("id", None) if isinstance(response, dict) else None
                    except Exception:
                        order_id = None
                    
                    # Recalculate PnL% using actual fill price if available
                    actual_pnl_pct = pnl_pct
                    if actual_price is not None and avg_cost_basis is not None and avg_cost_basis > 0:
                        actual_pnl_pct = ((actual_price - avg_cost_basis) / avg_cost_basis) * PERCENT_MULTIPLIER
                    
                    self._record_trade(
                        side="buy",
                        symbol=symbol,
                        qty=float(rounded_quantity),
                        price=float(fill_price),
                        avg_cost_basis=float(avg_cost_basis) if avg_cost_basis is not None else None,
                        pnl_pct=float(actual_pnl_pct) if actual_pnl_pct is not None else None,
                        tag=tag,
                        order_id=order_id,
                    )
                    # Immediately recalculate cost basis after successful buy
                    debug_print(f"[DEBUG] TRADER: Buy order successful, updating cost basis...")
                    time.sleep(2)  # Brief delay for order to fully settle
                    self.recalculate_single_cost_basis(symbol)
                    return response  # Successfully placed order

            except Exception as e:
                pass #print(traceback.format_exc())

            # Check for precision errors
            if response and "errors" in response:
                for error in response["errors"]:
                    if "has too much precision" in error.get("detail", ""):
                        # Extract required precision directly from the error message
                        detail = error["detail"]
                        nearest_value = detail.split("nearest ")[1].split(" ")[0]

                        decimal_places = len(nearest_value.split(".")[1].rstrip("0"))
                        asset_quantity = round(asset_quantity, decimal_places)
                        break
                    elif "must be greater than or equal to" in error.get("detail", ""):
                        return None

        return None

    def place_sell_order(
        self,
        client_order_id: str,
        side: str,
        order_type: str,
        symbol: str,
        asset_quantity: float,
        expected_price: Optional[float] = None,
        avg_cost_basis: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        tag: Optional[str] = None,
    ) -> Any:
        """
        Execute a market sell order for a specified quantity of cryptocurrency.
        
        Sells the entire position quantity (or partial for testing), records the trade with
        actual execution price for accurate realized PnL tracking, and updates the PnL ledger.
        In simulation mode, records a fake trade without hitting the API.
        
        The expected_price parameter is the current bid at order time and is used for GUI
        display and loss-prevention validation, but the actual fill price from the API response
        is used for PnL calculations to ensure accuracy despite slippage.
        
        Args:
            client_order_id: Unique order ID for idempotency
            side: Must be "sell"
            order_type: Must be "market"
            symbol: Trading pair (e.g., "BTC-USD")
            asset_quantity: Crypto quantity to sell (e.g., 0.00123456 BTC)
            expected_price: Current bid price for validation/display (not used for PnL)
            avg_cost_basis: Cost basis per unit for PnL calculation
            pnl_pct: Pre-calculated PnL percentage at sell time
            tag: Order classification ("TRAIL_SELL", "STOP_LOSS", etc.)
            
        Returns:
            Robinhood API response dict on success, None on failure
        """
        debug_print(f"[DEBUG] TRADER: Placing SELL order for {symbol}: {asset_quantity:.8f} units @ ${expected_price:.2f}" if expected_price else f"[DEBUG] TRADER: Placing SELL order for {symbol}: {asset_quantity:.8f} units")
        
        # Simulation mode: skip actual trade execution
        if _is_simulation_mode():
            fake_response = {
                "id": f"sim_{uuid.uuid4()}",
                "state": "filled",
                "side": "sell",
                "symbol": symbol
            }
            self._record_trade(
                side="sell",
                symbol=symbol,
                qty=float(asset_quantity),
                price=float(expected_price) if expected_price is not None else None,
                avg_cost_basis=float(avg_cost_basis) if avg_cost_basis is not None else None,
                pnl_pct=float(pnl_pct) if pnl_pct is not None else None,
                tag=f"SIM_{tag}" if tag else "SIM_SELL",
                order_id=fake_response["id"],
            )
            debug_print(f"[DEBUG] TRADER: [SIM] Sell order simulated, updating cost basis...")
            time.sleep(2)
            self.recalculate_single_cost_basis(symbol)
            return fake_response
        
        body = {
            "client_order_id": client_order_id,
            "side": side,
            "type": order_type,
            "symbol": symbol,
            "market_order_config": {
                "asset_quantity": f"{asset_quantity:.8f}"
            }
        }

        path = "/api/v1/crypto/trading/orders/"

        response = self.make_api_request("POST", path, json.dumps(body))

        if response and isinstance(response, dict) and "errors" not in response:
            order_id = response.get("id", None)
            
            # Extract actual execution price from response, fallback to estimated if not available
            actual_price = self._extract_execution_price(response)
            fill_price = actual_price if actual_price is not None else expected_price
            
            # Recalculate PnL% using actual fill price if available
            actual_pnl_pct = pnl_pct
            if actual_price is not None and avg_cost_basis is not None and avg_cost_basis > 0:
                actual_pnl_pct = ((actual_price - avg_cost_basis) / avg_cost_basis) * PERCENT_MULTIPLIER
            
            self._record_trade(
                side="sell",
                symbol=symbol,
                qty=float(asset_quantity),
                price=float(fill_price) if fill_price is not None else None,
                avg_cost_basis=float(avg_cost_basis) if avg_cost_basis is not None else None,
                pnl_pct=float(actual_pnl_pct) if actual_pnl_pct is not None else None,
                tag=tag,
                order_id=order_id,
            )
            # Immediately recalculate cost basis after successful sell
            debug_print(f"[DEBUG] TRADER: Sell order successful, updating cost basis...")
            time.sleep(2)  # Brief delay for order to fully settle
            self.recalculate_single_cost_basis(symbol)

        return response

    def manage_trades(self):
        """Execute one complete trading cycle: fetch positions, check signals, execute trades.
        
        This is the main trading loop called periodically by the runner. Each cycle:
        1. Fetches current account state (holdings, buying power, prices)
        2. Calculates gain/loss percentages for each held position
        3. Checks DCA triggers (neural levels + hardcoded percentages)
        4. Checks trailing profit margin sell conditions
        5. Executes any triggered trades and updates cost basis
        6. Writes status to GUI hub for display
        
        Circuit breaker protection: Skips trading if persistent API failures detected but
        continues monitoring (fetching prices/positions) to maintain GUI state. Automatically
        resumes trading after cooldown period elapses.
        
        Account value caching: Falls back to last known-good values if any component of account
        state fails to fetch. Prevents GUI chart corruption from transient API glitches while
        still attempting to update when possible.
        
        Hot-reload support: Reloads coin list and trading config from JSON files each cycle so
        settings changes take effect without restart.
        """
        debug_print("[DEBUG] TRADER: Starting trade management cycle...")
        
        # Check circuit breaker - skip trading but continue monitoring during network issues
        if _circuit_breaker["is_open"]:
            # Check if cooldown period has passed
            if _circuit_breaker["last_error_time"]:
                elapsed = time.time() - _circuit_breaker["last_error_time"]
                if elapsed < _circuit_breaker["cooldown_seconds"]:
                    debug_print(f"[DEBUG] TRADER: Circuit breaker open, skipping trade cycle ({int(_circuit_breaker['cooldown_seconds'] - elapsed)}s remaining)")
                    return
                else:
                    debug_print(f"[DEBUG] TRADER: Circuit breaker cooldown expired, attempting to resume trading...")
        
        trades_made = False  # Flag to track if any trade was made in this iteration

        # Clear cycle-level DCA tracking at start of each iteration
        self._dca_triggered_this_cycle = {}
        debug_print("[DEBUG] TRADER: Cleared DCA cycle tracking for new iteration")

        # Hot-reload coins list + paths from GUI settings while running
        try:
            _refresh_paths_and_symbols()
            self.path_map = dict(base_paths)
            self.coin_categories = dict(coin_categories)
        except Exception:
            pass

        # Fetch account details
        debug_print("[DEBUG] TRADER: Fetching account details...")
        account = self.get_account()
        
        # Reset circuit breaker after successful API call
        if account:
            reset_circuit_breaker()
        
        # Fetch holdings
        debug_print("[DEBUG] TRADER: Fetching current holdings...")
        holdings = self.get_holdings()
        # Fetch trading pairs
        debug_print("[DEBUG] TRADER: Fetching trading pairs...")
        trading_pairs = self.get_trading_pairs()

        # Use the stored cost_basis instead of recalculating
        cost_basis = self.cost_basis
        # Fetch current prices
        symbols = [holding["asset_code"] + "-USD" for holding in holdings.get("results", [])] if holdings else []

        # ALSO fetch prices for tracked coins even if not currently held (so GUI can show bid/ask lines)
        for s in crypto_symbols:
            full = f"{s}-USD"
            if full not in symbols:
                symbols.append(full)

        current_buy_prices, current_sell_prices, valid_symbols = self.get_price(symbols)

        # Calculate total account value (robust: never drop a held coin to $0 on transient API misses)
        snapshot_ok = True

        # buying power
        try:
            buying_power = float(account.get("buying_power", 0))
        except Exception:
            buying_power = 0.0
            snapshot_ok = False

        # holdings list (treat missing/invalid holdings payload as transient error)
        try:
            holdings_list = holdings.get("results", None) if isinstance(holdings, dict) else None
            if not isinstance(holdings_list, list):
                holdings_list = []
                snapshot_ok = False
        except Exception:
            holdings_list = []
            snapshot_ok = False

        holdings_buy_value = 0.0
        holdings_sell_value = 0.0

        for holding in holdings_list:
            try:
                asset = holding.get("asset_code")
                if asset == "USDC":
                    continue

                qty = float(holding.get("total_quantity", 0.0))
                if qty <= 0.0:
                    continue

                sym = f"{asset}-USD"
                bp = float(current_buy_prices.get(sym, 0.0) or 0.0)
                sp = float(current_sell_prices.get(sym, 0.0) or 0.0)

                # If any held asset is missing a usable price this tick, do NOT allow a new "low" snapshot
                if bp <= 0.0 or sp <= 0.0:
                    snapshot_ok = False
                    continue

                holdings_buy_value += qty * bp
                holdings_sell_value += qty * sp
            except Exception:
                snapshot_ok = False
                continue

        total_account_value = buying_power + holdings_sell_value
        in_use = (holdings_sell_value / total_account_value) * PERCENT_MULTIPLIER if total_account_value > 0 else 0.0

        # If this tick is incomplete, fall back to last known-good snapshot so the GUI chart never gets a bogus dip.
        if (not snapshot_ok) or (total_account_value <= 0.0):
            last = getattr(self, "_last_good_account_snapshot", None) or {}
            if last.get("total_account_value") is not None:
                total_account_value = float(last["total_account_value"])
                buying_power = float(last.get("buying_power", buying_power or 0.0))
                holdings_sell_value = float(last.get("holdings_sell_value", holdings_sell_value or 0.0))
                holdings_buy_value = float(last.get("holdings_buy_value", holdings_buy_value or 0.0))
                in_use = float(last.get("percent_in_trade", in_use or 0.0))
        else:
            # Save last complete snapshot
            self._last_good_account_snapshot = {
                "total_account_value": float(total_account_value),
                "buying_power": float(buying_power),
                "holdings_sell_value": float(holdings_sell_value),
                "holdings_buy_value": float(holdings_buy_value),
                "percent_in_trade": float(in_use),
            }

        # Periodic heartbeat message (every 90 seconds, but only when values change)
        current_time = time.time()
        if current_time - self._last_heartbeat >= 90:
            self._last_heartbeat = current_time
            num_positions = len([h for h in holdings_list if h.get("asset_code") != "USDC"])
            cb_status = "PAUSED" if _circuit_breaker["is_open"] else "ACTIVE"
            status_msg = f"{cb_status}   Positions: {num_positions}   Buying Power: ${buying_power:,.2f}   Total Value: ${total_account_value:,.2f}"
            # Only print if status changed (prevents spam of identical messages)
            if status_msg != self._last_status_message:
                print(status_msg, flush=True)
                self._last_status_message = status_msg

        os.system('cls' if os.name == 'nt' else 'clear')

        positions = {}
        for holding in holdings.get("results", []):
            symbol = holding["asset_code"]
            full_symbol = f"{symbol}-USD"

            if full_symbol not in valid_symbols or symbol == "USDC":
                continue

            quantity = float(holding["total_quantity"])
            current_buy_price = current_buy_prices.get(full_symbol, 0)
            current_sell_price = current_sell_prices.get(full_symbol, 0)
            avg_cost_basis = cost_basis.get(symbol, 0)

            # Validate prices before any calculations to prevent division errors
            if current_buy_price <= 0 or current_sell_price <= 0:
                print(f"⚠ [{symbol}] WARNING: Invalid prices (buy: ${current_buy_price}, sell: ${current_sell_price}) - Skipping")
                debug_print(f"[DEBUG] TRADER: Skipping {symbol} - invalid prices detected")
                # Still add to positions dict with zeros so GUI shows the holding exists
                positions[symbol] = {
                    "quantity": quantity,
                    "avg_cost_basis": avg_cost_basis,
                    "current_buy_price": 0.0,
                    "current_sell_price": 0.0,
                    "gain_loss_pct_buy": 0.0,
                    "gain_loss_pct_sell": 0.0,
                    "value_usd": 0.0,
                    "dca_triggered_stages": len(self.dca_levels_triggered.get(symbol, [])),
                    "next_dca_display": "Price unavailable",
                    "dca_line_price": 0.0,
                    "dca_line_source": "N/A",
                    "dca_line_pct": 0.0,
                    "trail_active": False,
                    "trail_line": 0.0,
                    "trail_peak": 0.0,
                    "dist_to_trail_pct": 0.0,
                }
                continue  # Skip all trading logic for this symbol

            if avg_cost_basis > 0:
                gain_loss_percentage_buy = ((current_buy_price - avg_cost_basis) / avg_cost_basis) * PERCENT_MULTIPLIER
                gain_loss_percentage_sell = ((current_sell_price - avg_cost_basis) / avg_cost_basis) * PERCENT_MULTIPLIER
            else:
                gain_loss_percentage_buy = 0
                gain_loss_percentage_sell = 0
                print(f"⚠ [{symbol}] Warning: Average cost basis is 0, P&L calculation skipped")

            value = quantity * current_sell_price
            triggered_levels_count = len(self.dca_levels_triggered.get(symbol, []))
            triggered_levels = triggered_levels_count  # Number of DCA levels triggered

            # Determine the next DCA trigger for this coin (hardcoded % and optional neural level)
            next_stage = triggered_levels_count  # stage 0 == first DCA after entry

            # Hardcoded % for this stage (repeat -50% after we reach it)
            hard_next = self.dca_levels[next_stage] if next_stage < len(self.dca_levels) else self.dca_levels[-1]

            # Neural DCA only applies to first 4 DCA stages:
            # DCA neural levels offset from entry level (e.g., if entry=4: stage 0->N4, stage 1->N5, etc.)
            neural_distance_info = ""
            if next_stage < 4:
                trading_cfg = _load_trading_config()
                long_min = max(1, min(7, int(trading_cfg.get("entry_signals", {}).get("long_signal_min", 4))))
                neural_next = long_min + next_stage
                # Calculate distance to next neural level
                blue_lines = self._read_long_price_levels(symbol)
                if blue_lines and neural_next <= len(blue_lines):
                    target_price = blue_lines[neural_next - 1]  # neural_next-1 for 0-based indexing
                    distance = current_buy_price - target_price
                    distance_pct = (distance / current_buy_price) * PERCENT_MULTIPLIER if current_buy_price > 0 else 0
                    neural_distance_info = f" ({self._fmt_price(target_price)}, ${distance:.2f} / {distance_pct:.2f}% away)"
                next_dca_display = f"{hard_next:.2f}% / N{neural_next}{neural_distance_info}"
            else:
                next_dca_display = f"{hard_next:.2f}%"

            # DCA display line calculation - picks whichever trigger line is higher (NEURAL vs HARD).
            # Hardcoded gives an actual price line: cost_basis * (1 + hard_next%).
            # Neural uses actual predicted price levels from low_bound_prices.dat.
            dca_line_source = "HARD"
            dca_line_price = 0.0
            dca_line_pct = 0.0

            if avg_cost_basis > 0:
                # Hardcoded trigger line price
                hard_line_price = avg_cost_basis * (1.0 + (hard_next / PERCENT_MULTIPLIER))
                dca_line_price = hard_line_price

                # Neural DCA: read actual price levels and use whichever is higher (triggers first as price drops)
                if next_stage < 4:
                    trading_cfg = _load_trading_config()
                    long_min = max(1, min(7, int(trading_cfg.get("entry_signals", {}).get("long_signal_min", 4))))
                    neural_level_needed_disp = long_min + next_stage  # Offset from entry level
                    neural_levels = self._read_long_price_levels(symbol)  # highest->lowest == N1..N7
                    
                    neural_line_price = 0.0
                    if len(neural_levels) >= neural_level_needed_disp:
                        neural_line_price = float(neural_levels[neural_level_needed_disp - 1])
                    
                    # Sanity check - only use neural price if it's realistic (within 15% of current price)
                    # Tightened from 50% to 15% to catch corrupted data earlier
                    # Protects against corrupted data or missing files causing bad triggers
                    if neural_line_price > 0 and current_buy_price > 0:
                        price_deviation = abs(neural_line_price - current_buy_price) / current_buy_price
                        if price_deviation > 0.15:  # Reject neural prices that deviate more than 15%
                            # Neural price is wildly off - ignore it and fall back to hardcoded
                            debug_print(
                                f"[DEBUG] TRADER: Rejecting neural DCA level N{neural_level_needed_disp} for {symbol}. "
                                f"Price ${neural_line_price:.2f} deviates {price_deviation*PERCENT_MULTIPLIER:.1f}% from current ${current_buy_price:.2f} (max 15%)"
                            )
                            neural_line_price = 0.0

                    # Only use neural if it's more conservative than hardcoded
                    # For longs (DCAing down), "more conservative" means higher price (triggers later)
                    # This prevents neural from triggering DCA too aggressively
                    if neural_line_price > 0 and neural_line_price < hard_line_price:
                        debug_print(
                            f"[DEBUG] TRADER: Rejecting neural DCA level N{neural_level_needed_disp} for {symbol}. "
                            f"Neural ${neural_line_price:.2f} is more aggressive than hardcoded ${hard_line_price:.2f}"
                        )
                        neural_line_price = 0.0

                    # Whichever is higher will be hit first as price drops
                    if neural_line_price > dca_line_price:
                        dca_line_price = neural_line_price
                        dca_line_source = f"NEURAL N{neural_level_needed_disp}"

                # PnL% shown alongside DCA is the normal buy-side PnL%
                # (same calculation as GUI "Buy Price PnL": current buy/ask vs avg cost basis)
                dca_line_pct = gain_loss_percentage_buy

            dca_line_price_disp = self._fmt_price(dca_line_price) if avg_cost_basis > 0 else "N/A"

            # Set color code:
            # - DCA is green if we're above the chosen DCA line, red if we're below it
            # - SELL stays based on profit vs cost basis (your original behavior)
            if dca_line_pct >= 0:
                color = Fore.GREEN
            else:
                color = Fore.RED

            if gain_loss_percentage_sell >= 0:
                color2 = Fore.GREEN
            else:
                color2 = Fore.RED

            # Trailing profit margin display for per-coin isolated state.
            # Display uses current state if present, otherwise shows the base PM start line.
            trail_status = "N/A"
            pm_start_pct_disp = 0.0
            base_pm_line_disp = 0.0
            trail_line_disp = 0.0
            trail_peak_disp = 0.0
            above_disp = False
            dist_to_trail_pct = 0.0

            if avg_cost_basis > 0:
                pm_start_pct_disp = self.pm_start_pct_no_dca if int(triggered_levels) == 0 else self.pm_start_pct_with_dca
                base_pm_line_disp = avg_cost_basis * (1.0 + (pm_start_pct_disp / PERCENT_MULTIPLIER))

                state = self.trailing_pm.get(symbol)
                if state is None:
                    trail_line_disp = base_pm_line_disp
                    trail_peak_disp = 0.0
                    active_disp = False
                else:
                    trail_line_disp = float(state.get("line", base_pm_line_disp))
                    trail_peak_disp = float(state.get("peak", 0.0))
                    active_disp = bool(state.get("active", False))

                above_disp = current_sell_price >= trail_line_disp
                # If we're already above the line, trailing is effectively "on/armed" (even if active flips this tick)
                trail_status = "ON" if (active_disp or above_disp) else "OFF"

                if trail_line_disp > 0:
                    dist_to_trail_pct = ((current_sell_price - trail_line_disp) / trail_line_disp) * PERCENT_MULTIPLIER
            
            # Write current price to hub_data folder
            try:
                price_file = os.path.join(HUB_DATA_DIR, symbol + '_current_price.txt')
                with open(price_file, 'w') as f:
                    f.write(str(current_buy_price))
            except Exception:
                pass
            
            positions[symbol] = {
                "quantity": quantity,
                "avg_cost_basis": avg_cost_basis,
                "current_buy_price": current_buy_price,
                "current_sell_price": current_sell_price,
                "gain_loss_pct_buy": gain_loss_percentage_buy,
                "gain_loss_pct_sell": gain_loss_percentage_sell,
                "value_usd": value,
                "dca_triggered_stages": int(triggered_levels_count),
                "next_dca_display": next_dca_display,
                "dca_line_price": float(dca_line_price) if dca_line_price else 0.0,
                "dca_line_source": dca_line_source,
                "dca_line_pct": float(dca_line_pct) if dca_line_pct else 0.0,
                "trail_active": True if (trail_status == "ON") else False,
                "trail_line": float(trail_line_disp) if trail_line_disp else 0.0,
                "trail_peak": float(trail_peak_disp) if trail_peak_disp else 0.0,
                "dist_to_trail_pct": float(dist_to_trail_pct) if dist_to_trail_pct else 0.0,
            }

            # Clean display output
            print(f"[{symbol}] Current Price: Buy ${self._fmt_price(current_buy_price)} | Sell ${self._fmt_price(current_sell_price)}")
            print(f"[{symbol}] Position: {quantity:.8f} @ ${self._fmt_price(avg_cost_basis)} avg cost | Value: ${value:.2f}")
            print(f"[{symbol}] P&L: Buy {color}{dca_line_pct:+.2f}%{Style.RESET_ALL} | Sell {color2}{gain_loss_percentage_sell:+.2f}%{Style.RESET_ALL}")
            print(f"[{symbol}] DCA: {triggered_levels} levels triggered | Next: {next_dca_display} | Line: {dca_line_price_disp} ({dca_line_source})")
            
            if avg_cost_basis > 0:
                trail_indicator = "[ON]" if trail_status == "ON" else "[OFF]"
                above_text = "ABOVE" if above_disp else "below"
                print(f"[{symbol}] Trailing PM: {trail_indicator} Line ${self._fmt_price(trail_line_disp)} | Price {above_text} line ({dist_to_trail_pct:+.2f}%)")
            else:
                print(f"[{symbol}] Trailing PM: N/A (no cost basis)")
            print()

            # STOP-LOSS CHECK: Emergency exit if loss exceeds configured threshold
            # This runs BEFORE trailing PM to catch catastrophic drops immediately
            if avg_cost_basis > 0 and gain_loss_percentage_sell <= self.stop_loss_pct:
                print(f"[{symbol}] ⚠ STOP-LOSS TRIGGERED: Loss {gain_loss_percentage_sell:.2f}% exceeds limit {self.stop_loss_pct:.2f}%")
                print(f"[{symbol}] Emergency sell at ${current_sell_price:.8f} (cost basis ${avg_cost_basis:.8f}){sim_prefix()}")
                debug_print(f"[DEBUG] TRADER: Stop-loss triggered for {symbol}, loss={gain_loss_percentage_sell:.2f}%")
                
                response = self.place_sell_order(
                    str(uuid.uuid4()),
                    "sell",
                    "market",
                    full_symbol,
                    quantity,
                    expected_price=current_sell_price,
                    avg_cost_basis=avg_cost_basis,
                    pnl_pct=gain_loss_percentage_sell,
                    tag="STOP_LOSS",
                )

                if response and isinstance(response, dict) and "errors" not in response:
                    trades_made = True
                    self.trailing_pm.pop(symbol, None)  # clear trailing state
                    self._reset_dca_window_for_trade(symbol, sold=True)
                    print(f"🛑 [{symbol}] STOP-LOSS EXECUTED: {quantity:.8f} {symbol} sold | Loss: {gain_loss_percentage_sell:+.2f}%{sim_prefix()}")
                    debug_print(f"[DEBUG] TRADER: Stop-loss sell successful for {symbol}, quantity={quantity}, loss={gain_loss_percentage_sell:.2f}%")
                    trading_cfg = _load_trading_config()
                    post_delay = trading_cfg.get("timing", {}).get("post_trade_delay_seconds", 30)
                    time.sleep(post_delay)
                    holdings = self.get_holdings()
                    continue
                else:
                    print(f"⚠ [{symbol}] WARNING: Stop-loss sell order FAILED")
                    debug_print(f"[DEBUG] TRADER: Stop-loss sell failed for {symbol}, response: {response}")

            # Trailing profit margin logic with 0.5% trail gap. The PM "start line" is the normal
            # 5% or 2.5% line depending on whether DCA occurred. Trailing activates once price
            # rises above the PM start line, then the line follows peaks upward by 0.5%. A forced
            # sell happens only when price crosses from above the trailing line to below it.
            if avg_cost_basis > 0:
                pm_start_pct = self.pm_start_pct_no_dca if int(triggered_levels) == 0 else self.pm_start_pct_with_dca
                base_pm_line = avg_cost_basis * (1.0 + (pm_start_pct / PERCENT_MULTIPLIER))
                trail_gap = self.trailing_gap_pct / PERCENT_MULTIPLIER  # 0.5% => 0.005

                state = self.trailing_pm.get(symbol)
                if state is None:
                    state = {"active": False, "line": base_pm_line, "peak": 0.0, "was_above": False}
                    self.trailing_pm[symbol] = state
                else:
                    # Ensure line stays at proper floor based on active state
                    if not state.get("active", False):
                        state["line"] = base_pm_line
                    else:
                        # Once trailing is active, the line should never be below the base PM start line
                        if state.get("line", 0.0) < base_pm_line:
                            state["line"] = base_pm_line

                # Use SELL price because that's what you actually get when you market sell
                above_now = current_sell_price >= state["line"]

                # Activate trailing once we first get above the base PM line
                if (not state["active"]) and above_now:
                    state["active"] = True
                    state["peak"] = current_sell_price

                # If active, update peak and move trailing line up behind it
                if state["active"]:
                    if current_sell_price > state["peak"]:
                        state["peak"] = current_sell_price

                    new_line = state["peak"] * (1.0 - trail_gap)
                    if new_line < base_pm_line:
                        new_line = base_pm_line
                    if new_line > state["line"]:
                        state["line"] = new_line

                    # Validate PM line is above cost basis before executing sell
                    # This protects against any race condition or stale cost basis data
                    if state["line"] < avg_cost_basis:
                        print(
                            f"  WARNING: PM line {state['line']:.8f} is below cost basis {avg_cost_basis:.8f} for {symbol}. "
                            f"Resetting to base PM line to prevent selling at loss."
                        )
                        debug_print(
                            f"[DEBUG] TRADER: PM floor violation detected for {symbol}, "
                            f"line={state['line']:.8f}, cost_basis={avg_cost_basis:.8f}, resetting line"
                        )
                        state["line"] = base_pm_line
                        state["active"] = False  # Deactivate trailing to force recalibration
                        state["was_above"] = False

                    # Forced sell on cross from ABOVE -> BELOW trailing line
                    if state["was_above"] and (current_sell_price < state["line"]):
                        # Final validation before sell - ensure we're actually in profit
                        expected_pnl_pct = ((current_sell_price - avg_cost_basis) / avg_cost_basis) * PERCENT_MULTIPLIER if avg_cost_basis > 0 else 0.0
                        
                        if expected_pnl_pct < 0:
                            print(
                                f"  WARNING: Trailing PM would sell {symbol} at LOSS ({expected_pnl_pct:.2f}%). "
                                f"Price={current_sell_price:.8f}, Cost={avg_cost_basis:.8f}. BLOCKING SELL."
                            )
                            debug_print(
                                f"[DEBUG] TRADER: Blocked trailing sell at loss for {symbol}, "
                                f"expected_pnl={expected_pnl_pct:.2f}%"
                            )
                            # Reset state to prevent repeated blocks
                            state["active"] = False
                            state["was_above"] = False
                        else:
                            # MTF Exit Confirmation: Check if we have bearish confirmation to exit
                            # Exception: Stop-loss ALWAYS executes (bypasses MTF check for safety)
                            trading_cfg = _load_trading_config()
                            stop_loss_threshold = float(trading_cfg.get("profit_margin", {}).get("stop_loss_pct", -40.0))
                            is_stop_loss = (expected_pnl_pct <= stop_loss_threshold)
                            
                            if is_stop_loss:
                                # Stop-loss triggered - bypass MTF check and sell immediately
                                print(f"🛑 [{symbol}] STOP-LOSS triggered at {expected_pnl_pct:+.2f}% (threshold: {stop_loss_threshold}%)")
                                print(f"[{symbol}] Bypassing MTF check for emergency exit{sim_prefix()}")
                                debug_print(f"[DEBUG] TRADER: Stop-loss bypass for {symbol}, executing sell without MTF check")
                            else:
                                # Normal profit exit - require MTF confirmation
                                exit_cfg = trading_cfg.get("exit_signals", {})
                                short_signal_min = int(exit_cfg.get("short_signal_min", 4))
                                long_signal_max = int(exit_cfg.get("long_signal_max", 0))
                                
                                short_signal = self._read_short_dca_signal(symbol)
                                long_signal = self._read_long_dca_signal(symbol)
                                
                                debug_print(
                                    f"[DEBUG] TRADER: MTF exit check for {symbol}: "
                                    f"short_signal={short_signal} (need >={short_signal_min}), "
                                    f"long_signal={long_signal} (need <={long_signal_max})"
                                )
                                
                                # Both conditions must be true to allow sell
                                mtf_confirmed = (short_signal >= short_signal_min) and (long_signal <= long_signal_max)
                                
                                if not mtf_confirmed:
                                    # MTF check failed - block sell
                                    if short_signal < short_signal_min:
                                        print(
                                            f"⏸️ [{symbol}] MTF BLOCK: Not enough bearish signals "
                                            f"({short_signal}/7 bearish, need >={short_signal_min})"
                                        )
                                        debug_print(
                                            f"[DEBUG] TRADER: MTF blocked sell for {symbol} - "
                                            f"insufficient bearish confirmation"
                                        )
                                    if long_signal > long_signal_max:
                                        print(
                                            f"⏸️ [{symbol}] MTF BLOCK: Still too bullish "
                                            f"({long_signal}/7 bullish, need <={long_signal_max})"
                                        )
                                        debug_print(
                                            f"[DEBUG] TRADER: MTF blocked sell for {symbol} - "
                                            f"excessive bullish signals"
                                        )
                                    # Don't reset state - allow trailing line to continue tracking
                                    # Price might come back up or MTF might confirm on next cycle
                                    continue
                                else:
                                    # MTF confirmed - proceed with sell
                                    print(
                                        f"✅ [{symbol}] MTF CONFIRMED: "
                                        f"{short_signal}/7 bearish, {long_signal}/7 bullish"
                                    )
                                    debug_print(f"[DEBUG] TRADER: MTF exit confirmation passed for {symbol}")
                            
                            # Check coin category - block profit sells for accumulation coins
                            coin_category = self.coin_categories.get(symbol, "active")
                            if coin_category == "accumulate":
                                print(f"[{symbol}] Trailing PM sell skipped: Accumulation coin (no profit sells)")
                                debug_print(f"[DEBUG] TRADER: Blocking trailing PM sell for {symbol} - accumulation coin")
                                # Reset was_above to prevent repeated attempts
                                state["was_above"] = False
                                continue
                            
                            # MTF check passed or stop-loss - execute sell
                            print(f"[{symbol}] Trailing PM triggered: Price ${current_sell_price:.8f} fell below line ${state['line']:.8f}")
                            print(f"[{symbol}] Expected P&L: {expected_pnl_pct:+.2f}%{sim_prefix()}")
                            debug_print(f"[DEBUG] TRADER: Executing trailing PM sell for {symbol}, expected_pnl={expected_pnl_pct:.2f}%")
                            response = self.place_sell_order(
                                str(uuid.uuid4()),
                                "sell",
                                "market",
                                full_symbol,
                                quantity,
                                expected_price=current_sell_price,
                                avg_cost_basis=avg_cost_basis,
                                pnl_pct=gain_loss_percentage_sell,
                                tag="TRAIL_SELL",
                            )

                            # Check if sell succeeded before clearing state
                            if response and isinstance(response, dict) and "errors" not in response:
                                trades_made = True
                                self.trailing_pm.pop(symbol, None)  # clear per-coin trailing state on exit

                                # Trade ended -> reset rolling 24h DCA window for this coin
                                self._reset_dca_window_for_trade(symbol, sold=True)

                                print(f"✅ [{symbol}] POSITION CLOSED: {quantity:.8f} {symbol} sold | P&L: {expected_pnl_pct:+.2f}%{sim_prefix()}")
                                debug_print(f"[DEBUG] TRADER: Trailing PM sell successful for {symbol}, quantity={quantity}, pnl={expected_pnl_pct:.2f}%")
                                trading_cfg = _load_trading_config()
                                post_delay = trading_cfg.get("timing", {}).get("post_trade_delay_seconds", 30)
                                time.sleep(post_delay)
                                holdings = self.get_holdings()
                                continue
                            else:
                                # Sell failed - reset was_above to prevent repeated attempts
                                print(f"⚠ [{symbol}] WARNING: Trailing sell order FAILED - resetting state")
                                debug_print(f"[DEBUG] TRADER: Trailing PM sell failed for {symbol}, response: {response}")
                                state["was_above"] = False
                                # Don't continue - allow normal state update below to proceed

                # Save this tick’s position relative to the line (needed for “above -> below” detection)
                state["was_above"] = above_now

            # DCA (NEURAL or hardcoded %, whichever hits first for the current stage)
            # Trade entry requires neural signal >= long_signal_min (default 4), then DCA stages:
            # Neural-driven DCA stages (max 4) offset from entry level:
            #   stage 0 => neural (entry+0) OR -2.5%  [e.g., N4 if entry=4]
            #   stage 1 => neural (entry+1) OR -5.0%  [e.g., N5 if entry=4]
            #   stage 2 => neural (entry+2) OR -10.0% [e.g., N6 if entry=4]
            #   stage 3 => neural (entry+3) OR -20.0% [e.g., N7 if entry=4]
            # After that: hardcoded only (-30, -40, -50, then repeat -50 forever).
            current_stage = len(self.dca_levels_triggered.get(symbol, []))

            # Hardcoded loss % for this stage (repeat last level after list ends)
            hard_level = self.dca_levels[current_stage] if current_stage < len(self.dca_levels) else self.dca_levels[-1]
            hard_hit = gain_loss_percentage_buy <= hard_level

            # Neural trigger only for first 4 DCA stages
            neural_level_needed = None
            neural_level_now = None
            neural_hit = False
            if current_stage < 4:
                trading_cfg = _load_trading_config()
                long_min = max(1, min(7, int(trading_cfg.get("entry_signals", {}).get("long_signal_min", 4))))
                neural_level_needed = long_min + current_stage
                neural_level_now = self._read_long_dca_signal(symbol)

                # Keep it sane: don't DCA from neural if we're not even below cost basis.
                neural_hit = (gain_loss_percentage_buy < 0) and (neural_level_now >= neural_level_needed)

            if hard_hit or neural_hit:
                # Check if this DCA stage already triggered this cycle
                if symbol in self._dca_triggered_this_cycle:
                    if current_stage in self._dca_triggered_this_cycle[symbol]:
                        print(f"[{symbol}] DCA Stage {current_stage + 1} already triggered this cycle - skipped")
                        debug_print(f"[DEBUG] TRADER: DCA stage {current_stage + 1} already triggered for {symbol} this cycle")
                        continue  # Skip to avoid double-triggering

                if neural_hit and hard_hit:
                    reason = f"NEURAL L{neural_level_now}>=L{neural_level_needed} OR HARD {hard_level:.2f}%"
                elif neural_hit:
                    reason = f"NEURAL L{neural_level_now}>=L{neural_level_needed}"
                else:
                    reason = f"HARD {hard_level:.2f}%"

                print(f"[{symbol}] DCA Stage {current_stage + 1} triggered via {reason}")
                debug_print(f"[DEBUG] TRADER: Attempting DCA stage {current_stage + 1} for {symbol} via {reason}")

                trading_cfg = _load_trading_config()
                dca_multiplier = trading_cfg.get("dca", {}).get("position_multiplier", 2.0)
                dca_amount = value * dca_multiplier
                print(f"[{symbol}] Current Value: ${value:.2f} | DCA Amount: ${dca_amount:.2f} | Buying Power: ${buying_power:.2f}")

                # Check coin category - skip DCA for liquidation coins (sell-only)
                coin_category = self.coin_categories.get(symbol, "active")
                if coin_category == "liquidate":
                    print(f"[{symbol}] DCA skipped: Liquidation coin (sell-only)")
                    debug_print(f"[DEBUG] TRADER: Skipping DCA for {symbol} - liquidation coin")
                elif self._dca_window_count(symbol) >= int(getattr(self, "max_dca_buys_per_window", 2)):
                    recent_dca = self._dca_window_count(symbol)
                    print(f"[{symbol}] DCA skipped: {recent_dca} buys in last {self.dca_window_seconds/3600:.0f}h (max {self.max_dca_buys_per_window})")

                elif dca_amount <= buying_power:
                    response = self.place_buy_order(
                        str(uuid.uuid4()),
                        "buy",
                        "market",
                        full_symbol,
                        dca_amount,
                        avg_cost_basis=avg_cost_basis,
                        pnl_pct=gain_loss_percentage_buy,
                        tag="DCA",
                    )

                    if response and "errors" not in response:
                        # record that we completed THIS stage (no matter what triggered it)
                        self.dca_levels_triggered.setdefault(symbol, []).append(current_stage)

                        # Mark this stage as triggered in current cycle
                        self._dca_triggered_this_cycle.setdefault(symbol, set()).add(current_stage)
                        
                        print(f"📊 [{symbol}] DCA BUY COMPLETED: Stage {current_stage + 1} | ${dca_amount:.2f} added to position{sim_prefix()}")
                        debug_print(f"[DEBUG] TRADER: DCA stage {current_stage + 1} completed for {symbol}, amount=${dca_amount:.2f}")

                        # Only record a DCA buy timestamp on success (so skips never advance anything)
                        self._note_dca_buy(symbol)

                        # DCA changes avg_cost_basis, so the PM line must be rebuilt from the new basis
                        # (this will re-init to 5% if DCA=0, or 2.5% if DCA>=1)
                        self.trailing_pm.pop(symbol, None)

                        trades_made = True
                        print(f"[{symbol}] DCA buy order successful{sim_prefix()}")
                    else:
                        print(f"[{symbol}] DCA buy order FAILED{sim_prefix()}")
                else:
                    print(f"[{symbol}] DCA skipped: Insufficient funds")

            else:
                pass

        # Ensure GUI gets bid/ask lines even for coins not currently held
        try:
            for sym in crypto_symbols:
                if sym in positions:
                    continue

                full_symbol = f"{sym}-USD"
                if full_symbol not in valid_symbols or sym == "USDC":
                    continue

                current_buy_price = current_buy_prices.get(full_symbol, 0.0)
                current_sell_price = current_sell_prices.get(full_symbol, 0.0)

                # Write current price to hub_data folder
                try:
                    price_file = os.path.join(HUB_DATA_DIR, sym + '_current_price.txt')
                    with open(price_file, 'w') as f:
                        f.write(str(current_buy_price))
                except Exception:
                    pass

                positions[sym] = {
                    "quantity": 0.0,
                    "avg_cost_basis": 0.0,
                    "current_buy_price": current_buy_price,
                    "current_sell_price": current_sell_price,
                    "gain_loss_pct_buy": 0.0,
                    "gain_loss_pct_sell": 0.0,
                    "value_usd": 0.0,
                    "dca_triggered_stages": int(len(self.dca_levels_triggered.get(sym, []))),
                    "next_dca_display": "",
                    "dca_line_price": 0.0,
                    "dca_line_source": "N/A",
                    "dca_line_pct": 0.0,
                    "trail_active": False,
                    "trail_line": 0.0,
                    "trail_peak": 0.0,
                    "dist_to_trail_pct": 0.0,
                }
        except Exception:
            pass

        if not trading_pairs:
            return

        trading_cfg = _load_trading_config()
        allocation_pct = trading_cfg.get("position_sizing", {}).get("initial_allocation_pct", 0.01)
        min_alloc = trading_cfg.get("position_sizing", {}).get("min_allocation_usd", DEFAULT_MIN_ALLOCATION_USD)
        allocation_in_usd = total_account_value * (allocation_pct / len(crypto_symbols))
        if allocation_in_usd < min_alloc:
            allocation_in_usd = min_alloc

        holding_full_symbols = [f"{h['asset_code']}-USD" for h in holdings.get("results", [])]

        # Check max concurrent positions limit once (only count active trading coins)
        trading_cfg = _load_trading_config()
        max_positions = trading_cfg.get("position_sizing", {}).get("max_concurrent_positions", 3)
        
        active_positions = len([
            h for h in holdings.get("results", []) 
            if h.get("asset_code") != "USDC" 
            and self.coin_categories.get(h.get("asset_code", ""), "active") == "active"
        ])
        
        if active_positions >= max_positions:
            debug_print(f"[DEBUG] TRADER: At max active positions ({active_positions}/{max_positions}), skipping all entry checks")
            return

        # Entry signal thresholds from config
        long_min = max(1, min(7, int(trading_cfg.get("entry_signals", {}).get("long_signal_min", 4))))
        short_max = max(0, min(7, int(trading_cfg.get("entry_signals", {}).get("short_signal_max", 0))))

        # Phase 1: Score all eligible coins (preserves coin list order for stable sort)
        eligible_coins = []
        for index, base_symbol in enumerate(crypto_symbols):
            base_symbol = base_symbol.upper().strip()
            full_symbol = f"{base_symbol}-USD"

            # Skip if already held
            if full_symbol in holding_full_symbols:
                debug_print(f"[DEBUG] TRADER: Skipping {base_symbol} - already held")
                continue

            # Check coin category - block entry for liquidation coins (sell-only)
            coin_category = self.coin_categories.get(base_symbol, "active")
            if coin_category == "liquidate":
                debug_print(f"[DEBUG] TRADER: Skipping {base_symbol} - liquidation coin (sell-only)")
                continue

            # Read neural signals
            buy_count = self._read_long_dca_signal(base_symbol)
            sell_count = self._read_short_dca_signal(base_symbol)

            debug_print(
                f"[DEBUG] TRADER: Entry gate check for {base_symbol}: "
                f"long_signal={buy_count} (need >={long_min}), "
                f"short_signal={sell_count} (need <={short_max})"
            )

            # Check entry requirements
            entry_allowed = (buy_count >= long_min) and (sell_count <= short_max)
            
            if not entry_allowed:
                if buy_count < long_min:
                    debug_print(
                        f"[DEBUG] TRADER: Entry blocked for {base_symbol} - "
                        f"insufficient bullish signals ({buy_count} < {long_min})"
                    )
                if sell_count > short_max:
                    debug_print(
                        f"[DEBUG] TRADER: Entry blocked for {base_symbol} - "
                        f"excessive bearish signals ({sell_count} > {short_max})"
                    )
                continue

            # Coin is eligible - calculate opportunity score (signal strength above threshold)
            score = buy_count
            eligible_coins.append((score, index, base_symbol, buy_count, sell_count))
            debug_print(f"[DEBUG] TRADER: {base_symbol} eligible with score={score}")

        # Phase 2: Sort by score (descending), with original index as tiebreaker (stable sort)
        # Python's sort is stable, so coins with same score maintain list order
        eligible_coins.sort(reverse=True, key=lambda x: (x[0], -x[1]))

        if eligible_coins:
            # Select best coin
            best_score, best_index, base_symbol, buy_count, sell_count = eligible_coins[0]
            full_symbol = f"{base_symbol}-USD"
            
            debug_print(
                f"[DEBUG] TRADER: Best opportunity: {base_symbol} (score={best_score}, "
                f"rank 1 of {len(eligible_coins)} eligible)"
            )
            
            # Re-fetch holdings before buy to prevent double-entry race
            try:
                holdings = self.get_holdings()
                holding_full_symbols = [f"{h['asset_code']}-USD" for h in holdings.get("results", [])]
                debug_print(f"[DEBUG] TRADER: Refreshed holdings before buy for {base_symbol}")
            except Exception as e:
                debug_print(f"[DEBUG] TRADER: Failed to refresh holdings: {e}")

            # Double-check still not held (could have changed during scoring phase)
            if full_symbol in holding_full_symbols:
                debug_print(f"[DEBUG] TRADER: {base_symbol} now held (race condition), aborting buy")
                return

            # Entry gate passed - proceed with buy
            print(
                f"✅ [{base_symbol}] BEST OPPORTUNITY (score {best_score}): "
                f"{buy_count}/7 bullish, {sell_count}/7 bearish"
            )
            debug_print(f"[DEBUG] TRADER: Entry confirmation passed for {base_symbol}, placing initial buy order")

            response = self.place_buy_order(
                str(uuid.uuid4()),
                "buy",
                "market",
                full_symbol,
                allocation_in_usd,
            )

            if response and "errors" not in response:
                trades_made = True
                # Do NOT pre-trigger any DCA levels. Hardcoded DCA will mark levels only when it hits your loss thresholds.
                self.dca_levels_triggered[base_symbol] = []

                # Fresh trade -> clear any rolling 24h DCA window for this coin
                self._reset_dca_window_for_trade(base_symbol, sold=False)

                # Reset trailing PM state for this coin (fresh trade, fresh trailing logic)
                self.trailing_pm.pop(base_symbol, None)

                print(f"💰 [{base_symbol}] INITIAL POSITION OPENED: ${allocation_in_usd:.2f} allocated{sim_prefix()}")
                debug_print(f"[DEBUG] TRADER: Initial buy order completed for {base_symbol}, amount=${allocation_in_usd:.2f}")
                trading_cfg = _load_trading_config()
                post_delay = trading_cfg.get("timing", {}).get("post_trade_delay_seconds", 30)
                time.sleep(post_delay)
        else:
            debug_print("[DEBUG] TRADER: No eligible coins for entry")

        # Cost basis now recalculated immediately after each trade
        # Only need to reinitialize DCA levels if trades were made
        if trades_made:
            trading_cfg = _load_trading_config()
            post_delay = trading_cfg.get("timing", {}).get("post_trade_delay_seconds", 30)
            time.sleep(post_delay)
            debug_print("[DEBUG] TRADER: Trades made this cycle, reinitializing DCA levels...")
            self.initialize_dca_levels()

        # Step 8: Dust cleanup for liquidation coins - auto-remove when balance drops below dust threshold
        try:
            # Dust threshold: remove when balance < $0.50 (true dust / fully liquidated)
            # This is separate from min_allocation_usd (which is for opening new positions)
            dust_threshold = 0.50
            
            # Check each liquidation coin for dust threshold
            for coin_symbol, category in self.coin_categories.items():
                if category != "liquidate":
                    continue
                
                # Get current balance for this liquidation coin
                pos = positions.get(coin_symbol)
                balance_usd = 0.0 if not pos else float(pos.get("value_usd", 0.0))
                
                # If balance is below dust threshold (including $0), remove from liquidation list
                if balance_usd < dust_threshold:
                    print(f"[{coin_symbol}] Dust cleanup: Balance ${balance_usd:.2f} below dust threshold (${dust_threshold:.2f}), removing from liquidation list")
                    
                    # Remove from coin_categories and update settings
                    try:
                        # Load current GUI settings file directly (not through cache which flattens the structure)
                        with open(_GUI_SETTINGS_PATH, "r", encoding="utf-8") as f:
                            full_settings = json.load(f) or {}
                        
                        coins_dict = full_settings.get("coins", {"active": [], "accumulate": [], "liquidate": []})
                        
                        if isinstance(coins_dict, dict) and coin_symbol in coins_dict.get("liquidate", []):
                            # Remove from liquidation list
                            coins_dict["liquidate"] = [c for c in coins_dict.get("liquidate", []) if c != coin_symbol]
                            
                            # Update settings file
                            full_settings["coins"] = coins_dict
                            
                            with open(_GUI_SETTINGS_PATH, "w", encoding="utf-8") as f:
                                json.dump(full_settings, f, indent=2)
                            
                            # Update local cache
                            self.coin_categories.pop(coin_symbol, None)
                            
                            # Force immediate reload of coin list to stop monitoring this coin
                            try:
                                _refresh_paths_and_symbols()
                                self.coin_categories = dict(coin_categories)
                            except Exception:
                                pass
                            
                            print(f"[{coin_symbol}] ✓ Removed from liquidation list (fully liquidated)")
                    except Exception as e:
                        print(f"[{coin_symbol}] ERROR: Failed to remove from liquidation list: {e}")
        except Exception:
            pass
        
        # Write combined holdings and zero-holdings status for GUI hub display
        try:
            status = {
                "timestamp": time.time(),
                "simulation_mode": _is_simulation_mode(),
                "account": {
                    "total_account_value": total_account_value,
                    "buying_power": buying_power,
                    "holdings_sell_value": holdings_sell_value,
                    "holdings_buy_value": holdings_buy_value,
                    "percent_in_trade": in_use,
                    # trailing PM config (matches what's printed above current trades)
                    "pm_start_pct_no_dca": float(getattr(self, "pm_start_pct_no_dca", 0.0)),
                    "pm_start_pct_with_dca": float(getattr(self, "pm_start_pct_with_dca", 0.0)),
                    "trailing_gap_pct": float(getattr(self, "trailing_gap_pct", 0.0)),
                },
                "positions": positions,
            }
            self._append_jsonl(
                ACCOUNT_VALUE_HISTORY_PATH,
                {"ts": status["timestamp"], "total_account_value": total_account_value},
            )
            self._write_trader_status(status)
        except Exception:
            pass

    def run(self):
        # Load coins from settings before printing startup message
        try:
            _refresh_paths_and_symbols()
        except Exception:
            pass
        
        print("Starting main loop...\n", flush=True)
        print(f"Monitoring {len(crypto_symbols)} coins: {', '.join(crypto_symbols)}", flush=True)
        print(f"Heartbeat interval: 90 seconds\n", flush=True)
        while True:
            try:
                self.manage_trades()
                trading_cfg = _load_trading_config()
                loop_delay = trading_cfg.get("timing", {}).get("main_loop_delay_seconds", DEFAULT_MAIN_LOOP_DELAY)
                time.sleep(loop_delay)
            except Exception as e:
                msg = traceback.format_exc()
                print(msg)
                # Log to file
                try:
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open("debug_trader.log", "a", encoding="utf-8") as f:
                        f.write(f"[{timestamp}] MAIN LOOP EXCEPTION:\n{msg}\n")
                except Exception:
                    pass

if __name__ == "__main__":
    trading_bot = CryptoAPITrading()
    trading_bot.run()
