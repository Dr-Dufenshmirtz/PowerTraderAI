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
	candle's outcome. The trainer generates predictions by weighted
	averaging the closest memory-pattern matches for the current pattern;
	this per-timeframe predicted candle yields the high/low lines used by
	the Thinker.
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
from contextlib import contextmanager

# Ensure clean console output (avoid encoding issues)
try:
	sys.stdout.reconfigure(encoding='utf-8')
except Exception:
	pass  # Not all systems support reconfigure; gracefully degrade

# third-party
import requests
from kucoin.client import Market
import psutil
from nacl.signing import SigningKey

# instantiate KuCoin market client (kept at top-level like original)
market = Market(url='https://api.kucoin.com')

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

# -----------------------------
# Helper function to clean training file strings
# -----------------------------
def _clean_training_string(text: str) -> str:
	"""Remove quotes, brackets, commas, and spaces from training file data."""
	for char in ("'", ',', '"', ']', '[', ' '):
		text = text.replace(char, '')
	return text

# -----------------------------
# Robinhood market-data (current ASK), same source as rhcb.py trader:
#   GET /api/v1/crypto/marketdata/best_bid_ask/?symbol=BTC-USD
#   use result["ask_inclusive_of_buy_spread"]
# -----------------------------
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

		if not data or "results" not in data or not data["results"]:
			raise RuntimeError(f"Robinhood best_bid_ask returned no results for {symbol}: {data}")

		result = data["results"][0]
		# EXACTLY like rhcb.py's get_price(): ask_inclusive_of_buy_spread
		return float(result["ask_inclusive_of_buy_spread"])

def robinhood_current_ask(symbol: str) -> float:
    """
    Returns Robinhood current BUY price (ask_inclusive_of_buy_spread) for symbols like 'BTC-USD'.
    Reads encrypted creds from rh_key.enc and rh_secret.enc in the same folder as this script.
    """
    global _RH_MD
    if _RH_MD is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        key_path = os.path.join(base_dir, "rh_key.enc")
        secret_path = os.path.join(base_dir, "rh_secret.enc")

        api_key = None
        priv_b64 = None
        
        # Read encrypted files
        try:
            if os.path.isfile(key_path):
                with open(key_path, "rb") as f:
                    api_key = _decrypt_with_dpapi(f.read())
        except Exception as e:
            print(f"[Thinker] Warning: Failed to read encrypted API key: {e}")
        
        try:
            if os.path.isfile(secret_path):
                with open(secret_path, "rb") as f:
                    priv_b64 = _decrypt_with_dpapi(f.read())
        except Exception as e:
            print(f"[Thinker] Warning: Failed to read encrypted private key: {e}")

        if not api_key or not priv_b64:
            raise RuntimeError(
                "Missing rh_key.enc and/or rh_secret.enc next to pt_thinker.py. "
                "Open the Hub and go to Settings → Robinhood API → Setup / Update to configure encrypted keys."
            )

        _RH_MD = RobinhoodMarketData(api_key=api_key, base64_private_key=priv_b64)

    return _RH_MD.get_current_ask(symbol)

def restart_program():
	"""Restarts the current program (no CLI args; uses hardcoded COIN_SYMBOLS)."""
	try:
		os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)])
	except Exception as e:
		print(f'Error during program restart: {e}')

# Utility: PrintException prints a helpful one-line context for exceptions
# by locating the source file/line and printing the offending line. This
# mirrors a similar helper in other scripts and is useful when debugging
# runtime errors during long-running loops.

def PrintException():
	exc_type, exc_obj, tb = sys.exc_info()

	# walk to the innermost frame (where the error actually happened)
	while tb and tb.tb_next:
		tb = tb.tb_next

	f = tb.tb_frame
	lineno = tb.tb_lineno
	filename = f.f_code.co_filename

	linecache.checkcache(filename)
	line = linecache.getline(filename, lineno, f.f_globals)
	msg = 'EXCEPTION IN (LINE {} "{}"): {}'.format(lineno, line.strip(), exc_obj)
	print(msg)
	# Always log exceptions to file (even without debug mode)
	try:
		import datetime
		timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		with open("debug_thinker.log", "a", encoding="utf-8") as f:
			f.write(f"[{timestamp}] {msg}\n")
	except Exception:
		pass

restarted = 'no'
short_started = 'no'
long_started = 'no'
minute = 0
last_minute = 0

# -----------------------------
# GUI SETTINGS (coins list)
# -----------------------------
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
		print(msg)
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


def handle_network_error(operation: str, error: Exception):
	"""Print network error and suggest enabling debug mode"""
	print(f"\n{'='*60}")
	print(f"NETWORK ERROR: {operation} failed")
	print(f"Error: {type(error).__name__}: {str(error)[:200]}")
	print(f"")
	print(f"The process will exit. Please check:")
	print(f"  1. Your internet connection")
	print(f"  2. API service status (KuCoin/Robinhood)")
	print(f"  3. Enable debug_mode in gui_settings.json for more details")
	print(f"{'='*60}\n")
	time.sleep(_get_sleep_timing("sleep_startup_error"))
	sys.exit(1)

def _load_gui_coins() -> list:
	"""
	Reads gui_settings.json and returns settings["coins"] as an uppercased list.
	Caches by mtime so it is cheap to call frequently.
	"""
	try:
		if not os.path.isfile(_GUI_SETTINGS_PATH):
			return list(_gui_settings_cache["coins"])

		mtime = os.path.getmtime(_GUI_SETTINGS_PATH)
		if _gui_settings_cache["mtime"] == mtime:
			return list(_gui_settings_cache["coins"])

		with open(_GUI_SETTINGS_PATH, "r", encoding="utf-8") as f:
			data = json.load(f) or {}

		coins = data.get("coins", None)
		if not isinstance(coins, list) or not coins:
			coins = list(_gui_settings_cache["coins"])

		coins = [str(c).strip().upper() for c in coins if str(c).strip()]
		if not coins:
			coins = list(_gui_settings_cache["coins"])

		# Remove any non-alphanumeric characters (including hidden Unicode)
		coins = [''.join(ch for ch in coin if ch.isalnum()) for coin in coins]
		coins = [c for c in coins if c]  # Remove any empty strings

		_gui_settings_cache["mtime"] = mtime
		_gui_settings_cache["coins"] = coins
		return list(coins)
	except Exception:
		return list(_gui_settings_cache["coins"])

# Initial coin list (will be kept live via _sync_coins_from_settings())
COIN_SYMBOLS = _load_gui_coins()
CURRENT_COINS = list(COIN_SYMBOLS)

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
	old_dir = os.getcwd()
	try:
		folder = coin_folder(sym)
		# Ensure folder exists before attempting to cd
		os.makedirs(folder, exist_ok=True)
		os.chdir(folder)
		yield
	finally:
		os.chdir(old_dir)


# --- training freshness gate (mirrors pt_hub.py) ---
_TRAINING_STALE_SECONDS = 14 * 24 * 60 * 60  # 14 days

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
		return (time.time() - ts) <= _TRAINING_STALE_SECONDS
	except OSError as e:
		debug_print(f"[DEBUG] {sym}: Failed to read trainer_last_training_time.txt: {e}")
		return False
	except Exception as e:
		debug_print(f"[DEBUG] {sym}: Unexpected error checking training status: {type(e).__name__}: {e}")
		return False

# --- GUI HUB "runner ready" gate file (read by gui_hub.py Autopilot toggle) ---

HUB_DIR = os.environ.get("POWERTRADER_HUB_DIR") or os.path.join(BASE_DIR, "hub_data")
try:
	os.makedirs(HUB_DIR, exist_ok=True)
except Exception:
	pass

RUNNER_READY_PATH = os.path.join(HUB_DIR, "runner_ready.json")

def _atomic_write_json(path: str, data: dict) -> None:
	try:
		tmp = path + ".tmp"
		with open(tmp, "w", encoding="utf-8") as f:
			json.dump(data, f, indent=2)
		os.replace(tmp, path)
	except Exception:
		pass

def _write_runner_ready(ready: bool, stage: str, ready_coins=None, total_coins: int = 0) -> None:
	obj = {
		"timestamp": time.time(),
		"ready": bool(ready),
		"stage": stage,
		"ready_coins": ready_coins or [],
		"total_coins": int(total_coins or 0),
	}
	_atomic_write_json(RUNNER_READY_PATH, obj)

# Ensure folders exist for the current configured coins
for _sym in CURRENT_COINS:
	os.makedirs(coin_folder(_sym), exist_ok=True)

# Required timeframes that the Thinker uses for predictions and neural levels
# These MUST be trained for the system to function properly
REQUIRED_THINKER_TIMEFRAMES = [
	'1hour', '2hour', '4hour', '8hour', '12hour', '1day', '1week'
]

distance = 0.5
tf_choices = REQUIRED_THINKER_TIMEFRAMES

def new_coin_state():
	return {
		'low_bound_prices': [0.0] * len(tf_choices),
		'high_bound_prices': [float('inf')] * len(tf_choices),

		'tf_times': [],
		'tf_choice_index': 0,

		'tf_update': ['yes'] * len(tf_choices),
		'messages': ['none'] * len(tf_choices),
		'last_messages': ['none'] * len(tf_choices),
		'margins': [0.25] * len(tf_choices),

		'high_tf_prices': [float('inf')] * len(tf_choices),
		'low_tf_prices': [0.0] * len(tf_choices),

		'tf_sides': ['none'] * len(tf_choices),
		'messaged': ['no'] * len(tf_choices),
		'updated': [0] * len(tf_choices),
		'perfects': ['active'] * len(tf_choices),
		'training_issues': [0] * len(tf_choices),

		# readiness gating (no placeholder-number checks; this is process-based)
		'bounds_version': 0,
		'last_display_bounds_version': -1,

	}

states = {}

display_cache = {sym: f"{sym}  (starting.)" for sym in CURRENT_COINS}

def _validate_coin_training(coin: str) -> tuple:
	"""Check if coin has all required timeframes trained. Returns (is_valid, missing_timeframes_list)."""
	missing = []
	folder = coin_folder(coin)
	
	if not os.path.isdir(folder):
		return (False, REQUIRED_THINKER_TIMEFRAMES)
	
	for tf in REQUIRED_THINKER_TIMEFRAMES:
		memory_file = os.path.join(folder, f"memories_{tf}.dat")
		weight_file = os.path.join(folder, f"memory_weights_{tf}.dat")
		threshold_file = os.path.join(folder, f"neural_perfect_threshold_{tf}.dat")
		
		# Check if all three files exist and are non-empty
		for fpath in [memory_file, weight_file, threshold_file]:
			if not os.path.isfile(fpath):
				if tf not in missing:
					missing.append(tf)
				break
			try:
				if os.path.getsize(fpath) < 10:  # At least 10 bytes
					if tf not in missing:
						missing.append(tf)
					break
			except Exception:
				if tf not in missing:
					missing.append(tf)
				break
	
	return (len(missing) == 0, missing)

# Track which coins have produced REAL predicted levels (not placeholder 0.0 / inf)
_ready_coins = set()

# We consider the runner "READY" only once it is ACTUALLY PRINTING real prediction messages
# (i.e. output lines start with WITHIN / LONG / SHORT). No numeric placeholder checks at all.
def _is_printing_real_predictions(messages) -> bool:
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
	Hot-reload coins from gui_settings.json while runner is running.

	- Adds new coins: creates folder + init_coin() + starts stepping them
	- Removes coins: stops stepping them (leaves state on disk untouched)
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
			display_cache[sym] = f"{sym}  (starting.)"
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

_write_runner_ready(False, stage="starting", ready_coins=[], total_coins=len(CURRENT_COINS))

def init_coin(sym: str):
	with coin_directory(sym):

		# per-coin "version" + on/off files (no collisions between coins)
		with open('alerts_version.txt', 'w+') as f:
			f.write('5/3/2022/9am')

		with open('futures_long_onoff.txt', 'w+') as f:
			f.write('OFF')

		with open('futures_short_onoff.txt', 'w+') as f:
			f.write('OFF')

		st = new_coin_state()

		coin = sym + '-USDT'
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
					history = str(market.get_kline(coin, tf_choices[ind])).replace(']]', '], ').replace('[[', '[')
					break
				except Exception as e:
					init_retry_count += 1
					if init_retry_count >= init_max_retries:
						debug_print(f"[DEBUG] {sym}: init_coin failed after {init_max_retries} retries on timeframe {tf_choices[ind]}")
						handle_network_error(f"init_coin for {sym} (timeframe {tf_choices[ind]})", e)
					time.sleep(_get_sleep_timing("sleep_api_retry"))
					if 'Requests' in str(e):
						pass
					else:
						PrintException()
					continue

			history_list = history.split("], [")
			ind += 1
			try:
				working_minute = str(history_list[1]).replace('"', '').replace("'", "").split(", ")
				the_time = working_minute[0].replace('[', '')
			except Exception:
				the_time = 0.0

			tf_times_local.append(the_time)
			if len(tf_times_local) >= len(tf_choices):
				break

		st['tf_times'] = tf_times_local
		states[sym] = st

# init all coins once (from GUI settings)
for _sym in CURRENT_COINS:
	init_coin(_sym)

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
	debug_print(f"[DEBUG] {sym}: step_coin() called")
	
	# Load state before training check so we can persist it properly
	st = states[sym]
	
	# --- training freshness gate (check BEFORE processing predictions) ---
	# If GUI would show NOT TRAINED (missing / stale trainer_last_training_time.txt),
	# skip this coin so no new trades can start until it is trained again.
	if not _coin_is_trained(sym):
		with coin_directory(sym):
			try:
				# Prevent new trades (and DCA) by forcing signals to 0 and keeping PM at baseline.
				with open('futures_long_profit_margin.txt', 'w+') as f:
					f.write('0.25')
				with open('futures_short_profit_margin.txt', 'w+') as f:
					f.write('0.25')
				with open('long_dca_signal.txt', 'w+') as f:
					f.write('0')
				with open('short_dca_signal.txt', 'w+') as f:
					f.write('0')
			except Exception:
				pass
		try:
			display_cache[sym] = sym + "  (NOT TRAINED / OUTDATED - run trainer)"
		except Exception:
			pass
		try:
			_ready_coins.discard(sym)
			all_ready = len(_ready_coins) >= len(CURRENT_COINS)
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
		training_issues = st.get('training_issues', [0] * len(tf_choices))
		# keep training_issues aligned to tf_choices
		if len(training_issues) < len(tf_choices):
			training_issues.extend([0] * (len(tf_choices) - len(training_issues)))
		elif len(training_issues) > len(tf_choices):
			del training_issues[len(tf_choices):]

		# ====== ORIGINAL: fetch current candle for this timeframe index ======
		debug_print(f"[DEBUG] {sym}: Fetching market data for timeframe {tf_choices[tf_choice_index]}...")
		kucoin_api_retry_count = 0
		kucoin_empty_data_count = 0
		kucoin_max_retries = 5
		while True:
			history_list = []
			while True:
				try:
					history = str(market.get_kline(coin, tf_choices[tf_choice_index])).replace(']]', '], ').replace('[[', '[')
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
			history_list = history.split("], [")
			# KuCoin can occasionally return an empty/short kline response.
			# Guard against history_list[1] raising IndexError.
			if len(history_list) < 2:
				kucoin_empty_data_count += 1
				if kucoin_empty_data_count >= kucoin_max_retries:
					debug_print(f"[DEBUG] {sym}: KuCoin returned empty data after {kucoin_max_retries} attempts")
					# Skip this cycle rather than crashing
					st['tf_choice_index'] = tf_choice_index
					states[sym] = st
					return
				time.sleep(_get_sleep_timing("sleep_data_validation_retry"))
				continue
			working_minute = str(history_list[1]).replace('"', '').replace("'", "").split(", ")
			try:
				openPrice = float(working_minute[1])
				closePrice = float(working_minute[2])
				break
			except Exception:
				continue

		# Prevent division by zero if openPrice is 0.0 (corrupt/stale data)
		if openPrice == 0:
			debug_print(f"[DEBUG] {sym}: openPrice is zero, skipping this cycle")
			st['tf_choice_index'] = tf_choice_index
			states[sym] = st
			return
		
		current_candle = 100 * ((closePrice - openPrice) / openPrice)
		debug_print(f"[DEBUG] {sym}: Market data fetched successfully. Current candle: {current_candle:.4f}%")

		# ====== ORIGINAL: load threshold + memories/weights and compute moves ======
		debug_print(f"[DEBUG] {sym}: Loading neural training files...")
		
		# Check if training files exist before attempting to open
		threshold_file = 'neural_perfect_threshold_' + tf_choices[tf_choice_index] + '.dat'
		if not os.path.exists(threshold_file):
			debug_print(f"[DEBUG] {sym}: Training file missing: {threshold_file}")
			training_issues[tf_choice_index] = 1
			st['training_issues'] = training_issues
			st['tf_choice_index'] = (tf_choice_index + 1) % len(tf_choices)
			states[sym] = st
			return
		
		with open(threshold_file, 'r') as file:
			perfect_threshold = float(file.read())

		try:
			# If we can read/parse training files, this timeframe is NOT a training-file issue.
			training_issues[tf_choice_index] = 0

			# Validate all required training files exist
			memories_file = 'memories_' + tf_choices[tf_choice_index] + '.dat'
			weights_file = 'memory_weights_' + tf_choices[tf_choice_index] + '.dat'
			weights_high_file = 'memory_weights_high_' + tf_choices[tf_choice_index] + '.dat'
			weights_low_file = 'memory_weights_low_' + tf_choices[tf_choice_index] + '.dat'
			
			if not os.path.exists(memories_file):
				debug_print(f"[DEBUG] {sym}: Training file missing: {memories_file}")
				raise FileNotFoundError(f"Missing {memories_file}")
			if not os.path.exists(weights_file):
				debug_print(f"[DEBUG] {sym}: Training file missing: {weights_file}")
				raise FileNotFoundError(f"Missing {weights_file}")
			if not os.path.exists(weights_high_file):
				debug_print(f"[DEBUG] {sym}: Training file missing: {weights_high_file}")
				raise FileNotFoundError(f"Missing {weights_high_file}")
			if not os.path.exists(weights_low_file):
				debug_print(f"[DEBUG] {sym}: Training file missing: {weights_low_file}")
				raise FileNotFoundError(f"Missing {weights_low_file}")

			with open(memories_file, 'r') as file:
				memory_list = file.read().replace("'", "").replace(',', '').replace('"', '').replace(']', '').replace('[', '').split('~')

			with open(weights_file, 'r') as file:
				weight_list = file.read().replace("'", "").replace(',', '').replace('"', '').replace(']', '').replace('[', '').split(' ')

			with open(weights_high_file, 'r') as file:
				high_weight_list = file.read().replace("'", "").replace(',', '').replace('"', '').replace(']', '').replace('[', '').split(' ')

			with open(weights_low_file, 'r') as file:
				low_weight_list = file.read().replace("'", "").replace(',', '').replace('"', '').replace(']', '').replace('[', '').split(' ')

			mem_ind = 0
			diffs_list = []
			any_perfect = 'no'
			perfect_dexs = []
			perfect_diffs = []
			moves = []
			move_weights = []
			unweighted = []
			high_unweighted = []
			low_unweighted = []
			high_moves = []
			low_moves = []

			debug_print(f"[DEBUG] {sym}: Processing {len(memory_list)} memory patterns...")
			while mem_ind < len(memory_list):
				# Validate training data format before parsing
				parts = memory_list[mem_ind].split('{}')
				if len(parts) < 3:
					# Corrupted/incomplete training data - skip this pattern
					debug_print(f"[DEBUG] {sym}: Skipping corrupted memory pattern at index {mem_ind} (insufficient parts: {len(parts)})")
					mem_ind += 1
					continue
				
				try:
					memory_pattern = _clean_training_string(parts[0]).split()
					if not memory_pattern:
						# Empty pattern after cleaning - skip
						mem_ind += 1
						continue
				except Exception:
					debug_print(f"[DEBUG] {sym}: Failed to parse memory pattern at index {mem_ind}")
					mem_ind += 1
					continue
				
				check_dex = 0
				memory_candle = float(memory_pattern[check_dex])

				# Calculate percentage difference with proper zero-denominator protection
				average = (current_candle + memory_candle) / 2
				if abs(average) < 1e-10:  # Essentially zero (handles +5 and -5 summing to 0)
					difference = 0.0
				else:
					try:
						difference = abs((abs(current_candle - memory_candle) / average) * 100)
					except Exception:
						difference = 0.0

				diff_avg = difference

				if diff_avg <= perfect_threshold:
					any_perfect = 'yes'
					# Use pre-validated parts from split (already verified len >= 3)
					try:
						high_diff = float(_clean_training_string(parts[1])) / 100
						low_diff = float(_clean_training_string(parts[2])) / 100
					except (ValueError, IndexError) as e:
						# Failed to parse high/low diffs - skip this pattern
						debug_print(f"[DEBUG] {sym}: Failed to parse high/low diffs at index {mem_ind}: {e}")
						mem_ind += 1
						continue

				# Validate weight list indices before accessing
				if mem_ind >= len(weight_list) or mem_ind >= len(high_weight_list) or mem_ind >= len(low_weight_list):
					debug_print(f"[DEBUG] {sym}: Weight list length mismatch at index {mem_ind} (memory: {len(memory_list)}, weight: {len(weight_list)}, high: {len(high_weight_list)}, low: {len(low_weight_list)}) - skipping pattern")
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

				# Check if we've processed all memory patterns
				if mem_ind >= len(memory_list):
					if any_perfect == 'no':
						final_moves = 0.0
						high_final_moves = 0.0
						low_final_moves = 0.0
						perfects[tf_choice_index] = 'inactive'
					elif moves and high_moves and low_moves:
						# Only calculate averages if lists are non-empty
						final_moves = sum(moves) / len(moves)
						high_final_moves = sum(high_moves) / len(high_moves)
						low_final_moves = sum(low_moves) / len(low_moves)
						perfects[tf_choice_index] = 'active'
					else:
						final_moves = 0.0
						high_final_moves = 0.0
						low_final_moves = 0.0
						perfects[tf_choice_index] = 'inactive'
					break

		except Exception:
			PrintException()
			training_issues[tf_choice_index] = 1
			final_moves = 0.0
			high_final_moves = 0.0
			low_final_moves = 0.0
			perfects[tf_choice_index] = 'inactive'

		# keep threshold persisted (original behavior)
		with open('neural_perfect_threshold_' + tf_choices[tf_choice_index] + '.dat', 'w+') as file:
			file.write(str(perfect_threshold))

		# ====== ORIGINAL: compute new high/low predictions ======
		price_list2 = [openPrice, closePrice]
		current_pattern = [price_list2[0], price_list2[1]]

		try:
			c_diff = final_moves / 100
			high_diff = high_final_moves
			low_diff = low_final_moves

			start_price = current_pattern[len(current_pattern) - 1]
			high_new_price = start_price + (start_price * high_diff)
			low_new_price = start_price + (start_price * low_diff)
		except Exception:
			start_price = current_pattern[len(current_pattern) - 1]
			high_new_price = start_price
			low_new_price = start_price

		if perfects[tf_choice_index] == 'inactive':
			high_tf_prices[tf_choice_index] = start_price
			low_tf_prices[tf_choice_index] = start_price
		else:
			high_tf_prices[tf_choice_index] = high_new_price
			low_tf_prices[tf_choice_index] = low_new_price

		# ====== advance tf index; if full sweep complete, compute signals ======
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

			# --- HARD GUARANTEE: all TF arrays stay length==len(tf_choices) (fallback placeholders) ---
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
						history = str(market.get_kline(coin, tf_choices[inder])).replace(']]', '], ').replace('[[', '[')
						break
					except Exception as e:
						time.sleep(_get_sleep_timing("sleep_api_retry"))
						if 'Requests' in str(e):
							pass
						else:
							PrintException()
						continue

				history_list = history.split("], [")
				try:
					working_minute = str(history_list[1]).replace('"', '').replace("'", "").split(", ")
					the_time = working_minute[0].replace('[', '')
				except Exception:
					the_time = 0.0

				# (original comparisons)
				if current > high_bound_prices[inder] and high_tf_prices[inder] != low_tf_prices[inder]:
					message = 'SHORT on ' + tf_choices[inder] + ' timeframe. ' + format(((high_bound_prices[inder] - current) / abs(current)) * 100, '.8f') + ' High Boundary: ' + str(high_bound_prices[inder])
					if messaged[inder] != 'yes':
						messaged[inder] = 'yes'
					margins[inder] = ((high_tf_prices[inder] - current) / abs(current)) * 100

					# Compare with old message to detect changes
					if 'SHORT' in old_message:
						messages[inder] = message
						updated[inder] = 0
					else:
						messages[inder] = message
						updated[inder] = 1

					tf_sides[inder] = 'short'

				elif current < low_bound_prices[inder] and high_tf_prices[inder] != low_tf_prices[inder]:
					message = 'LONG on ' + tf_choices[inder] + ' timeframe. ' + format(((low_bound_prices[inder] - current) / abs(current)) * 100, '.8f') + ' Low Boundary: ' + str(low_bound_prices[inder])
					if messaged[inder] != 'yes':
						messaged[inder] = 'yes'

					margins[inder] = ((low_tf_prices[inder] - current) / abs(current)) * 100

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
							message = 'INACTIVE (training data issue) on ' + tf_choices[inder] + ' timeframe.' + ' Low Boundary: ' + str(low_bound_prices[inder]) + ' High Boundary: ' + str(high_bound_prices[inder])
						else:
							message = 'INACTIVE on ' + tf_choices[inder] + ' timeframe.' + ' Low Boundary: ' + str(low_bound_prices[inder]) + ' High Boundary: ' + str(high_bound_prices[inder])
					else:
						message = 'WITHIN on ' + tf_choices[inder] + ' timeframe.' + ' Low Boundary: ' + str(low_bound_prices[inder]) + ' High Boundary: ' + str(high_bound_prices[inder])

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
				new_low_price = low_tf_prices[price_list_index] - (low_tf_prices[price_list_index] * (distance / 100))
				new_high_price = high_tf_prices[price_list_index] + (high_tf_prices[price_list_index] * (distance / 100))
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
								low_perc_diff = (abs(new_low_bound_prices[og_index] - new_low_bound_prices[og_index + 1]) / low_avg) * 100
							except Exception:
								low_perc_diff = 0.0
						
						high_avg = (new_high_bound_prices[og_index] + new_high_bound_prices[og_index + 1]) / 2
						if abs(high_avg) < 1e-10:
							high_perc_diff = 0.0
						else:
							try:
								high_perc_diff = (abs(new_high_bound_prices[og_index] - new_high_bound_prices[og_index + 1]) / high_avg) * 100
							except Exception:
								high_perc_diff = 0.0

						if low_perc_diff < 0.25 + gap_modifier or new_low_bound_prices[og_index + 1] > new_low_bound_prices[og_index]:
							new_price = new_low_bound_prices[og_index + 1] - (new_low_bound_prices[og_index + 1] * 0.0005)
							new_low_bound_prices[og_index + 1] = new_price
							continue

						if high_perc_diff < 0.25 + gap_modifier or new_high_bound_prices[og_index + 1] < new_high_bound_prices[og_index]:
							new_price = new_high_bound_prices[og_index + 1] + (new_high_bound_prices[og_index + 1] * 0.0005)
							new_high_bound_prices[og_index + 1] = new_price
							continue

					og_index += 1
					gap_modifier += 0.25
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

			with open('low_bound_prices.html', 'w+') as file:
				file.write(str(new_low_bound_prices).replace("', '", " ").replace("[", "").replace("]", "").replace("'", ""))
			with open('high_bound_prices.html', 'w+') as file:
				file.write(str(new_high_bound_prices).replace("', '", " ").replace("[", "").replace("]", "").replace("'", ""))

			# cache display text for this coin (main loop prints everything on one screen)
			try:
				display_cache[sym] = (
					sym + '  ' + str(current) + '\n' +
					str(messages).replace("', '", "\n")
				)

				# The GUI-visible messages were generated using the bounds_version that was in state at the
				# start of this full-sweep (before we rebuilt bounds above).
				st['last_display_bounds_version'] = bounds_version_used_for_messages

				# Only consider this coin "ready" once we've already rebuilt bounds at least once
				# AND we're now printing messages generated from those rebuilt bounds.
				if (st['last_display_bounds_version'] >= 1) and _is_printing_real_predictions(messages):
					_ready_coins.add(sym)
				else:
					_ready_coins.discard(sym)

				all_ready = len(_ready_coins) >= len(COIN_SYMBOLS)
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
						if pm < 0.25:
							pm = 0.25
					else:
						pm = 0.25
				except Exception:
					pm = 0.25

				with open('futures_long_profit_margin.txt', 'w+') as f:
					f.write(str(pm))
				with open('long_dca_signal.txt', 'w+') as f:
					f.write(str(longs))

				# short pm
				current_pms = [m for m in margins if m != 0]
				try:
					if len(current_pms) > 0:
						pm = sum(current_pms) / len(current_pms)
						if pm < 0.25:
							pm = 0.25
					else:
						pm = 0.25
				except Exception:
					pm = 0.25

				with open('futures_short_profit_margin.txt', 'w+') as f:
					f.write(str(abs(pm)))
				with open('short_dca_signal.txt', 'w+') as f:
					f.write(str(shorts))

			except Exception:
				PrintException()

			# ====== NON-BLOCKING candle update check (single pass) ======
			tf_update_index = 0
			while tf_update_index < len(tf_update):
				while True:
					try:
						history = str(market.get_kline(coin, tf_choices[tf_update_index])).replace(']]', '], ').replace('[[', '[')
						break
					except Exception as e:
						time.sleep(_get_sleep_timing("sleep_api_retry"))
						if 'Requests' in str(e):
							pass
						else:
							PrintException()
						continue

				history_list = history.split("], [")
				try:
					working_minute = str(history_list[1]).replace('"', '').replace("'", "").split(", ")
					the_time = working_minute[0].replace('[', '')
				except Exception:
					the_time = 0.0

				if the_time != tf_times[tf_update_index]:
					tf_update[tf_update_index] = 'yes'
					tf_times[tf_update_index] = the_time

				tf_update_index += 1

		# ====== save state back ======
		# Only persist state if at least one timeframe has valid predictions
		# (not all timeframes marked as having training issues)
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
		st['tf_sides'] = tf_sides
		st['messaged'] = messaged
		st['updated'] = updated
		st['perfects'] = perfects
		st['training_issues'] = training_issues

		states[sym] = st

try:
	while True:
		# Hot-reload coins from GUI settings while running
		_sync_coins_from_settings()
		
		# Capture coin list after sync to ensure new coins are processed immediately
		coins_this_iteration = list(CURRENT_COINS)

		for _sym in coins_this_iteration:
			step_coin(_sym)

		# clear + re-print one combined screen (so you don't see old output above new)
		os.system('cls' if os.name == 'nt' else 'clear')

		# Validate training status and display persistent warning if incomplete
		training_warnings = []
		for _sym in coins_this_iteration:
			is_valid, missing_tfs = _validate_coin_training(_sym)
			if not is_valid:
				training_warnings.append(f"{_sym}: Missing {', '.join(missing_tfs)}")

		if training_warnings:
			print("="*70)
			print("⚠ WARNING: INCOMPLETE TRAINING DETECTED ⚠")
			print("="*70)
			print("The following coins are missing required timeframe training:")
			print()
			for warning in training_warnings:
				print(f"  • {warning}")
			print()
			print("Required timeframes:", ", ".join(REQUIRED_THINKER_TIMEFRAMES))
			print()
			print("Predictions may be inaccurate or use placeholder values.")
			print("Please complete training for all coins before trading.")
			print("="*70)
			print()

		# Print all coins' display output (use cached display from step_coin which includes symbol header)
		for _sym in coins_this_iteration:
			print()  # Add spacing before each coin section
			output = display_cache.get(_sym, _sym + "  (no data yet)")
			# Strip BOM, zero-width spaces, and other invisible Unicode
			output = output.replace('\ufeff', '').replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
			# Convert to ASCII only
			output = output.encode('ascii', 'ignore').decode('ascii')
			# Remove any remaining control characters except newline and tab
			output = ''.join(ch for ch in output if ch.isprintable() or ch in '\n\t')
			print(output)
		
		# small sleep so you don't peg CPU when running many coins
		time.sleep(_get_sleep_timing("sleep_main_loop"))

except Exception:
	PrintException()


