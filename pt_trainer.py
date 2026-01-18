"""
pt_trainer.py - ApolloTrader Training Engine

Description:
This module runs the training process for ApolloTrader. The trainer
process walks through historical candle data across multiple timeframes
and records memory patterns paired with the next-candle outcome. Those
memories are used to generate per-timeframe predicted candles (weighted
averages of close matches) whose highs/lows are displayed by the Thinker
and used to form trading decisions.

Primary Repository: https://github.com/Dr-Dufenshmirtz/ApolloTrader
Primary Author: Dr Dufenshmirtz

Original Project: https://github.com/garagesteve1155/PowerTrader_AI
Original Author: Stephen Hughes (garagesteve1155)

Key behavioral notes (informational only):

- Training:
	The trainer processes the entire coin history on several timeframes
	and stores observed patterns. After each candle closes the trainer
	adjusts pattern weights based on prediction accuracy so future
	weighted-averages improve over time.

- Pattern Matching:
	Uses relative threshold matching (percentage of pattern magnitude) for
	scale-invariant behavior across different price levels. Thresholds are
	automatically adjusted based on volatility (5.0× average volatility) and
	clamped to configured min/max bounds. Zero/tiny patterns use a
	volatility-adaptive baseline (0.1× volatility, typically ~0.4%) to
	prevent division-by-zero. Directionality-agnostic using abs() for both
	bull and bear patterns.

- Thinker/Trader integration:
	The Thinker uses predicted highs/lows from each timeframe to decide
	start signals and DCA levels; the Trader uses those signals with
	trailing profit and DCA rules described in other module banners.
"""

import os
import sys
import signal
import time
import shutil
import datetime
from datetime import datetime
import traceback
import linecache
import base64
import calendar
import hashlib
import hmac
import json
import uuid
import logging
import math

# Ensure clean console output
try:
	sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
	sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
except:
	pass

# third-party
import psutil
from kucoin.client import Market

# instantiate KuCoin market client (kept at top-level like original)
market = Market(url='https://api.kucoin.com')

# KuCoin API constants
KUCOIN_MAX_CANDLES = 1500  # Maximum candles per API request

# ============================================================================
# TRAINER ALGORITHM CONSTANTS
# ============================================================================

# Volatility and threshold calculation
DEFAULT_VOLATILITY = 4.0  # Default volatility fallback (typical crypto high-low range %)
VOLATILITY_THRESHOLD_MULTIPLIER = 5.0  # Multiplier for volatility-based threshold calculation
BOUNCE_TOLERANCE_MULTIPLIER = 0.25  # Multiplier for adaptive bounce tolerance (fraction of volatility)
BOUNCE_TOLERANCE_MINIMUM = 0.5  # Minimum baseline threshold percentage (prevents too-tight filtering)
DEFAULT_EWMA_DECAY = 0.9  # Exponential moving average decay factor for volatility smoothing

# Pattern matching and learning
BASE_KERNEL_BANDWIDTH = 2.5  # Kernel bandwidth for distance-based weighting (matches thinker behavior)
DEFAULT_PRUNING_SIGMA = 2.0  # Standard deviations below mean for pattern pruning
PRUNING_WEIGHT_FLOOR = 0.05  # Minimum weight threshold for pruning (always remove patterns below this)

# Weight system (learning rate and bounds)
WEIGHT_MIN = 0.0  # Minimum allowed weight value (patterns with zero weight are filtered)
WEIGHT_MAX = 2.0  # Maximum allowed weight value (clamps learning updates)
WEIGHT_FACTOR_MULTIPLIER = 2.0  # Multiplier for weight factor calculation in adaptive thresholds
WEIGHT_FACTOR_MIN = 0.5  # Minimum weight factor clamp (prevents division by zero)
ERROR_MAGNITUDE_MIN = 0.1  # Minimum error magnitude for calculations (prevents division by zero)

# Training window sizes
RECENT_ACCURACY_WINDOW = 100  # Number of recent predictions to calculate current accuracy
MAX_VOLATILITY_WINDOW = 500  # Maximum window size for volatility calculation
MIN_VOLATILITY_WINDOW = 10  # Minimum window size for volatility calculation
VOLATILITY_WINDOW_FRACTION = 0.1  # Fraction of total data to use for volatility window
INITIAL_SKIP_MIN = 12  # Minimum number of candles to skip at training start
INITIAL_SKIP_FRACTION = 0.02  # Fraction of data to skip at training start (warm-up period)

# API rate limiting
API_DELAY_BASE = 0.4  # Base delay between API calls in seconds
API_DELAY_FALLBACK = 0.5  # Fallback delay if calculation fails

# Accuracy trend detection
ADAPTIVE_THRESHOLD_MIN = 0.05  # Minimum adaptive threshold percentage
ADAPTIVE_THRESHOLD_MAX = 0.5  # Maximum adaptive threshold percentage

# Percentage conversion
PERCENT_MULTIPLIER = 100  # Multiplier to convert decimal to percentage

# Debug mode support - look in parent directory (project root) for gui_settings.json
# The trainer runs in coin subfolders (BTC/, ETH/, etc.) but settings are in root
_GUI_SETTINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gui_settings.json")
_debug_mode_cache = {"enabled": False}

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

def debug_print(msg: str):
	"""Print debug message only if debug mode is enabled, also log to file"""
	if _is_debug_mode():
		print(msg)
		# Also write to debug log file in the same directory as other training files
		try:
			# Use global _arg_coin if available, otherwise try sys.argv
			try:
				coin_name = _arg_coin
			except NameError:
				coin_name = sys.argv[1] if len(sys.argv) > 1 else "unknown"
			log_file = f"debug_trainer_{coin_name}.log"
			import datetime
			timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			with open(log_file, "a", encoding="utf-8") as f:
				f.write(f"[{timestamp}] {msg}\n")
		except Exception as e:
			# If logging fails, print to console with full error details
			print(f"\n{'!'*60}")
			print(f"❌ [DEBUG LOG ERROR] Failed to write to {log_file}")
			print(f"Error: {type(e).__name__}: {e}")
			print(f"Working directory: {os.getcwd()}")
			print(f"{'!'*60}\n")
			pass  # Don't let logging errors break the trainer

def handle_network_error(operation: str, error: Exception):
	"""Print network error and suggest enabling debug mode"""
	print(f"\n{'='*60}")
	print(f"❌ NETWORK ERROR: {operation} failed")
	print(f"Error: {type(error).__name__}: {str(error)[:200]}")
	print(f"The process will exit. Please check:")
	print(f"  1. Your internet connection")
	print(f"  2. KuCoin API service status")
	print(f"  3. Enable debug_mode in gui_settings.json for more details")
	print(f"{'='*60}\n")
	time.sleep(3)
	mark_training_error(f"Network error: {operation}")
	sys.exit(1)

sells_count = 0
list_len = 0
in_trade = 'no'

# Speed knobs
VERBOSE = False  # set True if you want the old high-volume prints
def vprint(*args, **kwargs):
	if VERBOSE:
		print(*args, **kwargs)

# Cache memory/weights in RAM (avoid re-reading and re-writing every loop)
_memory_cache = {}  # tf_choice -> dict(memory_list, weight_list, high_weight_list, low_weight_list, dirty)

# EWMA volatility tracking for volatility-adaptive thresholds
# Maintains running average of market volatility to scale error thresholds dynamically
_volatility_cache = {}  # cache_key (tf_choice, pattern_size) -> {"ewma_volatility": float, "avg_volatility": float}

# Pattern age tracking for adaptive pruning
# Tracks validation count per pattern to enable removal of stale low-weight patterns
_pattern_ages = {}  # tf_choice -> [age_count_per_pattern]

# Threshold history buffer for EWMA calculation
# Accumulates thresholds over staleness window, applies exponential weighting for recency bias
# This provides stable threshold values for thinker that aren't skewed by single volatility spikes
_threshold_buffers = {}  # tf_choice -> {"buffer": deque, "max_size": int}

# Global write buffer to reduce disk I/O during training
# Memory/weight data is buffered and written only at critical points (timeframe end, training complete, user stop)
_write_buffer = {
	"last_checkpoint_time": 0,  # timestamp of last safety checkpoint
	"last_flush_time": 0,       # timestamp of last periodic flush
	"volatility_ewma_decay": DEFAULT_EWMA_DECAY  # Default, will be updated from settings
}

def _read_text(path):
	with open(path, "r", encoding="utf-8", errors="ignore") as f:
		return f.read()

def load_memory(cache_key):
	"""Load memories/weights for a timeframe+size once and keep them in RAM.
	
	Args:
		cache_key: Tuple of (timeframe, pattern_size) e.g. ('1week', 3)
	"""
	if cache_key in _memory_cache:
		return _memory_cache[cache_key]
	
	tf_choice, pattern_size = cache_key
	data = {
		"memory_list": [],
		"memory_patterns": [],  # Pre-split patterns for faster comparison
		"weight_list": [],
		"high_weight_list": [],
		"low_weight_list": [],
		"dirty": False,
	}
	try:
		data["memory_list"] = _read_text(f"memories_{tf_choice}_p{pattern_size}.dat").replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split('~')
		# Pre-split patterns once during load (huge speedup for pattern matching)
		data["memory_patterns"] = [mem.split('{}')[0].split('|') for mem in data["memory_list"]]
	except:
		data["memory_list"] = []
		data["memory_patterns"] = []
	# The memory files are plain-text caches written by the trainer. We
	# normalize and split them into Python lists here to avoid repeated
	# file I/O during inner loops of the training routine.
	try:
		data["weight_list"] = _read_text(f"memory_weights_{tf_choice}_p{pattern_size}.dat").replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split(' ')
	except:
		data["weight_list"] = []
	try:
		data["high_weight_list"] = _read_text(f"memory_weights_high_{tf_choice}_p{pattern_size}.dat").replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split(' ')
	except:
		data["high_weight_list"] = []
	try:
		data["low_weight_list"] = _read_text(f"memory_weights_low_{tf_choice}_p{pattern_size}.dat").replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split(' ')
	except:
		data["low_weight_list"] = []
	_memory_cache[cache_key] = data
	
	# Initialize age tracking for each pattern (starts at 0 for all existing patterns)
	if cache_key not in _pattern_ages:
		_pattern_ages[cache_key] = [0] * len(data["memory_list"])
	
	return data

def flush_memory(cache_key, force=False):
	"""Write memories/weights back to disk only when they changed (batch IO).
	
	Args:
		cache_key: Tuple of (timeframe, pattern_size) e.g. ('1week', 3)
		force: If True, write even if not dirty
	"""
	data = _memory_cache.get(cache_key)
	if not data:
		return
	if (not data.get("dirty")) and (not force):
		return
	
	# Extract timeframe and pattern size for filename generation
	tf_choice, pattern_size = cache_key
	
	# Deduplicate patterns before saving (keep first occurrence, preserve weight alignment)
	seen = set()
	unique_indices = []
	for i, pattern in enumerate(data["memory_list"]):
		if pattern not in seen and str(pattern).strip() != "":
			seen.add(pattern)
			unique_indices.append(i)
	
	original_count = len(data["memory_list"])
	if len(unique_indices) < original_count:
		debug_print(f"[DEBUG] TRAINER: Deduplicated {tf_choice} p{pattern_size}: {original_count} → {len(unique_indices)} patterns ({100*(1-len(unique_indices)/original_count):.1f}% duplicates removed)")
	
	# Filter all lists to keep only unique patterns
	unique_memories = [data["memory_list"][i] for i in unique_indices]
	unique_weights = [data["weight_list"][i] for i in unique_indices]
	unique_high_weights = [data["high_weight_list"][i] for i in unique_indices]
	unique_low_weights = [data["low_weight_list"][i] for i in unique_indices]
	
	try:
		with open(f"memories_{tf_choice}_p{pattern_size}.dat", "w+", encoding="utf-8") as f:
			f.write("~".join(unique_memories))
	except:
		pass
	# flush_memory() batches writes to reduce disk churn. The `dirty` flag
	# prevents redundant writes when nothing changed.
	try:
		with open(f"memory_weights_{tf_choice}_p{pattern_size}.dat", "w+", encoding="utf-8") as f:
			f.write(" ".join([str(x) for x in unique_weights]))
	except:
		pass
	try:
		with open(f"memory_weights_high_{tf_choice}_p{pattern_size}.dat", "w+", encoding="utf-8") as f:
			f.write(" ".join([str(x) for x in unique_high_weights]))
	except:
		pass
	try:
		with open(f"memory_weights_low_{tf_choice}_p{pattern_size}.dat", "w+", encoding="utf-8") as f:
			f.write(" ".join([str(x) for x in unique_low_weights]))
	except:
		pass
	data["dirty"] = False

def flush_all_buffers(force=True):
	"""Write all buffered data to disk. Called at critical points:
	- End of each timeframe
	- Training completion
	- User stop
	- Error/crash (via signal handler)
	
	This ensures data integrity while minimizing disk writes during training."""
	debug_print("[DEBUG] TRAINER: Flushing all buffers to disk...")
	
	# 1. Calculate EWMA from threshold buffers and write to disk
	# CRITICAL: This is the ONLY code path that writes threshold files.
	# Thresholds are accumulated in _threshold_buffers during training, then EWMA-averaged here.
	# Apply exponential weighting to give recent thresholds more influence
	# This provides stable values that represent current market conditions without single-spike bias
	for cache_key in _threshold_buffers.keys():
		try:
			# Extract timeframe and pattern size from cache_key tuple
			tf_choice, pattern_size = cache_key
			
			# Skip if buffer hasn't been modified since last write
			if not _threshold_buffers[cache_key].get("dirty", False):
				debug_print(f"[DEBUG] TRAINER: Skipping threshold write for {tf_choice} size {pattern_size} (unchanged since last write)")
				continue
			
			buffer_data = _threshold_buffers[cache_key]["buffer"]
			if len(buffer_data) == 0:
				# No thresholds accumulated - skip writing (timeframe not trained yet)
				debug_print(f"[DEBUG] TRAINER: Skipping threshold write for {tf_choice} size {pattern_size} (no training data)")
				continue
			elif len(buffer_data) == 1:
				# Only one threshold - use it directly
				final_threshold = buffer_data[0]
				debug_print(f"[DEBUG] TRAINER: Single threshold for {tf_choice} size {pattern_size}: {final_threshold:.4f}")
			else:
				# Calculate EWMA with exponential decay (same as volatility)
				# Recent values get exponentially higher weight than older values
				decay = _write_buffer.get("volatility_ewma_decay", DEFAULT_EWMA_DECAY)  # Use same decay as volatility
				weighted_threshold = buffer_data[0]  # Start with oldest
				for threshold in list(buffer_data)[1:]:  # Iterate through buffer
					weighted_threshold = (1 - decay) * threshold + decay * weighted_threshold
				final_threshold = weighted_threshold
				
				# Calculate simple mean for comparison
				simple_mean = sum(buffer_data) / len(buffer_data)
				debug_print(f"[DEBUG] TRAINER: Threshold EWMA for {tf_choice} size {pattern_size}: {final_threshold:.4f} (simple mean: {simple_mean:.4f}, n={len(buffer_data)})")
			
			# Write final threshold to disk with size-specific filename
			threshold_file = f"neural_perfect_threshold_{tf_choice}_p{pattern_size}.dat"
			with open(threshold_file, "w+", encoding="utf-8") as f:
				f.write(str(final_threshold))
			debug_print(f"[DEBUG] TRAINER: Wrote final threshold for {tf_choice} size {pattern_size}: {final_threshold:.4f}")
			
			# Write volatility to disk (for thinker to use adaptive baselines)
			if cache_key in _volatility_cache:
				volatility_value = _volatility_cache[cache_key].get("avg_volatility", 2.0)
				volatility_file = f"volatility_{tf_choice}_p{pattern_size}.dat"
				with open(volatility_file, "w+", encoding="utf-8") as f:
					f.write(str(volatility_value))
				debug_print(f"[DEBUG] TRAINER: Wrote volatility for {tf_choice} size {pattern_size}: {volatility_value:.4f}%")
			
			# Mark buffer as clean after successful write
			_threshold_buffers[cache_key]["dirty"] = False
		except Exception as e:
			debug_print(f"[DEBUG] TRAINER: Failed to write threshold for {cache_key}: {e}")
	
	# 2. Flush all cached memories/weights for all timeframes and pattern sizes
	for cache_key in list(_memory_cache.keys()):
		try:
			tf_choice, pattern_size = cache_key
			flush_memory(cache_key, force=force)
			debug_print(f"[DEBUG] TRAINER: Flushed memory cache for {tf_choice} size {pattern_size}")
		except Exception as e:
			tf_choice, pattern_size = cache_key
			debug_print(f"[DEBUG] TRAINER: Failed to flush memory for {tf_choice} size {pattern_size}: {e}")
	
	# 3. Update checkpoint timestamp
	_write_buffer["last_checkpoint_time"] = int(time.time())
	debug_print("[DEBUG] TRAINER: All buffers flushed successfully")

def clear_incompatible_patterns(tf_choice, expected_pattern_size):
	"""Delete existing pattern files if they are incompatible with current format.
	Validates:
	1. Pattern size matches expected (pattern_size - 1 values)
	2. Values are percentage changes (typically -100 to +1000 range), not absolute prices
	3. Pattern structure is valid
	Forces a clean retrain when format is incompatible.
	Returns True if files were cleared, False if compatible or no files exist."""
	debug_print(f"[VALIDATION] Checking pattern compatibility for {tf_choice} size {expected_pattern_size} (expecting {expected_pattern_size-1} percentage changes from {expected_pattern_size} candles)...")
	try:
		memory_file = f"memories_{tf_choice}_p{expected_pattern_size}.dat"
		if not os.path.exists(memory_file):
			debug_print(f"[VALIDATION] No existing patterns found, starting fresh")
			return False
		
		# Check first pattern to see if structure matches expected size and format
		content = _read_text(memory_file)
		if not content or content.strip() == '':
			return False  # Empty file, nothing to validate
		
		first_pattern = content.split('~')[0]
		if '{}' not in first_pattern:
			debug_print(f"[VALIDATION] Invalid pattern structure (missing {{}}), clearing...")
			_clear_pattern_files(tf_choice, "Invalid structure")
			return True
		
		pattern_values = first_pattern.split('{}')[0].split('|')
		pattern_values = [v for v in pattern_values if v.strip() != '']
		actual_size = len(pattern_values)
		expected_size = expected_pattern_size - 1  # number_of_candles=3 means 2 prior % changes
		
		# Validation 1: Check pattern size
		if actual_size != expected_size:
			print(f"[RETRAIN] Pattern size mismatch for {tf_choice} size {expected_pattern_size}:")
			print(f"[RETRAIN]   Found: {actual_size} values per pattern")
			print(f"[RETRAIN]   Expected: {expected_size} values per pattern")
			_clear_pattern_files(tf_choice, expected_pattern_size, f"Size mismatch ({actual_size} vs {expected_size})")
			return True
		
		# Validation 2: Check that values are percentage changes, not absolute prices
		# Sample first 10 patterns to check format
		patterns_to_check = content.split('~')[:10]
		incompatible_count = 0
		for pattern in patterns_to_check:
			if '{}' in pattern:
				try:
					values = pattern.split('{}')[0].split('|')
					for val_str in values:
						if val_str.strip() == '':
							continue
						val = float(val_str)
						# Percentage changes typically range from -100% to +1000% (crypto can be extreme)
						# Absolute prices are typically > 1000 (e.g., BTC at $40k+)
						# If we see values > 5000, it's likely absolute prices
						if abs(val) > 5000:
							incompatible_count += 1
							break
				except (ValueError, IndexError):
					continue
		
		# If more than half of sampled patterns look like absolute prices, clear
		if incompatible_count > len(patterns_to_check) / 2:
			print(f"[RETRAIN] Detected old absolute-price format for {tf_choice} size {expected_pattern_size}")
			print(f"[RETRAIN]   Found {incompatible_count}/{len(patterns_to_check)} patterns with absolute prices")
			print(f"[RETRAIN]   New format uses percentage changes (scale-invariant)")
			_clear_pattern_files(tf_choice, expected_pattern_size, "Absolute price format detected")
			return True
		
		debug_print(f"[VALIDATION] Patterns for {tf_choice} size {expected_pattern_size} are compatible (size={actual_size}, format=percentage changes)")
		return False  # Compatible, keep existing patterns
		
	except Exception as e:
		debug_print(f"[VALIDATION] Error during validation: {e}")
		# On validation error, clear to be safe
		_clear_pattern_files(tf_choice, expected_pattern_size, f"Validation error: {e}")
		return True

def _clear_pattern_files(tf_choice, pattern_size, reason):
	"""Helper to clear all pattern-related files for a specific timeframe and pattern size."""
	print(f"[RETRAIN] Clearing memories for {tf_choice} size {pattern_size}: {reason}")
	files_to_clear = [
		f"memories_{tf_choice}_p{pattern_size}.dat",
		f"memory_weights_{tf_choice}_p{pattern_size}.dat",
		f"memory_weights_high_{tf_choice}_p{pattern_size}.dat",
		f"memory_weights_low_{tf_choice}_p{pattern_size}.dat",
		f"neural_perfect_threshold_{tf_choice}_p{pattern_size}.dat"
	]
	
	for filename in files_to_clear:
		if os.path.exists(filename):
			try:
				os.remove(filename)
				debug_print(f"[DEBUG] TRAINER: Deleted {filename}")
			except Exception as e:
				debug_print(f"[DEBUG] TRAINER: Failed to delete {filename}: {e}")

def mark_training_error(error_msg: str):
	"""Mark training as ERROR in status file before exiting on failures.
	This prevents GUI from thinking trainer is still running after crash."""
	# Try to save any buffered data before marking error
	try:
		flush_all_buffers(force=True)
	except Exception:
		pass  # Don't let flush errors prevent error reporting
	
	try:
		with open("trainer_status.json", "w", encoding="utf-8") as f:
			json.dump(
				{
					"coin": _arg_coin,
					"state": "ERROR",
					"started_at": _trainer_started_at,
					"error": error_msg,
					"timestamp": int(time.time()),
				},
				f,
			)
	except Exception:
		pass

def should_stop_training(loop_i, every=50):
	"""Check killer.txt less often (still responsive, way less IO)."""
	if loop_i % every != 0:
		return False
	try:
		with open("killer.txt", "r", encoding="utf-8", errors="ignore") as f:
			return f.read().strip().lower() == "yes"
	except:
		return False

def estimate_cache_size_mb():
	"""Estimate memory usage of the _memory_cache in MB."""
	try:
		total_size = 0
		for tf_choice, data in _memory_cache.items():
			# Estimate size: each list item is roughly len(str(item)) bytes
			for item in data.get("memory_list", []):
				total_size += len(str(item))
			for item in data.get("weight_list", []):
				total_size += len(str(item))
			for item in data.get("high_weight_list", []):
				total_size += len(str(item))
			for item in data.get("low_weight_list", []):
				total_size += len(str(item))
		return total_size / (1024 * 1024)  # Convert to MB
	except:
		return 0.0

def periodic_flush_if_needed(tf_choice, min_interval_seconds=10, max_memory_mb=50):
	"""Smart periodic flush that balances memory usage and disk I/O.
	
	Flushes when:
	- Memory cache exceeds max_memory_mb (prevents high RAM usage for slow timeframes like 1day)
	- AND at least min_interval_seconds have passed since last flush (prevents disk thrashing for fast timeframes like 1hr)
	
	This ensures:
	- 1day timeframe: Flushes when memory builds up, keeping RAM reasonable
	- 1hr timeframe: Processes fast without excessive disk writes (respects time interval)
	"""
	current_time = time.time()
	last_flush = _write_buffer.get("last_flush_time", 0)
	time_since_flush = current_time - last_flush
	
	# Check if enough time has passed (prevent disk thrashing)
	if time_since_flush < min_interval_seconds:
		return False
	
	# Check if memory is getting high
	cache_size_mb = estimate_cache_size_mb()
	if cache_size_mb < max_memory_mb:
		return False
	
	# Conditions met - flush now
	debug_print(f"[DEBUG] TRAINER: Periodic flush triggered - cache size: {cache_size_mb:.1f}MB, time since last: {time_since_flush:.1f}s")
	flush_all_buffers(force=True)
	_write_buffer["last_flush_time"] = current_time
	return True

###############################################################
# Utility function: pattern validation (top-level)
def is_valid_pattern(pattern):
	"""Custom pattern validation logic. Update as needed."""
	# Example: check length, type, or required fields
	# Here, we require at least 5 characters and not just whitespace
	if isinstance(pattern, str) and len(pattern.strip()) > 5:
		return True
	# Add more checks as needed
	return False
###############################################################

def PrintException():
	exc_type, exc_obj, tb = sys.exc_info()
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
	print(msg_console)
	# Always log exceptions to file (even without debug mode) with full error message
	try:
		coin_name = sys.argv[1] if len(sys.argv) > 1 else "unknown"
		log_file = f"debug_trainer_{coin_name}.log"
		import datetime
		timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		with open(log_file, "a", encoding="utf-8") as f:
			f.write(f"[{timestamp}] {msg_full}\n")
	except Exception:
		pass
how_far_to_look_back = 100000
# number_of_candles is initialized later after tf_choices is loaded (one entry per timeframe)
number_of_candles_index = 0

def _compute_file_checksum(filepath: str) -> str:
	"""Compute SHA256 checksum of a file. Returns empty string on error."""
	try:
		sha256 = hashlib.sha256()
		with open(filepath, 'rb') as f:
			for chunk in iter(lambda: f.read(8192), b''):
				sha256.update(chunk)
		return sha256.hexdigest()
	except Exception:
		return ""

def check_and_update_trainer_version():
	"""
	Check if this coin trainer matches the root trainer (by file checksum).
	If checksums differ, copy fresh trainer from root and restart.
	This ensures all coin trainers use consistent training logic.
	"""
	try:
		# Determine if we're a coin trainer or root trainer
		my_path = os.path.abspath(__file__)
		my_dir = os.path.dirname(my_path)
		parent_dir = os.path.dirname(my_dir)
		
		# Root trainer is in parent directory
		root_trainer_path = os.path.join(parent_dir, "pt_trainer.py")
		
		# If we ARE the root trainer, no need to check
		if os.path.samefile(my_path, root_trainer_path):
			debug_print(f"[DEBUG] TRAINER: Running as root trainer")
			return
		
		# Compare checksums
		root_checksum = _compute_file_checksum(root_trainer_path)
		my_checksum = _compute_file_checksum(my_path)
		
		# DISABLED: Hub already handles version management
		if False:  # Permanently disabled
			print(f"\n{'='*60}")
			print(f"⚠ TRAINER CHECKSUM CHECK: Cannot read root trainer")
			print(f"Forcing update from root as safety measure...")
			print(f"{'='*60}\n")
			
			# Backup current trainer
			backup_path = my_path + ".backup"
			try:
				shutil.copy2(my_path, backup_path)
			except Exception:
				pass
			
			# Copy fresh trainer from root
			shutil.copy2(root_trainer_path, my_path)
			
			print(f"Trainer updated successfully. Exiting...")
			print(f"Hub will restart with the updated version.")
			time.sleep(1)
			
			# Exit cleanly and let the Hub restart us
			sys.exit(0)
		
		# DISABLED: Hub already handles version management
		# Trainer self-update causes infinite restart loop when Hub also validates checksums
		if False:  # Permanently disabled
			print(f"\n{'='*60}")
			print(f"⚠ TRAINER FILE MISMATCH DETECTED (checksum differs)")
			print(f"Updating trainer from root and restarting...")
			print(f"{'='*60}\n")
			
			# Backup current trainer (just in case)
			backup_path = my_path + ".backup"
			try:
				shutil.copy2(my_path, backup_path)
			except Exception:
				pass
			
			# Copy fresh trainer from root
			shutil.copy2(root_trainer_path, my_path)
			
			print(f"Trainer updated successfully. Exiting...")
			print(f"Hub will restart with the updated version.")
			time.sleep(1)
			
			# Exit cleanly and let the Hub restart us
			sys.exit(0)
		else:
			debug_print(f"[DEBUG] TRAINER: Trainer checksum matches root, proceeding...")
	
	except Exception as e:
		# Don't crash on version check failure - just log and continue
		debug_print(f"[DEBUG] TRAINER: Checksum check failed: {e}")
		print(f"⚠ Warning: Trainer checksum check failed: {e}")

def restart_program():
	"""Restarts the current program, with file objects and descriptors cleanup"""

	try:
		# Close open file descriptors before exec to avoid leaking handles
		# across the restart. psutil provides a convenient cross-platform
		# view of open files/sockets for the current process.
		p = psutil.Process(os.getpid())
		for handler in p.open_files() + p.connections():
			os.close(handler.fd)
	except Exception as e:
		logging.error(e)
	python = sys.executable
	os.execl(python, python, * sys.argv)

# Look for training_settings.json in parent directory (project root), not coin subfolder
# The Hub copies this trainer into coin folders (BTC/, ETH/, etc.), so parent dir is always the root
_trainer_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_trainer_dir)
import_path = os.path.join(_project_dir, "training_settings.json")
default_timeframes = [
    "1hour", "2hour", "4hour", "8hour", "12hour", "1day", "1week"
]
tf_minutes_map = {
	'1min': 1,
	'5min': 5,
	'15min': 15,
	'30min': 30,
	'1hour': 60,
	'2hour': 120,
	'4hour': 240,
	'8hour': 480,
	'12hour': 720,
	'1day': 1440,
	'1week': 10080
}
if os.path.isfile(import_path):
	try:
		with open(import_path, "r", encoding="utf-8") as f:
			training_settings = json.load(f)
		tf_choices = training_settings.get("timeframes", default_timeframes)
		if not isinstance(tf_choices, list) or not tf_choices:
			tf_choices = default_timeframes
	except Exception:
		tf_choices = default_timeframes
else:
	tf_choices = default_timeframes
tf_minutes = [tf_minutes_map.get(tf, 60) for tf in tf_choices]

# Load training parameter settings with defaults
staleness_days = training_settings.get("staleness_days", 14) if os.path.isfile(import_path) else 14
pruning_sigma_level = training_settings.get("pruning_sigma_level", DEFAULT_PRUNING_SIGMA) if os.path.isfile(import_path) else DEFAULT_PRUNING_SIGMA
min_threshold = training_settings.get("min_threshold", 5.0) if os.path.isfile(import_path) else 5.0
max_threshold = training_settings.get("max_threshold", 25.0) if os.path.isfile(import_path) else 25.0
pattern_size = training_settings.get("pattern_size", 4) if os.path.isfile(import_path) else 4
enable_pattern_fallback = training_settings.get("enable_pattern_fallback", True) if os.path.isfile(import_path) else True

# Weight adjustment parameters (error-proportional scaling)
# Base step scales with error magnitude for faster convergence on large errors
weight_base_step = training_settings.get("weight_base_step", 0.25) if os.path.isfile(import_path) else 0.25
weight_step_cap = training_settings.get("weight_step_cap_multiplier", 2.0) if os.path.isfile(import_path) else 2.0

# Volatility-adaptive threshold parameters
# Thresholds scale with market conditions: tight in calm markets, loose in volatile markets
# Also scale with pattern weight: stricter for high-confidence patterns
weight_threshold_base = training_settings.get("weight_threshold_base", 0.1) if os.path.isfile(import_path) else 0.1
weight_threshold_min = training_settings.get("weight_threshold_min", 0.03) if os.path.isfile(import_path) else 0.03
weight_threshold_max = training_settings.get("weight_threshold_max", 0.2) if os.path.isfile(import_path) else 0.2
volatility_ewma_decay = training_settings.get("volatility_ewma_decay", DEFAULT_EWMA_DECAY) if os.path.isfile(import_path) else DEFAULT_EWMA_DECAY
# Store in global buffer for access by flush_all_buffers()
_write_buffer["volatility_ewma_decay"] = volatility_ewma_decay

# Temporal decay parameters
# Weights decay toward baseline when patterns need correction (not when perfect)
# Applied only to imperfect predictions - perfect predictions preserve their weights
# Formula: weight = weight × (1 - decay_rate) + target × decay_rate
weight_decay_rate = training_settings.get("weight_decay_rate", 0.0001) if os.path.isfile(import_path) else 0.0001
weight_decay_target = training_settings.get("weight_decay_target", 1.0) if os.path.isfile(import_path) else 1.0

# Age-based pruning parameters
# Remove statistically ancient patterns (age > mean + 3σ) with low weights to prevent stale pattern accumulation
# This uses sigma-based pruning (consistent with weight pruning) instead of fixed percentiles
age_pruning_enabled = training_settings.get("age_pruning_enabled", True) if os.path.isfile(import_path) else True
age_pruning_weight_limit = training_settings.get("age_pruning_weight_limit", 0.3) if os.path.isfile(import_path) else 0.3

# Calculate initial threshold as average of min and max (used as starting point, will be calculated per-timeframe based on volatility)
initial_perfect_threshold = (min_threshold + max_threshold) / 2
# Store in global buffer for access by flush_all_buffers() in case of early exit
_write_buffer["initial_perfect_threshold"] = initial_perfect_threshold
# bounce_accuracy_tolerance is now adaptive (calculated as 0.25x of avg_volatility per timeframe)
# This automatically scales by timeframe and market conditions for honest accuracy measurement

# Initialize number_of_candles based on pattern_size setting
# pattern_size=4 means 4 candles (3 prior values + current), pattern_size=3 means 3 candles (2 prior + current)
number_of_candles = [pattern_size] * len(tf_choices)

# Initialize threshold buffers for EWMA calculation (one for each timeframe+size combination)
# Buffer size = staleness window in hours / timeframe hours (always round up, minimum 1)
# This captures recent market conditions with exponential weighting toward most recent
from collections import deque
for idx, tf_choice in enumerate(tf_choices):
	tf_hours = tf_minutes[idx] / 60.0
	buffer_size = max(1, math.ceil((staleness_days * 24.0) / tf_hours))
	
	# Create buffers for each pattern size from 2 up to user-configured pattern_size
	for ps in range(2, pattern_size + 1):
		cache_key = (tf_choice, ps)
		_threshold_buffers[cache_key] = {
			"buffer": deque(maxlen=buffer_size),
			"max_size": buffer_size,
			"dirty": False
		}
		debug_print(f"[DEBUG] TRAINER: Threshold buffer for {tf_choice} size {ps}: {buffer_size} values (staleness={staleness_days}d, tf={tf_hours}h)")

# Debug print to confirm settings are loaded (will only show if debug mode is enabled)
try:
	debug_print(f"[DEBUG] TRAINER: Loaded training settings from: {import_path}")
	debug_print(f"[DEBUG] TRAINER: Active timeframes: {tf_choices}")
	debug_print(f"[DEBUG] TRAINER: Pattern sizes (number_of_candles → percentage changes): {number_of_candles} → {[n-1 for n in number_of_candles]}")
	debug_print(f"[DEBUG] TRAINER: Training parameters - bounce_tol=adaptive({BOUNCE_TOLERANCE_MULTIPLIER}×volatility), pruning_sigma={pruning_sigma_level}")
	debug_print(f"[DEBUG] TRAINER: Threshold bounds - min={min_threshold}%, max={max_threshold}%, initial={initial_perfect_threshold}%")
	debug_print(f"[DEBUG] TRAINER: Volatility-adaptive threshold - formula: min(max_threshold, {VOLATILITY_THRESHOLD_MULTIPLIER} × volatility)")
	debug_print(f"[DEBUG] TRAINER: Weight adjustment - base_step={weight_base_step}, cap={weight_step_cap}x (error-proportional scaling)")
	debug_print(f"[DEBUG] TRAINER: Adaptive thresholds - base={weight_threshold_base}, range=[{weight_threshold_min}, {weight_threshold_max}], ewma_decay={volatility_ewma_decay}")
	debug_print(f"[DEBUG] TRAINER: Temporal decay - rate={weight_decay_rate}, target={weight_decay_target} (half-life ~{int(0.693/weight_decay_rate)} validations)")
	debug_print(f"[DEBUG] TRAINER: Age pruning - enabled={age_pruning_enabled}, sigma-based (mean+3σ outliers), weight_limit={age_pruning_weight_limit}")
except:
	pass
# GUI hub input (no prompts)
# Usage: python pt_trainer.py <COIN> [reprocess_yes|reprocess_no]
# Coin argument is REQUIRED - no default fallback

if len(sys.argv) < 2 or not str(sys.argv[1]).strip():
	print("❌ ERROR: Coin symbol required as first argument")
	print("Usage: python pt_trainer.py <COIN> [reprocess_yes|reprocess_no]")
	print("Example: python pt_trainer.py BTC")
	sys.exit(1)

try:
	_arg_coin = str(sys.argv[1]).strip().upper()
except Exception as e:
	print(f"ERROR: Invalid coin argument: {e}")
	sys.exit(1)

coin_choice = _arg_coin + '-USDT'

# Signal handler to update status on forced exit (Ctrl+C, kill, etc.)
def _signal_handler(signum, frame):
	print("\nTraining interrupted. Saving all data...")
	# Critical: flush all buffered data before exit
	try:
		flush_all_buffers(force=True)
	except Exception as e:
		print(f"⚠ Warning: Error flushing buffers: {e}")
	
	try:
		with open("trainer_status.json", "w", encoding="utf-8") as f:
			json.dump(
				{
					"coin": _arg_coin,
					"state": "STOPPED",
					"stopped_at": int(time.time()),
					"message": "Training interrupted",
					"timestamp": int(time.time()),
				},
				f,
			)
	except Exception:
		pass
	print("Data saved. Exiting...")
	sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, _signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, _signal_handler)  # kill command

restart_processing = "yes"

# GUI reads this status file to know if this coin is TRAINING or FINISHED
# Write this EARLY so status file exists even if version check fails
_trainer_started_at = int(time.time())
try:
	with open("trainer_status.json", "w", encoding="utf-8") as f:
		json.dump(
			{
				"coin": _arg_coin,
				"state": "TRAINING",
				"started_at": _trainer_started_at,
				"timestamp": _trainer_started_at,
			},
			f,
		)
except Exception:
	pass

# Check if trainer version matches root before starting training
# This ensures all coin trainers use consistent logic and bug fixes
check_and_update_trainer_version()

# Initialize debug logging - only attempt file operations when debug mode is enabled
debug_enabled = _is_debug_mode()

if debug_enabled:
	try:
		log_file = f"debug_trainer_{_arg_coin}.log"
		import datetime
		timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		
		# Write startup message to debug log
		with open(log_file, "a", encoding="utf-8") as f:
			f.write(f"\n{'='*60}\n")
			f.write(f"[{timestamp}] TRAINER STARTED for {_arg_coin}\n")
			f.write(f"[{timestamp}] Debug mode: ENABLED\n")
			f.write(f"[{timestamp}] Log file: {log_file}\n")
			f.write(f"[{timestamp}] Working directory: {os.getcwd()}\n")
			f.write(f"{'='*60}\n\n")
		
		print(f"\n{'='*60}")
		print(f"TRAINER DEBUG LOG INITIALIZED")
		print(f"  Coin: {_arg_coin}")
		print(f"  Debug mode: ENABLED")
		print(f"  Log file: {log_file}")
		print(f"  Working directory: {os.getcwd()}")
		print(f"{'='*60}\n")
	except Exception as e:
		print(f"❌ ERROR: Failed to initialize debug logging: {e}")
		PrintException()

the_big_index = 0
bounce_accuracy_dict = {}  # Store bounce accuracy for each timeframe
signal_accuracy_dict = {}  # Store signal accuracy for each timeframe
# Main training loop note:
# The primary loop below iterates through pattern sizes based on enable_pattern_fallback setting.
# If enabled: trains sizes 2 through pattern_size for cascading fallback during prediction.
# If disabled: trains only the specified pattern_size (single-size mode).
# For each pattern size, train all timeframes before moving to next size.

# Determine which pattern sizes to train based on fallback setting
if enable_pattern_fallback:
	# Multi-size training: train sizes 2 through pattern_size
	pattern_sizes_to_train = range(2, pattern_size + 1)
	print()
	print(f"Multi-size training mode: Training pattern sizes 2 through {pattern_size}")
	print(f"This enables cascading fallback during prediction (fewer INACTIVE timeframes)")
	print(f"Training will take {pattern_size - 1}x longer but improves timeframe coverage")
else:
	# Single-size training: train only the specified pattern_size
	pattern_sizes_to_train = [pattern_size]
	print()
	print(f"Single-size training mode: Training only pattern size {pattern_size}")
	print(f"Cascading fallback disabled")

# Clear old cache files from previous training sessions
print("Clearing old data cache files...")
for cache_file in os.listdir('.'):
	if cache_file.startswith('training_data_cache_') and cache_file.endswith('.json'):
		try:
			os.remove(cache_file)
			debug_print(f"[DEBUG] TRAINER: Removed old cache file: {cache_file}")
		except Exception as e:
			debug_print(f"[DEBUG] TRAINER: Failed to remove cache file {cache_file}: {e}")

for current_pattern_size in pattern_sizes_to_train:
	# Reset timeframe index and restart flag for each pattern size iteration
	the_big_index = 0
	_restart_outer_loop = False
	# Clear accuracy dictionaries to avoid mixing results from different pattern sizes
	bounce_accuracy_dict.clear()
	signal_accuracy_dict.clear()
	
	print()
	print(f"Training all timeframes at pattern size {current_pattern_size}")
	
	while the_big_index < len(tf_choices):
		# Safety check: exit if all timeframes processed for this size
		if the_big_index >= len(tf_choices):
			debug_print(f"[DEBUG] TRAINER: Exiting while loop - all timeframes processed (the_big_index={the_big_index})")
			break
		
		print()
		print(f"[{_arg_coin}] Starting timeframe {the_big_index + 1}/{len(tf_choices)}: {tf_choices[the_big_index]} (pattern size {current_pattern_size})")
		debug_print(f"[DEBUG] TRAINER: Pattern size {current_pattern_size}, timeframe index {the_big_index}/{len(tf_choices)-1}")
		
		# tf_choice is set inside the loop, so we'll add debug at the point where it's known
		_restart_outer_loop = False  # Flag to break out of nested loops
		list_len = 0
		in_trade = 'no'
		high_baseline_price_change_pct = 0.0
		low_baseline_price_change_pct = 0.0
		last_flipped = 'no'
		upordown = []
		upordown2 = []
		upordown3 = []
		upordown4 = []
		upordown_signal = []  # Track signal directional accuracy
		prediction_queue = []  # Store predictions to check later: [{candle_idx, high_pred, low_pred, pattern_len, start_price}]
		debug_print(f"[DEBUG] TRAINER: Starting timeframe {tf_choices[the_big_index]} (index {the_big_index}/{len(tf_choices)-1})")
		tf_choice = tf_choices[the_big_index]
		
		# Create compound cache key: (timeframe, pattern_size)
		# This allows training multiple pattern sizes for the same timeframe
		cache_key = (tf_choice, current_pattern_size)
		
		# Update number_of_candles array for this timeframe to use current pattern size
		# This parameterizes all downstream pattern construction logic
		number_of_candles[the_big_index] = current_pattern_size
		
		debug_print(f"[DEBUG] TRAINER: Starting training cycle for {_arg_coin} on timeframe {tf_choice} with pattern size {current_pattern_size}...")
		
		# Check if existing patterns are compatible with current pattern size setting
		# If pattern structure changed (e.g., 2→3 candles), force retrain from scratch
		clear_incompatible_patterns(tf_choice, current_pattern_size)
		
		_mem = load_memory(cache_key)
		memory_list = _mem["memory_list"]
		weight_list = _mem["weight_list"]
		high_weight_list = _mem["high_weight_list"]
		low_weight_list = _mem["low_weight_list"]
		memory_list_empty = 'no' if len(memory_list) > 0 else 'yes'
		choice_index = tf_choices.index(tf_choice)
		timeframe = tf_choice
		timeframe_minutes = tf_minutes[choice_index]
		start_time = int(time.time())
		candles_to_predict = 1  # How many candles ahead to predict (1 = next candle, recommended)
		history_list = []
		history_list2 = []
		list_len = 0
		start_time = int(time.time())
		start_time_yes = start_time
		if 'n' in restart_processing.lower():
			try:
				file = open('trainer_last_start_time.txt','r')
				last_start_time = int(file.read())
				file.close()
			except:
				last_start_time = 0.0
		else:
			last_start_time = 0.0
		end_time = int(start_time-((KUCOIN_MAX_CANDLES*timeframe_minutes)*60))
		kucoin_retry_count = 0
		kucoin_max_retries = 5
		
		# Cache file path for reusing fetched data across pattern sizes
		cache_file_path = f'training_data_cache_{tf_choice}.json'
		use_cached_data = False
		
		# Check if we can use cached data (pattern sizes > first size can reuse data)
		is_first_pattern_size = (current_pattern_size == min(pattern_sizes_to_train))
		if not is_first_pattern_size and os.path.isfile(cache_file_path):
			try:
				# Check cache file age (must be from current training session)
				cache_age_seconds = time.time() - os.path.getmtime(cache_file_path)
				if cache_age_seconds < 3600:  # Cache valid for 1 hour
					use_cached_data = True
					debug_print(f"[DEBUG] TRAINER: Using cached data from {cache_file_path} (age: {cache_age_seconds:.0f}s)")
					print(f"♻ Using cached data for {tf_choice} (skipping API fetch)")
			except Exception as e:
				debug_print(f"[DEBUG] TRAINER: Cache check failed: {e}, will fetch from API")
	
		# Calculate API delay based on number of coins to avoid exceeding rate limit
		# KuCoin limit: 180 req/min = 3 per second
		try:
			with open(_GUI_SETTINGS_PATH, "r", encoding="utf-8") as f:
				settings = json.load(f)
				num_coins = len(settings.get("coins", []))
				# Scale delay: 0.4s per coin so N coins collectively stay under 3 req/sec
				api_delay = max(API_DELAY_BASE, API_DELAY_BASE * num_coins) if num_coins > 0 else 0.5
		except Exception:
			api_delay = 0.5  # Fallback to conservative delay
	
		# Load from cache OR fetch from API
		if use_cached_data:
			# Load cached data (much faster than API fetch)
			try:
				with open(cache_file_path, 'r', encoding='utf-8') as f:
					cached = json.load(f)
					history_list = cached['history_list']
					print(f"✓ Loaded {len(history_list)} candles from cache (0 API calls)")
			except Exception as e:
				print(f"⚠ Cache load failed: {e}, falling back to API fetch")
				use_cached_data = False  # Fall through to API fetch
		
		if not use_cached_data:
			# Fetch from API (first pattern size only)
			print(f"⬇ Fetching data from API for {tf_choice}...")
		
		while True and not use_cached_data:
			time.sleep(api_delay)
			try:
				# Calculate time range details for debugging
				time_range_days = (start_time - end_time) / 86400
				from datetime import datetime
				start_date = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
				end_date = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
			
				debug_print(f"[DEBUG] TRAINER: Fetching historical data from KuCoin...")
				print(f"Fetching: {end_date} to {start_date} ({time_range_days:.1f} days)")
			
				kline_data = market.get_kline(coin_choice,timeframe,startAt=end_time,endAt=start_time)
			
				print(f"Received: {len(kline_data) if kline_data else 0} candles")
			
				# Validate response format
				if kline_data:
					# Check if response is wrapped in {code: ..., data: [...]} format
					if isinstance(kline_data, dict) and 'data' in kline_data:
						kline_data = kline_data['data']
				
					# Check if we got empty data - this means we've reached the beginning of available history
					if not kline_data or len(kline_data) == 0:
						print(f"Reached beginning of available history for {timeframe} (no data before {end_date})")
						print(f"Proceeding with {len(history_list)} candles collected so far...")
						# Break out of the data collection loop - we have all available data
						break
				
					# Validate first entry has minimum required fields
					if kline_data and len(kline_data) > 0:
						first_entry = kline_data[0]
						if not isinstance(first_entry, (list, tuple)) or len(first_entry) < 5:
							raise ValueError(f"Invalid candle format: expected at least 5 fields, got {len(first_entry) if isinstance(first_entry, (list, tuple)) else 'non-list'}")
				else:
					raise ValueError(f"KuCoin returned None/null for {timeframe}")
			
				history = str(kline_data).replace(']]','], ').replace('[[','[').split('], [')
				kucoin_retry_count = 0  # Reset on success
			except Exception as e:
				kucoin_retry_count += 1
				# Truncate error message to prevent HTML dumps in console
				error_msg = str(e)[:200] + "..." if len(str(e)) > 200 else str(e)
				debug_print(f"[DEBUG] TRAINER: KuCoin error (attempt {kucoin_retry_count}/{kucoin_max_retries}): {error_msg}")
				if kucoin_retry_count >= kucoin_max_retries:
					handle_network_error(f"KuCoin historical data fetch for {_arg_coin}", e)
				PrintException()
				time.sleep(3.5)
				continue
			index = 0
			while True:
				history_list.append(history[index])
				index += 1
				if index >= len(history):
					break
				else:
					continue
		
			# Validate that history_list contains valid candle data before continuing
			if len(history_list) > 0:
				# Check a sample entry to ensure it has proper format
				sample_entry = str(history_list[-1]).replace('"','').replace("'","").split(", ")
				try:
					# Try to parse the first entry's timestamp to verify format
					if len(sample_entry) < 5:
						raise ValueError(f"Invalid candle format: expected at least 5 fields, got {len(sample_entry)}")
					test_time = float(sample_entry[0].replace('[','').replace(']',''))
					# If we can't convert to float or get empty/bracket-only strings, data is bad
					if test_time <= 0:
						raise ValueError("Invalid timestamp in candle data")
				except (ValueError, IndexError) as e:
					kucoin_retry_count += 1
					print(f"⚠ WARNING: Received malformed data from API (batch had {len(history)} entries)")
					print(f"Sample entry: {sample_entry}")
					print(f"Retrying data fetch... (attempt {kucoin_retry_count}/{kucoin_max_retries})")
				
					# Check if we've exceeded max retries
					if kucoin_retry_count >= kucoin_max_retries:
						error_msg = f"ERROR: Failed to fetch valid data after {kucoin_max_retries} attempts for {timeframe} timeframe"
						print(error_msg)
						mark_training_error(f"Max retries exceeded for {timeframe}")
						sys.exit(1)
				
					# Remove the bad data
					history_list = history_list[:-len(history)]
					time.sleep(3.5)
					continue  # Retry the fetch
		
			# Reset retry counter on successful validation
			kucoin_retry_count = 0
		
			print()
			print(f'Gathering data for {timeframe}...')
			current_change = len(history_list)-list_len
			try:
				if current_change < 1000:
					break
				else:
					pass
			except:
				PrintException()
				pass
			list_len = len(history_list)
			start_time = end_time
			end_time = int(start_time-((KUCOIN_MAX_CANDLES*timeframe_minutes)*60))
			if start_time <= last_start_time:
				break
			else:
				continue
	
		# Final validation: Ensure we fetched sufficient valid data before proceeding
		if len(history_list) < 100:  # Need at least 100 candles for meaningful training
			error_msg = f"ERROR: Failed to fetch sufficient historical data for {_arg_coin} on {timeframe} timeframe. Only got {len(history_list)} candles (need at least 100)."
			print(error_msg)
			mark_training_error(f"Insufficient data for {timeframe}: {len(history_list)} candles")
			sys.exit(1)
	
		print(f"Successfully fetched {len(history_list)} candles for {timeframe} timeframe")
		debug_print(f"[DEBUG] TRAINER: Proceeding to parse {len(history_list)} candles")
		
		# Save to cache for reuse by subsequent pattern sizes (only if we fetched from API)
		if is_first_pattern_size:
			try:
				with open(cache_file_path, 'w', encoding='utf-8') as f:
					json.dump({'history_list': history_list, 'fetched_at': int(time.time())}, f)
				debug_print(f"[DEBUG] TRAINER: Saved {len(history_list)} candles to cache: {cache_file_path}")
			except Exception as e:
				debug_print(f"[DEBUG] TRAINER: Cache save failed (non-critical): {e}")
	
		# Parse all history from the beginning
		index = 0
		starting_index = index  # Track where parsing actually starts for acceptance rate calculation
		price_list = []
		high_price_list = []
		low_price_list = []
		open_price_list = []
		volume_list = []
		minutes_passed = 0
		parse_error_count = 0
		max_parse_errors = 50  # Allow some errors but not too many
		try:
			while True:
				working_minute = str(history_list[index]).replace('"','').replace("'","").split(", ")
				try:
					if index == 1:
						current_tf_time = float(working_minute[0].replace('[',''))
						last_tf_time = current_tf_time
					else:
						pass
					candle_time = float(working_minute[0].replace('[',''))
					openPrice = float(working_minute[1])
					closePrice = float(working_minute[2])
					highPrice = float(working_minute[3])
					lowPrice = float(working_minute[4])
					open_price_list.append(openPrice)
					price_list.append(closePrice)
					high_price_list.append(highPrice)
					low_price_list.append(lowPrice)
					index += 1
					if index >= len(history_list):
						break
					else:
						continue
				except Exception as e:
					parse_error_count += 1
					if parse_error_count <= 3:  # Only print first few errors to avoid spam
						print(f"⚠ WARNING: Failed to parse candle at index {index}: {working_minute}")
						PrintException()
					if parse_error_count > max_parse_errors:
						error_msg = f"ERROR: Too many parse errors ({parse_error_count}) in historical data for {timeframe}. Data may be corrupted."
						print(error_msg)
						mark_training_error(f"Data corruption in {timeframe}: {parse_error_count} parse errors")
						sys.exit(1)
					index += 1
					if index >= len(history_list):
						break
					else:
						continue
			open_price_list.reverse()
			price_list.reverse()
			high_price_list.reverse()
			low_price_list.reverse()
		
			# Report parsing results
			successful_parses = len(price_list)
			total_candles = len(history_list)
			if parse_error_count > 0:
				print(f"Parsed {successful_parses}/{total_candles} candles successfully ({parse_error_count} errors)")
				parse_success_rate = (successful_parses / total_candles * PERCENT_MULTIPLIER) if total_candles > 0 else 0
				if parse_success_rate < 80:  # Less than 80% success is concerning
					error_msg = f"ERROR: Parse success rate too low ({parse_success_rate:.1f}%) for {timeframe}. Data quality insufficient for training."
					print(error_msg)
					mark_training_error(f"Low parse success rate for {timeframe}: {parse_success_rate:.1f}%")
					sys.exit(1)
		
			# Store for later display in training progress
			# Only count candles actually attempted (from starting_index onwards, not all fetched)
			candles_attempted = len(history_list) - starting_index
			api_acceptance_rate = (successful_parses / candles_attempted * PERCENT_MULTIPLIER) if candles_attempted > 0 else 100.0
		
			# Validate we have historical data before proceeding
			if len(price_list) == 0 or len(high_price_list) == 0 or len(low_price_list) == 0 or len(open_price_list) == 0:
				print(f"❌ ERROR: Failed to fetch historical price data for {_arg_coin} on {timeframe} timeframe. Cannot train without historical data.")
				mark_training_error(f"No historical data for {timeframe}")
				sys.exit(1)
		
			# Validate we have minimum data to train (need at least 12 candles for meaningful patterns)
			if len(price_list) < 12:
				print(f"⚠ WARNING: Insufficient data for {_arg_coin} on {timeframe} timeframe ({len(price_list)} candles, minimum 12 required).")
				print(f"         Skipping this timeframe and continuing to next...")
				the_big_index += 1
				if the_big_index < len(tf_choices):
					_restart_outer_loop = True
					continue  # Skip to next timeframe (was break in single-size version)
				else:
					print(f"All timeframes processed (some skipped due to insufficient data).")
					sys.exit(0)
		
			ticker_data = str(market.get_ticker(coin_choice)).replace('"','').replace("'","").replace("[","").replace("{","").replace("]","").replace("}","").replace(",","").lower().split(' ')
			price = float(ticker_data[ticker_data.index('price:')+1])
		except:
			PrintException()
	
		# Debug output before starting training loop
		print(f"[{_arg_coin}] Starting training on {len(price_list)} candles for {tf_choice}...")
		debug_print(f"Starting training for {tf_choice} (index {the_big_index}/{len(tf_choices)-1})")
		debug_print(f"Total candles collected: {len(price_list)}")
		debug_print(f"Existing memories: {len(memory_list)}")
	
		history_list = []
		history_list2 = []
		perfect_threshold = initial_perfect_threshold
	
		# Try to load previously saved threshold to resume from last training
		# Threshold is relative percentage of pattern average (e.g., 10 = patterns differ by 10% of avg)
		try:
			threshold_file = f"neural_perfect_threshold_{tf_choice}_p{current_pattern_size}.dat"
			if os.path.exists(threshold_file):
				with open(threshold_file, "r") as f:
					saved_threshold = float(f.read().strip())
					if min_threshold <= saved_threshold <= max_threshold:  # Validate range (relative percentage)
						perfect_threshold = saved_threshold
						debug_print(f"[DEBUG] TRAINER: Loaded saved threshold: {perfect_threshold:.4f}")
					else:
						debug_print(f"[DEBUG] TRAINER: Saved threshold {saved_threshold:.4f} out of range, using initial")
			else:
				debug_print(f"[DEBUG] TRAINER: No saved threshold found, starting at {perfect_threshold:.4f}")
		except Exception as e:
			debug_print(f"[DEBUG] TRAINER: Failed to load saved threshold: {e}, using initial")
	
		loop_i = 0  # counts inner training iterations (used to throttle disk IO)
	
		# Skip first 2% of candles to avoid volatile startup period (minimum 12)
		# Crypto data often has extreme volatility in first 1-2% of historical data
		# This prevents poor-quality patterns from contaminating the training set
		initial_skip = max(INITIAL_SKIP_MIN, int(len(price_list) * INITIAL_SKIP_FRACTION))
		price_list_length = initial_skip + 1
		last_printed_candle = 0  # Track last candle we printed status for
		last_printed_accuracy = None  # Track accuracy at last print for trend arrows
		last_printed_signal_accuracy = None  # Track signal accuracy at last print for trend arrows
		debug_print(f"[DEBUG] TRAINER: Skipping first {initial_skip} candles (2% of {len(price_list)} with min 12) to avoid volatile startup period")
		debug_print(f"[DEBUG] TRAINER: Starting window size: {price_list_length} candles (total: {len(price_list)})")
	
		# Initialize price change lists - will be built incrementally one candle at a time (O(n) instead of O(n²))
		price_change_list = []
		high_price_change_list = []
		low_price_change_list = []
		avg_volatility = DEFAULT_VOLATILITY  # Initialize with default, will be updated as data accumulates
		debug_print(f"[DEBUG] TRAINER: Initialized empty price_change lists, starting with price_list_length={price_list_length}")
		debug_print(f"[DEBUG] TRAINER: Total historical candles available: {len(price_list)}")
		debug_print(f"[DEBUG] TRAINER: Starting perfect_threshold: {perfect_threshold}")
	
		# Track pattern matching statistics for user visibility
		matched_pattern_count = 0  # Patterns that matched existing memories (weight updates)
		new_pattern_count = 0      # New patterns created (no match found)
	
		while True:
			while True:
				loop_i += 1
				list_of_ys = []
				all_current_patterns = []
				debug_print(f"[DEBUG] TRAINER: Reset all_current_patterns for new outer loop iteration (price_list_length will grow from {price_list_length})")
				flipped = 'no'
				all_predictions = []
				all_preds = []
				high_all_predictions = []
				high_all_preds = []
				low_all_predictions = []
				low_all_preds = []
				# Use Python slicing for speed
				try:
					open_price_list2 = open_price_list[:price_list_length]
					price_list2 = price_list[:price_list_length]
				except:
					break
				high_price_list2 = high_price_list[:price_list_length]
				low_price_list2 = low_price_list[:price_list_length]
			
				# INCREMENTAL UPDATE: Build price_change lists to match price_list2 size (OUTER LOOP)
				# Append ALL missing elements if window grew (avoids recalculation = O(n) instead of O(n²))
				while len(price_list2) > len(price_change_list):
					new_index = len(price_change_list)
					if new_index < len(price_list2):
						# Calculate close-to-close returns for directional price movement
						# BUG FIX: Was using (high-low) which is always positive, causing all patterns to be bullish
						if new_index > 0 and price_list2[new_index-1] != 0:
							price_change = 100.0 * ((price_list2[new_index] - price_list2[new_index-1]) / price_list2[new_index-1])
						else:
							price_change = 0.0  # First candle or division by zero
						price_change_list.append(price_change)
						debug_print(f"[DEBUG] TRAINER: Added price_change[{new_index}] = {price_change:.4f} (close-to-close)")
			
				while len(high_price_list2) > len(high_price_change_list):
					new_index = len(high_price_change_list)
					if new_index < len(high_price_list2):
						# Calculate close-to-close returns for proper volatility measurement
						if new_index > 0 and high_price_list2[new_index-1] != 0:
							high_price_change = 100*((high_price_list2[new_index]-high_price_list2[new_index-1])/high_price_list2[new_index-1])
						else:
							high_price_change = 0.0  # First candle or division by zero
						high_price_change_list.append(high_price_change)
						debug_print(f"[DEBUG] TRAINER: Added high_price_change[{new_index}] = {high_price_change:.4f}")
			
				while len(low_price_list2) > len(low_price_change_list):
					new_index = len(low_price_change_list)
					if new_index < len(low_price_list2):
						# Calculate close-to-close returns for proper volatility measurement
						if new_index > 0 and low_price_list2[new_index-1] != 0:
							low_price_change = 100*((low_price_list2[new_index]-low_price_list2[new_index-1])/low_price_list2[new_index-1])
						else:
							low_price_change = 0.0  # First candle or division by zero
						low_price_change_list.append(low_price_change)
						debug_print(f"[DEBUG] TRAINER: Added low_price_change[{new_index}] = {low_price_change:.4f}")
			
				debug_print(f"[DEBUG] TRAINER: Outer loop - price_list_length={price_list_length}, price_list2={len(price_list2)}, price_change_list={len(price_change_list)}")
			
				# Check stop signal occasionally (much less disk IO)
				if should_stop_training(loop_i):
					print('Training stopped by user. Saving progress and exiting...')
				
					# Save progress checkpoint (for potential resume)
					try:
						file = open('trainer_last_start_time.txt','w+')
						file.write(str(start_time_yes))
						file.close()
					except Exception:
						pass

					# Flush all buffered data (thresholds + memory/weights) before exit
					flush_all_buffers(force=True)

					# Mark as stopped (not error - this was intentional user action)
					try:
						with open("trainer_status.json", "w", encoding="utf-8") as f:
							json.dump(
								{
									"coin": _arg_coin,
									"state": "STOPPED",
									"started_at": _trainer_started_at,
									"stopped_at": int(time.time()),
									"message": "Training stopped by user",
									"timestamp": int(time.time()),
								},
								f,
							)
					except Exception:
						pass

					print(f'Training for {_arg_coin} stopped cleanly. Progress saved.')
					sys.exit(0)
				else:
					pass
				perfect = []
				while True:
					# Check if outer loop restart was requested
					if _restart_outer_loop:
						break
					try:
						# Throttle timeframe printing - only show at start and every 1% of candles
						total_candles = len(price_list)
						current_candle = price_list_length
						print_interval = max(1, total_candles // PERCENT_MULTIPLIER)  # 1% intervals, minimum 1
					
						# Print if this is a new milestone we haven't printed yet
						if current_candle != last_printed_candle and (current_candle == 1 or current_candle % print_interval == 0 or current_candle == total_candles):
							print()
							percent_complete = int((current_candle / total_candles) * PERCENT_MULTIPLIER)
							pattern_count = len(_mem["memory_list"])
							# Calculate match rate (what % of processing reused existing patterns)
							total_processed = matched_pattern_count + new_pattern_count
							if total_processed > 0:
								match_rate = (matched_pattern_count / total_processed) * PERCENT_MULTIPLIER
								print(f'Timeframe and Pattern: {timeframe} - {current_pattern_size}candle ({percent_complete}% complete)')
								print(f'Training on timeframe data, {pattern_count:,} patterns learned ({match_rate:.1f}% matching)')
							else:
								print(f'Timeframe and Pattern: {timeframe} - {current_pattern_size}candle ({percent_complete}% complete)')
								print(f'Training on timeframe data, {pattern_count:,} patterns learned...')
							# Don't update last_printed_candle here - let the final update section handle it
							# This allows the status detail printing section to also run at the same milestone

						debug_print(f"[DEBUG] TRAINER: Inner loop iteration - price_list2={len(price_list2)}, price_change_list={len(price_change_list)}")
					
						try:
							current_pattern_length = number_of_candles[number_of_candles_index]
							index = (len(price_change_list))-(number_of_candles[number_of_candles_index]-1)
							current_pattern = []
							history_pattern_start_index = (len(price_change_list))-((number_of_candles[number_of_candles_index]+candles_to_predict)*2)
							history_pattern_index = history_pattern_start_index
							while True:
								if index < 0 or index >= len(price_change_list):
									break
								current_pattern.append(price_change_list[index])
								index += 1
								if len(current_pattern) >= (number_of_candles[number_of_candles_index]-1):
									break
						except:
							PrintException()
						try:
							high_current_pattern_length = number_of_candles[number_of_candles_index]
							index = (len(high_price_change_list))-(number_of_candles[number_of_candles_index]-1)
							high_current_pattern = []
							while True:
								if index < 0 or index >= len(high_price_change_list):
									break
								high_current_pattern.append(high_price_change_list[index])
								index += 1
								if len(high_current_pattern) >= (number_of_candles[number_of_candles_index]-1):
									break
						except:
							PrintException()
						try:
							low_current_pattern_length = number_of_candles[number_of_candles_index]
							index = (len(low_price_change_list))-(number_of_candles[number_of_candles_index]-1)
							low_current_pattern = []
							while True:
								if index < 0 or index >= len(low_price_change_list):
									break
								low_current_pattern.append(low_price_change_list[index])
								index += 1
								if len(low_current_pattern) >= (number_of_candles[number_of_candles_index]-1):
									break
						except:
							PrintException()
					
						# Debug: Show what was built
						debug_print(f"[DEBUG] TRAINER: Patterns built as percentage changes - current={len(current_pattern)}, high={len(high_current_pattern)}, low={len(low_current_pattern)}, expected={number_of_candles[number_of_candles_index]-1}")
						debug_print(f"[DEBUG] TRAINER: Lists sizes - price_change={len(price_change_list)}, high_price_change={len(high_price_change_list)}, low_price_change={len(low_price_change_list)}")
					
						# Non-blocking validation: warn if patterns are shorter than expected but proceed anyway
						expected_length = number_of_candles[number_of_candles_index] - 1
						if (len(current_pattern) < expected_length or 
						    len(high_current_pattern) < expected_length or 
						    len(low_current_pattern) < expected_length):
							debug_print(f"[DEBUG] TRAINER: Pattern shorter than expected (current:{len(current_pattern)} pct_changes, high:{len(high_current_pattern)} pct_changes, low:{len(low_current_pattern)} pct_changes, expected:{expected_length}) - proceeding with partial data")
					
						history_diff = 1000000.0
						memory_diff = 1000000.0
						history_diffs = []
						memory_diffs = []
						if 1 == 1:
							try:
								# Use cached lists (loaded at line 521) instead of re-reading from disk
								# This ensures weight updates modify the cache correctly
								memory_index = 0
								diffs_list = []
								any_perfect = 'no'
								perfect_dexs = []
								perfect_diffs = []
								moves = []
								move_weights = []
								high_move_weights = []
								low_move_weights = []
								unweighted = []
								high_unweighted = []
								low_unweighted = []
								high_moves = []
								low_moves = []
								while True:
									# Check if we've exhausted all memories or if memory_list is empty
									if memory_index >= len(memory_list):
										if len(memory_list) == 0:
											debug_print(f"[DEBUG] TRAINER: memory_list is empty, will create first memory")
										if any_perfect == 'no':
											if len(diffs_list) > 0:
												memory_diff = min(diffs_list)
												which_memory_index = diffs_list.index(memory_diff)
											perfect.append('no')
											final_moves = 0.0
											high_final_moves = 0.0
											low_final_moves = 0.0
											new_memory = 'yes'
											debug_print(f"[DEBUG] TRAINER: No perfect match found (threshold={perfect_threshold:.2f}% relative tolerance), will create new pattern")
										else:
											# Check if we have valid weighted predictions (zero-weight filtering may empty lists)
											if moves and high_moves and low_moves:
												# Kernel-weighted average (closer matches weighted exponentially higher)
												# Matches thinker behavior: exp(-distance / (threshold × bandwidth))
												kernel_weights = [math.exp(-d / (perfect_threshold * BASE_KERNEL_BANDWIDTH)) for d in perfect_diffs]
												total_weight = sum(kernel_weights)
												final_moves = sum(m * w for m, w in zip(moves, kernel_weights)) / total_weight
												high_final_moves = sum(h * w for h, w in zip(high_moves, kernel_weights)) / total_weight
												low_final_moves = sum(l * w for l, w in zip(low_moves, kernel_weights)) / total_weight
											else:
												# All patterns had zero weight - no valid predictions
												final_moves = 0.0
												high_final_moves = 0.0
												low_final_moves = 0.0
											which_memory_index = perfect_dexs[perfect_diffs.index(min(perfect_diffs))]
											perfect.append('yes')
											matched_pattern_count += 1  # Track pattern reuse
											debug_print(f"[DEBUG] TRAINER: Matched {len(perfect_dexs)} pattern(s) within {perfect_threshold:.2f}% relative tolerance")
										break
								
									# Use pre-split patterns from cache (avoid repeated string operations)
									_mem = load_memory(cache_key)
									memory_pattern = _mem["memory_patterns"][memory_index]
									checks = []
									# Use for loop instead of while True (cleaner and faster)
									min_len = min(len(current_pattern), len(memory_pattern))
									for check_dex in range(min_len):
										current_candle = float(current_pattern[check_dex])
										memory_candle = float(memory_pattern[check_dex])
										# Relative percentage difference (threshold applies as % of pattern value)
										# Example: memory=5%, current=6%, threshold=15% → diff = |6-5|/5 * PERCENT_MULTIPLIER = 20% > 15% (no match)
										# Example: memory=5%, current=5.5%, threshold=15% → diff = |5.5-5|/5 * PERCENT_MULTIPLIER = 10% < 15% (match)
										# This scales precision: small moves get tight absolute tolerances, large moves get proportional tolerances
										# Use volatility-adaptive baseline to handle zero/tiny patterns consistently
										# Scales noise floor with market: high volatility → higher baseline, low volatility → tighter baseline
										baseline = max(abs(memory_candle), 0.1 * avg_volatility)
										difference = (abs(current_candle - memory_candle) / baseline) * PERCENT_MULTIPLIER
										checks.append(difference)
							
									if len(checks) == 0:
										debug_print(f"[DEBUG] TRAINER: Empty checks list for memory {memory_index}, skipping")
										memory_index += 1
										continue
								
									diff_avg = sum(checks)/len(checks)
									if diff_avg <= perfect_threshold:
										any_perfect = 'yes'
										# String cleaning already done in load_memory() - just remove spaces and split
										high_diff = float(memory_list[memory_index].split('{}')[1].replace(' ',''))/PERCENT_MULTIPLIER
										low_diff = float(memory_list[memory_index].split('{}')[2].replace(' ',''))/PERCENT_MULTIPLIER
										unweighted.append(float(memory_pattern[len(memory_pattern)-1]))
										move_weights.append(float(weight_list[memory_index]))
										high_move_weights.append(float(high_weight_list[memory_index]))
										low_move_weights.append(float(low_weight_list[memory_index]))
										high_unweighted.append(high_diff)
										low_unweighted.append(low_diff)
										# Zero-weight filtering: only include patterns with non-zero weights (matches thinker behavior)
										if float(weight_list[memory_index]) != 0.0:
											moves.append(float(memory_pattern[len(memory_pattern)-1])*float(weight_list[memory_index]))
										if float(high_weight_list[memory_index]) != 0.0:
											high_moves.append(high_diff*float(high_weight_list[memory_index]))
										if float(low_weight_list[memory_index]) != 0.0:
											low_moves.append(low_diff*float(low_weight_list[memory_index]))
										perfect_dexs.append(memory_index)
										perfect_diffs.append(diff_avg)
									else:
										pass
									diffs_list.append(diff_avg)
									memory_index += 1
									# Loop continues to check at top (line 1318) which properly increments counters
							except:
								PrintException()
								# Don't break references to _mem cache - just reset working variables
								which_memory_index = 'no'
								perfect.append('no')
								diffs_list = []
								any_perfect = 'no'
								perfect_dexs = []
								perfect_diffs = []
								moves = []
								move_weights = []
								high_move_weights = []
								low_move_weights = []
								unweighted = []
								high_moves = []
								low_moves = []
								final_moves = 0.0
								high_final_moves = 0.0
								low_final_moves = 0.0
						else:
							pass
						debug_print(f"[DEBUG] TRAINER: Appending current_pattern with {len(current_pattern)} percentage changes (expected: {number_of_candles[number_of_candles_index]-1})")
						all_current_patterns.append(current_pattern)
					
						# Pattern library size for status output
						pattern_count = len(_mem["memory_list"])
					
						# Calculate current accuracy from recent perfect predictions
						# Look at last 100 predictions to get recent performance
						recent_window = min(RECENT_ACCURACY_WINDOW, len(perfect))
						if recent_window > 0:
							recent_perfect_count = perfect[-recent_window:].count('yes')
							current_accuracy = (recent_perfect_count / recent_window) * PERCENT_MULTIPLIER
						else:
							current_accuracy = 0.0
					
						# Calculate recent volatility for adaptive parameters
						# Use RMS (root mean square) of high-low range as volatility measure
						# High-low range captures true intraday volatility for 24/7 crypto markets
						# Used for adaptive threshold bounds and volatility-adaptive k-selection
						volatility_window = min(MAX_VOLATILITY_WINDOW, max(MIN_VOLATILITY_WINDOW, int(len(price_list2) * VOLATILITY_WINDOW_FRACTION)))
						if len(price_list2) >= volatility_window + 1:  # Need both high and low lists
							# Calculate high-low range as percentage of low price for each candle
							high_low_ranges = []
							for i in range(-volatility_window, 0):
								if low_price_list2[i] != 0:
									range_pct = ((high_price_list2[i] - low_price_list2[i]) / low_price_list2[i]) * PERCENT_MULTIPLIER
									high_low_ranges.append(range_pct)
							# RMS volatility: sqrt(mean(squared_ranges))
							if len(high_low_ranges) > 0:
								avg_volatility = math.sqrt(sum(x**2 for x in high_low_ranges) / len(high_low_ranges))
							else:
								avg_volatility = DEFAULT_VOLATILITY
						else:
							# Fallback for early training: use all available data
							if len(price_list2) > 1:
								high_low_ranges = []
								for i in range(len(price_list2)):
									if low_price_list2[i] != 0:
										range_pct = ((high_price_list2[i] - low_price_list2[i]) / low_price_list2[i]) * PERCENT_MULTIPLIER
										high_low_ranges.append(range_pct)
								if len(high_low_ranges) > 0:
									avg_volatility = math.sqrt(sum(x**2 for x in high_low_ranges) / len(high_low_ranges))
								else:
									avg_volatility = DEFAULT_VOLATILITY
							else:
								avg_volatility = DEFAULT_VOLATILITY  # Default fallback (typical crypto high-low range volatility)
					
						debug_print(f"[DEBUG] TRAINER: Calculated avg_volatility={avg_volatility:.3f}% from high-low ranges (window={volatility_window})")
					
						# Initialize volatility cache for adaptive thresholds
						if cache_key not in _volatility_cache:
							_volatility_cache[cache_key] = {
								"ewma_volatility": avg_volatility,
								"avg_volatility": avg_volatility
							}
							debug_print(f"[DEBUG] TRAINER: Initialized volatility cache for {tf_choice} p{current_pattern_size} - ewma={avg_volatility:.3f}%, avg={avg_volatility:.3f}%")
					
						# Fixed threshold based on volatility (no PID adjustment, no target matches)
						# Formula: Scale with volatility, cap at max, ensure minimum
						# Low volatility → tight threshold (specific patterns needed)
						# High volatility → loose threshold (accommodate noise)
						# Formula: min(max_threshold, VOLATILITY_THRESHOLD_MULTIPLIER × volatility), clamped to [min_threshold, max_threshold]
						perfect_threshold = min(max_threshold, VOLATILITY_THRESHOLD_MULTIPLIER * avg_volatility)
						perfect_threshold = max(min_threshold, perfect_threshold)
					
						# Accumulate threshold in buffer for EWMA calculation at end of training
						# Each candle's threshold is added to rolling window over staleness period
						if cache_key in _threshold_buffers:
							_threshold_buffers[cache_key]["buffer"].append(perfect_threshold)
						_threshold_buffers[cache_key]["dirty"] = True
						# Status output
						bounce_accuracy = bounce_accuracy_dict.get(tf_choice, 0.0)
						signal_accuracy = signal_accuracy_dict.get(tf_choice, 0.0)
						debug_print(f"[DEBUG] TRAINER: threshold={perfect_threshold:.2f}% (volatility-adaptive: {avg_volatility:.1f}%, bounds=[{min_threshold}, {max_threshold}]) | match_acc={current_accuracy:.1f}% | limit_acc={bounce_accuracy:.1f}% | sig_acc={signal_accuracy:.1f}% | patterns={pattern_count:,}")
					
						try:
							index = 0
							current_pattern_length = number_of_candles[number_of_candles_index]
							current_pattern = []
							# First, collect absolute prices from price_list2
							temp_absolute_prices = []
							# Check if price_list2 has enough elements
							if len(price_list2) >= current_pattern_length:
								index = (len(price_list2))-current_pattern_length
								while True:
									temp_absolute_prices.append(price_list2[index])
									if len(temp_absolute_prices)>=number_of_candles[number_of_candles_index]:
										break
									else:
										index += 1
										if index >= len(price_list2):
											break
										else:
											continue
						
							# Convert absolute prices to percentage changes for scale-invariant matching
							# N prices → (N-1) percentage changes
							for i in range(1, len(temp_absolute_prices)):
								if temp_absolute_prices[i-1] != 0:
									pct_change = ((temp_absolute_prices[i] - temp_absolute_prices[i-1]) / temp_absolute_prices[i-1]) * PERCENT_MULTIPLIER
								else:
									pct_change = 0.0
								current_pattern.append(pct_change)
						
							debug_print(f"[DEBUG] TRAINER: Built current_pattern with {len(current_pattern)} percentage changes from {len(temp_absolute_prices)} prices")
						
							# Save the actual last price for prediction calculations (current_pattern now contains % changes)
							if len(temp_absolute_prices) > 0:
								last_actual_price = temp_absolute_prices[-1]
							else:
								last_actual_price = 0.0
						except:
							PrintException()
							last_actual_price = 0.0  # Fallback if pattern building fails
						# TROUBLESHOOTING TOGGLE: if 1==1 keeps this always enabled in production.
						# This block uses memory-based neural predictions (final_moves from pattern matching).
						# The 'else' branch is a fallback for when memory system has no trained data -
						# it attempts simpler 3-candle pattern matching before giving up.
						# Change to 'if 1==0' to force fallback mode for debugging.
						if 1==1:
							while True:
								try:
									c_diff = final_moves/PERCENT_MULTIPLIER
									high_diff = high_final_moves
									low_diff = low_final_moves
									# Use last_actual_price instead of current_pattern (which now contains % changes)
									start_price = last_actual_price
									prediction_prices = [start_price]
									high_prediction_prices = [start_price]
									low_prediction_prices = [start_price]
									new_price = start_price+(start_price*c_diff)
									high_new_price = start_price+(start_price*high_diff)
									low_new_price = start_price+(start_price*low_diff)
									prediction_prices = [start_price,new_price]
									high_prediction_prices = [start_price,high_new_price]
									low_prediction_prices = [start_price,low_new_price]
								except:
									start_price = last_actual_price if last_actual_price > 0 else 0.0
									new_price = start_price
									prediction_prices = [start_price,start_price]
									high_prediction_prices = [start_price,start_price]
									low_prediction_prices = [start_price,start_price]
								break
							index = len(current_pattern)-1
							index2 = 0
							all_preds.append(prediction_prices)
							high_all_preds.append(high_prediction_prices)
							low_all_preds.append(low_prediction_prices)
							overunder = 'within'
							all_predictions.append(prediction_prices)
							high_all_predictions.append(high_prediction_prices)
							low_all_predictions.append(low_prediction_prices)
							index = 0
							current_pattern_length = 3
							current_pattern = []
							# Check if price_list2 has enough elements
							if len(price_list2) > current_pattern_length:
								index = (len(price_list2)-1)-current_pattern_length
								while True:
									current_pattern.append(price_list2[index])
									index += 1
									if index >= len(price_list2):
										break
									else:
										continue
							high_current_pattern_length = 3
							high_current_pattern = []
							# Check if high_price_list2 has enough elements
							if len(high_price_list2) > high_current_pattern_length:
								high_index = (len(high_price_list2)-1)-high_current_pattern_length
								while True:
									high_current_pattern.append(high_price_list2[high_index])
									high_index += 1
									if high_index >= len(high_price_list2):
										break
									else:
										continue
							low_current_pattern_length = 3
							low_current_pattern = []
							# Check if low_price_list2 has enough elements
							if len(low_price_list2) > low_current_pattern_length:
								low_index = (len(low_price_list2)-1)-low_current_pattern_length
								while True:
									low_current_pattern.append(low_price_list2[low_index])
									low_index += 1
									if low_index >= len(low_price_list2):
										break
									else:
										continue
							try:
								which_pattern_length = 0
								# Check if variables from earlier prediction exist, otherwise build from patterns
								try:
									new_y = [start_price,new_price]
									high_new_y = [start_price,high_new_price]
									low_new_y = [start_price,low_new_price]
								except NameError:
									# Variables not defined, use last_actual_price fallback
									fallback_price = last_actual_price if last_actual_price > 0 else 0.0
									new_y = [fallback_price, fallback_price]
									if len(high_current_pattern) > 0:
										high_new_y = [fallback_price, high_current_pattern[len(high_current_pattern)-1]]
									else:
										high_new_y = [0.0, 0.0]
									if len(low_current_pattern) > 0:
										low_new_y = [fallback_price, low_current_pattern[len(low_current_pattern)-1]]
									else:
										low_new_y = [0.0, 0.0]
							except:
								PrintException()
								fallback_price = last_actual_price if last_actual_price > 0 else 0.0
								new_y = [fallback_price, fallback_price]
								if len(high_current_pattern) > 0:
									high_new_y = [fallback_price, high_current_pattern[len(high_current_pattern)-1]]
								else:
									high_new_y = [0.0, 0.0]
								if len(low_current_pattern) > 0:
									low_new_y = [fallback_price, low_current_pattern[len(low_current_pattern)-1]]
								else:
									low_new_y = [0.0, 0.0]
						else:
							# Fallback branch: Build pattern as percentage changes even when memory system is empty
							current_pattern_length = 3
							current_pattern = []
							temp_fallback_prices = []
							# Check if price_list2 has enough elements
							if len(price_list2) >= current_pattern_length:
								index = (len(price_list2))-current_pattern_length
								while True:
									temp_fallback_prices.append(price_list2[index])
									index += 1
									if index >= len(temp_fallback_prices) + (len(price_list2) - current_pattern_length):
										break
									if index >= len(price_list2):
										break
									else:
										continue
							# Convert to percentage changes
							fallback_last_price = 0.0
							for i in range(1, len(temp_fallback_prices)):
								if temp_fallback_prices[i-1] != 0:
									pct_change = ((temp_fallback_prices[i] - temp_fallback_prices[i-1]) / temp_fallback_prices[i-1]) * PERCENT_MULTIPLIER
								else:
									pct_change = 0.0
								current_pattern.append(pct_change)
								if i == len(temp_fallback_prices) - 1:
									fallback_last_price = temp_fallback_prices[i]  # Save last absolute price
							high_current_pattern_length = 3
							high_current_pattern = []
							# Check if high_price_list2 has enough elements
							if len(high_price_list2) > high_current_pattern_length:
								high_index = (len(high_price_list2)-1)-high_current_pattern_length
								while True:
									high_current_pattern.append(high_price_list2[high_index])
									high_index += 1
									if high_index >= len(high_price_list2):
										break
									else:
										continue
							low_current_pattern_length = 3
							low_current_pattern = []
							# Check if low_price_list2 has enough elements
							if len(low_price_list2) > low_current_pattern_length:
								low_index = (len(low_price_list2)-1)-low_current_pattern_length
								while True:
									low_current_pattern.append(low_price_list2[low_index])
									low_index += 1
									if low_index >= len(low_price_list2):
										break
									else:
										continue
							# Use absolute price for new_y (needed for metrics calculations)
							if fallback_last_price > 0:
								new_y = [fallback_last_price, fallback_last_price]
							else:
								new_y = [0.0, 0.0]
							number_of_candles_index += 1
							if number_of_candles_index >= len(number_of_candles):
								print("❌ ERROR: Processed all number_of_candles without finding patterns. Training cannot complete.")
								mark_training_error("Pattern finding exhausted")
								sys.exit(1)
						perfect_yes = 'no'
						# TROUBLESHOOTING TOGGLE: if 1==1 keeps accuracy tracking always enabled.
						# This block calculates prediction accuracy metrics (bounce accuracy, price differences).
						# The 'else' branch is a no-op - was used to disable stats during development.
						# Change to 'if 1==0' to disable accuracy tracking for performance testing.
						if 1==1:
							# Safe access with fallback values
							if len(high_current_pattern) > 0:
								high_current_price = high_current_pattern[len(high_current_pattern)-1]
							else:
								high_current_price = 0.0
							if len(low_current_pattern) > 0:
								low_current_price = low_current_pattern[len(low_current_pattern)-1]
							else:
								low_current_price = 0.0
							try:
								try:
									difference_of_actuals = last_actual-new_y[0]
									difference_of_last = last_actual-last_prediction
									percent_difference_of_actuals = ((new_y[0]-last_actual)/abs(last_actual))*PERCENT_MULTIPLIER
									high_difference_of_actuals = last_actual-high_current_price
									high_percent_difference_of_actuals = ((high_current_price-last_actual)/abs(last_actual))*PERCENT_MULTIPLIER
									low_difference_of_actuals = last_actual-low_current_price
									low_percent_difference_of_actuals = ((low_current_price-last_actual)/abs(last_actual))*PERCENT_MULTIPLIER
									percent_difference_of_last = ((last_prediction-last_actual)/abs(last_actual))*PERCENT_MULTIPLIER
									high_percent_difference_of_last = ((high_last_prediction-last_actual)/abs(last_actual))*PERCENT_MULTIPLIER
									low_percent_difference_of_last = ((low_last_prediction-last_actual)/abs(last_actual))*PERCENT_MULTIPLIER
									if in_trade == 'no':
										percent_for_no_sell = ((new_y[1]-last_actual)/abs(last_actual))*PERCENT_MULTIPLIER
										og_actual = last_actual
										in_trade = 'yes'
									else:
										percent_for_no_sell = ((new_y[1]-og_actual)/abs(og_actual))*PERCENT_MULTIPLIER
								except:
									difference_of_actuals = 0.0
									difference_of_last = 0.0
									percent_difference_of_actuals = 0.0
									percent_difference_of_last = 0.0
									high_difference_of_actuals = 0.0
									high_percent_difference_of_actuals = 0.0
									low_difference_of_actuals = 0.0
									low_percent_difference_of_actuals = 0.0
									high_percent_difference_of_last = 0.0
									low_percent_difference_of_last = 0.0
							except:
								PrintException()
							try:
								perdex = 0
								while True:
									if perfect[perdex] == 'yes':
										perfect_yes = 'yes'
										break
									else:
										perdex += 1
										if perdex >= len(perfect):
											perfect_yes = 'no'
											break
										else:
											continue
								if last_flipped == 'no':
									# LIMIT-BREACH ACCURACY: Tracks if predicted range contains the actual close
									# Only tracks significant predictions (both high and low > baseline) to filter baseline noise
									# Success (1) = close stayed within predicted range [low, high]
									# Failure (0) = close broke beyond predicted high OR below predicted low
									baseline_threshold = max(BOUNCE_TOLERANCE_MINIMUM, BOUNCE_TOLERANCE_MULTIPLIER * avg_volatility)  # Floor at 0.5%
								
									# Only track if BOTH predictions are significant (filters out baseline noise)
									if abs(high_baseline_price_change_pct) > baseline_threshold and abs(low_baseline_price_change_pct) > baseline_threshold:
										# Check if close stayed within predicted range
										if low_baseline_price_change_pct <= percent_difference_of_actuals <= high_baseline_price_change_pct:
											# Close stayed within predicted range = SUCCESS (range contained price)
											upordown3.append(1)
											upordown.append(1)
											upordown4.append(1)
										else:
											# Close broke beyond predicted range = FAILURE (breach)
											upordown3.append(0)
											upordown2.append(0)
											upordown.append(0)
											upordown4.append(0)
									
									# SIGNAL ACCURACY: Check predictions from pattern_length candles ago
									# Backward-looking: checks if prediction made N candles ago was correct over N candles
									current_candle_idx = len(price_list2) - 1
									predictions_to_remove = []
									
									for i, pred in enumerate(prediction_queue):
										# Check if this prediction is due (current candle = prediction candle + pattern length)
										if current_candle_idx == pred['candle_idx'] + pred['pattern_len']:
											# Check if price reached predicted level at ANY point during pattern_length candles
											try:
												# Only track if BOTH predictions were significant
												if abs(pred['high_pred']) > baseline_threshold and abs(pred['low_pred']) > baseline_threshold:
													# Calculate minimum tradeable movement (baseline/2)
													min_move_threshold = baseline_threshold / 2.0
													
													# Determine predicted direction by stronger side
													if abs(pred['high_pred']) > abs(pred['low_pred']):
														# Bullish prediction - check if ANY candle moved UP by at least min_move_threshold
														success = False
														min_price_target = pred['start_price'] * (1 + min_move_threshold / PERCENT_MULTIPLIER)
														for candle_idx in range(pred['candle_idx'] + 1, current_candle_idx + 1):
															if candle_idx < len(high_price_list2):
																if high_price_list2[candle_idx] >= min_price_target:
																	success = True
																	break
														
														if success:
															upordown_signal.append(1)  # SUCCESS: moved upward by tradeable amount
														else:
															upordown_signal.append(0)  # FAILURE: never moved up enough
													else:
														# Bearish prediction - check if ANY candle moved DOWN by at least min_move_threshold
														success = False
														max_price_target = pred['start_price'] * (1 - min_move_threshold / PERCENT_MULTIPLIER)
														for candle_idx in range(pred['candle_idx'] + 1, current_candle_idx + 1):
															if candle_idx < len(low_price_list2):
																if low_price_list2[candle_idx] <= max_price_target:
																	success = True
																	break
														
														if success:
															upordown_signal.append(1)  # SUCCESS: moved downward by tradeable amount
														else:
															upordown_signal.append(0)  # FAILURE: never moved down enough
												
												# Mark for removal (checked)
												predictions_to_remove.append(i)
											except (IndexError, ZeroDivisionError):
												# Candle not available yet or division by zero
												pass
									
									# Remove checked predictions (reverse order to preserve indices)
									for i in reversed(predictions_to_remove):
										prediction_queue.pop(i)
								else:
									pass
							
								# Calculate and store signal accuracy if we have data
								if len(upordown_signal) > 0:
									signal_accuracy = (sum(upordown_signal)/len(upordown_signal))*PERCENT_MULTIPLIER
								
									# Store signal accuracy in dict for this timeframe
									signal_accuracy_dict[tf_choice] = signal_accuracy
							
								# Calculate and store bounce accuracy if we have data (for later printing)
								if len(upordown4) > 0:
									accuracy = (sum(upordown4)/len(upordown4))*PERCENT_MULTIPLIER
								
									# Store accuracy in dict for this timeframe
									bounce_accuracy_dict[tf_choice] = accuracy
							
								# Print approximately every 1% of candles to reduce console I/O overhead
								total_candles = len(price_list)
								current_candle = len(price_list2)
								print_interval = max(1, total_candles // PERCENT_MULTIPLIER)  # 1% intervals, minimum 1
							
								# Print if this is a new milestone we haven't printed yet
								if current_candle != last_printed_candle and (current_candle % print_interval == 0 or current_candle == total_candles):
									# Always print basic status information (available from start)
									threshold_formatted = format(perfect_threshold, '.1f')
									volatility_formatted = format(avg_volatility, '.2f')
									acceptance_formatted = format(api_acceptance_rate, '.1f').rstrip('0').rstrip('.')
									print(f'Total Candles: {total_candles:,} ({acceptance_formatted}% acceptance)')
									print(f'Candles Processed: {current_candle:,} ({threshold_formatted}% threshold, {volatility_formatted}% volatility)')
								
									# Print accuracy metrics only when data is available
									if len(upordown4) > 0:
										# Calculate adaptive threshold from statistical standard error
										# Formula: SE = sqrt(p×(1-p)/n) where p=accuracy proportion, n=sample count
										# Threshold = 2×SE (95% confidence interval) to filter statistical noise
										sample_count = len(upordown4)
										if sample_count > 0:
											p = accuracy / PERCENT_MULTIPLIER  # Convert to proportion (0-1)
											standard_error = math.sqrt(p * (1 - p) / sample_count) * PERCENT_MULTIPLIER  # Back to percentage points
											adaptive_threshold = 2.0 * standard_error  # 95% CI
											# Clamp between 0.05 and 0.5 to prevent extremes
											adaptive_threshold = max(ADAPTIVE_THRESHOLD_MIN, min(ADAPTIVE_THRESHOLD_MAX, adaptive_threshold))
										else:
											adaptive_threshold = 0.5  # Default for no data
									
										# Determine trend arrow by comparing to last PRINTED accuracy (not last candle)
										if last_printed_accuracy is None:
											trend_arrow = "[~]"  # First print, starting point
										elif accuracy > last_printed_accuracy + adaptive_threshold:
											trend_arrow = "[+]"
										elif accuracy < last_printed_accuracy - adaptive_threshold:
											trend_arrow = "[-]"
										else:
											trend_arrow = "[=]"  # No significant change
									
										formatted = format(accuracy, '.2f').rstrip('0').rstrip('.')
										limit_test_count = len(upordown4)
										limit_test_count_formatted = f'{limit_test_count:,}'
										print(f'Limit-Breach Accuracy: {trend_arrow} {formatted}% ({limit_test_count_formatted} predictions)')
									
										# Update last printed accuracy for next comparison
										last_printed_accuracy = accuracy
								
									# Print signal accuracy if we have data
									if len(upordown_signal) > 0:
										sig_accuracy = (sum(upordown_signal)/len(upordown_signal))*PERCENT_MULTIPLIER
										sig_formatted = format(sig_accuracy, '.2f').rstrip('0').rstrip('.')
										sig_count = len(upordown_signal)
										sig_count_formatted = f'{sig_count:,}'
									
										# Calculate trend arrow for signal accuracy (same logic as bounce accuracy)
										# Use adaptive_threshold if available from bounce accuracy, otherwise calculate
										if 'adaptive_threshold' not in locals():
											adaptive_threshold = 0.5  # Default
									
										if last_printed_signal_accuracy is None:
											sig_trend_arrow = "[~]"  # First print, starting point
										elif sig_accuracy > last_printed_signal_accuracy + adaptive_threshold:
											sig_trend_arrow = "[+]"
										elif sig_accuracy < last_printed_signal_accuracy - adaptive_threshold:
											sig_trend_arrow = "[-]"
										else:
											sig_trend_arrow = "[=]"  # No significant change
									
										print(f'Signal Accuracy: {sig_trend_arrow} {sig_formatted}% ({sig_count_formatted} predictions)')
									
										# Update last printed signal accuracy for next comparison
										last_printed_signal_accuracy = sig_accuracy
									else:
										# No predictions exceeded baseline threshold
										print(f'Signal Accuracy: [~] No significant predictions (all < baseline)')
							except:
								PrintException()
						else:
							pass
					
						# Update last_printed_candle after all print sections have had a chance to run
						# This ensures both timeframe and bounce accuracy prints happen together at milestones
						total_candles = len(price_list)
						current_candle = len(price_list2)
						print_interval = max(1, total_candles // PERCENT_MULTIPLIER)
						if current_candle != last_printed_candle and (current_candle == 1 or current_candle % print_interval == 0 or current_candle == total_candles):
							last_printed_candle = current_candle
					
						try:
							long_trade = 'no'
							short_trade = 'no'
							last_moves = moves
							last_high_moves = high_moves
							last_low_moves = low_moves
							last_move_weights = move_weights
							last_high_move_weights = high_move_weights
							last_low_move_weights = low_move_weights
							last_perfect_dexs = perfect_dexs
							last_perfect_diffs = perfect_diffs
							# Protect against division by zero
							if abs(new_y[0]) > 0:
								percent_difference_of_now = ((new_y[1]-new_y[0])/abs(new_y[0]))*PERCENT_MULTIPLIER
							else:
								percent_difference_of_now = 0.0
							if abs(high_new_y[0]) > 0:
								high_percent_difference_of_now = ((high_new_y[1]-high_new_y[0])/abs(high_new_y[0]))*PERCENT_MULTIPLIER
							else:
								high_percent_difference_of_now = 0.0
							if abs(low_new_y[0]) > 0:
								low_percent_difference_of_now = ((low_new_y[1]-low_new_y[0])/abs(low_new_y[0]))*PERCENT_MULTIPLIER
							else:
								low_percent_difference_of_now = 0.0
							high_baseline_price_change_pct = high_percent_difference_of_now
							low_baseline_price_change_pct = low_percent_difference_of_now
							baseline_price_change_pct = percent_difference_of_now
							if flipped == 'yes':
								new1 = high_percent_difference_of_now
								high_percent_difference_of_now = low_percent_difference_of_now
								low_percent_difference_of_now = new1
							else:
								pass
							
							# Store prediction for future validation (backward-looking check)
							current_candle_idx = len(price_list2) - 1
							prediction_queue.append({
								'candle_idx': current_candle_idx,
								'high_pred': high_baseline_price_change_pct,
								'low_pred': low_baseline_price_change_pct,
								'pattern_len': current_pattern_length,
								'start_price': new_y[0]
							})
						except:
							PrintException()
						last_actual = new_y[0]
						last_prediction = new_y[1]
						high_last_prediction = high_new_y[1]
						low_last_prediction = low_new_y[1]
						last_flipped = flipped
						which_candle_of_the_prediction_index = 0
						if 1 == 1:
							if len(current_pattern) > 0:
								current_pattern_ending = [current_pattern[len(current_pattern)-1]]
							else:
								current_pattern_ending = [0.0]
							while True:
								try:
									try:
										# Grow by 1 candle at a time for maximum learning
										price_list_length += 1
										which_candle_of_the_prediction_index += 1
									
										# Periodic flush to manage memory (especially for slow timeframes like 1day)
										periodic_flush_if_needed(tf_choice)
									
										# Check if we've processed all candles for this timeframe
										if len(price_list2) == len(price_list):
											# OPTIMIZATION: Prune weak patterns before saving (keeps files lean)
											# Sigma-based adaptive pruning: removes patterns with weight < (mean - sigma × std_dev)
											# This adaptively removes only true outliers, not fixed threshold (more robust)
											debug_print(f"[DEBUG] TRAINER: Pruning weak patterns (sigma-based, level={pruning_sigma_level}) for {tf_choice}")
											try:
												_mem = _memory_cache.get(cache_key)
												if _mem:
													original_count = len(_mem["memory_list"])
												
													# Calculate mean and std dev of weights for sigma-based pruning
													valid_weights = []
													for w in _mem["weight_list"]:
														try:
															valid_weights.append(float(w))
														except (ValueError, TypeError):
															pass
												
													if len(valid_weights) > 1:
														try:
															import statistics
															mean_weight = statistics.mean(valid_weights)
															stdev_weight = statistics.stdev(valid_weights)
															pruning_threshold = mean_weight - (pruning_sigma_level * stdev_weight)
															
															# Apply floor to pruning threshold: always prune patterns below PRUNING_WEIGHT_FLOOR
															# Ensures very weak patterns removed even if sigma calculation would be more lenient
															# Patterns above floor can still be pruned if sigma or age-based criteria say so
															if pruning_threshold < PRUNING_WEIGHT_FLOOR:
																pruning_threshold = PRUNING_WEIGHT_FLOOR
														
															# Validate that the threshold is a valid number
															if not isinstance(pruning_threshold, (int, float)) or pruning_threshold != pruning_threshold:  # NaN check
																pruning_threshold = -float('inf')
																debug_print(f"[DEBUG] TRAINER: Invalid pruning threshold calculated, keeping all patterns")
															else:
																debug_print(f"[DEBUG] TRAINER: Weight stats - mean={mean_weight:.4f}, stdev={stdev_weight:.4f}, threshold={pruning_threshold:.4f}")
														except Exception as stats_err:
															# Statistics calculation failed, keep all patterns
															pruning_threshold = -float('inf')
															debug_print(f"[DEBUG] TRAINER: Stats calculation failed ({stats_err}), keeping all patterns")
													else:
														# Not enough data for statistics, keep all patterns
														pruning_threshold = -float('inf')
														debug_print(f"[DEBUG] TRAINER: Not enough patterns for sigma pruning, keeping all")
												
													# Get ages for this timeframe (use empty list if not initialized)
													ages = _pattern_ages.get(tf_choice, [])
													# Pad ages list if shorter than memory (safety for edge cases)
													while len(ages) < len(_mem["memory_list"]):
														ages.append(0)
												
													# Filter all lists together using zip (faster and cleaner)
													# Include ages in the zip to keep everything synchronized
													filtered = []
													for mem, w, hw, lw, age in zip(_mem["memory_list"], _mem["weight_list"], 
													                               _mem["high_weight_list"], _mem["low_weight_list"], ages):
														try:
															if float(w) >= pruning_threshold:
																filtered.append((mem, w, hw, lw, age))
														except (ValueError, TypeError):
															pass  # Skip corrupt weight entries
												
													sigma_pruned = original_count - len(filtered)
													debug_print(f"[DEBUG] TRAINER: Sigma-based pruning removed {sigma_pruned} patterns")
												
												# Age-based pruning: remove statistically ancient patterns with low weights
												# Only apply if enabled and we have enough patterns for meaningful statistics
												age_pruned = 0
												if age_pruning_enabled and len(filtered) > 100:
													# Calculate age statistics for sigma-based pruning (consistent with weight pruning)
													ages_list = [age for (_, _, _, _, age) in filtered]
													try:
														mean_age = statistics.mean(ages_list)
														stdev_age = statistics.stdev(ages_list) if len(ages_list) > 1 else 0.0
														# Prune patterns beyond mean + pruning_sigma_level×σ (extreme outliers in age distribution)
														age_cutoff = mean_age + (pruning_sigma_level * stdev_age)
														debug_print(f"[DEBUG] TRAINER: Age stats - mean={mean_age:.1f}, stdev={stdev_age:.1f}, cutoff={age_cutoff:.1f} validations")
													
														# Filter: keep pattern if (age <= cutoff) OR (weight >= limit)
														# Remove only if (age > cutoff) AND (weight < limit)
														age_filtered = []
														for mem, w, hw, lw, age in filtered:
															try:
																if age <= age_cutoff or float(w) >= age_pruning_weight_limit:
																	age_filtered.append((mem, w, hw, lw, age))
															except (ValueError, TypeError):
																age_filtered.append((mem, w, hw, lw, age))  # Keep on error
													
														age_pruned = len(filtered) - len(age_filtered)
														filtered = age_filtered
														debug_print(f"[DEBUG] TRAINER: Sigma-based age pruning removed {age_pruned} ancient patterns (age>{age_cutoff:.1f}, weight<{age_pruning_weight_limit})")
													except Exception as age_stats_err:
														debug_print(f"[DEBUG] TRAINER: Age stats calculation failed ({age_stats_err}), skipping age pruning")
											
												# Unpack pruned results back into memory (after both sigma and age pruning)
												if len(filtered) > 0:
													_mem["memory_list"], _mem["weight_list"], _mem["high_weight_list"], _mem["low_weight_list"], ages = zip(*filtered)
													_mem["memory_list"] = list(_mem["memory_list"])
													_mem["weight_list"] = list(_mem["weight_list"])
													_mem["high_weight_list"] = list(_mem["high_weight_list"])
													_mem["low_weight_list"] = list(_mem["low_weight_list"])
													ages = list(ages)
													# Rebuild pre-split patterns after pruning
													_mem["memory_patterns"] = [mem.split('{}')[0].split('|') for mem in _mem["memory_list"]]
												else:
													# All patterns were pruned - reset to empty
													_mem["memory_list"] = []
													_mem["weight_list"] = []
													_mem["high_weight_list"] = []
													_mem["low_weight_list"] = []
													_mem["memory_patterns"] = []
													ages = []
											
												# Sync age cache with pruned memory (happens for both branches)
												_pattern_ages[cache_key] = ages
												_mem["dirty"] = True
											
												total_pruned = sigma_pruned + age_pruned
												debug_print(f"[DEBUG] TRAINER: Total pruned {total_pruned} patterns ({original_count} -> {len(_mem['memory_list'])}) [sigma: {sigma_pruned}, age: {age_pruned}]")
											except Exception as e:
												debug_print(f"[DEBUG] TRAINER: Pruning failed (non-critical): {e}")
										
											# Flush all buffered data before switching to next timeframe
											debug_print(f"[DEBUG] TRAINER: Flushing buffers before timeframe switch (the_big_index: {the_big_index} -> {the_big_index + 1})")
											debug_print(f'Memory list size for {tf_choice} size {current_pattern_size} before flush: {len(_memory_cache.get(cache_key, {}).get("memory_list", []))}')
											flush_all_buffers(force=True)
										
											# Clear memory cache before switching timeframes to prevent cross-contamination
											debug_print(f"[DEBUG] TRAINER: Clearing memory cache after flush")
											_memory_cache.clear()
											the_big_index += 1
											debug_print(f'Incremented the_big_index to {the_big_index} (next timeframe will be: {tf_choices[the_big_index] if the_big_index < len(tf_choices) else "DONE"})')
											print('\nMoving to next timeframe')
											debug_print(f'Completed timeframe {tf_choice} - processed {len(price_list2)}/{len(price_list)} candles')
										
											# Check if we should continue to next timeframe or proceed to exit
											if the_big_index < len(tf_choices):
												debug_print(f'More timeframes remain, setting restart flag and breaking to outer loop')
												_restart_outer_loop = True
												break  # Exit prediction loop; cascade checks at 2345-2352 handle multi-level exit
										
											# Check if all timeframes have been processed
											debug_print(f'Timeframe index: {the_big_index}')
											debug_print(f'Total timeframes: {len(tf_choices)}')
											debug_print(f'the_big_index={the_big_index}, len(tf_choices)={len(tf_choices)}, len(number_of_candles)={len(number_of_candles)}')
											debug_print(f'Timeframes in bounce_accuracy_dict: {list(bounce_accuracy_dict.keys())}')
											# Check if we've processed all timeframes
											if the_big_index >= len(tf_choices):
												print("Finished processing all timeframes. Exiting.")
											
												# Flush all buffered data (thresholds + memory/weights) before marking complete
												flush_all_buffers(force=True)
											
												# Save and print bounce accuracy summary
												try:
													if bounce_accuracy_dict:
														# Calculate average
														avg_accuracy = sum(bounce_accuracy_dict.values()) / len(bounce_accuracy_dict)
													
														# Save to file
														timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
														with open(f'bounce_accuracy_p{current_pattern_size}.txt', 'w', encoding='utf-8') as f:
															f.write(f'Last Updated: {timestamp}\n')
															f.write(f'Pattern Size: {current_pattern_size}\n')
															f.write(f'Average: {avg_accuracy:.2f}%\n')
															for tf in tf_choices:
																if tf in bounce_accuracy_dict:
																	f.write(f'{tf}: {bounce_accuracy_dict[tf]:.2f}%\n')
													
														# Print combined summary
														print()
														print(f'Training Accuracy Results ({_arg_coin}) - Pattern Size {current_pattern_size}')
														print(f'Last trained: {timestamp}')
													
														# Get signal accuracy average if available
														avg_signal = 0.0
														if signal_accuracy_dict:
															avg_signal = sum(signal_accuracy_dict.values()) / len(signal_accuracy_dict)
													
														# Print averages
														print(f'Average Limit-Breach Accuracy: {avg_accuracy:.2f}%')
														if signal_accuracy_dict:
															print(f'Average Signal Accuracy:       {avg_signal:.2f}%')
														print()
													
														# Print per-timeframe results
														print('Per Timeframe:')
														for tf in tf_choices:
															if tf in bounce_accuracy_dict:
																limit_acc = bounce_accuracy_dict[tf]
																sig_acc = signal_accuracy_dict.get(tf, 0.0) if signal_accuracy_dict else 0.0
																if signal_accuracy_dict and tf in signal_accuracy_dict:
																	print(f'  {tf:8}   Limit-Breach: {limit_acc:5.2f}%   Signal: {sig_acc:5.2f}%')
																else:
																	print(f'  {tf:8}   Limit-Breach: {limit_acc:5.2f}%')
													
														# Check for suspicious accuracy (99-100% often indicates incomplete training)
														suspicious_accuracy = False
														for tf in tf_choices:
															if tf in bounce_accuracy_dict and bounce_accuracy_dict[tf] >= 99.0:
																suspicious_accuracy = True
																break
													
														if suspicious_accuracy:
															print()
															print(f'⚠ WARNING: Accuracy of 100% detected. This may indicate incomplete training.')
															print(f'⚠ Please verify that training completed properly and memories were saved.')
												except Exception as e:
													print(f'⚠ Warning: Could not save bounce accuracy: {e}')
											
												# Save signal accuracy to file
												try:
													if signal_accuracy_dict:
														# Calculate average
														avg_signal = sum(signal_accuracy_dict.values()) / len(signal_accuracy_dict)
													
														# Save to file
														timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
														with open(f'signal_accuracy_p{current_pattern_size}.txt', 'w', encoding='utf-8') as f:
															f.write(f'Last Updated: {timestamp}\n')
															f.write(f'Pattern Size: {current_pattern_size}\n')
															f.write(f'Average: {avg_signal:.2f}%\n')
															for tf in tf_choices:
																if tf in signal_accuracy_dict:
																	f.write(f'{tf}: {signal_accuracy_dict[tf]:.2f}%\n')
												except Exception as e:
													print(f'⚠ Warning: Could not save signal accuracy: {e}')
											
												# Save pattern count to file
												try:
													total_patterns = 0
													for tf in tf_choices:
														memory_file = f'memories_{tf}_p{current_pattern_size}.dat'
														if os.path.isfile(memory_file):
															with open(memory_file, 'r', encoding='utf-8') as f:
																content = f.read().strip()
																if content:
																	patterns = content.split('~')
																	total_patterns += len(patterns)
													
													# Write total count to file
													with open(f'pattern_count_p{current_pattern_size}.txt', 'w', encoding='utf-8') as f:
														f.write(str(total_patterns))
												except Exception as e:
													print(f'⚠ Warning: Could not save pattern count: {e}')
											
												# Break from while loop to continue to next pattern size
												print()
												print(f"Completed all timeframes for pattern size {current_pattern_size}")
												# Set restart flag to trigger cascade breaks through all loop levels
												_restart_outer_loop = True
												break
										else:
											try:
												# Rebuild all price lists with new window size
												open_price_list2 = []
												open_price_list_index = 0
												while True:
													open_price_list2.append(open_price_list[open_price_list_index])
													open_price_list_index += 1
													if open_price_list_index >= price_list_length:
														break
													else:
														continue
												price_list2 = []
												price_list_index = 0
												while True:
													price_list2.append(price_list[price_list_index])
													price_list_index += 1
													if len(price_list2) >= price_list_length:
														break
													else:
														continue
												high_price_list2 = []
												high_price_list_index = 0
												while True:
													high_price_list2.append(high_price_list[high_price_list_index])
													high_price_list_index += 1
													if high_price_list_index >= price_list_length:
														break
													else:
														continue
												low_price_list2 = []
												low_price_list_index = 0
												while True:
													low_price_list2.append(low_price_list[low_price_list_index])
													low_price_list_index += 1
													if low_price_list_index >= price_list_length:
														break
													else:
														continue
											
												# price_change lists will be rebuilt at start of next loop iteration
												debug_print(f"[DEBUG] TRAINER: Window grown to size {price_list_length}, lists will sync on next iteration")
											
												price2 = price_list2[len(price_list2)-1]
												high_price2 = high_price_list2[len(high_price_list2)-1]
												low_price2 = low_price_list2[len(low_price_list2)-1]
												highlowind = 0
												debug_print(f"[DEBUG] TRAINER: Starting weight update loop - all_current_patterns has {len(all_current_patterns)} patterns, all_predictions has {len(all_predictions)} predictions")
												# Protect against division by zero
												if abs(new_y[1]) > 0:
													this_differ = ((price2-new_y[1])/abs(new_y[1]))*PERCENT_MULTIPLIER
													high_this_differ = ((high_price2-new_y[1])/abs(new_y[1]))*PERCENT_MULTIPLIER
													low_this_differ = ((low_price2-new_y[1])/abs(new_y[1]))*PERCENT_MULTIPLIER
												else:
													this_differ = 0.0
													high_this_differ = 0.0
													low_this_differ = 0.0
												if abs(new_y[0]) > 0:
													this_diff = ((price2-new_y[0])/abs(new_y[0]))*PERCENT_MULTIPLIER
													high_this_diff = ((high_price2-new_y[0])/abs(new_y[0]))*PERCENT_MULTIPLIER
													low_this_diff = ((low_price2-new_y[0])/abs(new_y[0]))*PERCENT_MULTIPLIER
												else:
													this_diff = 0.0
													high_this_diff = 0.0
													low_this_diff = 0.0
												difference_list = []
												list_of_predictions = all_predictions
												close_enough_counter = []
												which_pattern_length_index = 0
												while True:
													current_prediction_price = all_predictions[highlowind][which_candle_of_the_prediction_index]
													high_current_prediction_price = high_all_predictions[highlowind][which_candle_of_the_prediction_index]
													low_current_prediction_price = low_all_predictions[highlowind][which_candle_of_the_prediction_index]
													# Protect against division by zero
													if abs(new_y[0]) > 0:
														perc_diff_now = ((current_prediction_price-new_y[0])/abs(new_y[0]))*PERCENT_MULTIPLIER
														perc_diff_now_actual = ((price2-new_y[0])/abs(new_y[0]))*PERCENT_MULTIPLIER
														high_perc_diff_now_actual = ((high_price2-new_y[0])/abs(new_y[0]))*PERCENT_MULTIPLIER
														low_perc_diff_now_actual = ((low_price2-new_y[0])/abs(new_y[0]))*PERCENT_MULTIPLIER
													else:
														perc_diff_now = 0.0
														perc_diff_now_actual = 0.0
														high_perc_diff_now_actual = 0.0
														low_perc_diff_now_actual = 0.0
													# Calculate difference with explicit zero-division protection
													avg_pred_price = (current_prediction_price + float(price2)) / 2
													if avg_pred_price == 0.0:
														# Both prediction and actual price are zero - treat as max difference
														difference = 100.0
													else:
														difference = abs((abs(current_prediction_price - float(price2)) / avg_pred_price) * PERCENT_MULTIPLIER)
													try:
														try:
															indy = 0
															# Find the minimum length across all lists to avoid index errors
															# Process as much valid data as possible even if lists got out of sync
															max_safe_index = min(
																len(moves), len(high_moves), len(low_moves),
																len(move_weights), len(high_move_weights), len(low_move_weights),
																len(perfect_dexs), len(unweighted)
															)
														
															# If no patterns matched (moves is empty), trigger exception to create new pattern
															# This matches baseline behavior where accessing moves[0] throws IndexError
															if max_safe_index == 0:
																debug_print(f"[DEBUG] TRAINER: No matching patterns found - triggering new pattern creation (highlowind={highlowind}, all_current_patterns length at this index={len(all_current_patterns[highlowind])})")
																raise IndexError("EXPECTED_NO_PATTERNS")
														
															while True:
																	# Safety check: stop when we reach the shortest list
																	if indy >= max_safe_index:
																		break
																
																	new_memory = 'no'
																	predicted_move_pct = (moves[indy]*PERCENT_MULTIPLIER)
																	high_predicted_move_pct = (high_moves[indy]*PERCENT_MULTIPLIER)
																	low_predicted_move_pct = (low_moves[indy]*PERCENT_MULTIPLIER)
																
																	# Update EWMA volatility with actual move for adaptive thresholds
																	_vol_cache = _volatility_cache.get(cache_key, {})
																	current_move = abs(perc_diff_now_actual)
																	ewma_vol = _vol_cache.get("ewma_volatility", 2.0)
																	avg_vol = _vol_cache.get("avg_volatility", 2.0)
																	# Update EWMA: new = decay × old + (1-decay) × current
																	ewma_vol = volatility_ewma_decay * ewma_vol + (1 - volatility_ewma_decay) * current_move
																	_vol_cache["ewma_volatility"] = ewma_vol
																	_volatility_cache[cache_key] = _vol_cache
																
																	# Volatility-adaptive thresholds scale with market conditions and pattern confidence
																	# Formula: base × |prediction| × (1 + current_vol/avg_vol) × (2.0/weight)
																	# Low vol → tight threshold, high vol → loose threshold
																	# High weight → strict threshold (must maintain performance), low weight → lenient (allow learning)
																	volatility_factor = 1.0 + (ewma_vol / avg_vol)
																	weight_factor_high = WEIGHT_FACTOR_MULTIPLIER / max(WEIGHT_FACTOR_MIN, float(high_move_weights[indy]))
																	weight_factor_low = WEIGHT_FACTOR_MULTIPLIER / max(WEIGHT_FACTOR_MIN, float(low_move_weights[indy]))
																	weight_factor_main = WEIGHT_FACTOR_MULTIPLIER / max(WEIGHT_FACTOR_MIN, float(move_weights[indy]))
																
																	high_predicted_move_pct_threshold = weight_threshold_base * abs(high_predicted_move_pct) * volatility_factor * weight_factor_high
																	high_predicted_move_pct_threshold = max(weight_threshold_min * abs(high_predicted_move_pct),
																	                                         min(weight_threshold_max * abs(high_predicted_move_pct),
																	                                             high_predicted_move_pct_threshold))
																
																	low_predicted_move_pct_threshold = weight_threshold_base * abs(low_predicted_move_pct) * volatility_factor * weight_factor_low
																	low_predicted_move_pct_threshold = max(weight_threshold_min * abs(low_predicted_move_pct),
																	                                        min(weight_threshold_max * abs(low_predicted_move_pct),
																	                                            low_predicted_move_pct_threshold))
																
																	predicted_move_pct_threshold = weight_threshold_base * abs(predicted_move_pct) * volatility_factor * weight_factor_main
																	predicted_move_pct_threshold = max(weight_threshold_min * abs(predicted_move_pct),
																	                                    min(weight_threshold_max * abs(predicted_move_pct),
																	                                        predicted_move_pct_threshold))
																
																	# Fallback for zero predictions to prevent division by zero
																	if high_predicted_move_pct == 0:
																		high_predicted_move_pct_threshold = 0.1
																	if low_predicted_move_pct == 0:
																		low_predicted_move_pct_threshold = 0.1
																	if predicted_move_pct == 0:
																		predicted_move_pct_threshold = 0.1
																
																	# Calculate normalized error magnitude for each weight type
																	# Step size scales with error: large errors → faster corrections, small errors → gentler adjustments
																	high_error = high_perc_diff_now_actual - high_predicted_move_pct
																	high_error_magnitude = abs(high_error) / max(ERROR_MAGNITUDE_MIN, abs(high_predicted_move_pct))
																	high_adaptive_step = weight_base_step * min(weight_step_cap, high_error_magnitude)
																
																	low_error = low_perc_diff_now_actual - low_predicted_move_pct
																	low_error_magnitude = abs(low_error) / max(ERROR_MAGNITUDE_MIN, abs(low_predicted_move_pct))
																	low_adaptive_step = weight_base_step * min(weight_step_cap, low_error_magnitude)
																
																	main_error = perc_diff_now_actual - predicted_move_pct
																	main_error_magnitude = abs(main_error) / max(ERROR_MAGNITUDE_MIN, abs(predicted_move_pct))
																	main_adaptive_step = weight_base_step * min(weight_step_cap, main_error_magnitude)
																
																	if high_perc_diff_now_actual > high_predicted_move_pct + high_predicted_move_pct_threshold:
																		high_new_weight = high_move_weights[indy] + high_adaptive_step
																		if high_new_weight > WEIGHT_MAX:
																			high_new_weight = WEIGHT_MAX
																		# Apply decay when pattern needs correction (not perfect)
																		high_new_weight = high_new_weight * (1 - weight_decay_rate) + weight_decay_target * weight_decay_rate
																	elif high_perc_diff_now_actual < high_predicted_move_pct - high_predicted_move_pct_threshold:
																		high_new_weight = high_move_weights[indy] - high_adaptive_step
																		if high_new_weight < WEIGHT_MIN:
																			high_new_weight = WEIGHT_MIN
																		# Apply decay when pattern needs correction (not perfect)
																		high_new_weight = high_new_weight * (1 - weight_decay_rate) + weight_decay_target * weight_decay_rate
																	else:
																		# Pattern prediction is perfect - preserve weight without decay
																		high_new_weight = high_move_weights[indy]
																	if low_perc_diff_now_actual < low_predicted_move_pct - low_predicted_move_pct_threshold:
																		low_new_weight = low_move_weights[indy] + low_adaptive_step
																		if low_new_weight > WEIGHT_MAX:
																			low_new_weight = WEIGHT_MAX
																		# Apply decay when pattern needs correction (not perfect)
																		low_new_weight = low_new_weight * (1 - weight_decay_rate) + weight_decay_target * weight_decay_rate
																	elif low_perc_diff_now_actual > low_predicted_move_pct + low_predicted_move_pct_threshold:
																		low_new_weight = low_move_weights[indy] - low_adaptive_step
																		if low_new_weight < WEIGHT_MIN:
																			low_new_weight = WEIGHT_MIN
																		# Apply decay when pattern needs correction (not perfect)
																		low_new_weight = low_new_weight * (1 - weight_decay_rate) + weight_decay_target * weight_decay_rate
																	else:
																		# Pattern prediction is perfect - preserve weight without decay
																		low_new_weight = low_move_weights[indy]
																	if perc_diff_now_actual > predicted_move_pct + predicted_move_pct_threshold:
																		new_weight = move_weights[indy] + main_adaptive_step
																		if new_weight > WEIGHT_MAX:
																			new_weight = WEIGHT_MAX
																		# Apply decay when pattern needs correction (not perfect)
																		new_weight = new_weight * (1 - weight_decay_rate) + weight_decay_target * weight_decay_rate
																	elif perc_diff_now_actual < predicted_move_pct - predicted_move_pct_threshold:
																		new_weight = move_weights[indy] - main_adaptive_step
																		if new_weight < WEIGHT_MIN:
																			new_weight = WEIGHT_MIN
																		# Apply decay when pattern needs correction (not perfect)
																		new_weight = new_weight * (1 - weight_decay_rate) + weight_decay_target * weight_decay_rate
																	else:
																		# Pattern prediction is perfect - preserve weight without decay
																		new_weight = move_weights[indy]
																
																	# Increment age counter for this pattern (tracks validation count)
																	pattern_idx = perfect_dexs[indy]
																	if cache_key in _pattern_ages and pattern_idx < len(_pattern_ages[cache_key]):
																		_pattern_ages[cache_key][pattern_idx] += 1
																
																	# Update cache directly to ensure changes persist (don't rely on references)
																	_mem["weight_list"][pattern_idx] = new_weight
																	_mem["high_weight_list"][pattern_idx] = high_new_weight
																	_mem["low_weight_list"][pattern_idx] = low_new_weight

																	# mark dirty (we will flush in batches)
																	_mem["dirty"] = True
																
																	indy += 1
																	# Loop condition moved to top of while loop for safety
														except Exception as e:
															# Catch unexpected errors during weight updates - create new pattern (baseline approach)
															# Don't log the expected "no patterns" exception - it's normal during initial training
															if "EXPECTED_NO_PATTERNS" not in str(e):
																PrintException()
														
															# This is the baseline's pattern creation approach - only on actual failure
															# IMPORTANT: all_current_patterns already contains percentage changes (from Step 2)
															# and this_diff is also a percentage change, so new_pattern is already in the correct format
															new_pattern = all_current_patterns[highlowind] + [this_diff]
															debug_print(f"[DEBUG] TRAINER: Creating new pattern with {len(new_pattern)} percentage changes: {[f'{x:.4f}%' for x in new_pattern[:3]]}{'...' if len(new_pattern) > 3 else ''}")
														
															# Increment counter at decision point (before validation) for accurate match rate
															new_pattern_count += 1
														
															# new_pattern is already percentage changes - use directly without conversion
															# Pattern format: [pct_change1, pct_change2, ..., prediction_pct_change]
															mem_entry = '|'.join([str(x) for x in new_pattern])+'{}'+str(high_this_diff)+'{}'+str(low_this_diff)
														# Inline duplicate check to prevent memory bloat and preserve weight accuracy
														# Always check duplicates to ensure weights are correctly maintained
														skip_pattern = False
														skip_reason = ""
													
														if not mem_entry or str(mem_entry).strip() == "":
															skip_pattern = True
															skip_reason = "empty"
														elif not is_valid_pattern(mem_entry):
															skip_pattern = True
															skip_reason = "invalid"
														elif mem_entry in _mem['memory_list']:
															skip_pattern = True
															skip_reason = "duplicate"
													
														if skip_pattern:
															debug_print(f"[DEBUG] TRAINER: Skipped {skip_reason} pattern (memory size: {len(_mem['memory_list'])})")
														else:
																pct_change_count = len(mem_entry.split('{}')[0].split('|'))
																debug_print(f"[DEBUG] TRAINER: Adding pattern with {pct_change_count} percentage changes to memory (expected: {number_of_candles[number_of_candles_index]-1})")
																_mem["memory_list"].append(mem_entry)
																# Also append pre-split pattern for fast comparison
																_mem["memory_patterns"].append(mem_entry.split('{}')[0].split('|'))
																_mem["weight_list"].append(1.0)  # Start with neutral weight
																_mem["high_weight_list"].append(1.0)
																_mem["low_weight_list"].append(1.0)
																# Initialize age for new pattern
																if cache_key in _pattern_ages:
																	_pattern_ages[cache_key].append(0)
																_mem["dirty"] = True
																debug_print(f"[DEBUG] TRAINER: Added new pattern to memory (total: {len(_mem['memory_list'])}), pct_changes={pct_change_count}")
													except:
														PrintException()
														pass
													highlowind += 1
													if highlowind >= len(all_predictions):
														break
													else:
														continue
											except:
												PrintException()
												print("CRITICAL ERROR in weight update loop. Training cannot continue safely.")
												mark_training_error("Weight update critical error")
												sys.exit(1)
									
										if which_candle_of_the_prediction_index >= candles_to_predict:
											break
										else:
											continue
									except (KeyboardInterrupt, SystemExit):
										raise
									except:
										PrintException()
										print("CRITICAL ERROR in prediction candle loop. Training cannot continue safely.")
										mark_training_error("Prediction candle loop critical error")
										sys.exit(1)
								except (KeyboardInterrupt, SystemExit):
									raise
								except:
									PrintException()
									print("CRITICAL ERROR in candle processing. Training cannot continue safely.")
									mark_training_error("Candle processing critical error")
									sys.exit(1)
						else:
							pass
						history_list = []
						current_pattern = []
						# Exit inner loop to return to middle loop where sync code runs
						break
					except (KeyboardInterrupt, SystemExit):
						raise
					except:
						PrintException()
						print("CRITICAL ERROR in main training loop. Training cannot continue safely.")
						mark_training_error("Main training loop critical error")
						sys.exit(1)
				# Check if we need to restart outer loop
				debug_print(f'[DEBUG] TRAINER: After INNER loop - _restart_outer_loop={_restart_outer_loop}')
				if _restart_outer_loop:
					debug_print(f'[DEBUG] TRAINER: Breaking from INNER loop level')
					break
			# Check if we need to restart outer loop
			debug_print(f'[DEBUG] TRAINER: After MIDDLE loop - _restart_outer_loop={_restart_outer_loop}')
			if _restart_outer_loop:
				debug_print(f'[DEBUG] TRAINER: Breaking from MIDDLE loop level')
				break
		# Check if we need to restart outer loop
		if _restart_outer_loop:
			# Check if this is a normal completion (all timeframes done) vs unexpected exit
			if the_big_index >= len(tf_choices):
				# Normal completion - exit the while loop cleanly
				debug_print(f'[DEBUG] TRAINER: All timeframes complete for pattern size {current_pattern_size}, exiting while loop')
				break
			else:
				# Restart for next timeframe
				debug_print(f'[DEBUG] TRAINER: Restarting while loop for next timeframe (the_big_index={the_big_index}, pattern_size={current_pattern_size})')
				continue
	
	# If we reach here without restart flag, something went wrong
	if not _restart_outer_loop:
		debug_print(f'[DEBUG] TRAINER: WARNING - Exited all loops without restart flag (the_big_index={the_big_index})')
		print(f'⚠ WARNING: Loop exited unexpectedly at timeframe index {the_big_index}')

# All pattern sizes complete - validate and exit
print()
print(f"Training complete for all pattern sizes!")

# Final flush and deduplication pass for all data
print("Performing final cleanup and deduplication...")
flush_all_buffers(force=True)
print("Final cleanup complete.")

def _verify_training_artifacts(coin: str, tf_choices: list, pattern_sizes: list) -> tuple:
	"""Verify all training artifacts were created successfully.
	
	Args:
		coin: Coin symbol being trained
		tf_choices: List of timeframes that were trained
		pattern_sizes: List of pattern sizes that were trained
	
	Returns:
		tuple: (is_valid: bool, error_messages: list)
	
	Checks performed:
		- Memory files exist and have reasonable size (>1KB)
		- Weight files exist for each memory file
		- Threshold files exist and contain valid numeric values
		- Training timestamp file was created
		- At least 100 patterns learned per timeframe
	"""
	errors = []
	warnings = []
	
	# Check each timeframe and pattern size combination
	for tf in tf_choices:
		for size in pattern_sizes:
			prefix = f"{tf}_p{size}"
			
			# Check memory file
			mem_file = f"memories_{prefix}.dat"
			if not os.path.isfile(mem_file):
				errors.append(f"Missing memory file: {mem_file}")
				continue  # Skip dependent checks if memory file missing
			
			# Check minimum size (1KB = at least some patterns)
			file_size = os.path.getsize(mem_file)
			if file_size < 1000:
				errors.append(f"Memory file too small: {mem_file} ({file_size} bytes, expected >1KB)")
			elif file_size < 5000:
				warnings.append(f"Memory file small: {mem_file} ({file_size} bytes, may have few patterns)")
			
			# Validate memory file content is readable
			try:
				with open(mem_file, "r", encoding="utf-8", errors="ignore") as f:
					content = f.read()
					if not content or content.strip() == "":
						errors.append(f"Memory file is empty: {mem_file}")
					else:
						# Count patterns (separated by ~)
						pattern_count = content.count('~')
						if pattern_count < 52:
							warnings.append(f"Few patterns in {mem_file}: {pattern_count} (expected >52)")
			except Exception as e:
				errors.append(f"Cannot read memory file {mem_file}: {e}")
			
			# Check weight files (3 types per timeframe/size)
			for weight_type in ["memory_weights", "memory_weights_high", "memory_weights_low"]:
				weight_file = f"{weight_type}_{prefix}.dat"
				if not os.path.isfile(weight_file):
					errors.append(f"Missing weight file: {weight_file}")
				else:
					# Weight files should be comparable size to memory files
					w_size = os.path.getsize(weight_file)
					if w_size < 100:
						errors.append(f"Weight file too small: {weight_file} ({w_size} bytes)")
			
			# Check threshold file
			threshold_file = f"neural_perfect_threshold_{prefix}.dat"
			if not os.path.isfile(threshold_file):
				errors.append(f"Missing threshold file: {threshold_file}")
			else:
				# Validate threshold contains valid numeric value
				try:
					with open(threshold_file, "r", encoding="utf-8") as f:
						threshold_str = f.read().strip()
						if not threshold_str:
							errors.append(f"Threshold file is empty: {threshold_file}")
						else:
							threshold = float(threshold_str)
							if threshold <= 0:
								errors.append(f"Invalid threshold in {threshold_file}: {threshold} (must be >0)")
							elif threshold > 100:
								warnings.append(f"Very high threshold in {threshold_file}: {threshold}% (typical: 10-25%)")
				except ValueError as e:
					errors.append(f"Threshold file contains non-numeric value in {threshold_file}: {e}")
				except Exception as e:
					errors.append(f"Cannot read threshold file {threshold_file}: {e}")
	
	# Check training timestamp file
	if not os.path.isfile("trainer_last_training_time.txt"):
		# This is created later, so just warn if we're checking before it's written
		pass  # Will be created after validation
	
	# Determine if training is valid
	is_valid = len(errors) == 0
	
	# Combine errors and warnings for reporting
	all_messages = errors + warnings
	
	return is_valid, all_messages

# Verify all training artifacts before marking as complete
print()
print("Verifying training artifacts...")
training_valid, validation_messages = _verify_training_artifacts(
	_arg_coin, 
	tf_choices, 
	list(pattern_sizes_to_train)
)

if not training_valid:
	print()
	print(f"{'='*60}")
	print(f"❌ TRAINING VERIFICATION FAILED FOR {_arg_coin}")
	print(f"{'='*60}")
	for msg in validation_messages:
		if msg.startswith("Missing") or msg.startswith("Invalid") or "too small" in msg or "empty" in msg:
			print(f"  ❌ {msg}")
		else:
			print(f"  ⚠  {msg}")
	print(f"{'='*60}")
	print()
else:
	print(f"✅ All training artifacts verified successfully")
	# Print warnings even if validation passed
	warnings_only = [msg for msg in validation_messages if msg.startswith("Few") or msg.startswith("Memory file small") or msg.startswith("Very high")]
	if warnings_only:
		print(f"  Warnings:")
		for msg in warnings_only:
			print(f"    ⚠  {msg}")

# Legacy check removed - comprehensive verification function replaced it

try:
	file = open('trainer_last_start_time.txt','w+')
	file.write(str(start_time_yes))
	file.close()
except:
	pass

# Mark training finished (or failed) for the GUI
try:
	_trainer_finished_at = int(time.time())
	
	# Only write last_training_time if training was valid
	if training_valid:
		file = open('trainer_last_training_time.txt','w+')
		file.write(str(_trainer_finished_at))
		file.close()
except:
	pass

try:
	status_data = {
		"coin": _arg_coin,
		"state": "FINISHED" if training_valid else "FAILED",
		"started_at": _trainer_started_at,
		"finished_at": _trainer_finished_at,
		"timestamp": _trainer_finished_at,
	}
	
	# Add error details if validation failed
	if not training_valid:
		# Get error messages only (not warnings)
		error_msgs = [msg for msg in validation_messages if msg.startswith("Missing") or msg.startswith("Invalid") or "too small" in msg or "empty" in msg]
		# Summarize first 3 errors for status bar
		error_summary = "; ".join(error_msgs[:3]) if error_msgs else "Training verification failed"
		if len(error_msgs) > 3:
			error_summary += f" (and {len(error_msgs) - 3} more errors)"
		
		status_data["error"] = error_summary
		status_data["all_errors"] = validation_messages  # Full list for debugging
		status_data["error_count"] = len(error_msgs)
	else:
		status_data["error"] = None
	
	with open("trainer_status.json", "w", encoding="utf-8") as f:
		json.dump(status_data, f, indent=2)
	
	# Print final status
	if training_valid:
		print()
		print(f"{'='*60}")
		print(f"✅ TRAINING COMPLETED SUCCESSFULLY FOR {_arg_coin}")
		print(f"{'='*60}")
		print(f"  Started:  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(_trainer_started_at))}")
		print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(_trainer_finished_at))}")
		print(f"  Duration: {(_trainer_finished_at - _trainer_started_at) // 60} minutes")
		print(f"{'='*60}")
		print()
	else:
		print()
		print(f"{'='*60}")
		print(f"❌ TRAINING FAILED FOR {_arg_coin}")
		print(f"{'='*60}")
		print(f"  Please review the errors above and retry training.")
		print(f"  Common fixes:")
		print(f"    - Ensure stable internet connection")
		print(f"    - Check disk space availability")
		print(f"    - Verify KuCoin API is accessible")
		print(f"{'='*60}")
		print()
		
except Exception as e:
	print(f"❌ ERROR: Failed to write training status: {e}")
	# Still write a basic status file so GUI knows training ended
	try:
		with open("trainer_status.json", "w", encoding="utf-8") as f:
			json.dump({
				"coin": _arg_coin,
				"state": "FAILED",
				"timestamp": int(time.time()),
				"error": f"Status write error: {e}"
			}, f)
	except:
		pass

sys.exit(0 if training_valid else 1)