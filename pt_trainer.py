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

# Ensure clean console output
try:
	sys.stdout.reconfigure(encoding='utf-8')
except:
	pass

# third-party
import psutil
from kucoin.client import Market

# instantiate KuCoin market client (kept at top-level like original)
market = Market(url='https://api.kucoin.com')

# KuCoin API constants
KUCOIN_MAX_CANDLES = 1500  # Maximum candles per API request

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
			print(f"[DEBUG LOG ERROR] Failed to write to {log_file}")
			print(f"Error: {type(e).__name__}: {e}")
			print(f"Working directory: {os.getcwd()}")
			print(f"{'!'*60}\n")
			pass  # Don't let logging errors break the trainer

def handle_network_error(operation: str, error: Exception):
	"""Print network error and suggest enabling debug mode"""
	print(f"\n{'='*60}")
	print(f"NETWORK ERROR: {operation} failed")
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
_last_threshold_written = {}  # tf_choice -> float

# Global write buffer to reduce disk I/O during training
# Thresholds are buffered and written only at critical points (timeframe end, training complete, user stop)
_write_buffer = {
	"thresholds": {},           # {timeframe: threshold_value}
	"last_checkpoint_time": 0,  # timestamp of last safety checkpoint
	"last_flush_time": 0,       # timestamp of last periodic flush
}

def _read_text(path):
	with open(path, "r", encoding="utf-8", errors="ignore") as f:
		return f.read()

def load_memory(tf_choice):
	"""Load memories/weights for a timeframe once and keep them in RAM."""
	if tf_choice in _memory_cache:
		return _memory_cache[tf_choice]
	data = {
		"memory_list": [],
		"memory_patterns": [],  # Pre-split patterns for faster comparison
		"weight_list": [],
		"high_weight_list": [],
		"low_weight_list": [],
		"dirty": False,
	}
	try:
		data["memory_list"] = _read_text(f"memories_{tf_choice}.dat").replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split('~')
		# Pre-split patterns once during load (huge speedup for pattern matching)
		data["memory_patterns"] = [mem.split('{}')[0].split(' ') for mem in data["memory_list"]]
	except:
		data["memory_list"] = []
		data["memory_patterns"] = []
	# The memory files are plain-text caches written by the trainer. We
	# normalize and split them into Python lists here to avoid repeated
	# file I/O during inner loops of the training routine.
	try:
		data["weight_list"] = _read_text(f"memory_weights_{tf_choice}.dat").replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split(' ')
	except:
		data["weight_list"] = []
	try:
		data["high_weight_list"] = _read_text(f"memory_weights_high_{tf_choice}.dat").replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split(' ')
	except:
		data["high_weight_list"] = []
	try:
		data["low_weight_list"] = _read_text(f"memory_weights_low_{tf_choice}.dat").replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split(' ')
	except:
		data["low_weight_list"] = []
	_memory_cache[tf_choice] = data
	return data

def flush_memory(tf_choice, force=False):
	"""Write memories/weights back to disk only when they changed (batch IO)."""
	data = _memory_cache.get(tf_choice)
	if not data:
		return
	if (not data.get("dirty")) and (not force):
		return
	try:
		with open(f"memories_{tf_choice}.dat", "w+", encoding="utf-8") as f:
			f.write("~".join([x for x in data["memory_list"] if str(x).strip() != ""]))
	except:
		pass
	# flush_memory() batches writes to reduce disk churn. The `dirty` flag
	# prevents redundant writes when nothing changed.
	try:
		with open(f"memory_weights_{tf_choice}.dat", "w+", encoding="utf-8") as f:
			f.write(" ".join([str(x) for x in data["weight_list"] if str(x).strip() != ""]))
	except:
		pass
	try:
		with open(f"memory_weights_high_{tf_choice}.dat", "w+", encoding="utf-8") as f:
			f.write(" ".join([str(x) for x in data["high_weight_list"] if str(x).strip() != ""]))
	except:
		pass
	try:
		with open(f"memory_weights_low_{tf_choice}.dat", "w+", encoding="utf-8") as f:
			f.write(" ".join([str(x) for x in data["low_weight_list"] if str(x).strip() != ""]))
	except:
		pass
	data["dirty"] = False

def buffer_threshold(tf_choice, perfect_threshold):
	"""Buffer threshold in memory instead of writing to disk immediately.
	This dramatically reduces disk I/O during training loops."""
	_write_buffer["thresholds"][tf_choice] = perfect_threshold
	_last_threshold_written[tf_choice] = perfect_threshold

def flush_all_buffers(force=True):
	"""Write all buffered data to disk. Called at critical points:
	- End of each timeframe
	- Training completion
	- User stop
	- Error/crash (via signal handler)
	
	This ensures data integrity while minimizing disk writes during training."""
	debug_print("[DEBUG] TRAINER: Flushing all buffers to disk...")
	
	# 1. Write all buffered thresholds
	for tf_choice, threshold_value in _write_buffer["thresholds"].items():
		try:
			with open(f"neural_perfect_threshold_{tf_choice}.dat", "w+", encoding="utf-8") as f:
				f.write(str(threshold_value))
			debug_print(f"[DEBUG] TRAINER: Wrote threshold for {tf_choice}: {threshold_value:.4f}")
		except Exception as e:
			debug_print(f"[DEBUG] TRAINER: Failed to write threshold for {tf_choice}: {e}")
	
	# 2. Flush all cached memories/weights for all timeframes
	for tf_choice in list(_memory_cache.keys()):
		try:
			flush_memory(tf_choice, force=force)
			debug_print(f"[DEBUG] TRAINER: Flushed memory cache for {tf_choice}")
		except Exception as e:
			debug_print(f"[DEBUG] TRAINER: Failed to flush memory for {tf_choice}: {e}")
	
	# 3. Update checkpoint timestamp
	_write_buffer["last_checkpoint_time"] = int(time.time())
	debug_print("[DEBUG] TRAINER: All buffers flushed successfully")

def clear_incompatible_patterns(tf_choice, expected_pattern_size):
	"""Delete existing pattern files if they were created with a different number_of_candles.
	This forces a clean retrain when pattern structure changes (e.g., 2 candles → 3 candles).
	Returns True if files were cleared, False if compatible or no files exist."""
	debug_print(f"[VALIDATION] Checking pattern compatibility for {tf_choice} (expecting {expected_pattern_size} candles)...")
	try:
		memory_file = f"memories_{tf_choice}.dat"
		if not os.path.exists(memory_file):
			debug_print(f"[VALIDATION] No existing patterns found, starting fresh")
		
		# Check first pattern to see if structure matches expected size
		content = _read_text(memory_file)
		if not content or content.strip() == '':
			return False  # Empty file, nothing to validate
		
		first_pattern = content.split('~')[0]
		if '{}' in first_pattern:
			pattern_values = first_pattern.split('{}')[0].split(' ')
			pattern_values = [v for v in pattern_values if v.strip() != '']
			actual_size = len(pattern_values)
			expected_size = expected_pattern_size - 1  # number_of_candles=3 means 2 prior values
			
			if actual_size == expected_size:
				debug_print(f"[DEBUG] TRAINER: Existing patterns for {tf_choice} are compatible (size={actual_size})")
				return False  # Compatible, keep existing patterns
			
			# Incompatible structure detected - clear all related files
			print(f"[RETRAIN] Pattern structure changed for {tf_choice}:")
			print(f"[RETRAIN]   Old: {actual_size} values per pattern")
			print(f"[RETRAIN]   New: {expected_size} values per pattern")
			print(f"[RETRAIN] Clearing old memories and starting fresh...")
			
			files_to_clear = [
				f"memories_{tf_choice}.dat",
				f"memory_weights_{tf_choice}.dat",
				f"memory_weights_high_{tf_choice}.dat",
				f"memory_weights_low_{tf_choice}.dat",
				f"neural_perfect_threshold_{tf_choice}.dat"
			]
			
			for filename in files_to_clear:
				if os.path.exists(filename):
					os.remove(filename)
					debug_print(f"[DEBUG] TRAINER: Deleted {filename}")
			
			return True  # Files cleared
	except Exception as e:
		debug_print(f"[DEBUG] TRAINER: Error checking pattern compatibility: {e}")
		return False

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
	msg = 'EXCEPTION IN (LINE {} "{}"): {}'.format(lineno, line.strip(), exc_obj)
	print(msg)
	# Always log exceptions to file (even without debug mode)
	try:
		coin_name = sys.argv[1] if len(sys.argv) > 1 else "unknown"
		log_file = f"debug_trainer_{coin_name}.log"
		import datetime
		timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		with open(log_file, "a", encoding="utf-8") as f:
			f.write(f"[{timestamp}] {msg}\n")
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
			print(f"TRAINER CHECKSUM CHECK: Cannot read root trainer")
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
			print(f"TRAINER FILE MISMATCH DETECTED (checksum differs)")
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
		print(f"Warning: Trainer checksum check failed: {e}")

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
bounce_accuracy_tolerance = training_settings.get("bounce_accuracy_tolerance", 0.5) if os.path.isfile(import_path) else 0.5
bounce_accuracy_target = training_settings.get("bounce_accuracy_target", 70.0) if os.path.isfile(import_path) else 70.0
min_pattern_weight = training_settings.get("min_pattern_weight", 0.1) if os.path.isfile(import_path) else 0.1
base_learning_rate = training_settings.get("learning_rate", 0.01) if os.path.isfile(import_path) else 0.01
initial_perfect_threshold = training_settings.get("initial_perfect_threshold", 1.0) if os.path.isfile(import_path) else 1.0
perfect_threshold_target = training_settings.get("perfect_threshold_target", 10.0) if os.path.isfile(import_path) else 10.0
target_matches_small = training_settings.get("target_matches_small", 10) if os.path.isfile(import_path) else 10
target_matches_medium = training_settings.get("target_matches_medium", 15) if os.path.isfile(import_path) else 15
target_matches_large = training_settings.get("target_matches_large", 20) if os.path.isfile(import_path) else 20

# Initialize number_of_candles based on number of timeframes being trained
# Each timeframe starts with minimum pattern size (2 candles = 1 prior value, 3 candles = 2 prior values)
number_of_candles = [2] * len(tf_choices)

# Debug print to confirm settings are loaded (will only show if debug mode is enabled)
try:
	debug_print(f"[DEBUG] TRAINER: Loaded training settings from: {import_path}")
	debug_print(f"[DEBUG] TRAINER: Active timeframes: {tf_choices}")
	debug_print(f"[DEBUG] TRAINER: Pattern sizes (number_of_candles): {number_of_candles}")
	debug_print(f"[DEBUG] TRAINER: Training parameters - bounce_tol={bounce_accuracy_tolerance}, bounce_target={bounce_accuracy_target}, min_weight={min_pattern_weight}, learning_rate={base_learning_rate}, threshold_target={perfect_threshold_target}")
	debug_print(f"[DEBUG] TRAINER: Target matches - small={target_matches_small}, medium={target_matches_medium}, large={target_matches_large}")
except:
	pass
# GUI hub input (no prompts)
# Usage: python pt_trainer.py <COIN> [reprocess_yes|reprocess_no]
# Coin argument is REQUIRED - no default fallback

if len(sys.argv) < 2 or not str(sys.argv[1]).strip():
	print("ERROR: Coin symbol required as first argument")
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
		print(f"Warning: Error flushing buffers: {e}")
	
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

# TEST DEBUG LOGGING - Write a startup message to verify debug logging is working
try:
	debug_enabled = _is_debug_mode()
	log_file = f"debug_trainer_{_arg_coin}.log"
	import datetime
	timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	
	# Always write startup message (even if debug is off) to confirm file writing works
	with open(log_file, "a", encoding="utf-8") as f:
		f.write(f"\n{'='*60}\n")
		f.write(f"[{timestamp}] TRAINER STARTED for {_arg_coin}\n")
		f.write(f"[{timestamp}] Debug mode: {'ENABLED' if debug_enabled else 'DISABLED'}\n")
		f.write(f"[{timestamp}] Log file: {log_file}\n")
		f.write(f"[{timestamp}] Working directory: {os.getcwd()}\n")
		f.write(f"{'='*60}\n\n")
	
	print(f"\n{'='*60}")
	print(f"TRAINER DEBUG LOG INITIALIZED")
	print(f"  Coin: {_arg_coin}")
	print(f"  Debug mode: {'ENABLED' if debug_enabled else 'DISABLED'}")
	print(f"  Log file: {log_file}")
	print(f"  Working directory: {os.getcwd()}")
	print(f"{'='*60}\n")
	
	if not debug_enabled:
		print(f"WARNING: Debug mode is DISABLED in gui_settings.json")
		print(f"         Enable it to see detailed training logs")
		print(f"         Startup message written to {log_file} for verification\n")
except Exception as e:
	print(f"ERROR: Failed to initialize debug logging: {e}")
	PrintException()

the_big_index = 0
bounce_accuracy_dict = {}  # Store bounce accuracy for each timeframe
# Main training loop note:
# The primary loop below (`while True:`) resets per-iteration state and
# then performs training/analysis for the configured `tf_choice`. It is
# designed to be long-running and guarded by small I/O checkpoints so
# external 'killer' files, status files, or restarts can be coordinated.
while True:
	debug_print(f"[DEBUG] TRAINER: Outer loop restart - the_big_index={the_big_index}")
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
	debug_print(f"[DEBUG] TRAINER: Starting timeframe {tf_choices[the_big_index]} (index {the_big_index}/{len(tf_choices)-1})")
	tf_choice = tf_choices[the_big_index]
	debug_print(f"[DEBUG] TRAINER: Starting training cycle for {_arg_coin} on timeframe {tf_choice}...")
	
	# Check if existing patterns are compatible with current number_of_candles setting
	# If pattern structure changed (e.g., 2→3 candles), force retrain from scratch
	clear_incompatible_patterns(tf_choice, number_of_candles[the_big_index])
	
	_mem = load_memory(tf_choice)
	memory_list = _mem["memory_list"]
	weight_list = _mem["weight_list"]
	high_weight_list = _mem["high_weight_list"]
	low_weight_list = _mem["low_weight_list"]
	memory_list_empty = 'no' if len(memory_list) > 0 else 'yes'
	choice_index = tf_choices.index(tf_choice)
	timeframe = tf_choice
	timeframe_minutes = tf_minutes[choice_index]
	start_time = int(time.time())
	candles_to_predict = 1#droplet setting (Max is half of number_of_candles)(Min is 2)
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
	
	# Calculate API delay based on number of coins to avoid exceeding rate limit
	# KuCoin limit: 180 req/min = 3 per second
	try:
		with open(_GUI_SETTINGS_PATH, "r", encoding="utf-8") as f:
			settings = json.load(f)
			num_coins = len(settings.get("coins", []))
			# Scale delay: 0.4s per coin so N coins collectively stay under 3 req/sec
			api_delay = max(0.4, 0.4 * num_coins) if num_coins > 0 else 0.5
	except Exception:
		api_delay = 0.5  # Fallback to conservative delay
	
	while True:
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
			debug_print(f"[DEBUG] TRAINER: KuCoin error (attempt {kucoin_retry_count}/{kucoin_max_retries}): {e}")
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
				print(f"WARNING: Received malformed data from API (batch had {len(history)} entries)")
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
					print(f"WARNING: Failed to parse candle at index {index}: {working_minute}")
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
			parse_success_rate = (successful_parses / total_candles * 100) if total_candles > 0 else 0
			if parse_success_rate < 80:  # Less than 80% success is concerning
				error_msg = f"ERROR: Parse success rate too low ({parse_success_rate:.1f}%) for {timeframe}. Data quality insufficient for training."
				print(error_msg)
				mark_training_error(f"Low parse success rate for {timeframe}: {parse_success_rate:.1f}%")
				sys.exit(1)
		
		# Store for later display in training progress
		# Only count candles actually attempted (from starting_index onwards, not all fetched)
		candles_attempted = len(history_list) - starting_index
		api_acceptance_rate = (successful_parses / candles_attempted * 100) if candles_attempted > 0 else 100.0
		
		# Validate we have historical data before proceeding
		if len(price_list) == 0 or len(high_price_list) == 0 or len(low_price_list) == 0 or len(open_price_list) == 0:
			print(f"ERROR: Failed to fetch historical price data for {_arg_coin} on {timeframe} timeframe. Cannot train without historical data.")
			mark_training_error(f"No historical data for {timeframe}")
			sys.exit(1)
		
		ticker_data = str(market.get_ticker(coin_choice)).replace('"','').replace("'","").replace("[","").replace("{","").replace("]","").replace("}","").replace(",","").lower().split(' ')
		price = float(ticker_data[ticker_data.index('price:')+1])
	except:
		PrintException()
	
	# Debug output before starting training loop
	debug_print(f"Starting training for {tf_choice} (index {the_big_index}/{len(tf_choices)-1})")
	debug_print(f"Total candles collected: {len(price_list)}")
	debug_print(f"Existing memories: {len(memory_list)}")
	
	history_list = []
	history_list2 = []
	perfect_threshold = initial_perfect_threshold
	loop_i = 0  # counts inner training iterations (used to throttle disk IO)
	# Start with 1 candle and grow 1 at a time for maximum learning and self-correction
	price_list_length = 1
	last_printed_candle = 0  # Track last candle we printed status for
	debug_print(f"[DEBUG] TRAINER: Starting window size: {price_list_length} candles (total: {len(price_list)})")
	
	# Initialize price change lists - will be built incrementally one candle at a time (O(n) instead of O(n²))
	price_change_list = []
	high_price_change_list = []
	low_price_change_list = []
	debug_print(f"[DEBUG] TRAINER: Initialized empty price_change lists, starting with price_list_length={price_list_length}")
	debug_print(f"[DEBUG] TRAINER: Total historical candles available: {len(price_list)}")
	debug_print(f"[DEBUG] TRAINER: Starting perfect_threshold: {perfect_threshold}")
	
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
				if new_index < len(price_list2) and new_index < len(open_price_list2):
					price_change = 100*((price_list2[new_index]-open_price_list2[new_index])/open_price_list2[new_index])
					price_change_list.append(price_change)
					debug_print(f"[DEBUG] TRAINER: Added price_change[{new_index}] = {price_change:.4f}")
			
			while len(high_price_list2) > len(high_price_change_list):
				new_index = len(high_price_change_list)
				if new_index < len(high_price_list2) and new_index < len(open_price_list2):
					high_price_change = 100*((high_price_list2[new_index]-open_price_list2[new_index])/open_price_list2[new_index])
					high_price_change_list.append(high_price_change)
					debug_print(f"[DEBUG] TRAINER: Added high_price_change[{new_index}] = {high_price_change:.4f}")
			
			while len(low_price_list2) > len(low_price_change_list):
				new_index = len(low_price_change_list)
				if new_index < len(low_price_list2) and new_index < len(open_price_list2):
					low_price_change = 100*((low_price_list2[new_index]-open_price_list2[new_index])/open_price_list2[new_index])
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
					print_interval = max(1, total_candles // 100)  # 1% intervals, minimum 1
					
					# Print if this is a new milestone we haven't printed yet
					if current_candle != last_printed_candle and (current_candle == 1 or current_candle % print_interval == 0 or current_candle == total_candles):
						print()
						percent_complete = int((current_candle / total_candles) * 100)
						print(f'Timeframe: {timeframe} ({percent_complete}% complete)')
						pattern_count = len(_mem["memory_list"])
						print(f'Training on timeframe data, {pattern_count:,} patterns learned...')
					
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
					debug_print(f"[DEBUG] TRAINER: Patterns built - current={len(current_pattern)}, high={len(high_current_pattern)}, low={len(low_current_pattern)}, expected={number_of_candles[number_of_candles_index]-1}")
					debug_print(f"[DEBUG] TRAINER: Lists sizes - price_change={len(price_change_list)}, high_price_change={len(high_price_change_list)}, low_price_change={len(low_price_change_list)}")
					
					# Non-blocking validation: warn if patterns are shorter than expected but proceed anyway
					expected_length = number_of_candles[number_of_candles_index] - 1
					if (len(current_pattern) < expected_length or 
					    len(high_current_pattern) < expected_length or 
					    len(low_current_pattern) < expected_length):
						debug_print(f"[DEBUG] TRAINER: Pattern shorter than expected (current:{len(current_pattern)}, high:{len(high_current_pattern)}, low:{len(low_current_pattern)}, expected:{expected_length}) - proceeding with partial data")
					
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
										debug_print(f"[DEBUG] TRAINER: No perfect match found (threshold={perfect_threshold:.4f}), will create new pattern")
									else:
										try:
											final_moves = sum(moves)/len(moves)
											high_final_moves = sum(high_moves)/len(high_moves)
											low_final_moves = sum(low_moves)/len(low_moves)
										except:
											final_moves = 0.0
											high_final_moves = 0.0
											low_final_moves = 0.0
										which_memory_index = perfect_dexs[perfect_diffs.index(min(perfect_diffs))]
										perfect.append('yes')
										debug_print(f"[DEBUG] TRAINER: Found {len(perfect_dexs)} perfect match(es) (threshold={perfect_threshold:.4f})")
									break
								
								# Use pre-split patterns from cache (avoid repeated string operations)
								_mem = load_memory(tf_choice)
								memory_pattern = _mem["memory_patterns"][memory_index]
								checks = []
								# Use for loop instead of while True (cleaner and faster)
								min_len = min(len(current_pattern), len(memory_pattern))
								for check_dex in range(min_len):
									current_candle = float(current_pattern[check_dex])
									memory_candle = float(memory_pattern[check_dex])
									# Calculate difference with explicit zero-division protection
									avg_value = (current_candle + memory_candle) / 2
									if avg_value == 0.0:
										# Both values are zero or sum to zero - no difference
										difference = 0.0
									else:
										difference = abs((abs(current_candle - memory_candle) / avg_value) * 100)
									checks.append(difference)
								
								# Protect against division by zero if checks list is empty
								if len(checks) == 0:
									debug_print(f"[DEBUG] TRAINER: Empty checks list for memory {memory_index}, skipping")
									memory_index += 1
									continue
								
								diff_avg = sum(checks)/len(checks)
								if diff_avg <= perfect_threshold:
									any_perfect = 'yes'
									# String cleaning already done in load_memory() - just remove spaces and split
									high_diff = float(memory_list[memory_index].split('{}')[1].replace(' ',''))/100
									low_diff = float(memory_list[memory_index].split('{}')[2].replace(' ',''))/100
									unweighted.append(float(memory_pattern[len(memory_pattern)-1]))
									move_weights.append(float(weight_list[memory_index]))
									high_move_weights.append(float(high_weight_list[memory_index]))
									low_move_weights.append(float(low_weight_list[memory_index]))
									high_unweighted.append(high_diff)
									low_unweighted.append(low_diff)
									moves.append(float(memory_pattern[len(memory_pattern)-1])*float(weight_list[memory_index]))
									high_moves.append(high_diff*float(high_weight_list[memory_index]))
									low_moves.append(low_diff*float(low_weight_list[memory_index]))
									perfect_dexs.append(memory_index)
									perfect_diffs.append(diff_avg)
								else:
									pass
								diffs_list.append(diff_avg)
								memory_index += 1
								if memory_index >= len(memory_list):
									if any_perfect == 'no':
										memory_diff = min(diffs_list)
										which_memory_index = diffs_list.index(memory_diff)
										perfect.append('no')
										final_moves = 0.0
										high_final_moves = 0.0
										low_final_moves = 0.0
										new_memory = 'yes'
									else:
										try:
											final_moves = sum(moves)/len(moves)
											high_final_moves = sum(high_moves)/len(high_moves)
											low_final_moves = sum(low_moves)/len(low_moves)
										except:
											final_moves = 0.0
											high_final_moves = 0.0
											low_final_moves = 0.0
										which_memory_index = perfect_dexs[perfect_diffs.index(min(perfect_diffs))]
										perfect.append('yes')
									break
								else:
									continue
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
					debug_print(f"[DEBUG] TRAINER: Appending current_pattern with {len(current_pattern)} candles (expected: {number_of_candles[number_of_candles_index]-1})")
					all_current_patterns.append(current_pattern)
					
					# Adjust perfect_threshold based on match count, pattern library size, AND accuracy
					# With strict validation, we have fewer but higher-quality patterns
					# Key insight: If accuracy is high, we have enough good matches - don't force more
					match_count = len(unweighted)
					pattern_count = len(_mem["memory_list"])
					
					# Calculate current accuracy from recent perfect predictions
					# Look at last 100 predictions to get recent performance
					recent_window = min(100, len(perfect))
					if recent_window > 0:
						recent_perfect_count = perfect[-recent_window:].count('yes')
						current_accuracy = (recent_perfect_count / recent_window) * 100
					else:
						current_accuracy = 0.0
					
					# Target matches: scale with pattern count
					# Small library (<5k): target 10 matches
					# Medium library (5k-50k): target medium matches  
					# Large library (>50k): target large matches
					if pattern_count < 5000:
						target_matches = target_matches_small
					elif pattern_count < 50000:
						target_matches = target_matches_medium
					else:
						target_matches = target_matches_large
					
					# MATHEMATICAL threshold adjustment (continuous feedback loop)
					# All multipliers are continuous functions, no discrete tiers
					
					# Dataset size: smaller datasets need faster convergence
					dataset_size = len(price_list)
					if dataset_size < 2000:
						size_multiplier = 2.0  # Fast convergence for small datasets
					elif dataset_size < 5000:
						size_multiplier = 1.5  # Medium convergence
					else:
						size_multiplier = 1.0  # Standard convergence
					
					# Bounce accuracy feedback: quality_multiplier = target / actual
					# Low accuracy (50%) → target/50 = 1.4× (aggressive)
					# At target → target/target = 1.0× (neutral)
					# High accuracy (85%) → target/85 = 0.82× (gentle)
					target_bounce_accuracy = bounce_accuracy_target
					bounce_accuracy = bounce_accuracy_dict.get(tf_choice, 0.0)
					if bounce_accuracy > 0:
						quality_multiplier = target_bounce_accuracy / max(10.0, bounce_accuracy)
						quality_multiplier = max(0.3, min(2.0, quality_multiplier))  # Clamp to sane range
					else:
						quality_multiplier = 1.0  # No data yet, neutral
					
					# Match error: encodes both magnitude and direction
					# Too many matches (15 vs 10) → (10-15)/10 = -0.5 → decrease threshold
					# Too few matches (5 vs 10) → (10-5)/10 = +0.5 → increase threshold
					# At target (10 vs 10) → 0 → no change
					match_ratio = (target_matches - match_count) / target_matches
					
					# Learning rate: very conservative since this runs every candle (1497 iterations)
					# When close to target, divide by 10 for fine-tuning
					if perfect_threshold < perfect_threshold_target:
						learning_rate = base_learning_rate / 10
					else:
						learning_rate = base_learning_rate
					
					# Final adjustment: learning_rate × match_error × multipliers
					adjustment = learning_rate * match_ratio * size_multiplier * quality_multiplier
					
					# Accuracy-aware damping: reduce increases when accuracy is already excellent
					if adjustment > 0 and current_accuracy >= 90.0 and match_count >= 5:
						adjustment *= 0.05  # 95% reduction for excellent accuracy
					
					perfect_threshold += adjustment
					
					# Clamp between 0.01 (prevent negative/zero) and 0.90 (prevent runaway)
					perfect_threshold = max(0.01, min(90, perfect_threshold))
					
					debug_print(f"[DEBUG] TRAINER: threshold={perfect_threshold:.4f} | matches={match_count}/{target_matches} | match_acc={current_accuracy:.1f}% | bounce_acc={bounce_accuracy:.1f}% (×{quality_multiplier:.1f}) | patterns={pattern_count:,}")
					
					# Buffer threshold in memory instead of writing to disk
					buffer_threshold(tf_choice, perfect_threshold)

					try:
						index = 0
						current_pattern_length = number_of_candles[number_of_candles_index]
						current_pattern = []
						# Check if price_list2 has enough elements
						if len(price_list2) >= current_pattern_length:
							index = (len(price_list2))-current_pattern_length
							while True:
								current_pattern.append(price_list2[index])
								if len(current_pattern)>=number_of_candles[number_of_candles_index]:
									break
								else:
									index += 1
									if index >= len(price_list2):
										break
									else:
										continue
					except:
						PrintException()
					# TROUBLESHOOTING TOGGLE: if 1==1 keeps this always enabled in production.
					# This block uses memory-based neural predictions (final_moves from pattern matching).
					# The 'else' branch is a fallback for when memory system has no trained data -
					# it attempts simpler 3-candle pattern matching before giving up.
					# Change to 'if 1==0' to force fallback mode for debugging.
					if 1==1:
						while True:
							try:
								c_diff = final_moves/100
								high_diff = high_final_moves
								low_diff = low_final_moves
								prediction_prices = [current_pattern[len(current_pattern)-1]]
								high_prediction_prices = [current_pattern[len(current_pattern)-1]]
								low_prediction_prices = [current_pattern[len(current_pattern)-1]]
								start_price = current_pattern[len(current_pattern)-1]
								new_price = start_price+(start_price*c_diff)
								high_new_price = start_price+(start_price*high_diff)
								low_new_price = start_price+(start_price*low_diff)
								prediction_prices = [start_price,new_price]
								high_prediction_prices = [start_price,high_new_price]
								low_prediction_prices = [start_price,low_new_price]
							except:
								if len(current_pattern) > 0:
									start_price = current_pattern[len(current_pattern)-1]
								else:
									start_price = 0.0
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
								# Variables not defined, use pattern fallback
								if len(current_pattern) > 0:
									new_y = [current_pattern[len(current_pattern)-1],current_pattern[len(current_pattern)-1]]
								else:
									new_y = [0.0, 0.0]
								if len(high_current_pattern) > 0:
									high_new_y = [current_pattern[len(current_pattern)-1] if len(current_pattern) > 0 else 0.0, high_current_pattern[len(high_current_pattern)-1]]
								else:
									high_new_y = [0.0, 0.0]
								if len(low_current_pattern) > 0:
									low_new_y = [current_pattern[len(current_pattern)-1] if len(current_pattern) > 0 else 0.0, low_current_pattern[len(low_current_pattern)-1]]
								else:
									low_new_y = [0.0, 0.0]
						except:
							PrintException()
							if len(current_pattern) > 0:
								new_y = [current_pattern[len(current_pattern)-1],current_pattern[len(current_pattern)-1]]
							else:
								new_y = [0.0, 0.0]
							if len(high_current_pattern) > 0:
								high_new_y = [current_pattern[len(current_pattern)-1] if len(current_pattern) > 0 else 0.0, high_current_pattern[len(high_current_pattern)-1]]
							else:
								high_new_y = [0.0, 0.0]
							if len(low_current_pattern) > 0:
								low_new_y = [current_pattern[len(current_pattern)-1] if len(current_pattern) > 0 else 0.0, low_current_pattern[len(low_currentPattern)-1]]
							else:
								low_new_y = [0.0, 0.0]
					else:
						current_pattern_length = 3
						current_pattern = []
						# Check if price_list2 has enough elements
						if len(price_list2) >= current_pattern_length:
							index = (len(price_list2))-current_pattern_length
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
						# Use last element if pattern exists, otherwise use a default
						if len(current_pattern) > 0:
							new_y = [current_pattern[len(current_pattern)-1],current_pattern[len(current_pattern)-1]]
						else:
							new_y = [0.0, 0.0]
						number_of_candles_index += 1
						if number_of_candles_index >= len(number_of_candles):
							print("ERROR: Processed all number_of_candles without finding patterns. Training cannot complete.")
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
								percent_difference_of_actuals = ((new_y[0]-last_actual)/abs(last_actual))*100
								high_difference_of_actuals = last_actual-high_current_price
								high_percent_difference_of_actuals = ((high_current_price-last_actual)/abs(last_actual))*100
								low_difference_of_actuals = last_actual-low_current_price
								low_percent_difference_of_actuals = ((low_current_price-last_actual)/abs(last_actual))*100
								percent_difference_of_last = ((last_prediction-last_actual)/abs(last_actual))*100
								high_percent_difference_of_last = ((high_last_prediction-last_actual)/abs(last_actual))*100
								low_percent_difference_of_last = ((low_last_prediction-last_actual)/abs(last_actual))*100
								if in_trade == 'no':
									percent_for_no_sell = ((new_y[1]-last_actual)/abs(last_actual))*100
									og_actual = last_actual
									in_trade = 'yes'
								else:
									percent_for_no_sell = ((new_y[1]-og_actual)/abs(og_actual))*100
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
								# BOUNCE ACCURACY: Tracks only candles where price approached predicted limits
								# Success (1) = price touched limit but close bounced back inside
								# Failure (0) = price touched limit and close broke through
								# Tolerance: percentage points (additive, not multiplicative)
								tolerance = bounce_accuracy_tolerance
								
								# Case 1: High limit approached, close bounced back (SUCCESS)
								if high_percent_difference_of_actuals >= high_baseline_price_change_pct + tolerance and percent_difference_of_actuals < high_baseline_price_change_pct:
									upordown3.append(1)
									upordown.append(1)
									upordown4.append(1)
									if len(upordown4) > 100:
										del upordown4[0]
									else:
										pass
								# Case 2: Low limit approached, close bounced back (SUCCESS)
								elif low_percent_difference_of_actuals <= low_baseline_price_change_pct - tolerance and percent_difference_of_actuals > low_baseline_price_change_pct:
									upordown.append(1)
									upordown3.append(1)
									upordown4.append(1)
									if len(upordown4) > 100:
										del upordown4[0]
									else:
										pass
								# Case 3: High limit approached, close broke through (FAILURE)
								elif high_percent_difference_of_actuals >= high_baseline_price_change_pct + tolerance and percent_difference_of_actuals > high_baseline_price_change_pct:
									upordown3.append(0)
									upordown2.append(0)
									upordown.append(0)
									upordown4.append(0)
									if len(upordown4) > 100:
										del upordown4[0]
									else:
										pass
								# Case 4: Low limit approached, close broke through (FAILURE)
								elif low_percent_difference_of_actuals <= low_baseline_price_change_pct - tolerance and percent_difference_of_actuals < low_baseline_price_change_pct:
									upordown3.append(0)
									upordown2.append(0)
									upordown.append(0)
									upordown4.append(0)
									if len(upordown4) > 100:
										del upordown4[0]
									else:
										pass
								else:
									# Price stayed within normal range - not tracked for bounce accuracy
									pass
							else:
								pass
							# Calculate and display bounce accuracy if we have data
							if len(upordown4) > 0:
								accuracy = (sum(upordown4)/len(upordown4))*100
								# Store accuracy in dict for this timeframe
								bounce_accuracy_dict[tf_choice] = accuracy
								
								# Print approximately every 1% of candles to reduce console I/O overhead
								total_candles = len(price_list)
								current_candle = len(price_list2)
								print_interval = max(1, total_candles // 100)  # 1% intervals, minimum 1
								
								# Print if this is a new milestone we haven't printed yet
								if current_candle != last_printed_candle and (current_candle % print_interval == 0 or current_candle == total_candles):
									formatted = format(accuracy, '.2f').rstrip('0').rstrip('.')
									limit_test_count = len(upordown4)
									print(f'Accuracy (last 100 breakout candles): {formatted}%')
									threshold_formatted = format(perfect_threshold, '.2f')
									print(f'Candles Processed: {current_candle} ({threshold_formatted}% threshold)')
									acceptance_formatted = format(api_acceptance_rate, '.1f').rstrip('0').rstrip('.')
									print(f'Total Candles: {total_candles} ({acceptance_formatted}% acceptance)')
						except:
							PrintException()
					else:
						pass
					
					# Update last_printed_candle after all print sections have had a chance to run
					# This ensures both timeframe and bounce accuracy prints happen together at milestones
					total_candles = len(price_list)
					current_candle = len(price_list2)
					print_interval = max(1, total_candles // 100)
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
							percent_difference_of_now = ((new_y[1]-new_y[0])/abs(new_y[0]))*100
						else:
							percent_difference_of_now = 0.0
						if abs(high_new_y[0]) > 0:
							high_percent_difference_of_now = ((high_new_y[1]-high_new_y[0])/abs(high_new_y[0]))*100
						else:
							high_percent_difference_of_now = 0.0
						if abs(low_new_y[0]) > 0:
							low_percent_difference_of_now = ((low_new_y[1]-low_new_y[0])/abs(low_new_y[0]))*100
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
										# Remove patterns with weight below threshold (they contribute almost nothing to predictions)
										debug_print(f"[DEBUG] TRAINER: Pruning weak patterns (weight < {min_pattern_weight}) for {tf_choice}")
										try:
											_mem = _memory_cache.get(tf_choice)
											if _mem:
												original_count = len(_mem["memory_list"])
												# Filter all lists together using zip (faster and cleaner)
												filtered = []
												for mem, w, hw, lw in zip(_mem["memory_list"], _mem["weight_list"], 
												                          _mem["high_weight_list"], _mem["low_weight_list"]):
													try:
														if float(w) >= min_pattern_weight:
															filtered.append((mem, w, hw, lw))
													except (ValueError, TypeError):
														pass  # Skip corrupt weight entries
												
												# Unzip filtered results and rebuild memory_patterns
												if filtered:
													_mem["memory_list"], _mem["weight_list"], _mem["high_weight_list"], _mem["low_weight_list"] = zip(*filtered)
													_mem["memory_list"] = list(_mem["memory_list"])
													_mem["weight_list"] = list(_mem["weight_list"])
													_mem["high_weight_list"] = list(_mem["high_weight_list"])
													_mem["low_weight_list"] = list(_mem["low_weight_list"])
													# Rebuild pre-split patterns after pruning
													_mem["memory_patterns"] = [mem.split('{}')[0].split(' ') for mem in _mem["memory_list"]]
												else:
													_mem["memory_list"] = []
													_mem["weight_list"] = []
													_mem["high_weight_list"] = []
													_mem["low_weight_list"] = []
													_mem["memory_patterns"] = []
												_mem["dirty"] = True
												
												pruned_count = original_count - len(_mem["memory_list"])
												if pruned_count > 0:
													debug_print(f"[DEBUG] TRAINER: Pruned {pruned_count} weak patterns ({original_count} -> {len(_mem['memory_list'])})")
										except Exception as e:
											debug_print(f"[DEBUG] TRAINER: Pruning failed (non-critical): {e}")
										
										# Flush all buffered data before switching to next timeframe
										debug_print(f"[DEBUG] TRAINER: Flushing buffers before timeframe switch (the_big_index: {the_big_index} -> {the_big_index + 1})")
										debug_print(f'Memory list size for {tf_choice} before flush: {len(_memory_cache.get(tf_choice, {}).get("memory_list", []))}')
										flush_all_buffers(force=True)
										
										# Clear memory cache before switching timeframes to prevent cross-contamination
										debug_print(f"[DEBUG] TRAINER: Clearing memory cache after flush")
										_memory_cache.clear()
										the_big_index += 1
										debug_print(f'Incremented the_big_index to {the_big_index} (next timeframe will be: {tf_choices[the_big_index] if the_big_index < len(tf_choices) else "DONE"})')
										print('Moving to next timeframe')
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
											if len(number_of_candles) == 1:
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
														with open('bounce_accuracy.txt', 'w', encoding='utf-8') as f:
															f.write(f'Last Updated: {timestamp}\n')
															f.write(f'Average: {avg_accuracy:.2f}%\n')
															for tf in tf_choices:
																if tf in bounce_accuracy_dict:
																	f.write(f'{tf}: {bounce_accuracy_dict[tf]:.2f}%\n')
														
														# Print summary
														print()  # Empty line before bounce accuracy
														print(f'Limit-Breach Accuracy Results ({_arg_coin})')
														print(f'Last trained: {timestamp}')
														print(f'Average accuracy: {avg_accuracy:.2f}%')
														tf_results = ', '.join([f'{tf}={bounce_accuracy_dict[tf]:.2f}%' for tf in tf_choices if tf in bounce_accuracy_dict])
														print(f'Per timeframe: {tf_results}')
														
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
													print(f'Warning: Could not save bounce accuracy: {e}')
												
												# Check if memory files are empty (indicates training failure)
												training_valid = True
												empty_memories = []
												try:
													for tf in tf_choices:
														memory_file = f"memories_{tf}.dat"
														if os.path.isfile(memory_file):
															file_size = os.path.getsize(memory_file)
															if file_size < 100:  # Less than 100 bytes is effectively empty
																empty_memories.append(tf)
																training_valid = False
														else:
															# Memory file doesn't exist at all
															empty_memories.append(tf)
															training_valid = False
													
													if not training_valid:
														print()
														print(f'❌ ERROR: Training failed - memory files are empty or missing for: {", ".join(empty_memories)}')
														print(f'❌ Training will be marked as FAILED. Please retry training.')
												except Exception as e:
													print(f'Warning: Could not validate memory files: {e}')
												
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
													with open("trainer_status.json", "w", encoding="utf-8") as f:
														json.dump(
															{
																"coin": _arg_coin,
																"state": "FINISHED" if training_valid else "FAILED",
																"started_at": _trainer_started_at,
																"finished_at": _trainer_finished_at,
																"timestamp": _trainer_finished_at,
																"error": "Empty memory files" if not training_valid else None,
															},
															f,
														)
												except Exception:
													pass

												sys.exit(0 if training_valid else 1)
										else:
											pass
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
												this_differ = ((price2-new_y[1])/abs(new_y[1]))*100
												high_this_differ = ((high_price2-new_y[1])/abs(new_y[1]))*100
												low_this_differ = ((low_price2-new_y[1])/abs(new_y[1]))*100
											else:
												this_differ = 0.0
												high_this_differ = 0.0
												low_this_differ = 0.0
											if abs(new_y[0]) > 0:
												this_diff = ((price2-new_y[0])/abs(new_y[0]))*100
												high_this_diff = ((high_price2-new_y[0])/abs(new_y[0]))*100
												low_this_diff = ((low_price2-new_y[0])/abs(new_y[0]))*100
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
													perc_diff_now = ((current_prediction_price-new_y[0])/abs(new_y[0]))*100
													perc_diff_now_actual = ((price2-new_y[0])/abs(new_y[0]))*100
													high_perc_diff_now_actual = ((high_price2-new_y[0])/abs(new_y[0]))*100
													low_perc_diff_now_actual = ((low_price2-new_y[0])/abs(new_y[0]))*100
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
													difference = abs((abs(current_prediction_price - float(price2)) / avg_pred_price) * 100)
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
																predicted_move_pct = (moves[indy]*100)
																high_predicted_move_pct = (high_moves[indy]*100)
																low_predicted_move_pct = (low_moves[indy]*100)
																
																# Safe threshold calculation - prevent division/comparison issues with zero values
																# When predicted_move_pct=0 (flat price), use small absolute threshold instead
																predicted_move_pct_threshold = abs(predicted_move_pct) * 0.1 if predicted_move_pct != 0 else 0.1
																high_predicted_move_pct_threshold = abs(high_predicted_move_pct) * 0.1 if high_predicted_move_pct != 0 else 0.1
																low_predicted_move_pct_threshold = abs(low_predicted_move_pct) * 0.1 if low_predicted_move_pct != 0 else 0.1
																
																if high_perc_diff_now_actual > high_predicted_move_pct + high_predicted_move_pct_threshold:
																	high_new_weight = high_move_weights[indy] + 0.25
																	if high_new_weight > 2.0:
																		high_new_weight = 2.0
																	else:
																		pass
																elif high_perc_diff_now_actual < high_predicted_move_pct - high_predicted_move_pct_threshold:
																	high_new_weight = high_move_weights[indy] - 0.25
																	if high_new_weight < 0.0:
																		high_new_weight = 0.0
																	else:
																		pass
																else:
																	high_new_weight = high_move_weights[indy]
																if low_perc_diff_now_actual < low_predicted_move_pct - low_predicted_move_pct_threshold:
																	low_new_weight = low_move_weights[indy] + 0.25
																	if low_new_weight > 2.0:
																		low_new_weight = 2.0
																	else:
																		pass
																elif low_perc_diff_now_actual > low_predicted_move_pct + low_predicted_move_pct_threshold:
																	low_new_weight = low_move_weights[indy] - 0.25
																	if low_new_weight < 0.0:
																		low_new_weight = 0.0
																	else:
																		pass
																else:
																	low_new_weight = low_move_weights[indy]
																if perc_diff_now_actual > predicted_move_pct + predicted_move_pct_threshold:
																	new_weight = move_weights[indy] + 0.25
																	if new_weight > 2.0:
																		new_weight = 2.0
																	else:
																		pass
																elif perc_diff_now_actual < predicted_move_pct - predicted_move_pct_threshold:
																	new_weight = move_weights[indy] - 0.25
																	if new_weight < (0.0-2.0):
																		new_weight = (0.0-2.0)
																	else:
																		pass
																else:
																	new_weight = move_weights[indy]
																
																# Update cache directly to ensure changes persist (don't rely on references)
																_mem["weight_list"][perfect_dexs[indy]] = new_weight
																_mem["high_weight_list"][perfect_dexs[indy]] = high_new_weight
																_mem["low_weight_list"][perfect_dexs[indy]] = low_new_weight

																# mark dirty (we will flush in batches)
																_mem["dirty"] = True

																indy += 1
																# Loop condition moved to top of while loop for safety
													except Exception as e:
														# Catch unexpected errors during weight updates - create new pattern (baseline approach)
														# Don't log the expected "no patterns" exception - it's normal during initial training
														if "EXPECTED_NO_PATTERNS" not in str(e):
															PrintException()
														debug_print(f"[DEBUG] TRAINER: Weight update failed - creating new memory pattern")
														
														# Create new memory when weight update fails (not when moves is empty)
														# This is the baseline's pattern creation approach - only on actual failure
														# Create a copy of the pattern with this_diff appended
														new_pattern = all_current_patterns[highlowind] + [this_diff]
														debug_print(f"[DEBUG] TRAINER: new_pattern before mem_entry: {len(new_pattern)} candles, highlowind={highlowind}, all_current_patterns[{highlowind}]={len(all_current_patterns[highlowind])} candles")
														
														mem_entry = str(new_pattern).replace("'","").replace(',','').replace('"','').replace(']','').replace('[','')+'{}'+str(high_this_diff)+'{}'+str(low_this_diff)
														
														# Strict pattern creation: only append valid, unique, non-empty patterns
														# However, when memory is small (<1000 patterns), be more permissive to allow initial learning
														skip_pattern = False
														skip_reason = ""
														
														if not mem_entry or str(mem_entry).strip() == "":
															skip_pattern = True
															skip_reason = "empty"
														elif not is_valid_pattern(mem_entry):
															skip_pattern = True
															skip_reason = "invalid"
														elif len(_mem["memory_list"]) > 1000 and mem_entry in _mem["memory_list"]:
															# Only check duplicates when we have a healthy memory size
															# This prevents infinite loops during initial training
															skip_pattern = True
															skip_reason = "duplicate"
														
														if skip_pattern:
															debug_print(f"[DEBUG] TRAINER: Skipped {skip_reason} pattern (memory size: {len(_mem['memory_list'])})")
														else:
															candle_count = len(mem_entry.split('{}')[0].split(' '))
															debug_print(f"[DEBUG] TRAINER: Adding pattern with {candle_count} candles to memory (expected: {number_of_candles[number_of_candles_index]})")
															_mem["memory_list"].append(mem_entry)
															# Also append pre-split pattern for fast comparison
															_mem["memory_patterns"].append(mem_entry.split('{}')[0].split(' '))
															_mem["weight_list"].append(1.0)  # Start with neutral weight
															_mem["high_weight_list"].append(1.0)
															_mem["low_weight_list"].append(1.0)
															_mem["dirty"] = True
															debug_print(f"[DEBUG] TRAINER: Added new pattern to memory (total: {len(_mem['memory_list'])}), candles={candle_count}")
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
			if _restart_outer_loop:
				break
		# Check if we need to restart outer loop
		if _restart_outer_loop:
			break
	# Check if we need to restart outer loop
	if _restart_outer_loop:
		continue