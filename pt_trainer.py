"""
pt_trainer.py

Description:
This module runs the training process for PowerTraderAI. The trainer
process walks through historical candle data across multiple timeframes
and records memory patterns paired with the next-candle outcome. Those
memories are used to generate per-timeframe predicted candles (weighted
averages of close matches) whose highs/lows are displayed by the Thinker
and used to form trading decisions.

Repository: https://github.com/garagesteve1155/PowerTrader_AI
Author: Stephen Hughes (garagesteve1155)
Contributors: Dr-Dufenshmirtz

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

# -----------------------------
# DEBUG MODE SUPPORT
# -----------------------------
_GUI_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "gui_settings.json")
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
		# Also write to debug log file
		try:
			# Get coin name from command line args if available
			coin_name = sys.argv[1] if len(sys.argv) > 1 else "unknown"
			log_file = f"debug_trainer_{coin_name}.log"
			import datetime
			timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
			with open(log_file, "a", encoding="utf-8") as f:
				f.write(f"[{timestamp}] {msg}\n")
		except Exception:
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

avg50 = []
sells_count = 0
prediction_prices_avg_list = []
list_len = 0
in_trade = 'no'

# ===== COMMENTED OUT - REDUNDANT MODULE-LEVEL VARIABLES =====
# These variables are immediately overwritten when the main loop starts at line ~458
# They are redefined at the start of each loop iteration (lines ~460-537)
# DELETE IF NO ISSUES AFTER TESTING
# updowncount = 0
# updowncount1 = 0
# updowncount1_2 = 0
# updowncount1_3 = 0
# updowncount1_4 = 0
# high_baseline_price_change_pct = 0.0
# low_baseline_price_change_pct = 0.0
# last_flipped = 'no'
# starting_amounth02 = 100.0
# starting_amounth05 = 100.0
# starting_amounth10 = 100.0
# starting_amounth20 = 100.0
# starting_amounth50 = 100.0
# starting_amount = 100.0
# starting_amount1 = 100.0
# starting_amount1_2 = 100.0
# starting_amount1_3 = 100.0
# starting_amount1_4 = 100.0
# starting_amount2 = 100.0
# starting_amount2_2 = 100.0
# starting_amount2_3 = 100.0
# starting_amount2_4 = 100.0
# starting_amount3 = 100.0
# starting_amount3_2 = 100.0
# starting_amount3_3 = 100.0
# starting_amount3_4 = 100.0
# starting_amount4 = 100.0
# starting_amount4_2 = 100.0
# starting_amount4_3 = 100.0
# starting_amount4_4 = 100.0
# profit_list = []
# profit_list1 = []
# profit_list1_2 = []
# profit_list1_3 = []
# profit_list1_4 = []
# profit_list2 = []
# profit_list2_2 = []
# profit_list2_3 = []
# profit_list2_4 = []
# profit_list3 = []
# profit_list3_2 = []
# profit_list3_3 = []
# profit_list4 = []
# profit_list4_2 = []
# good_hits = []
# good_preds = []
# good_preds2 = []
# good_preds3 = []
# good_preds4 = []
# good_preds5 = []
# good_preds6 = []
# big_good_preds = []
# big_good_preds2 = []
# big_good_preds3 = []
# big_good_preds4 = []
# big_good_preds5 = []
# big_good_preds6 = []
# big_good_hits = []
# upordown = []
# upordown1 = []
# upordown1_2 = []
# upordown1_3 = []
# upordown1_4 = []
# upordown2 = []
# upordown2_2 = []
# upordown2_3 = []
# upordown2_4 = []
# upordown3 = []
# upordown3_2 = []
# upordown3_3 = []
# upordown3_4 = []
# upordown4 = []
# upordown4_2 = []
# upordown4_3 = []
# upordown4_4 = []
# upordown5 = []
# ===== END COMMENTED OUT SECTION =====

# ---- speed knobs ----
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
		"weight_list": [],
		"high_weight_list": [],
		"low_weight_list": [],
		"dirty": False,
	}
	try:
		data["memory_list"] = _read_text(f"memories_{tf_choice}.dat").replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split('~')
	except:
		data["memory_list"] = []
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

def write_threshold_sometimes(tf_choice, perfect_threshold, loop_i, every=200):
	"""Avoid writing neural_perfect_threshold_* every single loop."""
	last = _last_threshold_written.get(tf_choice)
	# write occasionally, or if it changed meaningfully
	if (loop_i % every != 0) and (last is not None) and (abs(perfect_threshold - last) < 0.05):
		return
	try:
		with open(f"neural_perfect_threshold_{tf_choice}.dat", "w+", encoding="utf-8") as f:
			f.write(str(perfect_threshold))
		_last_threshold_written[tf_choice] = perfect_threshold
	except:
		pass

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
		
		# If can't read root checksum, force update (safer to copy fresh)
		if not root_checksum:
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
			
			print(f"Trainer updated successfully. Restarting...")
			time.sleep(2)
			
			# Restart with same arguments
			python = sys.executable
			os.execl(python, python, *sys.argv)
		
		# Compare checksums - if different, update
		if root_checksum != my_checksum:
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
			
			print(f"Trainer updated successfully. Restarting...")
			time.sleep(2)
			
			# Restart with same arguments
			python = sys.executable
			os.execl(python, python, *sys.argv)
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

# Initialize number_of_candles based on number of timeframes being trained
# Each timeframe starts with minimum pattern size of 2 candles
number_of_candles = [2] * len(tf_choices)

# Debug print to confirm settings are loaded (will only show if debug mode is enabled)
try:
	if _is_debug_mode():
		print(f"[DEBUG] TRAINER: Loaded training settings from: {import_path}")
		print(f"[DEBUG] TRAINER: Active timeframes: {tf_choices}")
		print(f"[DEBUG] TRAINER: Pattern sizes (number_of_candles): {number_of_candles}")
except:
	pass
# --- GUI HUB INPUT (NO PROMPTS) ---
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
	updowncount = 0
	updowncount1 = 0
	updowncount1_2 = 0
	updowncount1_3 = 0
	updowncount1_4 = 0
	high_baseline_price_change_pct = 0.0
	low_baseline_price_change_pct = 0.0
	last_flipped = 'no'
	starting_amounth02 = 100.0
	starting_amounth05 = 100.0
	starting_amounth10 = 100.0
	starting_amounth20 = 100.0
	starting_amounth50 = 100.0
	starting_amount = 100.0
	starting_amount1 = 100.0
	starting_amount1_2 = 100.0
	starting_amount1_3 = 100.0
	starting_amount1_4 = 100.0
	starting_amount2 = 100.0
	starting_amount2_2 = 100.0
	starting_amount2_3 = 100.0
	starting_amount2_4 = 100.0
	starting_amount3 = 100.0
	starting_amount3_2 = 100.0
	starting_amount3_3 = 100.0
	starting_amount3_4 = 100.0
	starting_amount4 = 100.0
	starting_amount4_2 = 100.0
	starting_amount4_3 = 100.0
	starting_amount4_4 = 100.0
	profit_list = []
	profit_list1 = []
	profit_list1_2 = []
	profit_list1_3 = []
	profit_list1_4 = []
	profit_list2 = []
	profit_list2_2 = []
	profit_list2_3 = []
	profit_list2_4 = []
	profit_list3 = []
	profit_list3_2 = []
	profit_list3_3 = []
	profit_list4 = []
	profit_list4_2 = []
	good_hits = []
	good_preds = []
	good_preds2 = []
	good_preds3 = []
	good_preds4 = []
	good_preds5 = []
	good_preds6 = []
	big_good_preds = []
	big_good_preds2 = []
	big_good_preds3 = []
	big_good_preds4 = []
	big_good_preds5 = []
	big_good_preds6 = []
	big_good_hits = []
	upordown = []
	upordown1 = []
	upordown1_2 = []
	upordown1_3 = []
	upordown1_4 = []
	upordown2 = []
	upordown2_2 = []
	upordown2_3 = []
	upordown2_4 = []
	upordown3 = []
	upordown3_2 = []
	upordown3_3 = []
	upordown3_4 = []
	upordown4 = []
	upordown4_2 = []
	upordown4_3 = []
	upordown4_4 = []
	upordown5 = []
	debug_print(f"[DEBUG] TRAINER: Starting timeframe {tf_choices[the_big_index]} (index {the_big_index}/{len(tf_choices)-1})")
	tf_choice = tf_choices[the_big_index]
	debug_print(f"[DEBUG] TRAINER: Starting training cycle for {_arg_coin} on timeframe {tf_choice}...")
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
	success_rate = 85
	volume_success_rate = 60
	candles_to_predict = 1#droplet setting (Max is half of number_of_candles)(Min is 2)
	max_difference = .5
	preferred_difference = .4 #droplet setting (max profit_margin) (Min 0.01)
	min_good_matches = 1#droplet setting (Max 100) (Min 4)
	max_good_matches = 1#droplet setting (Max 100) (Min is min_good_matches)
	prediction_expander = 1.33
	prediction_expander2 = 1.5
	prediction_adjuster = 0.0
	diff_avg_setting = 0.01
	min_success_rate = 90
	histories = 'off'
	coin_choice_index = 0
	list_of_ys_count = 0
	last_difference_between = 0.0
	history_list = []
	history_list2 = []
	len_avg = []
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
	end_time = int(start_time-((1500*timeframe_minutes)*60))
	perc_comp = format((len(history_list2)/how_far_to_look_back)*100,'.2f')
	last_perc_comp = perc_comp+'kjfjakjdakd'
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
		
		perc_comp = format((len(history_list)/how_far_to_look_back)*100,'.2f')
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
		len_avg.append(current_change)
		list_len = len(history_list)
		last_perc_comp = perc_comp
		start_time = end_time
		end_time = int(start_time-((1500*timeframe_minutes)*60))
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
	
	# Start from halfway point in history for all timeframes
	index = int(len(history_list)/2)
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
	perfect_threshold = 1.0
	loop_i = 0  # counts inner training iterations (used to throttle disk IO)
	# Start with 1 candle and grow 1 at a time for maximum learning and self-correction
	price_list_length = 1
	debug_print(f"[DEBUG] TRAINER: Starting window size: {price_list_length} candles (total: {len(price_list)})")
	while True:
		while True:
			loop_i += 1
			matched_patterns_count = 0
			list_of_ys = []
			list_of_ys_count = 0
			next_coin = 'no'
			all_current_patterns = []
			memory_or_history = []
			memory_weights = []

			high_memory_weights = []
			low_memory_weights = []
			final_moves = 0.0
			high_final_moves = 0.0
			low_final_moves = 0.0
			memory_indexes = []
			matches_yep = []
			flipped = 'no'
			last_minute = int(time.time()/60)
			overunder = 'nothing'
			overunder2 = 'nothing'
			list_of_ys = []
			all_predictions = []
			all_preds = []
			high_all_predictions = []
			high_all_preds = []
			low_all_predictions = []
			low_all_preds = []
			try:
				open_price_list2 = []
				open_price_list_index = 0
				while True:
					open_price_list2.append(open_price_list[open_price_list_index])
					open_price_list_index += 1
					if open_price_list_index >= price_list_length:
						break
					else:
						continue
			except:
				break
			low_all_preds = []
			try:
				price_list2 = []
				price_list_index = 0
				while True:
					price_list2.append(price_list[price_list_index])
					price_list_index += 1
					if price_list_index >= price_list_length:
						break
					else:
						continue
			except:
				break
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
			index = 0
			index2 = index+1
			price_change_list = []
			while True:
				price_change = 100*((price_list2[index]-open_price_list2[index])/open_price_list2[index])
				price_change_list.append(price_change)
				index += 1
				if index >= len(price_list2):
					break
				else:
					continue
			index = 0
			index2 = index+1
			high_price_change_list = []
			while True:
				high_price_change = 100*((high_price_list2[index]-open_price_list2[index])/open_price_list2[index])
				high_price_change_list.append(high_price_change)
				index += 1
				if index >= len(price_list2):
					break
				else:
					continue
			index = 0
			index2 = index+1
			low_price_change_list = []
			while True:
				low_price_change = 100*((low_price_list2[index]-open_price_list2[index])/open_price_list2[index])
				low_price_change_list.append(low_price_change)
				index += 1
				if index >= len(price_list2):
					break
				else:
					continue
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

				# Mark as stopped (not ERROR - this was intentional user action)
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
					print()
					print(f'Timeframe: {timeframe}')
					
					# Rebuild price_change lists from current price_list2 at start of each iteration
					# This ensures they stay in sync even if cleared at end of previous iteration
					index = 0
					price_change_list = []
					while True:
						if index >= len(price_list2) or index >= len(open_price_list2):
							break
						price_change = 100*((price_list2[index]-open_price_list2[index])/open_price_list2[index])
						price_change_list.append(price_change)
						index += 1
					
					index = 0
					high_price_change_list = []
					while True:
						if index >= len(high_price_list2) or index >= len(open_price_list2):
							break
						high_price_change = 100*((high_price_list2[index]-open_price_list2[index])/open_price_list2[index])
						high_price_change_list.append(high_price_change)
						index += 1
					
					index = 0
					low_price_change_list = []
					while True:
						if index >= len(low_price_list2) or index >= len(open_price_list2):
							break
						low_price_change = 100*((low_price_list2[index]-open_price_list2[index])/open_price_list2[index])
						low_price_change_list.append(low_price_change)
						index += 1
					
					debug_print(f"[DEBUG] TRAINER: Loop iteration - price_list2={len(price_list2)}, price_change_list={len(price_change_list)}")
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
							else:
								continue
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
							else:
								continue
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
							else:
								continue
					except:
						PrintException()
					
					# Debug: Show what was built
					debug_print(f"[DEBUG] TRAINER: Patterns built - current={len(current_pattern)}, high={len(high_current_pattern)}, low={len(low_current_pattern)}, expected={number_of_candles[number_of_candles_index]-1}")
					debug_print(f"[DEBUG] TRAINER: Lists sizes - price_change={len(price_change_list)}, high_price_change={len(high_price_change_list)}, low_price_change={len(low_price_change_list)}")
					
					# Validate patterns were built successfully before continuing
					expected_length = number_of_candles[number_of_candles_index] - 1
					if (len(current_pattern) < expected_length or 
					    len(high_current_pattern) < expected_length or 
					    len(low_current_pattern) < expected_length):
						debug_print(f"[DEBUG] TRAINER: Insufficient pattern data - current:{len(current_pattern)}, high:{len(high_current_pattern)}, low:{len(low_current_pattern)}, expected:{expected_length}")
						debug_print(f"[DEBUG] TRAINER: Skipping this iteration, waiting for more data...")
						time.sleep(5)  # Prevents tight loop while waiting for candle data
						continue  # Skip this iteration and wait for more candles
					
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
								
								memory_pattern = memory_list[memory_index].split('{}')[0].replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').split(' ')
								checks = []
								check_dex = 0
								while True:
									# Check bounds before accessing
									if check_dex >= len(current_pattern) or check_dex >= len(memory_pattern):
										break
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
									check_dex += 1
								
								# Protect against division by zero if checks list is empty
								if len(checks) == 0:
									debug_print(f"[DEBUG] TRAINER: Empty checks list for memory {memory_index}, skipping")
									memory_index += 1
									continue
								
								diff_avg = sum(checks)/len(checks)
								if diff_avg <= perfect_threshold:
									any_perfect = 'yes'
									high_diff = float(memory_list[memory_index].split('{}')[1].replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').replace(' ',''))/100
									low_diff = float(memory_list[memory_index].split('{}')[2].replace("'","").replace(',','').replace('"','').replace(']','').replace('[','').replace(' ',''))/100
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
					all_current_patterns.append(current_pattern)
					
					# Adjust perfect_threshold based on match count
					# Too many matches (>20) = threshold too loose, tighten it
					# Too few matches (<20) = threshold too tight, loosen it
					# Bounded to reasonable range: 0.01% to 10.0% (prevents extremes)
					if len(unweighted) > 20:
						if perfect_threshold < 0.1:
							perfect_threshold -= 0.001
						else:
							perfect_threshold -= 0.01
					else:
						if perfect_threshold < 0.1:
							perfect_threshold += 0.001
						else:
							perfect_threshold += 0.01
					
					# Clamp to prevent negative or zero threshold (would break matching logic)
					# No upper limit during training - let each timeframe discover its natural threshold
					# Adaptive logic self-regulates based on match counts
					perfect_threshold = max(0.01, perfect_threshold)
					
					debug_print(f"[DEBUG] TRAINER: perfect_threshold adjusted to {perfect_threshold:.4f} (matches: {len(unweighted)})")
					
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
								if high_percent_difference_of_actuals >= high_baseline_price_change_pct+(high_baseline_price_change_pct*0.005) and percent_difference_of_actuals < high_baseline_price_change_pct:
									upordown3.append(1)
									upordown.append(1)
									upordown4.append(1)
									if len(upordown4) > 100:
										del upordown4[0]
									else:
										pass
								elif low_percent_difference_of_actuals <= low_baseline_price_change_pct-(low_baseline_price_change_pct*0.005) and percent_difference_of_actuals > low_baseline_price_change_pct:
									upordown.append(1)
									upordown3.append(1)
									upordown4.append(1)
									if len(upordown4) > 100:
										del upordown4[0]
									else:
										pass
								elif high_percent_difference_of_actuals >= high_baseline_price_change_pct+(high_baseline_price_change_pct*0.005) and percent_difference_of_actuals > high_baseline_price_change_pct:
									upordown3.append(0)
									upordown2.append(0)
									upordown.append(0)
									upordown4.append(0)
									if len(upordown4) > 100:
										del upordown4[0]
									else:
										pass
								elif low_percent_difference_of_actuals <= low_baseline_price_change_pct-(low_baseline_price_change_pct*0.005) and percent_difference_of_actuals < low_baseline_price_change_pct:
									upordown3.append(0)
									upordown2.append(0)
									upordown.append(0)
									upordown4.append(0)
									if len(upordown4) > 100:
										del upordown4[0]
									else:
										pass
								else:
									pass
							else:
								pass
							# Calculate and display bounce accuracy if we have data
							if len(upordown4) > 0:
								accuracy = (sum(upordown4)/len(upordown4))*100
								formatted = format(accuracy, '.2f').rstrip('0').rstrip('.')
								print(f'Bounce Accuracy for last 100 Over Limit Candles: {formatted}%')
								# Store accuracy in dict for this timeframe
								bounce_accuracy_dict[tf_choice] = accuracy
							try:
								print('current candle: '+str(len(price_list2)))
							except:
								pass
							try:
								print('Total Candles: '+str(int(len(price_list))))
							except:
								pass
						except:
							PrintException()
					else:
						pass
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
					prediction_adjuster = 0.0
					prediction_expander2 = 1.5
					ended_on = number_of_candles_index
					next_coin = 'yes'
					profit_hit = 'no'
					long_profit = 0
					short_profit = 0
					"""
					expander_move = input('Expander good? yes or new number: ')
					if expander_move == 'yes':
						pass
					else:
						prediction_expander = expander_move
						continue
					"""
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
									periodic_flush_if_needed(tf_choice, min_interval_seconds=10, max_memory_mb=50)
									
									# Check if we've processed all candles for this timeframe
									if len(price_list2) == len(price_list):
										# OPTIMIZATION: Prune weak patterns before saving (keeps files lean)
										# Remove patterns with weight < 0.1 (they contribute almost nothing to predictions)
										debug_print(f"[DEBUG] TRAINER: Pruning weak patterns (weight < 0.1) for {tf_choice}")
										try:
											_mem = _memory_cache.get(tf_choice)
											if _mem:
												original_count = len(_mem["memory_list"])
												# Find indices of patterns worth keeping (handles corrupt weights gracefully)
												good_indices = []
												for i in range(len(_mem["weight_list"])):
													try:
														if i < len(_mem["weight_list"]) and float(_mem["weight_list"][i]) >= 0.1:
															good_indices.append(i)
													except (ValueError, TypeError):
														pass  # Skip corrupt weight entries
												
												# Keep only good patterns
												_mem["memory_list"] = [_mem["memory_list"][i] for i in good_indices if i < len(_mem["memory_list"])]
												_mem["weight_list"] = [_mem["weight_list"][i] for i in good_indices if i < len(_mem["weight_list"])]
												_mem["high_weight_list"] = [_mem["high_weight_list"][i] for i in good_indices if i < len(_mem["high_weight_list"])]
												_mem["low_weight_list"] = [_mem["low_weight_list"][i] for i in good_indices if i < len(_mem["low_weight_list"])]
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
											break  # Exit all the way to outer loop to pick up next timeframe
										
										# If the_big_index >= len(tf_choices), fall through to final exit/save logic
										# NOTE: The ~80 lines of variable resets that were here have been removed.
										# They were unreachable in multi-timeframe mode (break above executes first)
										# and pointless before exit (variables reset at outer loop top anyway).
										
										# Jump directly to final checks (all variables properly initialized at outer loop)
										debug_print(f'Timeframe index: {the_big_index}')
										debug_print(f'Total timeframes: {len(tf_choices)}')
										debug_print(f'the_big_index={the_big_index}, len(tf_choices)={len(tf_choices)}, len(number_of_candles)={len(number_of_candles)}')
										debug_print(f'Timeframes in bounce_accuracy_dict: {list(bounce_accuracy_dict.keys())}')
										# Check if we've processed all timeframes
										if the_big_index >= len(tf_choices):
											if len(number_of_candles) == 1:
												print("Finished processing all timeframes (number_of_candles has only one entry). Exiting.")
												
												# CRITICAL: Flush all buffered data (thresholds + memory/weights) before marking complete
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
														print(f'Bounce Accuracy Results ({_arg_coin})')
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
														# Ensure all lists have data before attempting weight updates
														if len(moves) == 0 or len(high_moves) == 0 or len(low_moves) == 0:
															debug_print(f"[DEBUG] TRAINER: No matching patterns for weight update - creating new memory")
															
															# Create new memory when no patterns match
															# CRITICAL: Create a COPY of the pattern with this_diff appended
															# Do NOT modify all_current_patterns[highlowind] in place!
															new_pattern = all_current_patterns[highlowind] + [this_diff]
															
															mem_entry = str(new_pattern).replace("'","").replace(',','').replace('"','').replace(']','').replace('[','')+'{}'+str(high_this_diff)+'{}'+str(low_this_diff)
															
															# Add new pattern to memory cache
															_mem["memory_list"].append(mem_entry)
															_mem["weight_list"].append(1.0)  # Start with neutral weight
															_mem["high_weight_list"].append(1.0)
															_mem["low_weight_list"].append(1.0)
															_mem["dirty"] = True
															debug_print(f"[DEBUG] TRAINER: Added new pattern to memory (total: {len(_mem['memory_list'])})")
														else:
															# Find the minimum length across all lists to avoid index errors
															# Process as much valid data as possible even if lists got out of sync
															max_safe_index = min(
																len(moves), len(high_moves), len(low_moves),
																len(move_weights), len(high_move_weights), len(low_move_weights),
																len(perfect_dexs), len(unweighted)
															)
															
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
																
																debug_print(f"[DEBUG] TRAINER: Weight update - predicted_move_pct={predicted_move_pct:.4f}, threshold={predicted_move_pct_threshold:.4f}")
																
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
													except:
														# Catch unexpected errors during weight updates
														PrintException()
														pass

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
					coin_choice_index += 1
					history_list = []
					price_change_list = []
					current_pattern = []
				except (KeyboardInterrupt, SystemExit):
					raise
				except:
					PrintException()
					print("CRITICAL ERROR in main training loop. Training cannot continue safely.")
					mark_training_error("Main training loop critical error")
					sys.exit(1)
			# After candle processing loop - check if we need to restart outer loop
			if _restart_outer_loop:
				break
		# After window size loop - check if we need to restart outer loop
		if _restart_outer_loop:
			break
	# After middle loop - check if we need to restart outer loop
	if _restart_outer_loop:
		continue