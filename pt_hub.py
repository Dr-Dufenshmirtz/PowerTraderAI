from __future__ import annotations
"""
pt_hub.py - ApolloTrader Control Hub

Description:
This module implements the Tkinter GUI hub for ApolloTrader. It provides
widgets, layout helpers, and the bridge logic that reads/writes hub_data
files to monitor and control the `trainer`, `thinker`, and `trader`
processes. The hub displays predicted levels, trade status, and provides
controls to start/stop components.

Primary Repository: https://github.com/Dr-Dufenshmirtz/ApolloTrader
Primary Author: Dr Dufenshmirtz

Original Project: https://github.com/garagesteve1155/PowerTrader_AI
Original Author: Stephen Hughes (garagesteve1155)

Relevant behavior notes (informational only):

- Visualization:
    The hub presents predicted highs/lows (from the trainer/thinker) as
    colored horizontal lines on charts and provides live status panels for
    each coin.

- Coordination:
    The GUI communicates with other processes through status files and the
    `hub_data` directory so each component can run independently while the
    hub shows a unified view.
"""

import os
import sys
import json
import time
import math
import queue
import threading
import subprocess
import shutil
import glob
import bisect
import psutil
import base64
import requests

# Ensure CREATE_NO_WINDOW is available (added in Python 3.7)
if not hasattr(subprocess, 'CREATE_NO_WINDOW'):
    subprocess.CREATE_NO_WINDOW = 0x08000000

# Ensure STARTUPINFO flags are available for console hiding
if not hasattr(subprocess, 'STARTF_USESHOWWINDOW'):
    subprocess.STARTF_USESHOWWINDOW = 0x00000001
if not hasattr(subprocess, 'SW_HIDE'):
    subprocess.SW_HIDE = 0

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory

# Configure matplotlib for crisp text rendering
matplotlib.rcParams['text.antialiased'] = False
matplotlib.rcParams['text.hinting'] = 'none'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']

# Version: YY.MMDDHH (Year, Month, Day, Hour of last save)
VERSION = "26.011117"

# Windows DPAPI encryption helpers
def _encrypt_with_dpapi(data: str) -> bytes:
    """Encrypt string using Windows DPAPI (Data Protection API).
    Returns encrypted bytes that can only be decrypted by this Windows user account."""
    import ctypes
    from ctypes import wintypes
    
    class DATA_BLOB(ctypes.Structure):
        _fields_ = [('cbData', wintypes.DWORD), ('pbData', ctypes.POINTER(ctypes.c_char))]
    
    buffer = data.encode('utf-8')
    blob_in = DATA_BLOB(len(buffer), ctypes.cast(ctypes.c_char_p(buffer), ctypes.POINTER(ctypes.c_char)))
    blob_out = DATA_BLOB()
    
    if ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)
    ):
        encrypted = ctypes.string_at(blob_out.pbData, blob_out.cbData)
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)
        return encrypted
    else:
        raise RuntimeError("Failed to encrypt data with Windows DPAPI")

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

# Default theme colors (Modern Dark)
DARK_BG = "#0D1117"
DARK_BG2 = "#161B22"
DARK_PANEL = "#1A1F29"
DARK_PANEL2 = "#21262D"
DARK_BORDER = "#30363D"
DARK_FG = "#E6EDF3"
LIVE_OUTPUT_FG = "#00E5FF"  # Live output text color (bright blue like button highlights)
DARK_MUTED = "#8B949E"
DARK_ACCENT = "#FFD54F"
DARK_ACCENT2 = "#00E5FF"
DARK_SELECT_BG = "#2D333B"
DARK_SELECT_FG = "#FFD54F"

# Chart colors
CHART_CANDLE_UP = "#00E676"
CHART_CANDLE_DOWN = "#FF4081"
CHART_NEURAL_LONG = "#00E5FF"
CHART_NEURAL_SHORT = "#FFD54F"
CHART_SELL_LINE = "#00E676"
CHART_DCA_LINE = "#FF4081"
CHART_ASK_LINE = "#FFD54F"
CHART_BID_LINE = "#00E5FF"
CHART_ACCOUNT_LINE = "#00E5FF"

# Font settings (Modern Segoe UI)
LOG_FONT_FAMILY = "Cascadia Mono"
LOG_FONT_SIZE = 8
CHART_FONT_FAMILY = "Segoe UI"
CHART_LABEL_FONT_SIZE = 9

THEME_SETTINGS_FILE = "theme_settings.json"

DEFAULT_THEME = {
    "bg": DARK_BG,
    "bg2": DARK_BG2,
    "panel": DARK_PANEL,
    "panel2": DARK_PANEL2,
    "border": DARK_BORDER,
    "fg": DARK_FG,
    "live_output_fg": LIVE_OUTPUT_FG,
    "muted": DARK_MUTED,
    "accent": DARK_ACCENT,
    "accent2": DARK_ACCENT2,
    "select_bg": DARK_SELECT_BG,
    "select_fg": DARK_SELECT_FG,
    "chart_candle_up": CHART_CANDLE_UP,
    "chart_candle_down": CHART_CANDLE_DOWN,
    "chart_neural_long": CHART_NEURAL_LONG,
    "chart_neural_short": CHART_NEURAL_SHORT,
    "chart_sell_line": CHART_SELL_LINE,
    "chart_dca_line": CHART_DCA_LINE,
    "chart_ask_line": CHART_ASK_LINE,
    "chart_bid_line": CHART_BID_LINE,
    "chart_account_line": CHART_ACCOUNT_LINE,
    "font_family": "Segoe UI",
    "font_size": 11,
    "log_font_family": "Cascadia Mono",
    "log_font_size": 8,
    "chart_font_family": "Segoe UI",
    "chart_label_font_size": 9
}

def _load_theme_settings():
    """Load custom theme settings if theme path is provided, override global color constants."""
    global DARK_BG, DARK_BG2, DARK_PANEL, DARK_PANEL2, DARK_BORDER
    global DARK_FG, DARK_MUTED, DARK_ACCENT, DARK_ACCENT2, DARK_SELECT_BG, DARK_SELECT_FG
    global CHART_CANDLE_UP, CHART_CANDLE_DOWN, CHART_NEURAL_LONG, CHART_NEURAL_SHORT
    global CHART_SELL_LINE, CHART_DCA_LINE, CHART_ASK_LINE, CHART_BID_LINE, CHART_ACCOUNT_LINE
    global LOG_FONT_FAMILY, LOG_FONT_SIZE, CHART_FONT_FAMILY, CHART_LABEL_FONT_SIZE
    
    # Check if custom theme path is provided
    try:
        settings = _safe_read_json(SETTINGS_FILE)
        theme_path = (settings.get("theme_path") or "").strip() if settings else ""
        
        # If no theme path provided, use defaults
        if not theme_path:
            return
        
        # Resolve relative paths from project directory
        if not os.path.isabs(theme_path):
            theme_path = os.path.join(os.path.dirname(__file__), theme_path)
        
        if not os.path.isfile(theme_path):
            return
        
        theme = _safe_read_json(theme_path)
        if not theme:
            return
        
        # Override color constants
        global DARK_BG, DARK_BG2, DARK_PANEL, DARK_PANEL2, DARK_BORDER, DARK_FG, LIVE_OUTPUT_FG
        global DARK_MUTED, DARK_ACCENT, DARK_ACCENT2, DARK_SELECT_BG, DARK_SELECT_FG
        global CHART_CANDLE_UP, CHART_CANDLE_DOWN, CHART_NEURAL_LONG, CHART_NEURAL_SHORT
        global CHART_SELL_LINE, CHART_DCA_LINE, CHART_ASK_LINE, CHART_BID_LINE, CHART_ACCOUNT_LINE
        global LOG_FONT_FAMILY, LOG_FONT_SIZE, CHART_FONT_FAMILY, CHART_LABEL_FONT_SIZE
        
        DARK_BG = theme.get("bg", DARK_BG)
        DARK_BG2 = theme.get("bg2", DARK_BG2)
        DARK_PANEL = theme.get("panel", DARK_PANEL)
        DARK_PANEL2 = theme.get("panel2", DARK_PANEL2)
        DARK_BORDER = theme.get("border", DARK_BORDER)
        DARK_FG = theme.get("fg", DARK_FG)
        LIVE_OUTPUT_FG = theme.get("live_output_fg", DARK_FG)
        DARK_MUTED = theme.get("muted", DARK_MUTED)
        DARK_ACCENT = theme.get("accent", DARK_ACCENT)
        DARK_ACCENT2 = theme.get("accent2", DARK_ACCENT2)
        DARK_SELECT_BG = theme.get("select_bg", DARK_SELECT_BG)
        DARK_SELECT_FG = theme.get("select_fg", DARK_SELECT_FG)
        CHART_CANDLE_UP = theme.get("chart_candle_up", CHART_CANDLE_UP)
        CHART_CANDLE_DOWN = theme.get("chart_candle_down", CHART_CANDLE_DOWN)
        CHART_NEURAL_LONG = theme.get("chart_neural_long", CHART_NEURAL_LONG)
        CHART_NEURAL_SHORT = theme.get("chart_neural_short", CHART_NEURAL_SHORT)
        CHART_SELL_LINE = theme.get("chart_sell_line", CHART_SELL_LINE)
        CHART_DCA_LINE = theme.get("chart_dca_line", CHART_DCA_LINE)
        CHART_ASK_LINE = theme.get("chart_ask_line", CHART_ASK_LINE)
        CHART_BID_LINE = theme.get("chart_bid_line", CHART_BID_LINE)
        CHART_ACCOUNT_LINE = theme.get("chart_account_line", CHART_ACCOUNT_LINE)
        LOG_FONT_FAMILY = theme.get("log_font_family", LOG_FONT_FAMILY)
        LOG_FONT_SIZE = theme.get("log_font_size", LOG_FONT_SIZE)
        CHART_FONT_FAMILY = theme.get("chart_font_family", CHART_FONT_FAMILY)
        CHART_LABEL_FONT_SIZE = theme.get("chart_label_font_size", CHART_LABEL_FONT_SIZE)
    except Exception:
        pass

@dataclass
class _WrapItem:
    w: tk.Widget
    padx: Tuple[int, int] = (0, 0)
    pady: Tuple[int, int] = (0, 0)

class WrapFrame(ttk.Frame):

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._items: List[_WrapItem] = []
        self._reflow_pending = False
        self._in_reflow = False
        self.bind("<Configure>", self._schedule_reflow)

    def add(self, widget: tk.Widget, padx=(0, 0), pady=(0, 0)) -> None:
        self._items.append(_WrapItem(widget, padx=padx, pady=pady))
        self._schedule_reflow()

# `WrapFrame` is a small layout helper used by the GUI to place many
# small widgets on a wrapping row/column pattern. It reflows children
# automatically when the container resizes. Keeping the helper here keeps
# layout logic centralized for the GUI panels below.

    def clear(self, destroy_widgets: bool = True) -> None:

        for it in list(self._items):
            try:
                it.w.grid_forget()
            except Exception:
                pass
            if destroy_widgets:
                try:
                    it.w.destroy()
                except Exception:
                    pass
        self._items = []
        self._schedule_reflow()

    def _schedule_reflow(self, event=None) -> None:
        if self._reflow_pending:
            return
        self._reflow_pending = True
        self.after_idle(self._reflow)

    def _reflow(self) -> None:
        if self._in_reflow:
            self._reflow_pending = False
            return

        self._reflow_pending = False
        self._in_reflow = True
        try:
            # Determine the usable width inside the frame; subtract a small
            # padding value to avoid touching the edges. If the width is
            # extremely small we skip layout until a real size is available.
            width = self.winfo_width()
            if width <= 1:
                return
            usable_width = max(1, width - 6)

            for it in self._items:
                it.w.grid_forget()

            row = 0
            col = 0
            x = 0

            for it in self._items:
                # Compute the required width for this widget including
                # estimated horizontal padding; the constant `10` is a
                # small inter-widget gap used in the original layout.
                reqw = max(it.w.winfo_reqwidth(), it.w.winfo_width())

                needed = 10 + reqw + it.padx[0] + it.padx[1]

                if col > 0 and (x + needed) > usable_width:
                    row += 1
                    col = 0
                    x = 0

                it.w.grid(row=row, column=col, sticky="w", padx=it.padx, pady=it.pady)
                x += needed
                col += 1
        finally:
            self._in_reflow = False

class NeuralSignalTile(ttk.Frame):

    def __init__(self, parent: tk.Widget, coin: str, bar_height: int = 52, levels: int = 8):
        super().__init__(parent)
        self.coin = coin

        self._hover_on = False
        self._selected_on = False
        self._normal_canvas_bg = DARK_PANEL2
        self._hover_canvas_bg = DARK_PANEL
        self._selected_canvas_bg = DARK_PANEL
        self._normal_border = DARK_BORDER
        self._hover_border = DARK_ACCENT2
        self._selected_border = DARK_ACCENT2
        self._normal_fg = DARK_FG
        self._hover_fg = DARK_ACCENT2
        self._selected_fg = DARK_ACCENT2

        self._levels = max(2, int(levels))
        self._display_levels = self._levels - 1

        self._bar_h = int(bar_height)
        self._bar_w = 12
        self._gap = 16
        self._pad = 6

        self._base_fill = DARK_PANEL
        self._long_fill = CHART_NEURAL_LONG
        self._short_fill = CHART_NEURAL_SHORT

        self.title_lbl = ttk.Label(self, text=coin)
        self.title_lbl.pack(anchor="center")

        w = (self._pad * 2) + (self._bar_w * 2) + self._gap
        h = (self._pad * 2) + self._bar_h

        self.canvas = tk.Canvas(
            self,
            width=w,
            height=h,
            bg=self._normal_canvas_bg,
            highlightthickness=1,
            highlightbackground=self._normal_border,
        )
        self.canvas.pack(padx=2, pady=(2, 0))

        x0 = self._pad
        x1 = x0 + self._bar_w
        x2 = x1 + self._gap
        x3 = x2 + self._bar_w
        yb = self._pad + self._bar_h

        # Build segmented bars: 7 segments for levels 1..7 (level 0 is "no highlight")
        self._long_segs: List[int] = []
        self._short_segs: List[int] = []

        for seg in range(self._display_levels):
            # seg=0 is bottom segment (level 1), seg=display_levels-1 is top segment (level 7)
            y_top = int(round(yb - ((seg + 1) * self._bar_h / self._display_levels)))
            y_bot = int(round(yb - (seg * self._bar_h / self._display_levels)))

            self._long_segs.append(
                self.canvas.create_rectangle(
                    x0, y_top, x1, y_bot,
                    fill=self._base_fill,
                    outline=DARK_BORDER,
                    width=1,
                )
            )
            self._short_segs.append(
                self.canvas.create_rectangle(
                    x2, y_top, x3, y_bot,
                    fill=self._base_fill,
                    outline=DARK_BORDER,
                    width=1,
                )
            )

        # Marker lines based on entry signal config
        # LONG side marker (shows minimum long signal to start trades)
        trading_cfg = _load_trading_config()
        long_min = trading_cfg.get("entry_signals", {}).get("long_signal_min", 4)
        long_min = max(1, min(7, int(long_min)))  # Clamp to valid range 1-7
        
        long_trade_y = int(round(yb - ((long_min - 1) * self._bar_h / self._display_levels)))
        self._long_marker = self.canvas.create_line(x0, long_trade_y, x1, long_trade_y, fill=DARK_FG, width=2)
        
        # SHORT side marker (shows maximum short signal allowed for long entries)
        short_max = trading_cfg.get("entry_signals", {}).get("short_signal_max", 0)
        short_max = max(0, min(7, int(short_max)))  # Clamp to valid range 0-7
        
        short_trade_y = int(round(yb - (short_max * self._bar_h / self._display_levels)))
        self._short_marker = self.canvas.create_line(x2, short_trade_y, x3, short_trade_y, fill=DARK_FG, width=2)
        
        # Store last known config values for change detection
        self._last_long_min = long_min
        self._last_short_max = short_max

        self.value_lbl = ttk.Label(self, text="L:0 S:0")
        self.value_lbl.pack(anchor="center", pady=(1, 0))

        self.set_values(0, 0)

    def set_hover(self, on: bool) -> None:
        """Visually highlight the tile on hover (like a button hover state)."""
        if bool(on) == bool(self._hover_on):
            return
        self._hover_on = bool(on)
        self._apply_visual_state()

    def set_selected(self, on: bool) -> None:
        """Visually highlight the tile as the currently displayed chart coin."""
        if bool(on) == bool(self._selected_on):
            return
        self._selected_on = bool(on)
        self._apply_visual_state()

    def _apply_visual_state(self) -> None:
        """Apply visual styling based on selected/hover state priority."""
        try:
            # Hover changes both border and text (transient)
            # Selected changes only border, not text (persistent)
            # Priority: hover overrides selected for text color
            if self._hover_on:
                # Full hover effect: border + text color
                self.canvas.configure(
                    bg=self._hover_canvas_bg,
                    highlightbackground=self._hover_border,
                    highlightthickness=2,
                )
                self.title_lbl.configure(foreground=self._hover_fg)
                self.value_lbl.configure(foreground=self._hover_fg)
            elif self._selected_on:
                # Selected effect: border only, text stays normal
                self.canvas.configure(
                    bg=self._selected_canvas_bg,
                    highlightbackground=self._selected_border,
                    highlightthickness=2,
                )
                self.title_lbl.configure(foreground=self._normal_fg)
                self.value_lbl.configure(foreground=self._normal_fg)
            else:
                # Normal state
                self.canvas.configure(
                    bg=self._normal_canvas_bg,
                    highlightbackground=self._normal_border,
                    highlightthickness=1,
                )
                self.title_lbl.configure(foreground=self._normal_fg)
                self.value_lbl.configure(foreground=self._normal_fg)
        except Exception:
            pass

    def _clamp_level(self, value: Any) -> int:
        try:
            v = int(float(value))
        except Exception:
            v = 0
        return max(0, min(v, self._levels - 1))  # logical clamp: 0..7

    def _set_level(self, seg_ids: List[int], level: int, active_fill: str) -> None:
        # Reset all segments to base
        for rid in seg_ids:
            self.canvas.itemconfigure(rid, fill=self._base_fill)

        # Level 0 -> show nothing (no highlight)
        if level <= 0:
            return

        # Level 1..7 -> fill from bottom up through the current level
        idx = level - 1  # level 1 maps to seg index 0
        if idx < 0:
            return
        if idx >= len(seg_ids):
            idx = len(seg_ids) - 1

        for i in range(idx + 1):
            self.canvas.itemconfigure(seg_ids[i], fill=active_fill)

    def set_values(self, long_sig: Any, short_sig: Any) -> None:
        ls = self._clamp_level(long_sig)
        ss = self._clamp_level(short_sig)

        self.value_lbl.config(text=f"L:{ls} S:{ss}")
        self._set_level(self._long_segs, ls, self._long_fill)
        self._set_level(self._short_segs, ss, self._short_fill)

    def update_marker_lines(self) -> None:
        """Update marker line positions based on current trading config."""
        trading_cfg = _load_trading_config()
        long_min = trading_cfg.get("entry_signals", {}).get("long_signal_min", 4)
        long_min = max(1, min(7, int(long_min)))  # Clamp to valid range 1-7
        
        short_max = trading_cfg.get("entry_signals", {}).get("short_signal_max", 0)
        short_max = max(0, min(7, int(short_max)))  # Clamp to valid range 0-7
        
        # Only update if values changed
        if long_min != self._last_long_min or short_max != self._last_short_max:
            self._last_long_min = long_min
            self._last_short_max = short_max
            
            # Calculate geometry
            yb = self._pad + self._bar_h
            x0 = self._pad
            x1 = x0 + self._bar_w
            x2 = x1 + self._gap
            x3 = x2 + self._bar_w
            
            # Update LONG marker position
            long_trade_y = int(round(yb - ((long_min - 1) * self._bar_h / self._display_levels)))
            self.canvas.coords(self._long_marker, x0, long_trade_y, x1, long_trade_y)
            
            # Update SHORT marker position
            short_trade_y = int(round(yb - (short_max * self._bar_h / self._display_levels)))
            self.canvas.coords(self._short_marker, x2, short_trade_y, x3, short_trade_y)

# Default configuration settings and file paths
DEFAULT_SETTINGS = {
    "main_neural_dir": "",  # if blank, defaults to script directory
    "coins": ["BTC", "ETH", "XRP", "BNB", "DOGE"],
    "default_timeframe": "1hour",
    "timeframes": [
        "1min", "5min", "15min", "30min",
        "1hour", "2hour", "4hour", "8hour", "12hour",
        "1day", "1week"
    ],
    "candles_limit": 120,
    "ui_refresh_seconds": 1.0,
    "chart_refresh_seconds": 10.0,
    "hub_data_dir": "",  # if blank, defaults to <this_dir>/hub_data
    "script_neural_runner2": "pt_thinker.py",
    "script_neural_trainer": "pt_trainer.py",
    "script_trader": "pt_trader.py",
    "auto_start_scripts": False,
    "theme_path": "",
    "auto_switch": {
        "enabled": True,
        "threshold_pct": 2.0
    },
}

SETTINGS_FILE = "gui_settings.json"
TRADING_SETTINGS_FILE = "trading_settings.json"
TRAINING_SETTINGS_FILE = "training_settings.json"

# Default trading config
DEFAULT_TRADING_CONFIG = {
    "dca": {
        "levels": [-2.5, -5.0, -10.0, -20.0],
        "max_buys_per_window": 2,
        "window_hours": 24,
        "position_multiplier": 2.0
    },
    "profit_margin": {
        "trailing_gap_pct": 0.5,
        "target_no_dca_pct": 5.0,
        "target_with_dca_pct": 2.5,
        "stop_loss_pct": -40.0
    },
    "entry_signals": {
        "long_signal_min": 4,
        "short_signal_max": 0
    },
    "position_sizing": {
        "initial_allocation_pct": 0.01,
        "min_allocation_usd": 0.5,
        "max_concurrent_positions": 3
    },
    "timing": {
        "main_loop_delay_seconds": 0.5,
        "post_trade_delay_seconds": 30
    }
}

# Default training config
# Required timeframes that the Thinker uses for predictions and neural levels
# These MUST be trained for the system to function properly
REQUIRED_THINKER_TIMEFRAMES = [
    "1hour",
    "2hour",
    "4hour",
    "8hour",
    "12hour",
    "1day",
    "1week"
]

DEFAULT_TRAINING_CONFIG = {
    "staleness_days": 14,
    "auto_train_when_stale": True,
    "pruning_sigma_level": 2.0,
    "pid_kp": 0.5,
    "pid_ki": 0.005,
    "pid_kd": 0.2,
    "pid_integral_limit": 20,
    "min_threshold": 5.0,
    "max_threshold": 25.0,
    "pattern_size": 4,
    "weight_base_step": 0.25,
    "weight_step_cap_multiplier": 2.0,
    "weight_threshold_base": 0.1,
    "weight_threshold_min": 0.03,
    "weight_threshold_max": 0.2,
    "volatility_ewma_decay": 0.9,
    "weight_decay_rate": 0.001,
    "weight_decay_target": 1.0,
    "age_pruning_enabled": True,
    "age_pruning_percentile": 0.10,
    "age_pruning_weight_limit": 1.0,
    "timeframes": REQUIRED_THINKER_TIMEFRAMES
}

# Cache for trading config
_trading_config_cache = {
    "mtime": None,
    "config": dict(DEFAULT_TRADING_CONFIG)
}

# Cache for training config
_training_config_cache = {
    "mtime": None,
    "config": dict(DEFAULT_TRAINING_CONFIG)
}

def _safe_read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _safe_write_json(path: str, data: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def _load_trading_config() -> dict:
    """Load trading config with caching (similar to gui_settings pattern)."""
    try:
        if not os.path.isfile(TRADING_SETTINGS_FILE):
            return dict(_trading_config_cache["config"])
        
        mtime = os.path.getmtime(TRADING_SETTINGS_FILE)
        if _trading_config_cache["mtime"] == mtime:
            return dict(_trading_config_cache["config"])
        
        data = _safe_read_json(TRADING_SETTINGS_FILE)
        if not data:
            return dict(_trading_config_cache["config"])
        
        # Merge with defaults to ensure all keys exist
        config = dict(DEFAULT_TRADING_CONFIG)
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

def _load_training_config() -> dict:
    """Load training config with caching (similar to gui_settings pattern)."""
    try:
        if not os.path.isfile(TRAINING_SETTINGS_FILE):
            return dict(_training_config_cache["config"])
        
        mtime = os.path.getmtime(TRAINING_SETTINGS_FILE)
        if _training_config_cache["mtime"] == mtime:
            return dict(_training_config_cache["config"])
        
        data = _safe_read_json(TRAINING_SETTINGS_FILE)
        if not data:
            return dict(_training_config_cache["config"])
        
        # Merge with defaults to ensure all keys exist
        config = dict(DEFAULT_TRAINING_CONFIG)
        for key in data:
            if isinstance(data[key], dict) and isinstance(config.get(key), dict):
                config[key].update(data[key])
            else:
                config[key] = data[key]
        
        _training_config_cache["mtime"] = mtime
        _training_config_cache["config"] = config
        return dict(config)
    except Exception:
        return dict(_training_config_cache["config"])

def _read_trade_history_jsonl(path: str) -> List[dict]:
    """
    Reads hub_data/trade_history.jsonl written by pt_trader.py.
    Returns a list of dicts (only buy/sell rows).
    """
    out: List[dict] = []
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        obj = json.loads(ln)
                        side = str(obj.get("side", "")).lower().strip()
                        if side not in ("buy", "sell"):
                            continue
                        out.append(obj)
                    except Exception:
                        continue
    except Exception:
        pass
    return out

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _fmt_money(x: float) -> str:
    """Format a USD *amount* (account value, position value, etc.) as dollars with 2 decimals."""
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "N/A"

def _fmt_price(x: Any) -> str:
    """
    Format a USD *price/level* with dynamic decimals based on magnitude.
    Examples:
      50234.12   -> $50,234.12
      123.4567   -> $123.457
      1.234567   -> $1.2346
      0.06234567 -> $0.062346
      0.00012345 -> $0.00012345
    """
    try:
        if x is None:
            return "N/A"

        v = float(x)
        if not math.isfinite(v):
            return "N/A"

        sign = "-" if v < 0 else ""
        av = abs(v)

        # Choose decimals by magnitude (more detail for smaller prices).
        if av >= 1000:
            dec = 2
        elif av >= 100:
            dec = 3
        elif av >= 1:
            dec = 4
        elif av >= 0.1:
            dec = 5
        elif av >= 0.01:
            dec = 6
        elif av >= 0.001:
            dec = 7
        else:
            dec = 8

        s = f"{av:,.{dec}f}"

        return f"{sign}${s}"
    except Exception:
        return "N/A"

def _fmt_pct(x: float) -> str:
    try:
        return f"{float(x):+.2f}%"
    except Exception:
        return "N/A"

def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

# Helper functions for neural folder detection and coin subfolder mapping
def build_coin_folders(main_dir: str, coins: List[str]) -> Dict[str, str]:
    """
    All coins use subfolders, including BTC.
    Auto-detects coin folders inside main_dir.

    Returns { "BTC": "...", "ETH": "...", ... }
    """
    out: Dict[str, str] = {}
    main_dir = main_dir or os.getcwd()

    # Auto-detect subfolders for all coins
    if os.path.isdir(main_dir):
        for name in os.listdir(main_dir):
            p = os.path.join(main_dir, name)
            if not os.path.isdir(p):
                continue
            sym = name.upper().strip()
            if sym in coins:
                out[sym] = p

    # Fallbacks for missing ones
    for c in coins:
        c = c.upper().strip()
        if c not in out:
            out[c] = os.path.join(main_dir, c)  # best-effort fallback

    return out

def read_price_levels_from_html(path: str) -> List[Tuple[str, float]]:
    """
    pt_thinker writes labeled boundaries into low_bound_prices.txt / high_bound_prices.txt.

    Format: "1h:43210.1 2h:43100.0 4h:42950.5"

    Returns list of (timeframe_label, price) tuples.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        if not raw:
            return []

        vals: List[Tuple[str, float]] = []
        for tok in raw.split():
            try:
                if ':' in tok:
                    label, price_str = tok.split(':', 1)
                    v = float(price_str)
                    
                    # Filter sentinel values
                    if v <= 0 or v >= 9e15:
                        continue
                    
                    vals.append((label, v))
            except Exception:
                pass
        
        return vals
    except Exception:
        return []

def filter_neural_levels_by_timeframe(levels: List[Tuple[str, float]], selected_tf: str) -> List[Tuple[str, float]]:
    """
    Filter neural levels to only show timeframes equal to or larger than selected_tf.
    This helps declutter the chart by hiding shorter-timeframe predictions.
    
    Special case: For sub-hourly charts (1min, 5min, 15min, 30min), show ALL neural levels
    since those timeframes don't have their own predictions but benefit from seeing all longer-term context.
    """
    # Timeframe hierarchy (smallest to largest)
    tf_order = ['1min', '5min', '15min', '30min', '1h', '2h', '4h', '8h', '12h', '1d', '1w']
    # Also accept 'hour' variants
    tf_map = {
        '1hour': '1h', '2hour': '2h', '4hour': '4h', '8hour': '8h', '12hour': '12h',
        '1day': '1d', '1week': '1w'
    }
    
    # Normalize selected timeframe
    norm_selected = tf_map.get(selected_tf, selected_tf)
    
    try:
        selected_idx = tf_order.index(norm_selected)
    except ValueError:
        # If unknown timeframe, show all levels
        return levels
    
    # If viewing sub-hourly chart (< 1h), show all neural levels for context
    if selected_idx < 4:  # 1h is at index 4
        return levels
    
    filtered = []
    for label, price in levels:
        # Normalize level label
        norm_label = tf_map.get(label, label)
        try:
            label_idx = tf_order.index(norm_label)
            # Only include if this level's timeframe >= selected timeframe
            if label_idx >= selected_idx:
                filtered.append((label, price))
        except ValueError:
            # If label not in hierarchy, include it
            filtered.append((label, price))
    
    return filtered

def read_int_from_file(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        return int(float(raw))
    except Exception:
        return 0

def read_short_signal(folder: str) -> int:
    txt = os.path.join(folder, "short_dca_signal.txt")
    if os.path.isfile(txt):
        return read_int_from_file(txt)
    else:
        return 0

# KuCoin API integration for fetching candlestick chart data
class CandleFetcher:
    """
    Uses kucoin-python if available; otherwise falls back to KuCoin REST via requests.
    """
    def __init__(self):
        self._mode = "kucoin_client"
        self._market = None
        try:
            from kucoin.client import Market # type: ignore
            self._market = Market(url="https://api.kucoin.com")
        except Exception:
            self._mode = "rest"
            self._market = None

        if self._mode == "rest":
            import requests # local import
            self._requests = requests

        # Small in-memory cache to keep timeframe switching snappy.
        # key: (pair, timeframe, limit) -> (saved_time_epoch, candles)
        self._cache: Dict[Tuple[str, str, int], Tuple[float, List[dict]]] = {}
        self._cache_ttl_seconds: float = 10.0

    def get_klines(self, symbol: str, timeframe: str, limit: int = 120) -> List[dict]:
        """
        Returns candles oldest->newest as:
          [{"ts": int, "open": float, "high": float, "low": float, "close": float}, ...]
        """
        symbol = symbol.upper().strip()

        # Your neural uses USDT pairs on KuCoin (ex: BTC-USDT)
        pair = f"{symbol}-USDT"
        limit = int(limit or 0)

        now = time.time()
        cache_key = (pair, timeframe, limit)
        cached = self._cache.get(cache_key)
        if cached and (now - float(cached[0])) <= float(self._cache_ttl_seconds):
            return cached[1]

        # rough window (timeframe-dependent) so we get enough candles
        tf_seconds = {
            "1min": 60, "5min": 300, "15min": 900, "30min": 1800,
            "1hour": 3600, "2hour": 7200, "4hour": 14400, "8hour": 28800, "12hour": 43200,
            "1day": 86400, "1week": 604800
        }.get(timeframe, 3600)

        end_at = int(now)
        start_at = end_at - (tf_seconds * max(200, (limit + 50) if limit else 250))

        if self._mode == "kucoin_client" and self._market is not None:
            try:
                # IMPORTANT: limit the server response by passing startAt/endAt.
                # This avoids downloading a huge default kline set every switch.
                try:
                    raw = self._market.get_kline(pair, timeframe, startAt=start_at, endAt=end_at)  # type: ignore
                except Exception:
                    # fallback if that client version doesn't accept kwargs
                    raw = self._market.get_kline(pair, timeframe)  # returns newest->oldest

                candles: List[dict] = []
                for row in raw:
                    # KuCoin kline row format:
                    # [time, open, close, high, low, volume, turnover]
                    ts = int(float(row[0]))
                    o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
                    candles.append({"ts": ts, "open": o, "high": h, "low": l, "close": c})
                candles.sort(key=lambda x: x["ts"])
                if limit and len(candles) > limit:
                    candles = candles[-limit:]

                self._cache[cache_key] = (now, candles)
                return candles
            except Exception:
                return []

        # REST fallback
        try:
            url = "https://api.kucoin.com/api/v1/market/candles"
            params = {"symbol": pair, "type": timeframe, "startAt": start_at, "endAt": end_at}
            resp = self._requests.get(url, params=params, timeout=10)
            j = resp.json()
            data = j.get("data", [])  # newest->oldest
            candles: List[dict] = []
            for row in data:
                ts = int(float(row[0]))
                o = float(row[1]); c = float(row[2]); h = float(row[3]); l = float(row[4])
                candles.append({"ts": ts, "open": o, "high": h, "low": l, "close": c})
            candles.sort(key=lambda x: x["ts"])
            if limit and len(candles) > limit:
                candles = candles[-limit:]

            self._cache[cache_key] = (now, candles)
            return candles
        except Exception:
            return []

# Tkinter chart widget for displaying candlestick charts with neural level overlays
class CandleChart(ttk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        fetcher: CandleFetcher,
        coin: str,
        settings_getter,
        trade_history_path: str,
    ):
        super().__init__(parent)
        self.fetcher = fetcher
        self.coin = coin
        self.settings_getter = settings_getter
        self.trade_history_path = trade_history_path

        self.timeframe_var = tk.StringVar(value=self.settings_getter()["default_timeframe"])

        top = ttk.Frame(self)
        top.pack(fill="x", padx=6, pady=6)

        ttk.Label(top, text=f"{coin} chart").pack(side="left")

        ttk.Label(top, text="Timeframe:").pack(side="left", padx=(12, 4))
        self.tf_combo = ttk.Combobox(
            top,
            textvariable=self.timeframe_var,
            values=self.settings_getter()["timeframes"],
            state="readonly",
            width=10,
        )
        self.tf_combo.pack(side="left")

        # Debounce rapid timeframe changes so redraws don't stack
        self._tf_after_id = None

        def _debounced_tf_change(*_):
            try:
                if self._tf_after_id:
                    self.after_cancel(self._tf_after_id)
            except Exception:
                pass

            def _do():
                # Ask the hub to refresh charts on the next tick (single refresh)
                try:
                    self.event_generate("<<TimeframeChanged>>", when="tail")
                except Exception:
                    pass

            self._tf_after_id = self.after(120, _do)

        self.tf_combo.bind("<<ComboboxSelected>>", _debounced_tf_change)

        self.neural_status_label = ttk.Label(top, text="Neural: N/A")
        self.neural_status_label.pack(side="left", padx=(12, 0))

        self.last_update_label = ttk.Label(top, text="Last neurals: N/A")
        self.last_update_label.pack(side="right")
        
        # Priority alert label (shown when coin is close to trigger point)
        self.priority_alert_label = ttk.Label(
            top,
            text="",
            font=("Segoe UI", 10, "bold"),
            foreground="#FF6B00"  # Orange color for visibility
        )
        self.priority_alert_label.pack(side="right", padx=(0, 12))

        # Figure
        # IMPORTANT: keep a stable DPI and resize the figure to the widget's pixel size.
        # On Windows scaling, trying to "sync DPI" via winfo_fpixels("1i") can produce the
        # exact right-side blank/covered region you're seeing.
        self.fig = Figure(figsize=(6.5, 3.5), dpi=100)
        self.fig.patch.set_facecolor(DARK_BG)

        # CandleChart padding: Reserve bottom space for date+time x tick labels,
        # right space for price labels (Bid/Ask/DCA/Sell), and top space for title.
        self.fig.subplots_adjust(left=0.08, bottom=0.15, right=0.88, top=0.94)

        self.ax = self.fig.add_subplot(111)
        self._apply_dark_chart_style()
        self.ax.set_title(f"{coin}", color=DARK_FG, fontsize=CHART_LABEL_FONT_SIZE, fontfamily=CHART_FONT_FAMILY)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        canvas_w = self.canvas.get_tk_widget()
        canvas_w.configure(bg=DARK_BG)

        # Remove horizontal padding here so the chart widget truly fills the container.
        canvas_w.pack(fill="both", expand=True, padx=0, pady=(0, 6))

        # Keep the matplotlib figure EXACTLY the same pixel size as the Tk widget.
        # FigureCanvasTkAgg already sizes its backing PhotoImage to e.width/e.height.
        # Multiplying by tk scaling here makes the renderer larger than the PhotoImage,
        # which produces the "blank/covered strip" on the right.
        self._last_canvas_px = (0, 0)
        self._resize_after_id = None

        def _on_canvas_configure(e):
            try:
                w = int(e.width)
                h = int(e.height)
                if w <= 1 or h <= 1:
                    return

                if (w, h) == self._last_canvas_px:
                    return
                self._last_canvas_px = (w, h)

                dpi = float(self.fig.get_dpi() or 100.0)
                self.fig.set_size_inches(w / dpi, h / dpi, forward=True)

                # Debounce redraws during live resize
                if self._resize_after_id:
                    try:
                        self.after_cancel(self._resize_after_id)
                    except Exception:
                        pass
                self._resize_after_id = self.after_idle(self.canvas.draw_idle)
            except Exception:
                pass

        canvas_w.bind("<Configure>", _on_canvas_configure, add="+")

        self._last_refresh = 0.0

    def _apply_dark_chart_style(self) -> None:
        """Apply dark styling (called on init and after every ax.clear())."""
        try:
            self.fig.patch.set_facecolor(DARK_BG)
            self.ax.set_facecolor(DARK_PANEL)
            self.ax.tick_params(colors=DARK_FG, labelsize=CHART_LABEL_FONT_SIZE)
            for spine in self.ax.spines.values():
                spine.set_color(DARK_BORDER)
            self.ax.grid(True, color=DARK_BORDER, linewidth=0.6, alpha=1.0)
        except Exception:
            pass

    def refresh(
        self,
        coin_folders: Dict[str, str],
        current_buy_price: Optional[float] = None,
        current_sell_price: Optional[float] = None,
        trail_line: Optional[float] = None,
        dca_line_price: Optional[float] = None,
    ) -> None:

        cfg = self.settings_getter()

        tf = self.timeframe_var.get().strip()
        limit = int(cfg.get("candles_limit", 120))

        candles = self.fetcher.get_klines(self.coin, tf, limit=limit)

        folder = coin_folders.get(self.coin, "")
        low_path = os.path.join(folder, "low_bound_prices.txt")
        high_path = os.path.join(folder, "high_bound_prices.txt")

        # --- Cached neural reads (per path, by mtime) ---
        if not hasattr(self, "_neural_cache"):
            self._neural_cache = {}  # path -> (mtime, value)

        def _cached(path: str, loader, default):
            try:
                mtime = os.path.getmtime(path)
            except Exception:
                return default
            hit = self._neural_cache.get(path)
            if hit and hit[0] == mtime:
                return hit[1]
            v = loader(path)
            self._neural_cache[path] = (mtime, v)
            return v

        long_levels_all = _cached(low_path, read_price_levels_from_html, []) if folder else []
        short_levels_all = _cached(high_path, read_price_levels_from_html, []) if folder else []
        
        # Filter to only show timeframes >= selected timeframe (declutter chart)
        long_levels_labeled = filter_neural_levels_by_timeframe(long_levels_all, tf)
        short_levels_labeled = filter_neural_levels_by_timeframe(short_levels_all, tf)

        long_sig_path = os.path.join(folder, "long_dca_signal.txt")
        long_sig = _cached(long_sig_path, read_int_from_file, 0) if folder else 0
        short_sig = read_short_signal(folder) if folder else 0

        # --- Avoid full ax.clear() (expensive). Just clear artists. ---
        try:
            self.ax.lines.clear()
            self.ax.patches.clear()
            self.ax.collections.clear()  # scatter dots live here
            self.ax.texts.clear()        # labels/annotations live here
        except Exception:
            # fallback if matplotlib version lacks .clear() on these lists
            self.ax.cla()
            self._apply_dark_chart_style()

        if not candles:
            self.ax.set_title(f"{self.coin} ({tf}) - no candles", color=DARK_FG, fontsize=CHART_LABEL_FONT_SIZE, fontfamily=CHART_FONT_FAMILY)
            self.canvas.draw_idle()
            return

        # Candlestick drawing (green up / red down) - batch rectangles
        xs = getattr(self, "_xs", None)
        if not xs or len(xs) != len(candles):
            xs = list(range(len(candles)))
            self._xs = xs

        rects = []
        for i, c in enumerate(candles):
            o = float(c["open"])
            cl = float(c["close"])
            h = float(c["high"])
            l = float(c["low"])

            up = cl >= o
            candle_color = CHART_CANDLE_UP if up else CHART_CANDLE_DOWN

            # wick
            self.ax.plot([i, i], [l, h], linewidth=1, color=candle_color)

            # body
            bottom = min(o, cl)
            height = abs(cl - o)
            if height < 1e-12:
                height = 1e-12

            rects.append(
                Rectangle(
                    (i - 0.35, bottom),
                    0.7,
                    height,
                    facecolor=candle_color,
                    edgecolor=candle_color,
                    linewidth=1,
                    alpha=1.0,
                )
            )

        for r in rects:
            self.ax.add_patch(r)

        # Lock y-limits to candle range so overlay lines can go offscreen without expanding the chart.
        try:
            y_low = min(float(c["low"]) for c in candles)
            y_high = max(float(c["high"]) for c in candles)
            pad = (y_high - y_low) * 0.03
            if not math.isfinite(pad) or pad <= 0:
                pad = max(abs(y_low) * 0.001, 1e-6)
            self.ax.set_ylim(y_low - pad, y_high + pad)
        except Exception:
            pass

        # Overlay Neural levels (blue long, orange short)
        for label, lv in long_levels_labeled:
            try:
                self.ax.axhline(y=float(lv), linewidth=1, color=CHART_NEURAL_LONG, alpha=1.0)
            except Exception:
                pass

        for label, lv in short_levels_labeled:
            try:
                self.ax.axhline(y=float(lv), linewidth=1, color=CHART_NEURAL_SHORT, alpha=1.0)
            except Exception:
                pass

        # Overlay Trailing PM line (sell) and next DCA line
        try:
            if trail_line is not None and float(trail_line) > 0:
                self.ax.axhline(y=float(trail_line), linewidth=1.5, color=CHART_SELL_LINE, alpha=1.0)
        except Exception:
            pass

        try:
            if dca_line_price is not None and float(dca_line_price) > 0:
                self.ax.axhline(y=float(dca_line_price), linewidth=1.5, color=CHART_DCA_LINE, alpha=1.0)
        except Exception:
            pass

        # Overlay current ask/bid prices
        try:
            if current_buy_price is not None and float(current_buy_price) > 0:
                self.ax.axhline(y=float(current_buy_price), linewidth=1.5, color=CHART_ASK_LINE, alpha=1.0)
        except Exception:
            pass

        try:
            if current_sell_price is not None and float(current_sell_price) > 0:
                self.ax.axhline(y=float(current_sell_price), linewidth=1.5, color=CHART_BID_LINE, alpha=1.0)
        except Exception:
            pass

        # Right-side price labels (so you can read Bid/Ask/DCA/Sell at a glance)
        try:
            trans = blended_transform_factory(self.ax.transAxes, self.ax.transData)
            used_y: List[float] = []
            boxed_y: List[float] = []  # Track positions with boxes (need more clearance)
            y0, y1 = self.ax.get_ylim()
            y_pad = max((y1 - y0) * 0.012, 1e-9)
            y_box_pad = y_pad * 3.0  # Boxes need 3x more space

            def _label_right(y: Optional[float], tag: str, color: str, fill_color: Optional[str] = None) -> None:
                if y is None:
                    return
                try:
                    yy = float(y)
                    if (not math.isfinite(yy)) or yy <= 0:
                        return
                except Exception:
                    return

                # Nudge labels apart if levels are very close
                for prev in used_y:
                    if abs(yy - prev) < y_pad:
                        yy = prev + y_pad
                used_y.append(yy)
                boxed_y.append(yy)  # Mark this as a boxed label

                self.ax.text(
                    1.01,
                    yy,
                    f"{tag} {_fmt_price(yy)}",
                    transform=trans,
                    ha="left",
                    va="center",
                    fontsize=CHART_LABEL_FONT_SIZE,
                    fontfamily=CHART_FONT_FAMILY,
                    color=color,
                    bbox=dict(
                        facecolor=fill_color if fill_color else DARK_BG2,
                        edgecolor=color,
                        boxstyle="square,pad=0.3",
                        alpha=1.0,
                    ),
                    zorder=20,
                    clip_on=False,
                )

            def _label_neural(y: Optional[float], tag: str, color: str) -> None:
                """Label for neural levels - no box, only show if visible on chart and no overlap"""
                if y is None:
                    return
                try:
                    yy = float(y)
                    if (not math.isfinite(yy)) or yy <= 0:
                        return
                    # Skip if outside visible chart range
                    if yy < y0 or yy > y1:
                        return
                except Exception:
                    return

                # Check if this label would overlap with existing labels - if so, skip it
                for prev in used_y:
                    pad_to_use = y_box_pad if prev in boxed_y else y_pad
                    if abs(yy - prev) < pad_to_use:
                        return  # Skip this label to avoid overlap
                
                used_y.append(yy)

                self.ax.text(
                    1.01,
                    yy,
                    tag,
                    transform=trans,
                    ha="left",
                    va="center",
                    fontsize=CHART_LABEL_FONT_SIZE,
                    fontfamily=CHART_FONT_FAMILY,
                    color=color,
                    zorder=20,
                    clip_on=False,
                )

            # Map to your terminology: Ask=buy line, Bid=sell line
            _label_right(current_buy_price, "ASK", CHART_ASK_LINE)
            _label_right(current_sell_price, "BID", CHART_BID_LINE)
            _label_right(dca_line_price, "DCA", CHART_DCA_LINE)
            _label_right(trail_line, "SELL", CHART_SELL_LINE)
            
            # Add neural level labels (no boxes, only if visible)
            for label, lv in long_levels_labeled:
                _label_neural(lv, label, CHART_NEURAL_LONG)
            
            for label, lv in short_levels_labeled:
                _label_neural(lv, label, CHART_NEURAL_SHORT)
        except Exception:
            pass

        # --- Trade dots (BUY / DCA / SELL) for THIS coin only ---
        try:
            trades = _read_trade_history_jsonl(self.trade_history_path) if self.trade_history_path else []
            if trades:
                candle_ts = [int(c["ts"]) for c in candles]  # oldest->newest
                t_min = float(candle_ts[0])
                t_max = float(candle_ts[-1])

                for tr in trades:
                    sym = str(tr.get("symbol", "")).upper()
                    base = sym.split("-")[0].strip() if sym else ""
                    if base != self.coin.upper().strip():
                        continue

                    side = str(tr.get("side", "")).lower().strip()
                    tag = str(tr.get("tag") or "").upper().strip()

                    if side == "buy":
                        label = "DCA" if tag == "DCA" else "BUY"
                        color = CHART_ASK_LINE if tag == "DCA" else CHART_DCA_LINE
                    elif side == "sell":
                        label = "SELL"
                        color = CHART_SELL_LINE
                    else:
                        continue

                    tts = tr.get("ts", None)
                    if tts is None:
                        continue
                    try:
                        tts = float(tts)
                    except Exception:
                        continue
                    if tts < t_min or tts > t_max:
                        continue

                    i = bisect.bisect_left(candle_ts, tts)
                    if i <= 0:
                        idx = 0
                    elif i >= len(candle_ts):
                        idx = len(candle_ts) - 1
                    else:
                        idx = i if abs(candle_ts[i] - tts) < abs(tts - candle_ts[i - 1]) else (i - 1)

                    # y = trade price if present, else candle close
                    y = None
                    try:
                        p = tr.get("price", None)
                        if p is not None and float(p) > 0:
                            y = float(p)
                    except Exception:
                        y = None
                    if y is None:
                        try:
                            y = float(candles[idx].get("close", 0.0))
                        except Exception:
                            y = None
                    if y is None:
                        continue

                    x = idx
                    self.ax.scatter([x], [y], s=35, color=color, zorder=6)
                    self.ax.annotate(
                        label,
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=CHART_LABEL_FONT_SIZE,
                        fontfamily=CHART_FONT_FAMILY,
                        color=DARK_FG,
                        zorder=7,
                    )
        except Exception:
            pass

        self.ax.set_xlim(-0.5, (len(candles) - 0.5) + 0.6)

        self.ax.set_title(f"{self.coin} ({tf})", color=DARK_FG, fontsize=CHART_LABEL_FONT_SIZE, fontfamily=CHART_FONT_FAMILY)

        # Format y-axis with dollar signs (tiered precision: $1000+=0, $100-999=2, $10-99=3, <$10=4)
        try:
            def format_price(y, _pos):
                if y >= 1000:
                    return f"${y:,.0f}"
                elif y >= 100:
                    return f"${y:,.2f}"
                elif y >= 10:
                    return f"${y:,.3f}"
                else:
                    return f"${y:,.4f}"
            self.ax.yaxis.set_major_formatter(FuncFormatter(format_price))
        except Exception:
            pass

        # x tick labels (date + time) - evenly spaced, never overlapping duplicates
        n = len(candles)
        want = 5  # keep it readable even when the window is narrow
        if n <= want:
            idxs = list(range(n))
        else:
            step = (n - 1) / float(want - 1)
            idxs = []
            last = -1
            for j in range(want):
                i = int(round(j * step))
                if i <= last:
                    i = last + 1
                if i >= n:
                    i = n - 1
                idxs.append(i)
                last = i

        tick_x = [xs[i] for i in idxs]
        tick_lbl = [
            time.strftime("%Y-%m-%d\n%H:%M", time.localtime(int(candles[i].get("ts", 0))))
            for i in idxs
        ]

        try:
            self.ax.minorticks_off()
            self.ax.set_xticks(tick_x)
            self.ax.set_xticklabels(tick_lbl)
            self.ax.tick_params(axis="x", labelsize=CHART_LABEL_FONT_SIZE)
        except Exception:
            pass

        self.canvas.draw_idle()

        self.neural_status_label.config(text=f"Neural: long = {long_sig}/{len(long_levels_labeled)}  short = {short_sig}/{len(short_levels_labeled)}")

        # show file update time if possible
        last_ts = None
        try:
            if os.path.isfile(low_path):
                last_ts = os.path.getmtime(low_path)
            elif os.path.isfile(high_path):
                last_ts = os.path.getmtime(high_path)
        except Exception:
            last_ts = None

        if last_ts:
            self.last_update_label.config(text=f"Last neurals: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_ts))}")
        else:
            self.last_update_label.config(text="Last neurals: N/A")

# Account value chart widget for displaying account value over time with trade markers
class AccountValueChart(ttk.Frame):
    """
    Matplotlib-based chart displaying account value over time.
    
    Features:
    - Line plot with shaded fill underneath (gradient effect)
    - Trade markers (BUY/DCA/SELL dots with coin labels)
    - Dynamic y-axis formatting ($1000+ = no decimals, <$1000 = 2 decimals)
    - Time-series x-axis with date + time labels
    - Mtime-based caching (only redraws when data files change)
    - Downsampling to max 250 points (prevents performance degradation)
    - Auto-resizing to match container dimensions exactly
    
    Data Sources:
    - account_value_history.jsonl: Time-series of total account value
    - trade_history.jsonl: Trade events for marker overlays
    
    Performance Optimizations:
    - Mtime caching: Skip redraw if files unchanged (checked every tick)
    - Bucket averaging: Downsample large datasets by averaging buckets
    - Debounced resize: Batch resize events during window dragging
    - Sparse plotting: Max 250 data points regardless of history size
    
    Visual Design:
    - Dark theme with cyan (#00E5FF) line and 15% transparent fill
    - Trade dots color-coded: Green (SELL), Purple (DCA), Red (BUY)
    - Coin-tagged markers: "BTC BUY", "ETH SELL", etc.
    - Grid lines for readability without clutter
    """
    
    def __init__(self, parent: tk.Widget, history_path: str, trade_history_path: str, max_points: int = 250):
        """
        Initialize account value chart widget.
        
        Args:
            parent: Parent Tkinter widget
            history_path: Path to account_value_history.jsonl
            trade_history_path: Path to trade_history.jsonl (for marker overlays)
            max_points: Maximum data points to display (hard-capped at 250)
        """
        super().__init__(parent)
        self.history_path = history_path
        self.trade_history_path = trade_history_path
        # Hard-cap to 250 points max (prevents matplotlib slowdown with large datasets)
        self.max_points = min(int(max_points or 0) or 250, 250)
        # Mtime cache: tracks last modification time of data files to skip unnecessary redraws
        self._last_mtime: Optional[float] = None

        top = ttk.Frame(self)
        top.pack(fill="x", padx=6, pady=6)

        ttk.Label(top, text="Account value").pack(side="left")
        self.last_update_label = ttk.Label(top, text="Last status: N/A")
        self.last_update_label.pack(side="right")
        
        # Priority alert label (shown when coin is close to trigger point)
        self.priority_alert_label = ttk.Label(
            top,
            text="",
            font=("Segoe UI", 10, "bold"),
            foreground="#FF6B00"  # Orange color for visibility
        )
        self.priority_alert_label.pack(side="right", padx=(0, 12))

        self.fig = Figure(figsize=(6.5, 3.5), dpi=100)
        self.fig.patch.set_facecolor(DARK_BG)

        # AccountValueChart padding: Reserve bottom space for date+time x tick labels,
        # right space for price labels, and top space for title.
        self.fig.subplots_adjust(left=0.08, bottom=0.15, right=0.94, top=0.94)

        self.ax = self.fig.add_subplot(111)
        self._apply_dark_chart_style()
        self.ax.set_title("Account Value", color=DARK_FG, fontsize=CHART_LABEL_FONT_SIZE, fontfamily=CHART_FONT_FAMILY)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        canvas_w = self.canvas.get_tk_widget()
        canvas_w.configure(bg=DARK_BG)

        # Remove horizontal padding here so the chart widget truly fills the container.
        canvas_w.pack(fill="both", expand=True, padx=0, pady=(0, 6))

        # Keep the matplotlib figure EXACTLY the same pixel size as the Tk widget.
        # FigureCanvasTkAgg already sizes its backing PhotoImage to e.width/e.height.
        # Multiplying by tk scaling here makes the renderer larger than the PhotoImage,
        # which produces the "blank/covered strip" on the right.
        self._last_canvas_px = (0, 0)
        self._resize_after_id = None

        def _on_canvas_configure(e):
            try:
                w = int(e.width)
                h = int(e.height)
                if w <= 1 or h <= 1:
                    return

                if (w, h) == self._last_canvas_px:
                    return
                self._last_canvas_px = (w, h)

                dpi = float(self.fig.get_dpi() or 100.0)
                self.fig.set_size_inches(w / dpi, h / dpi, forward=True)

                # Debounce redraws during live resize
                if self._resize_after_id:
                    try:
                        self.after_cancel(self._resize_after_id)
                    except Exception:
                        pass
                self._resize_after_id = self.after_idle(self.canvas.draw_idle)
            except Exception:
                pass

        canvas_w.bind("<Configure>", _on_canvas_configure, add="+")

    def _apply_dark_chart_style(self) -> None:
        try:
            self.fig.patch.set_facecolor(DARK_BG)
            self.ax.set_facecolor(DARK_PANEL)
            self.ax.tick_params(colors=DARK_FG, labelsize=CHART_LABEL_FONT_SIZE)
            for spine in self.ax.spines.values():
                spine.set_color(DARK_BORDER)
            self.ax.grid(True, color=DARK_BORDER, linewidth=0.6, alpha=1.0)
        except Exception:
            pass

    def refresh(self) -> None:
        """
        Redraw account value chart from data files (mtime-cached).
        
        Performance Strategy:
        - Mtime caching: Only redraws if account_value_history.jsonl OR trade_history.jsonl changed
        - Called every 10 seconds (default) by _tick() loop
        - Early return if no file changes detected (zero overhead)
        
        Data Processing Pipeline:
        1. Load & Parse: Read all lines from account_value_history.jsonl
        2. Validate: Drop invalid/malformed data points (NaN, negative values, bad timestamps)
        3. Sort: Ensure chronological order (handles out-of-order writes)
        4. Dedupe: Remove duplicate timestamps (keeps latest value per timestamp)
        5. Downsample: Average into max 250 buckets (prevents matplotlib slowdown)
        6. Plot: Draw line + shaded fill + trade markers
        
        Downsampling Algorithm (Bucket Averaging):
        - Divides N points into 250 evenly-sized buckets
        - Averages timestamp and value within each bucket
        - Result: Smooth chart that responds to new data without visual "jumping"
        - Alternative (rejected): Random sampling causes visual instability
        
        Trade Markers:
        - Overlays colored dots for BUY/DCA/SELL trades from all coins
        - Uses bisect to find nearest account value point for each trade timestamp
        - Color coding: Green (SELL), Purple (DCA), Red/Magenta (BUY)
        - Labels: "BTC BUY", "ETH SELL", etc. (coin-tagged for clarity)
        
        Visual Formatting:
        - Y-axis: Smart dollar formatting ($1000+ = no decimals, <$1000 = 2 decimals)
        - X-axis: 5 evenly-spaced time labels with date + time (YYYY-MM-DD HH:MM:SS)
        - Title: "Account Value ($X.XX)" - shows current value
        - Last update label: "Last: HH:MM:SS" - shows timestamp of most recent data point
        - Fill: 15% transparent cyan under line (extends to chart edges)
        
        Edge Cases:
        - No data: Displays "Account Value - no data" title
        - Single data point: Still draws chart (matplotlib handles gracefully)
        - Large history (>250 points): Downsampled to prevent slowdown
        - Duplicate timestamps: Latest value wins (handles rapid updates)
        """
        path = self.history_path

        # Mtime-Based Cache Check
        # Only redraw if account history OR trade history files have changed
        # This prevents expensive matplotlib redraws when no new data exists
        try:
            m_hist = os.path.getmtime(path)
        except Exception:
            m_hist = None

        try:
            m_trades = os.path.getmtime(self.trade_history_path) if self.trade_history_path else None
        except Exception:
            m_trades = None

        # Use max(mtimes) so changes to either file trigger refresh
        candidates = [m for m in (m_hist, m_trades) if m is not None]
        mtime = max(candidates) if candidates else None

        # Early return: No file changes since last refresh
        if mtime is not None and self._last_mtime == mtime:
            return
        self._last_mtime = mtime

        # Data Loading
        # Load all account value history points (timestamp, value) tuples
        points: List[Tuple[float, float]] = []

        try:
            if os.path.isfile(path):
                # Read the FULL history so the chart shows from the very beginning
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()

                for ln in lines:
                    try:
                        obj = json.loads(ln)
                        ts = obj.get("ts", None)
                        v = obj.get("total_account_value", None)
                        if ts is None or v is None:
                            continue

                        tsf = float(ts)
                        vf = float(v)

                        # Drop obviously invalid points early
                        if (not math.isfinite(tsf)) or (not math.isfinite(vf)) or (vf <= 0.0):
                            continue

                        points.append((tsf, vf))
                    except Exception:
                        continue
        except Exception:
            points = []

        # Clean up account value history to remove invalid data points and single-tick anomalies
        if points:
            # Ensure chronological order
            points.sort(key=lambda x: x[0])

            # De-dupe identical timestamps (keep the latest occurrence)
            dedup: List[Tuple[float, float]] = []
            for tsf, vf in points:
                if dedup and tsf == dedup[-1][0]:
                    dedup[-1] = (tsf, vf)
                else:
                    dedup.append((tsf, vf))
            points = dedup

        # Downsample to <= 250 points by AVERAGING buckets instead of skipping points.
        # This keeps the chart visually stable when new nearby values arrive.
        max_keep = min(max(2, int(self.max_points or 250)), 250)
        n = len(points)

        if n > max_keep:
            bucket_size = n / float(max_keep)
            new_points: List[Tuple[float, float]] = []

            for i in range(max_keep):
                start = int(i * bucket_size)
                end = int((i + 1) * bucket_size)
                if end <= start:
                    end = start + 1
                if start >= n:
                    break
                if end > n:
                    end = n

                bucket = points[start:end]
                if not bucket:
                    continue

                # Average timestamp and account value within the bucket
                avg_ts = sum(p[0] for p in bucket) / len(bucket)
                avg_val = sum(p[1] for p in bucket) / len(bucket)

                new_points.append((avg_ts, avg_val))

            points = new_points

        # clear artists (fast) / fallback to cla()
        try:
            self.ax.lines.clear()
            self.ax.patches.clear()
            self.ax.collections.clear()  # scatter dots live here
            self.ax.texts.clear()        # labels/annotations live here
        except Exception:
            self.ax.cla()
            self._apply_dark_chart_style()

        if not points:
            self.ax.set_title("Account Value - no data", color=DARK_FG, fontsize=CHART_LABEL_FONT_SIZE, fontfamily=CHART_FONT_FAMILY)
            self.last_update_label.config(text="Last status: N/A")
            self.canvas.draw_idle()
            return

        xs = list(range(len(points)))
        # Only show cent-level changes (hide sub-cent noise)
        ys = [round(p[1], 2) for p in points]

        # Draw the line
        self.ax.plot(xs, ys, linewidth=1.5, color=CHART_ACCOUNT_LINE)
        
        # Fill area under the line (shaded) - creates a gradient effect
        self.ax.fill_between(xs, ys, alpha=0.15, color=CHART_ACCOUNT_LINE)

        # --- Trade dots (BUY / DCA / SELL) for ALL coins ---
        try:
            trades = _read_trade_history_jsonl(self.trade_history_path) if self.trade_history_path else []
            if trades:
                ts_list = [float(p[0]) for p in points]  # matches xs/ys indices
                t_min = ts_list[0]
                t_max = ts_list[-1]

                for tr in trades:
                    # Determine label/color
                    side = str(tr.get("side", "")).lower().strip()
                    tag = str(tr.get("tag", "")).upper().strip()

                    if side == "buy":
                        action_label = "DCA" if tag == "DCA" else "BUY"
                        color = CHART_ASK_LINE if tag == "DCA" else CHART_DCA_LINE
                    elif side == "sell":
                        action_label = "SELL"
                        color = CHART_SELL_LINE
                    else:
                        continue

                    # Prefix with coin (so the dot says which coin it is)
                    sym = str(tr.get("symbol", "")).upper().strip()
                    coin_tag = (sym.split("-")[0].split("/")[0].strip() if sym else "") or (sym or "?")
                    label = f"{coin_tag} {action_label}"

                    tts = tr.get("ts")
                    try:
                        tts = float(tts)
                    except Exception:
                        continue
                    if tts < t_min or tts > t_max:
                        continue

                    # nearest account-value point
                    i = bisect.bisect_left(ts_list, tts)
                    if i <= 0:
                        idx = 0
                    elif i >= len(ts_list):
                        idx = len(ts_list) - 1
                    else:
                        idx = i if abs(ts_list[i] - tts) < abs(tts - ts_list[i - 1]) else (i - 1)

                    x = idx
                    y = ys[idx]

                    self.ax.scatter([x], [y], s=30, color=color, zorder=6)
                    self.ax.annotate(
                        label,
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=CHART_LABEL_FONT_SIZE,
                        fontfamily=CHART_FONT_FAMILY,
                        color=DARK_FG,
                        zorder=7,
                    )

        except Exception:
            pass

        # Account value formatting (real dollars: $1000+=0 decimals, <$1000=2 decimals)
        try:
            def format_price(y, _pos):
                if y >= 1000:
                    return f"${y:,.0f}"
                else:
                    return f"${y:,.2f}"
            self.ax.yaxis.set_major_formatter(FuncFormatter(format_price))
        except Exception:
            pass

        # x labels: show a few timestamps (date + time) - evenly spaced, never overlapping duplicates
        n = len(points)
        want = 5
        if n <= want:
            idxs = list(range(n))
        else:
            step = (n - 1) / float(want - 1)
            idxs = []
            last = -1
            for j in range(want):
                i = int(round(j * step))
                if i <= last:
                    i = last + 1
                if i >= n:
                    i = n - 1
                idxs.append(i)
                last = i

        tick_x = [xs[i] for i in idxs]
        tick_lbl = [time.strftime("%Y-%m-%d\n%H:%M", time.localtime(points[i][0])) for i in idxs]
        try:
            self.ax.minorticks_off()
            self.ax.set_xticks(tick_x)
            self.ax.set_xticklabels(tick_lbl)
            self.ax.tick_params(axis="x", labelsize=CHART_LABEL_FONT_SIZE)
        except Exception:
            pass

        # Set x-axis limits to align with data points (fill extends to edges)
        self.ax.set_xlim(xs[0], xs[-1])

        try:
            self.ax.set_title(f"Account Value ({_fmt_money(ys[-1])})", color=DARK_FG, fontsize=CHART_LABEL_FONT_SIZE, fontfamily=CHART_FONT_FAMILY)
        except Exception:
            self.ax.set_title("Account Value", color=DARK_FG, fontsize=CHART_LABEL_FONT_SIZE, fontfamily=CHART_FONT_FAMILY)

        try:
            self.last_update_label.config(
                text=f"Last status: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(points[-1][0]))}"
            )
        except Exception:
            self.last_update_label.config(text="Last status: N/A")

        self.canvas.draw_idle()

# Main Hub application class and supporting dataclasses
@dataclass
class ProcInfo:
    name: str
    path: str
    proc: Optional[subprocess.Popen] = None

@dataclass
class LogProc:
    """
    A running process with a live log queue for stdout/stderr lines.
    """
    info: ProcInfo
    log_q: "queue.Queue[str]"
    thread: Optional[threading.Thread] = None
    is_trainer: bool = False
    coin: Optional[str] = None

class ApolloHub(tk.Tk):
    """
    Main GUI application for ApolloTrader.
    
    Responsibilities:
    - Display real-time account status, charts, and trade history
    - Launch and monitor trainer, thinker, and trader subprocesses
    - Provide user controls for starting/stopping components
    - Manage configuration via settings dialogs
    - Coordinate "Autopilot" mode (auto-train → auto-start thinking/trading)
    - Fetch and display Robinhood account data independently of trader
    
    Architecture:
    - Uses Tkinter for GUI with forced dark theme
    - Communicates with subprocesses via JSON files in hub_data/
    - Employs mtime caching to minimize file I/O and UI redraws
    - Runs periodic _tick() loop for UI updates (default 1Hz)
    """
    
    def __init__(self):
        """
        Initialize the Hub GUI and set up all components.
        
        Initialization sequence:
        1. Create main window with custom icon and minimum size
        2. Load settings from gui_settings.json
        3. Apply dark theme styling
        4. Set up file paths for data exchange with subprocesses
        5. Initialize state tracking variables
        6. Build UI layout (menus, panels, charts, logs)
        7. Start tick loop and schedule startup tasks
        """
        super().__init__()
        self.title(f"Apollo Trader {VERSION.split('.')[0]}")
        
        # Set initial size and ensure window opens within visible screen area
        win_width = 1400
        win_height = 820
        
        # Get screen dimensions
        try:
            screen_width = self.winfo_screenwidth()
            screen_height = self.winfo_screenheight()
            
            # Clamp window size to screen if needed (leave 50px margin)
            actual_width = min(win_width, screen_width - 50)
            actual_height = min(win_height, screen_height - 50)
            
            # Center window on screen
            x = max(0, (screen_width - actual_width) * 2 // 3)
            y = max(0, (screen_height - actual_height) // 3)
            
            self.geometry(f"{actual_width}x{actual_height}+{x}+{y}")
        except Exception:
            # Fallback to simple geometry without position
            self.geometry("1400x820")

        # Set custom taskbar icon if at.ico exists in project directory
        # This gives the app a professional appearance in Windows taskbar
        icon_path = os.path.join(os.path.dirname(__file__), "at.ico")
        if os.path.isfile(icon_path):
            try:
                self.iconbitmap(icon_path)
                # Set AppUserModelID so Windows taskbar displays our custom icon
                # instead of Python's default icon (Windows-specific)
                import ctypes
                myappid = 'apollotrader.cryptoai.26'  # Arbitrary unique string
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            except Exception:
                pass  # Silently fail if icon can't be loaded

        # Set minimum window size to prevent UI from becoming unusable when resized
        # These dimensions ensure all panels remain visible and functional
        self.minsize(980, 640)
        
        # Menu Checkbox Variables
        # These BooleanVars are bound to menu checkboxes and persist to gui_settings.json
        
        # Auto-start: Launches thinker/trader automatically when Hub opens
        self.auto_start_var = tk.BooleanVar()
        
        # Debug mode: Enables verbose console output for troubleshooting
        self.debug_mode_var = tk.BooleanVar()
        
        # Simulation mode: Prevents real trades (testing/development safety)
        self.simulation_mode_var = tk.BooleanVar()

        # UI Performance
        # Debounce map prevents rapid-fire panedwindow resize events from causing lag
        # Keys are pane widget IDs, values are after() callback IDs for cancellation
        self._paned_clamp_after_ids: Dict[str, str] = {}

        # Configuration Loading
        # Load user settings from gui_settings.json (coins, directories, API keys, etc.)
        self.settings = self._load_settings()
        
        # Load custom theme (cyber/midnight) if user has selected one
        # This overrides default color constants defined at module level
        _load_theme_settings()

        # Apply forced dark mode styling to all Tk/ttk widgets
        # Must run after theme load to respect custom color schemes
        self._apply_forced_dark_mode()

        # Directory Structure
        # Project directory: Location of pt_hub.py (base for relative paths)
        self.project_dir = os.path.abspath(os.path.dirname(__file__))

        # Hub data directory: Shared communication channel between all processes
        # Contains JSON/JSONL files written by trader/thinker, read by hub
        hub_dir = self.settings.get("hub_data_dir") or os.path.join(self.project_dir, "hub_data")
        self.hub_dir = os.path.abspath(hub_dir)
        _ensure_dir(self.hub_dir)

        # Inter-Process Communication Files
        # These files are the primary data exchange mechanism between Hub and subprocesses
        
        # Written by pt_trader.py - Account status and open positions
        self.trader_status_path = os.path.join(self.hub_dir, "trader_status.json")
        
        # Written by pt_trader.py - Complete trade history (append-only log)
        self.trade_history_path = os.path.join(self.hub_dir, "trade_history.jsonl")
        
        # Written by pt_trader.py - Realized profit/loss ledger
        self.pnl_ledger_path = os.path.join(self.hub_dir, "pnl_ledger.json")
        
        # Written by pt_trader.py AND pt_hub.py - Account value time series for chart
        # Hub writes when trader is stopped, trader writes when running (no conflicts)
        self.account_value_history_path = os.path.join(self.hub_dir, "account_value_history.jsonl")

        # Written by pt_thinker.py - Signals when thinker is ready to provide predictions
        # Used by "Start All" to coordinate thinker → trader startup sequence
        self.runner_ready_path = os.path.join(self.hub_dir, "runner_ready.json")

        # Startup Coordination State
        # When "Start All" is clicked, Hub starts thinker first, then waits for runner_ready.json
        # before starting trader. This flag tracks whether trader is waiting for thinker.
        self._auto_start_trader_pending = False

        # Autopilot Mode State
        # Autopilot orchestrates: auto-train untrained coins → start thinker → start trader
        # This provides a "one-click" experience for first-time users
        self._auto_mode_active = False          # Currently in Autopilot workflow
        self._auto_mode_phase = ""              # Current phase: "training", "thinking", "trading"
        self._auto_mode_pending_coins: set = set()  # Coins waiting for training to complete

        # Auto-Retrain State (Proactive Training)
        # Monitors training data staleness and auto-retrains before it becomes critical
        # Prevents thinker/trader from using expired AI memory patterns
        self._auto_retraining_active = False       # Currently in auto-retrain workflow
        self._auto_retrain_pending_coins: set = set()  # Coins being retrained
        self._last_auto_retrain_check = 0.0       # Timestamp of last staleness check

        # Live Trading State Cache
        # Stores most recent position data for each coin to overlay on charts
        # Format: {coin: {current_buy_price, current_sell_price, trail_line, dca_line_price}}
        self._last_positions: Dict[str, dict] = {}

        # Trading config cache (mtime-based to avoid repeated file reads)
        self._cached_trading_config: Optional[dict] = None
        self._cached_trading_config_mtime: Optional[float] = None

        # Auto-Switch State
        # Tracks manual chart switches to prevent auto-switch from overriding user intent
        # Auto-switch pauses for 2 minutes after manual selection
        self._last_manual_chart_switch: float = 0.0  # Timestamp of last manual chart selection
        self._manual_override_duration: float = 120.0  # 2 minutes in seconds
        
        # Log Auto-Scroll State
        # Tracks if user has scrolled away from bottom to prevent forced auto-scroll
        self._trainer_log_user_scrolled_away = False
        self._runner_log_user_scrolled_away = False
        self._trader_log_user_scrolled_away = False

        # account value chart widget (created in _build_layout)
        self.account_chart = None

        # coin folders (neural outputs)
        self.coins = [c.upper().strip() for c in self.settings["coins"]]

        # On startup (like on Settings-save), create missing alt folders and copy the trainer into them.
        self._ensure_alt_coin_folders_and_trainer_on_startup()
        
        # Clear stale training status from interrupted sessions
        self._clear_stale_training_status()

        # Rebuild folder map after potential folder creation
        self.coin_folders = build_coin_folders(self.settings["main_neural_dir"], self.coins)

        # scripts
        self.proc_neural = ProcInfo(
            name="Thinker",
            path=os.path.abspath(os.path.join(self.project_dir, self.settings["script_neural_runner2"]))
        )
        self.proc_trader = ProcInfo(
            name="Trader",
            path=os.path.abspath(os.path.join(self.project_dir, self.settings["script_trader"]))
        )

        self.proc_trainer_path = os.path.abspath(os.path.join(self.project_dir, self.settings["script_neural_trainer"]))

        # live log queues
        self.runner_log_q: "queue.Queue[str]" = queue.Queue()
        self.trader_log_q: "queue.Queue[str]" = queue.Queue()

        # trainers: coin -> LogProc
        self.trainers: Dict[str, LogProc] = {}
        # trainer log history: coin -> list of log lines (for switching between coins)
        self.trainer_log_history: Dict[str, list] = {}

        self.fetcher = CandleFetcher()

        self._build_menu()
        self._build_layout()
        
        # Set auto-start variable after settings are loaded
        self.auto_start_var.set(bool(self.settings.get("auto_start_scripts", False)))
        
        # Set debug mode variable after settings are loaded
        self.debug_mode_var.set(bool(self.settings.get("debug_mode", False)))
        
        # Set simulation mode variable after settings are loaded
        self.simulation_mode_var.set(bool(self.settings.get("simulation_mode", False)))
        
        # Show simulation mode banner immediately if enabled (don't wait for first account fetch)
        if bool(self.settings.get("simulation_mode", False)):
            try:
                self.lbl_simulation_banner.config(text="⚠️ SIMULATION MODE ⚠️")
                self.lbl_simulation_banner.pack(anchor="w", padx=6, pady=(4, 4), before=self.lbl_acct_total_value)
            except Exception:
                pass

        # Refresh charts immediately when a timeframe is changed (don't wait for the 10s throttle).
        self.bind_all("<<TimeframeChanged>>", self._on_timeframe_changed)

        self._last_chart_refresh = 0.0

        if bool(self.settings.get("auto_start_scripts", False)):
            self.start_all_scripts()

        self.after(250, self._tick)
        
        # Display accuracy results on startup (after GUI is ready)
        # Increased delay to ensure all widgets are fully initialized
        self.after(500, self._display_startup_accuracy)
        
        # Fetch initial account info if API keys are present (displays immediately without starting trader)
        # Delay increased to ensure all widgets are initialized
        self.after(1500, self._fetch_initial_account_info)
        
        # Check if first-time setup is needed and open settings dialog
        self.after(1000, self._check_first_time_setup)
        
        # Auto-start thinker if coins are trained (delayed to ensure UI is ready)
        self.after(2000, self._auto_start_thinker_if_trained)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # Theme styling methods
    def _apply_forced_dark_mode(self) -> None:
        """Force a single, global, non-optional dark theme."""
        # Root background (handles the areas behind ttk widgets)
        try:
            self.configure(bg=DARK_BG)
        except Exception:
            pass

        # Defaults for classic Tk widgets (Text/Listbox/Menu) created later
        try:
            self.option_add("*Text.background", DARK_PANEL)
            self.option_add("*Text.foreground", DARK_FG)
            self.option_add("*Text.insertBackground", DARK_FG)
            self.option_add("*Text.selectBackground", DARK_SELECT_BG)
            self.option_add("*Text.selectForeground", DARK_SELECT_FG)

            self.option_add("*Listbox.background", DARK_PANEL)
            self.option_add("*Listbox.foreground", DARK_FG)
            self.option_add("*Listbox.selectBackground", DARK_SELECT_BG)
            self.option_add("*Listbox.selectForeground", DARK_SELECT_FG)

            self.option_add("*Menu.background", DARK_BG2)
            self.option_add("*Menu.foreground", DARK_FG)
            self.option_add("*Menu.activeBackground", DARK_SELECT_BG)
            self.option_add("*Menu.activeForeground", DARK_SELECT_FG)
        except Exception:
            pass

        style = ttk.Style(self)

        # Pick a theme that is actually recolorable (Windows 'vista' theme ignores many color configs)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # Base defaults
        try:
            style.configure(".", background=DARK_BG, foreground=DARK_FG)
        except Exception:
            pass

        # Containers / text
        for name in ("TFrame", "TLabel", "TCheckbutton", "TRadiobutton"):
            try:
                style.configure(name, background=DARK_BG, foreground=DARK_FG)
            except Exception:
                pass

        try:
            style.configure("TLabelframe", background=DARK_BG, foreground=DARK_FG, bordercolor=DARK_BORDER)
            style.configure("TLabelframe.Label", background=DARK_BG, foreground=DARK_ACCENT)
        except Exception:
            pass

        try:
            style.configure("TSeparator", background=DARK_BORDER)
        except Exception:
            pass

        # Buttons
        try:
            style.configure(
                "TButton",
                background=DARK_BG2,
                foreground=DARK_FG,
                bordercolor=DARK_BORDER,
                focusthickness=1,
                focuscolor=DARK_ACCENT,
                padding=(10, 6),
            )
            style.map(
                "TButton",
                background=[
                    ("active", DARK_PANEL2),
                    ("pressed", DARK_PANEL),
                    ("disabled", DARK_BG2),
                ],
                foreground=[
                    ("active", DARK_ACCENT),
                    ("disabled", DARK_MUTED),
                ],
                bordercolor=[
                    ("active", DARK_ACCENT2),
                    ("focus", DARK_ACCENT),
                ],
            )
        except Exception:
            pass

        # Entries / combos
        try:
            style.configure(
                "TEntry",
                fieldbackground=DARK_PANEL,
                foreground=DARK_FG,
                bordercolor=DARK_BORDER,
                insertcolor=DARK_FG,
            )
        except Exception:
            pass

        try:
            style.configure(
                "TCombobox",
                fieldbackground=DARK_PANEL,
                background=DARK_PANEL,
                foreground=DARK_FG,
                bordercolor=DARK_BORDER,
                arrowcolor=DARK_ACCENT,
            )
            style.map(
                "TCombobox",
                fieldbackground=[
                    ("readonly", DARK_PANEL),
                    ("focus", DARK_PANEL2),
                ],
                foreground=[("readonly", DARK_FG)],
                background=[("readonly", DARK_PANEL)],
            )
        except Exception:
            pass

        # Notebooks
        try:
            style.configure("TNotebook", background=DARK_BG, bordercolor=DARK_BORDER)
            style.configure("TNotebook.Tab", background=DARK_BG2, foreground=DARK_FG, padding=(10, 6))
            style.map(
                "TNotebook.Tab",
                background=[
                    ("selected", DARK_PANEL),
                    ("active", DARK_PANEL2),
                ],
                foreground=[
                    ("selected", DARK_ACCENT),
                    ("active", DARK_ACCENT2),
                ],
            )

            # Charts tabs need to wrap to multiple lines. ttk.Notebook can't do that,
            # so we hide the Notebook's native tabs and render our own wrapping tab bar.
            #
            # IMPORTANT: the layout must exclude Notebook.tab entirely, and on some themes
            # you must keep Notebook.padding for proper sizing; otherwise the tab strip
            # can still render.
            style.configure("HiddenTabs.TNotebook", tabmargins=0)
            style.layout(
                "HiddenTabs.TNotebook",
                [
                    (
                        "Notebook.padding",
                        {
                            "sticky": "nswe",
                            "children": [
                                ("Notebook.client", {"sticky": "nswe"}),
                            ],
                        },
                    )
                ],
            )

            # Wrapping chart-tab buttons (normal + selected)
            style.configure(
                "ChartTab.TButton",
                background=DARK_BG2,
                foreground=DARK_FG,
                bordercolor=DARK_BORDER,
                padding=(10, 6),
            )
            style.map(
                "ChartTab.TButton",
                background=[("active", DARK_PANEL2), ("pressed", DARK_PANEL)],
                foreground=[("active", DARK_ACCENT2)],
                bordercolor=[("active", DARK_ACCENT2), ("focus", DARK_ACCENT)],
            )

            style.configure(
                "ChartTabSelected.TButton",
                background=DARK_PANEL,
                foreground=DARK_ACCENT,
                bordercolor=DARK_ACCENT2,
                padding=(10, 6),
            )
        except Exception:
            pass

        # Treeview (Current Trades table)
        try:
            style.configure(
                "Treeview",
                background=DARK_PANEL,
                fieldbackground=DARK_PANEL,
                foreground=DARK_FG,
                bordercolor=DARK_BORDER,
                lightcolor=DARK_BORDER,
                darkcolor=DARK_BORDER,
            )
            style.map(
                "Treeview",
                background=[("selected", DARK_SELECT_BG)],
                foreground=[("selected", DARK_SELECT_FG)],
            )

            style.configure("Treeview.Heading", background=DARK_BG2, foreground=DARK_ACCENT, relief="flat")
            style.map(
                "Treeview.Heading",
                background=[("active", DARK_PANEL2)],
                foreground=[("active", DARK_ACCENT2)],
            )
        except Exception:
            pass

        # Panedwindows / scrollbars
        try:
            style.configure("TPanedwindow", background=DARK_BG)
        except Exception:
            pass

        for sb in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
            try:
                style.configure(
                    sb,
                    background=DARK_BG2,
                    troughcolor=DARK_BG,
                    bordercolor=DARK_BORDER,
                    arrowcolor=DARK_ACCENT,
                )
            except Exception:
                pass

    # Configuration management methods for loading and saving settings
    def _load_settings(self) -> dict:
        """
        Load Hub configuration from gui_settings.json with intelligent defaults.
        
        Configuration Hierarchy:
        1. DEFAULT_SETTINGS (module-level constants) - Base configuration
        2. gui_settings.json (user overrides) - Persisted user preferences
        3. Normalization (uppercase coins, absolute paths) - Post-processing
        
        Key Settings:
        - coins: List of cryptocurrencies to trade (["BTC", "ETH"], etc.)
        - main_neural_dir: Base directory for coin folders and AI memory files
        - script_neural_trainer: Path to pt_trainer.py
        - script_neural_runner: Path to pt_thinker.py
        - script_trader: Path to pt_trader.py
        - ui_refresh_seconds: Tick loop frequency (default 1.0s)
        - chart_refresh_seconds: Chart update frequency (default 10.0s)
        - account_refresh_seconds: Account data fetch interval (default 10.0s)
        - auto_start_scripts: Launch thinker/trader on Hub startup
        - debug_mode: Enable verbose console logging
        - simulation_mode: Prevent real trades (safety mode)
        
        Fallback Behavior:
        - If gui_settings.json missing or malformed: Uses DEFAULT_SETTINGS
        - If main_neural_dir empty: Defaults to project directory
        - If coins list empty: System still functions (user prompted to configure)
        
        Returns:
            Merged dictionary with defaults + user settings + normalizations
        """
        data = _safe_read_json(SETTINGS_FILE)
        if not isinstance(data, dict):
            data = {}

        # Merge user settings on top of defaults (preserves all default keys)
        merged = dict(DEFAULT_SETTINGS)
        merged.update(data)
        
        # Ensure main_neural_dir has a sensible default (project directory)
        if not merged.get("main_neural_dir"):
            merged["main_neural_dir"] = os.path.abspath(os.path.dirname(__file__))
        
        # Normalize coin symbols to uppercase and strip whitespace
        # This ensures consistent coin folder names and case-insensitive matching
        merged["coins"] = [c.upper().strip() for c in merged.get("coins", [])]
        return merged

    def _save_settings(self) -> None:
        """
        Persist current settings to gui_settings.json (atomic write).
        
        Write Strategy:
        - Uses _safe_write_json helper for atomic write (temp file + os.replace)
        - Preserves all settings keys (not just changed values)
        - Pretty-printed JSON with 2-space indentation for readability
        
        Trigger Points:
        - Settings dialog "Save" button
        - Menu checkbox toggles (auto-start, debug mode, simulation mode)
        - Coin list modifications
        - Directory path changes
        
        Thread Safety:
        - Atomic write prevents partial reads by other processes
        - No explicit locking (file operations are atomic at OS level)
        """
        _safe_write_json(SETTINGS_FILE, self.settings)

    def _settings_getter(self) -> dict:
        return self.settings

    def _ensure_alt_coin_folders_and_trainer_on_startup(self) -> None:
        """
        Startup behavior (mirrors Settings-save behavior):
        - For every coin in the coin list that does NOT have its folder yet:
            - create the folder
            - copy neural_trainer.py from the MAIN folder into the new coin folder
        """
        try:
            coins = [str(c).strip().upper() for c in (self.settings.get("coins") or []) if str(c).strip()]
            main_dir = (self.settings.get("main_neural_dir") or self.project_dir or os.getcwd()).strip()

            trainer_name = os.path.basename(str(self.settings.get("script_neural_trainer", "neural_trainer.py")))

            # Source trainer: MAIN folder (preferred location)
            src_main_trainer = os.path.join(main_dir, trainer_name)

            # Best-effort fallback if the main folder doesn't have it (keeps behavior robust)
            src_cfg_trainer = str(self.settings.get("script_neural_trainer", trainer_name))
            src_trainer_path = src_main_trainer if os.path.isfile(src_main_trainer) else src_cfg_trainer

            for coin in coins:
                coin_dir = os.path.join(main_dir, coin)

                created = False
                if not os.path.isdir(coin_dir):
                    os.makedirs(coin_dir, exist_ok=True)
                    created = True

                # Only copy into folders created at startup (per your request)
                if created:
                    dst_trainer_path = os.path.join(coin_dir, trainer_name)
                    if (not os.path.isfile(dst_trainer_path)) and os.path.isfile(src_trainer_path):
                        shutil.copy2(src_trainer_path, dst_trainer_path)
        except Exception:
            pass

    def _clear_stale_training_status(self) -> None:
        """
        Clear any trainer_status.json files showing TRAINING state on Hub startup.
        This prevents showing stale "training" status from interrupted sessions.
        """
        try:
            main_dir = (self.settings.get("main_neural_dir") or self.project_dir or os.getcwd()).strip()
            coins = [str(c).strip().upper() for c in (self.settings.get("coins") or []) if str(c).strip()]
            
            for coin in coins:
                coin_dir = os.path.join(main_dir, coin)
                status_file = os.path.join(coin_dir, "trainer_status.json")
                
                if os.path.isfile(status_file):
                    try:
                        with open(status_file, "r", encoding="utf-8") as f:
                            status = json.load(f)
                        
                        # If status shows TRAINING, clear it (training process isn't actually running)
                        if status.get("state") == "TRAINING":
                            os.remove(status_file)
                    except Exception:
                        pass  # Ignore errors reading/removing individual status files
        except Exception:
            pass  # Don't let startup fail if status cleanup fails

    # GUI construction methods for menus and layout
    def _build_menu(self) -> None:
        menubar = tk.Menu(
            self,
            bg=DARK_BG2,
            fg=DARK_FG,
            activebackground=DARK_SELECT_BG,
            activeforeground=DARK_SELECT_FG,
            bd=0,
            relief="flat",
        )

        m_scripts = tk.Menu(
            menubar,
            tearoff=0,
            bg=DARK_BG2,
            fg=DARK_FG,
            activebackground=DARK_SELECT_BG,
            activeforeground=DARK_SELECT_FG,
            selectcolor=DARK_FG,
        )
        m_scripts.add_command(label="🚀 Engage Autopilot", command=self.start_all_scripts)
        m_scripts.add_command(label="Stop All", command=self.stop_all_scripts)
        m_scripts.add_separator()
        m_scripts.add_command(label="Start Thinker", command=self.start_neural)
        m_scripts.add_command(label="Stop Thinker", command=self.stop_neural)
        m_scripts.add_separator()
        m_scripts.add_command(label="Start Trader", command=self.start_trader)
        m_scripts.add_command(label="Stop Trader", command=self.stop_trader)
        m_scripts.add_separator()
        m_scripts.add_checkbutton(
            label="🚀 Engage Autopilot on Launch",
            command=self.toggle_auto_start,
            variable=self.auto_start_var
        )
        menubar.add_cascade(label="Scripts", menu=m_scripts)

        m_settings = tk.Menu(
            menubar,
            tearoff=0,
            bg=DARK_BG2,
            fg=DARK_FG,
            activebackground=DARK_SELECT_BG,
            activeforeground=DARK_SELECT_FG,
            selectcolor=DARK_FG,
        )
        m_settings.add_command(label="Hub Settings...", command=self.open_settings_dialog)
        m_settings.add_command(label="Trading Settings...", command=self.open_trading_settings_dialog)
        m_settings.add_command(label="Training Settings...", command=self.open_training_settings_dialog)
        m_settings.add_separator()
        m_settings.add_checkbutton(
            label="Debug Mode",
            command=self.toggle_debug_mode,
            variable=self.debug_mode_var
        )
        m_settings.add_checkbutton(
            label="Simulation Mode",
            command=self.toggle_simulation_mode,
            variable=self.simulation_mode_var
        )
        menubar.add_cascade(label="Settings", menu=m_settings)

        m_file = tk.Menu(
            menubar,
            tearoff=0,
            bg=DARK_BG2,
            fg=DARK_FG,
            activebackground=DARK_SELECT_BG,
            activeforeground=DARK_SELECT_FG,
        )
        m_file.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="Help", menu=m_file)

        self.config(menu=menubar)

    def _build_layout(self) -> None:
        # status bar - pack FIRST so it reserves space before the main content
        self.status = ttk.Label(self, text="Ready", anchor="w")
        self.status.pack(fill="x", side="bottom")

        outer = ttk.Panedwindow(self, orient="horizontal")
        outer.pack(fill="both", expand=True)

        # LEFT + RIGHT panes
        left = ttk.Frame(outer)
        right = ttk.Frame(outer)

        outer.add(left, weight=1)
        outer.add(right, weight=2)

        # Prevent the outer (left/right) panes from being collapsible to 0 width
        try:
            outer.paneconfig(left, minsize=360)
            outer.paneconfig(right, minsize=520)
        except Exception:
            pass

        # LEFT: vertical split (Controls, Live Output)
        left_split = ttk.Panedwindow(left, orient="vertical")
        left_split.pack(fill="both", expand=True, padx=8, pady=8)

        # RIGHT: vertical split (Charts on top, Trades+History underneath)
        right_split = ttk.Panedwindow(right, orient="vertical")
        right_split.pack(fill="both", expand=True, padx=8, pady=8)

        # Keep references so we can clamp sash positions later
        self._pw_outer = outer
        self._pw_left_split = left_split
        self._pw_right_split = right_split

        # Clamp panes when the user releases a sash or the window resizes
        outer.bind("<Configure>", lambda e: self._schedule_paned_clamp(self._pw_outer))
        outer.bind("<ButtonRelease-1>", lambda e: (
            setattr(self, "_user_moved_outer", True),
            self._schedule_paned_clamp(self._pw_outer),
        ))

        left_split.bind("<Configure>", lambda e: self._schedule_paned_clamp(self._pw_left_split))
        left_split.bind("<ButtonRelease-1>", lambda e: (
            setattr(self, "_user_moved_left_split", True),
            self._schedule_paned_clamp(self._pw_left_split),
        ))

        right_split.bind("<Configure>", lambda e: self._schedule_paned_clamp(self._pw_right_split))
        right_split.bind("<ButtonRelease-1>", lambda e: (
            setattr(self, "_user_moved_right_split", True),
            self._schedule_paned_clamp(self._pw_right_split),
        ))

        # Set a startup default width that matches the screenshot (so left has room for Neural Levels).
        def _init_outer_sash_once():
            try:
                if getattr(self, "_did_init_outer_sash", False):
                    return

                # If the user already moved it, never override it.
                if getattr(self, "_user_moved_outer", False):
                    self._did_init_outer_sash = True
                    return

                total = outer.winfo_width()
                if total <= 2:
                    self.after(10, _init_outer_sash_once)
                    return

                min_left = 360
                min_right = 520
                desired_left = 470  # ~matches your screenshot
                target = max(min_left, min(total - min_right, desired_left))
                outer.sashpos(0, int(target))

                self._did_init_outer_sash = True
            except Exception:
                pass

        self.after_idle(_init_outer_sash_once)

        # Global safety: on some themes/platforms, the mouse events land on the sash element,
        # not the panedwindow widget, so the widget-level binds won't always fire.
        self.bind_all("<ButtonRelease-1>", lambda e: (
            self._schedule_paned_clamp(getattr(self, "_pw_outer", None)),
            self._schedule_paned_clamp(getattr(self, "_pw_left_split", None)),
            self._schedule_paned_clamp(getattr(self, "_pw_right_split", None)),
            self._schedule_paned_clamp(getattr(self, "_pw_right_bottom_split", None)),
        ))

        # Mission Control panel: status display and primary action buttons
        top_controls = ttk.LabelFrame(left_split, text="Mission Control")

        info_row = ttk.Frame(top_controls)
        info_row.pack(fill="x", expand=False, padx=6, pady=6)

        # LEFT column (status + account/profit)
        controls_left = ttk.Frame(info_row)
        controls_left.pack(side="left", fill="both", expand=True)

        # RIGHT column (training)
        training_section = ttk.LabelFrame(info_row, text="Training")
        training_section.pack(side="right", fill="both", expand=False, padx=6, pady=6)

        training_left = ttk.Frame(training_section)
        training_left.pack(side="left", fill="both", expand=True)

        # Train coin selector (so you can choose what "Train Selected" targets)
        train_row = ttk.Frame(training_left)
        train_row.pack(fill="x", padx=6, pady=(6, 0))

        self.train_coin_var = tk.StringVar(value=(self.coins[0] if self.coins else ""))
        ttk.Label(train_row, text="Train coin:").pack(side="left")
        self.train_coin_combo = ttk.Combobox(
            train_row,
            textvariable=self.train_coin_var,
            values=self.coins,
            width=8,
            state="readonly",
        )
        self.train_coin_combo.pack(side="left", padx=(6, 0))

        def _sync_train_coin(*_):
            try:
                # keep the Trainers tab dropdown in sync (if present)
                self.trainer_coin_var.set(self.train_coin_var.get())
            except Exception:
                pass

        self.train_coin_combo.bind("<<ComboboxSelected>>", _sync_train_coin)
        _sync_train_coin()

        BTN_W = 22  # Standard button width for alignment

        # Status display - Last status comes first
        self.lbl_last_status = ttk.Label(controls_left, text="Last status: N/A")
        self.lbl_last_status.pack(anchor="w", padx=6, pady=(0, 6))

        # Training controls panel with coin selection and training buttons
        train_buttons_row = ttk.Frame(training_left)
        train_buttons_row.pack(fill="x", padx=6, pady=(6, 6))

        ttk.Button(train_buttons_row, text="Train Selected", width=BTN_W, command=self.train_selected_coin).pack(anchor="w", pady=(0, 6))
        ttk.Button(train_buttons_row, text="Train All", width=BTN_W, command=self.train_all_coins).pack(anchor="w")

        # Training status (per-coin + gating reason)
        self.lbl_training_overview = ttk.Label(training_left, text="Training: N/A")
        self.lbl_training_overview.pack(anchor="w", padx=6, pady=(0, 2))

        self.lbl_flow_hint = ttk.Label(training_left, text="Next: TRAIN → THINK → TRADE")
        self.lbl_flow_hint.pack(anchor="w", padx=6, pady=(0, 6))

        self.training_list = tk.Listbox(
            training_left,
            height=4,
            bg=DARK_PANEL,
            fg=DARK_FG,
            selectbackground=DARK_SELECT_BG,
            selectforeground=DARK_SELECT_FG,
            highlightbackground=DARK_BORDER,
            highlightcolor=DARK_ACCENT,
            activestyle="none",
        )
        self.training_list.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # Start All (moved here: LEFT side of the dual section, directly above Account)
        start_all_row = ttk.Frame(controls_left)
        start_all_row.pack(fill="x", padx=6, pady=(0, 6))

        # Create bold style for autopilot button
        bold_style = ttk.Style()
        bold_style.configure("Bold.TButton", font=("Segoe UI", 10, "bold"))
        
        self.btn_toggle_all = ttk.Button(
            start_all_row,
            text="🚀 ENGAGE AUTOPILOT",
            width=BTN_W,
            style="Bold.TButton",
            command=self.toggle_all_scripts,
        )
        self.btn_toggle_all.pack(side="left")

        # Thinker and Trader status - displayed after Autopilot button
        self.lbl_neural = ttk.Label(controls_left, text="Thinker: stopped")
        self.lbl_neural.pack(anchor="w", padx=6, pady=(6, 2))

        self.lbl_trader = ttk.Label(controls_left, text="Trader: stopped")
        self.lbl_trader.pack(anchor="w", padx=6, pady=(0, 6))

        # Account info (LEFT column, under status) - reduced top padding
        acct_box = ttk.LabelFrame(controls_left, text="Account")
        acct_box.pack(fill="x", padx=6, pady=(0, 6))
        
        # Simulation mode banner (prominent warning) - initially hidden
        self.lbl_simulation_banner = ttk.Label(
            acct_box, 
            text="", 
            foreground="#FF6B00",  # Orange
            font=("TkDefaultFont", 10, "bold")
        )
        # Don't pack initially - only show when simulation mode is active

        self.lbl_acct_total_value = ttk.Label(acct_box, text="Total Account Value: N/A")
        self.lbl_acct_total_value.pack(anchor="w", padx=6, pady=(4, 0))

        self.lbl_acct_holdings_value = ttk.Label(acct_box, text="Holdings Value: N/A")
        self.lbl_acct_holdings_value.pack(anchor="w", padx=6, pady=(2, 0))

        self.lbl_acct_buying_power = ttk.Label(acct_box, text="Buying Power: N/A")
        self.lbl_acct_buying_power.pack(anchor="w", padx=6, pady=(2, 0))

        self.lbl_acct_percent_in_trade = ttk.Label(acct_box, text="Percent In Trade: N/A")
        self.lbl_acct_percent_in_trade.pack(anchor="w", padx=6, pady=(2, 0))

        # DCA affordability
        self.lbl_acct_dca_spread = ttk.Label(acct_box, text="DCA Levels (spread): N/A")
        self.lbl_acct_dca_spread.pack(anchor="w", padx=6, pady=(2, 0))

        self.lbl_acct_dca_single = ttk.Label(acct_box, text="DCA Levels (single): N/A")
        self.lbl_acct_dca_single.pack(anchor="w", padx=6, pady=(2, 0))

        self.lbl_pnl = ttk.Label(acct_box, text="Total realized: N/A")
        self.lbl_pnl.pack(anchor="w", padx=6, pady=(2, 2))

        # Neural levels overview (spans FULL width under the dual section)
        # Shows the current LONG/SHORT level (0..7) for every coin at once.
        neural_box = ttk.LabelFrame(top_controls, text="Neural Levels (0–7)")
        neural_box.pack(fill="x", expand=False, padx=12, pady=(0, 10))

        legend = ttk.Frame(neural_box)
        legend.pack(fill="x", padx=6, pady=(4, 0))

        ttk.Label(legend, text="Level bars:   0 = bottom  7 = top").pack(side="left")
        ttk.Label(legend, text="  ").pack(side="left")
        ttk.Label(legend, text="L = long").pack(side="left")
        ttk.Label(legend, text="  ").pack(side="left")
        ttk.Label(legend, text="S = short").pack(side="left")

        self.lbl_neural_overview_last = ttk.Label(legend, text="Last status: N/A")
        self.lbl_neural_overview_last.pack(side="right")

        # Scrollable area for tiles (auto-hides the scrollbar if everything fits)
        self._neural_viewport_frame = ttk.Frame(neural_box)
        self._neural_viewport_frame.pack(fill="x", expand=False, padx=6, pady=(4, 6))
        self._neural_viewport_frame.grid_rowconfigure(0, weight=0)  # Don't expand row, let content size naturally
        self._neural_viewport_frame.grid_columnconfigure(0, weight=1)

        self._neural_overview_canvas = tk.Canvas(
            self._neural_viewport_frame,
            bg=DARK_PANEL2,
            highlightthickness=1,
            highlightbackground=DARK_BORDER,
            bd=0,
            height=115,  # Height for one row of tiles with minimal padding
        )
        self._neural_overview_canvas.grid(row=0, column=0, sticky="ew")

        self._neural_overview_scroll = ttk.Scrollbar(
            self._neural_viewport_frame,
            orient="vertical",
            command=self._neural_overview_canvas.yview,
        )
        self._neural_overview_scroll.grid(row=0, column=1, sticky="ns")

        self._neural_overview_canvas.configure(yscrollcommand=self._neural_overview_scroll.set)

        self.neural_wrap = WrapFrame(self._neural_overview_canvas)
        self._neural_overview_window = self._neural_overview_canvas.create_window(
            (0, 0),
            window=self.neural_wrap,
            anchor="nw",
        )

        def _update_neural_overview_scrollbars(event=None) -> None:
            """Update scrollregion + hide/show the scrollbar depending on overflow."""
            try:
                c = self._neural_overview_canvas
                win = self._neural_overview_window

                c.update_idletasks()
                bbox = c.bbox(win)
                if not bbox:
                    self._neural_overview_scroll.grid_remove()
                    return

                c.configure(scrollregion=bbox)
                content_h = int(bbox[3] - bbox[1])
                view_h = int(c.winfo_height())

                if content_h > (view_h + 1):
                    self._neural_overview_scroll.grid()
                else:
                    self._neural_overview_scroll.grid_remove()
                    try:
                        c.yview_moveto(0)
                    except Exception:
                        pass
            except Exception:
                pass

        def _on_neural_canvas_configure(e) -> None:
            # Keep the inner wrap frame exactly the canvas width so wrapping is correct.
            try:
                self._neural_overview_canvas.itemconfigure(self._neural_overview_window, width=int(e.width))
            except Exception:
                pass
            _update_neural_overview_scrollbars()

        self._neural_overview_canvas.bind("<Configure>", _on_neural_canvas_configure, add="+")
        self.neural_wrap.bind("<Configure>", _update_neural_overview_scrollbars, add="+")
        self._update_neural_overview_scrollbars = _update_neural_overview_scrollbars

        # Mousewheel scroll inside the tiles area
        def _wheel(e):
            try:
                if self._neural_overview_scroll.winfo_ismapped():
                    self._neural_overview_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
            except Exception:
                pass

        self._neural_overview_canvas.bind("<Enter>", lambda _e: self._neural_overview_canvas.focus_set(), add="+")
        self._neural_overview_canvas.bind("<MouseWheel>", _wheel, add="+")

        # tiles by coin
        self.neural_tiles: Dict[str, NeuralSignalTile] = {}
        # small cache: path -> (mtime, value)
        self._neural_overview_cache: Dict[str, Tuple[float, Any]] = {}

        self._rebuild_neural_overview()
        try:
            self.after_idle(self._update_neural_overview_scrollbars)
        except Exception:
            pass

        # Live Output panel: tabs for Thinker, Trader, and Trainer logs
        # Configurable font for live logs (Thinker/Trader/Trainers)
        try:
            _base = tkfont.nametofont("Cascadia Code")
            self._live_log_font = _base.copy()
        except Exception:
            self._live_log_font = tkfont.Font(family="Cascadia Code", size=LOG_FONT_SIZE)
        self._live_log_font.configure(size=LOG_FONT_SIZE)

        logs_frame = ttk.LabelFrame(left_split, text="Live Output")
        self.logs_nb = ttk.Notebook(logs_frame)
        self.logs_nb.pack(fill="both", expand=True, padx=6, pady=6)

        # Thinker tab
        runner_tab = ttk.Frame(self.logs_nb)
        self.logs_nb.add(runner_tab, text="Thinker")
        self.runner_text = tk.Text(
            runner_tab,
            height=8,
            wrap="none",
            font=self._live_log_font,
            bg=DARK_PANEL,
            fg=LIVE_OUTPUT_FG,
            insertbackground=DARK_FG,
            selectbackground=DARK_SELECT_BG,
            selectforeground=DARK_SELECT_FG,
            highlightbackground=DARK_BORDER,
            highlightcolor=DARK_ACCENT,
        )

        runner_scroll = ttk.Scrollbar(runner_tab, orient="vertical", command=self.runner_text.yview)
        
        # Bind scroll events to track user scrolling away from bottom
        def _on_runner_scroll(*args):
            """Track when user scrolls and check if they're at the bottom."""
            try:
                runner_scroll.set(*args)
                yview = self.runner_text.yview()
                at_bottom = (yview[1] >= 0.98)
                self._runner_log_user_scrolled_away = not at_bottom
            except Exception:
                pass
        
        self.runner_text.configure(yscrollcommand=_on_runner_scroll)
        self.runner_text.pack(side="left", fill="both", expand=True)
        runner_scroll.pack(side="right", fill="y")

        # Trader tab
        trader_tab = ttk.Frame(self.logs_nb)
        self.logs_nb.add(trader_tab, text="Trader")
        self.trader_text = tk.Text(
            trader_tab,
            height=8,
            wrap="none",
            font=self._live_log_font,
            bg=DARK_PANEL,
            fg=LIVE_OUTPUT_FG,
            insertbackground=DARK_FG,
            selectbackground=DARK_SELECT_BG,
            selectforeground=DARK_SELECT_FG,
            highlightbackground=DARK_BORDER,
            highlightcolor=DARK_ACCENT,
        )

        trader_scroll = ttk.Scrollbar(trader_tab, orient="vertical", command=self.trader_text.yview)
        
        # Bind scroll events to track user scrolling away from bottom
        def _on_trader_scroll(*args):
            """Track when user scrolls and check if they're at the bottom."""
            try:
                trader_scroll.set(*args)
                yview = self.trader_text.yview()
                at_bottom = (yview[1] >= 0.98)
                self._trader_log_user_scrolled_away = not at_bottom
            except Exception:
                pass
        
        self.trader_text.configure(yscrollcommand=_on_trader_scroll)
        self.trader_text.pack(side="left", fill="both", expand=True)
        trader_scroll.pack(side="right", fill="y")

        # Trainers tab (multi-coin)
        trainer_tab = ttk.Frame(self.logs_nb)
        self.logs_nb.add(trainer_tab, text="Trainers")

        top_bar = ttk.Frame(trainer_tab)
        top_bar.pack(fill="x", padx=6, pady=6)

        self.trainer_coin_var = tk.StringVar(value=(self.coins[0] if self.coins else "BTC"))
        ttk.Label(top_bar, text="Coin:").pack(side="left")
        self.trainer_coin_combo = ttk.Combobox(
            top_bar,
            textvariable=self.trainer_coin_var,
            values=self.coins,
            state="readonly",
            width=8
        )
        self.trainer_coin_combo.pack(side="left", padx=(6, 12))

        def _on_trainer_coin_changed(*_):
            try:
                # Restore the historical log for the selected coin
                selected_coin = (self.trainer_coin_var.get() or "").strip().upper()
                self.trainer_text.delete("1.0", "end")
                
                # Display the historical log lines for this coin
                if selected_coin in self.trainer_log_history:
                    history = self.trainer_log_history[selected_coin]
                    if history:
                        self.trainer_text.insert("end", "\n".join(history))
                        if not history[-1].endswith("\n"):
                            self.trainer_text.insert("end", "\n")
                        self.trainer_text.see("end")
                        # Reset scroll flag when switching coins (start at bottom)
                        self._trainer_log_user_scrolled_away = False
                
                # Sync with train_coin_combo if present
                if hasattr(self, "train_coin_var"):
                    self.train_coin_var.set(self.trainer_coin_var.get())
            except Exception:
                pass

        self.trainer_coin_combo.bind("<<ComboboxSelected>>", _on_trainer_coin_changed)

        ttk.Button(top_bar, text="Start Trainer", command=self.start_trainer_for_selected_coin).pack(side="left")
        ttk.Button(top_bar, text="Stop Trainer", command=self.stop_trainer_for_selected_coin).pack(side="left", padx=(6, 0))

        self.trainer_status_lbl = ttk.Label(top_bar, text="(no trainers running)")
        self.trainer_status_lbl.pack(side="left", padx=(12, 0))

        self.trainer_text = tk.Text(
            trainer_tab,
            height=8,
            wrap="none",
            font=self._live_log_font,
            bg=DARK_PANEL,
            fg=LIVE_OUTPUT_FG,
            insertbackground=DARK_FG,
            selectbackground=DARK_SELECT_BG,
            selectforeground=DARK_SELECT_FG,
            highlightbackground=DARK_BORDER,
            highlightcolor=DARK_ACCENT,
        )

        trainer_scroll = ttk.Scrollbar(trainer_tab, orient="vertical", command=self.trainer_text.yview)
        
        # Bind scroll events to track user scrolling away from bottom
        def _on_trainer_scroll(*args):
            """Track when user scrolls and check if they're at the bottom."""
            try:
                # Update the scrollbar
                trainer_scroll.set(*args)
                
                # Check if at bottom (within 2% tolerance)
                yview = self.trainer_text.yview()
                at_bottom = (yview[1] >= 0.98)
                
                # Update flag: if at bottom, allow auto-scroll; if not, disable it
                self._trainer_log_user_scrolled_away = not at_bottom
            except Exception:
                pass
        
        self.trainer_text.configure(yscrollcommand=_on_trainer_scroll)
        self.trainer_text.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=(0, 6))
        trainer_scroll.pack(side="right", fill="y", padx=(0, 6), pady=(0, 6))

        # Auto-scroll to bottom when switching to a live output tab
        def on_tab_selected(event):
            try:
                selected_tab = self.logs_nb.select()
                selected_index = self.logs_nb.index(selected_tab)
                # 0=Thinker, 1=Trader, 2=Trainers
                if selected_index == 0:
                    self.runner_text.after(50, lambda: self.runner_text.yview_moveto(1.0))
                    # Reset scroll flag when switching to this tab
                    self._runner_log_user_scrolled_away = False
                elif selected_index == 1:
                    self.trader_text.after(50, lambda: self.trader_text.yview_moveto(1.0))
                    # Reset scroll flag when switching to this tab
                    self._trader_log_user_scrolled_away = False
                elif selected_index == 2:
                    self.trainer_text.after(50, lambda: self.trainer_text.yview_moveto(1.0))
                    # Reset scroll flag when switching to this tab
                    self._trainer_log_user_scrolled_away = False
            except Exception:
                pass
        
        self.logs_nb.bind("<<NotebookTabChanged>>", on_tab_selected)

        # Add left panes (no trades/history on the left anymore)
        # Controls/Health stays fixed size (determined by neural tiles), Live Output fills remaining space
        left_split.add(top_controls, weight=0)  # Fixed size based on content
        left_split.add(logs_frame, weight=1)    # Expands to fill remaining vertical space

        try:
            # Ensure reasonable minimums
            left_split.paneconfig(top_controls, minsize=450)
            left_split.paneconfig(logs_frame, minsize=180)
        except Exception:
            pass

        def _init_left_split_sash_once():
            try:
                if getattr(self, "_did_init_left_split_sash", False):
                    return

                # If the user already moved the sash, never override it.
                if getattr(self, "_user_moved_left_split", False):
                    self._did_init_left_split_sash = True
                    return

                total = left_split.winfo_height()
                if total <= 2:
                    self.after(10, _init_left_split_sash_once)
                    return

                min_top = 450
                min_bottom = 180

                # Position sash just below Neural Levels content
                # Measure actual height needed for Controls/Health section
                try:
                    top_controls.update_idletasks()
                    controls_height = top_controls.winfo_reqheight()
                    target = controls_height
                except Exception:
                    # Fallback: fixed position for controls + neural tiles
                    target = 500

                target = max(min_top, min(total - min_bottom, target))
                left_split.sashpos(0, int(target))
                self._did_init_left_split_sash = True
            except Exception:
                pass

        self.after_idle(_init_left_split_sash_once)

        # Charts panel: tabbed interface for displaying price charts with neural overlays
        charts_frame = ttk.LabelFrame(right_split, text="Charts (neural lines overlaid)")
        self._charts_frame = charts_frame

        # Multi-row "tabs" (WrapFrame)
        self.chart_tabs_bar = WrapFrame(charts_frame)
        # Keep left padding, remove right padding so tabs can reach the edge
        self.chart_tabs_bar.pack(fill="x", padx=(6, 0), pady=(6, 0))

        # Page container (no ttk.Notebook, so there are NO native tabs to show)
        self.chart_pages_container = ttk.Frame(charts_frame)
        # Keep left padding, remove right padding so charts fill to the edge
        self.chart_pages_container.pack(fill="both", expand=True, padx=(6, 0), pady=(0, 6))

        self._chart_tab_buttons: Dict[str, ttk.Button] = {}
        self.chart_pages: Dict[str, ttk.Frame] = {}
        self._current_chart_page: str = "ACCOUNT"

        def _show_page(name: str, is_auto_switch: bool = False) -> None:
            # Track manual chart switches for auto-switch override logic
            if not is_auto_switch:
                import time
                self._last_manual_chart_switch = time.time()
            
            self._current_chart_page = name
            # hide all pages
            for f in self.chart_pages.values():
                try:
                    f.pack_forget()
                except Exception:
                    pass
            # show selected
            f = self.chart_pages.get(name)
            if f is not None:
                f.pack(fill="both", expand=True)

            # style selected tab
            for txt, b in self._chart_tab_buttons.items():
                try:
                    b.configure(style=("ChartTabSelected.TButton" if txt == name else "ChartTab.TButton"))
                except Exception:
                    pass
            
            # Update neural tiles to highlight the selected coin
            if hasattr(self, "neural_tiles"):
                for coin, tile in self.neural_tiles.items():
                    try:
                        tile.set_selected(coin == name)
                    except Exception:
                        pass

            # Immediately refresh the newly shown coin chart so candles appear right away
            # (even if trader/neural scripts are not running yet).
            try:
                tab = str(name or "").strip().upper()
                if tab and tab != "ACCOUNT":
                    coin = tab
                    chart = self.charts.get(coin)
                    if chart:
                        def _do_refresh_visible():
                            try:
                                # Ensure coin folders exist (best-effort; fast)
                                try:
                                    cf_sig = (self.settings.get("main_neural_dir"), tuple(self.coins))
                                    if getattr(self, "_coin_folders_sig", None) != cf_sig:
                                        self._coin_folders_sig = cf_sig
                                        self.coin_folders = build_coin_folders(self.settings["main_neural_dir"], self.coins)
                                except Exception:
                                    pass

                                pos = self._last_positions.get(coin, {}) if isinstance(self._last_positions, dict) else {}
                                buy_px = pos.get("current_buy_price", None)
                                sell_px = pos.get("current_sell_price", None)
                                trail_line = pos.get("trail_line", None)
                                dca_line_price = pos.get("dca_line_price", None)

                                chart.refresh(
                                    self.coin_folders,
                                    current_buy_price=buy_px,
                                    current_sell_price=sell_px,
                                    trail_line=trail_line,
                                    dca_line_price=dca_line_price,
                                )
                            except Exception:
                                pass

                        self.after(1, _do_refresh_visible)
            except Exception:
                pass

        self._show_chart_page = _show_page  # used by _rebuild_coin_chart_tabs()

        # ACCOUNT page
        acct_page = ttk.Frame(self.chart_pages_container)
        self.chart_pages["ACCOUNT"] = acct_page

        acct_btn = ttk.Button(
            self.chart_tabs_bar,
            text="ACCOUNT",
            style="ChartTab.TButton",
            command=lambda: self._show_chart_page("ACCOUNT"),
        )
        self.chart_tabs_bar.add(acct_btn, padx=(0, 6), pady=(0, 6))
        self._chart_tab_buttons["ACCOUNT"] = acct_btn

        self.account_chart = AccountValueChart(
            acct_page,
            self.account_value_history_path,
            self.trade_history_path,
        )
        self.account_chart.pack(fill="both", expand=True)

        # Coin pages
        self.charts: Dict[str, CandleChart] = {}
        for coin in self.coins:
            page = ttk.Frame(self.chart_pages_container)
            self.chart_pages[coin] = page

            btn = ttk.Button(
                self.chart_tabs_bar,
                text=coin,
                style="ChartTab.TButton",
                command=lambda c=coin: self._show_chart_page(c),
            )
            self.chart_tabs_bar.add(btn, padx=(0, 6), pady=(0, 6))
            self._chart_tab_buttons[coin] = btn

            chart = CandleChart(page, self.fetcher, coin, self._settings_getter, self.trade_history_path)
            chart.pack(fill="both", expand=True)
            self.charts[coin] = chart

        # show initial page
        self._show_chart_page("ACCOUNT")

        # Trading panels: current positions and historical trade log display
        right_bottom_split = ttk.Panedwindow(right_split, orient="vertical")
        self._pw_right_bottom_split = right_bottom_split

        right_bottom_split.bind("<Configure>", lambda e: self._schedule_paned_clamp(self._pw_right_bottom_split))
        right_bottom_split.bind("<ButtonRelease-1>", lambda e: (
            setattr(self, "_user_moved_right_bottom_split", True),
            self._schedule_paned_clamp(self._pw_right_bottom_split),
        ))

        # Current trades (top)
        trades_frame = ttk.LabelFrame(right_bottom_split, text="Current Trades")

        cols = (
            "coin",
            "qty",
            "value",          # <-- right after qty
            "avg_cost",
            "buy_price",
            "buy_pnl",
            "sell_price",
            "sell_pnl",
            "dca_stages",
            "dca_24h",
            "next_dca",
            "trail_line",     # keep trail line column
        )

        header_labels = {
            "coin": "Coin",
            "qty": "Qty",
            "value": "Value",
            "avg_cost": "Avg Cost",
            "buy_price": "Ask Price",
            "buy_pnl": "DCA PnL",
            "sell_price": "Bid Price",
            "sell_pnl": "Sell PnL",
            "dca_stages": "DCA Stage",
            "dca_24h": "DCA 24h",
            "next_dca": "Next DCA",
            "trail_line": "Trail Line",
        }

        trades_table_wrap = ttk.Frame(trades_frame)
        trades_table_wrap.pack(fill="both", expand=True, padx=6, pady=6)

        self.trades_tree = ttk.Treeview(
            trades_table_wrap,
            columns=cols,
            show="headings",
            height=10
        )
        for c in cols:
            self.trades_tree.heading(c, text=header_labels.get(c, c))
            self.trades_tree.column(c, width=110, anchor="center", stretch=True)

        # Reasonable starting widths (they will be dynamically scaled on resize)
        self.trades_tree.column("coin", width=70)
        self.trades_tree.column("qty", width=95)
        self.trades_tree.column("value", width=110)
        self.trades_tree.column("next_dca", width=160)
        self.trades_tree.column("dca_stages", width=90)
        self.trades_tree.column("dca_24h", width=80)

        ysb = ttk.Scrollbar(trades_table_wrap, orient="vertical", command=self.trades_tree.yview)
        xsb = ttk.Scrollbar(trades_table_wrap, orient="horizontal", command=self.trades_tree.xview)
        self.trades_tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)

        self.trades_tree.pack(side="top", fill="both", expand=True)
        xsb.pack(side="bottom", fill="x")
        ysb.pack(side="right", fill="y")

        def _resize_trades_columns(*_):
            # Scale the initial column widths proportionally so the table always fits the current window.
            try:
                total_w = int(self.trades_tree.winfo_width())
            except Exception:
                return
            if total_w <= 1:
                return

            try:
                sb_w = int(ysb.winfo_width() or 0)
            except Exception:
                sb_w = 0

            avail = max(200, total_w - sb_w - 8)

            base = {
                "coin": 70,
                "qty": 95,
                "value": 110,
                "avg_cost": 110,
                "buy_price": 110,
                "buy_pnl": 110,
                "sell_price": 110,
                "sell_pnl": 110,
                "dca_stages": 90,
                "dca_24h": 80,
                "next_dca": 160,
                "trail_line": 110,
            }
            base_total = sum(base.get(c, 110) for c in cols) or 1
            scale = avail / base_total

            for c in cols:
                w = int(base.get(c, 110) * scale)
                self.trades_tree.column(c, width=max(60, min(420, w)))

        self.trades_tree.bind("<Configure>", lambda e: self.after_idle(_resize_trades_columns))
        self.after_idle(_resize_trades_columns)

        # Trade history (bottom)
        hist_frame = ttk.LabelFrame(right_bottom_split, text="Trade History (scroll)")

        hist_wrap = ttk.Frame(hist_frame)
        hist_wrap.pack(fill="both", expand=True, padx=6, pady=6)

        self.hist_list = tk.Listbox(
            hist_wrap,
            height=10,
            bg=DARK_PANEL,
            fg=DARK_FG,
            selectbackground=DARK_SELECT_BG,
            selectforeground=DARK_SELECT_FG,
            highlightbackground=DARK_BORDER,
            highlightcolor=DARK_ACCENT,
            activestyle="none",
        )
        ysb2 = ttk.Scrollbar(hist_wrap, orient="vertical", command=self.hist_list.yview)
        xsb2 = ttk.Scrollbar(hist_wrap, orient="horizontal", command=self.hist_list.xview)
        self.hist_list.configure(yscrollcommand=ysb2.set, xscrollcommand=xsb2.set)

        self.hist_list.pack(side="left", fill="both", expand=True)
        ysb2.pack(side="right", fill="y")
        xsb2.pack(side="bottom", fill="x")

        # Assemble right side
        right_split.add(charts_frame, weight=3)
        right_split.add(right_bottom_split, weight=2)

        right_bottom_split.add(trades_frame, weight=2)
        right_bottom_split.add(hist_frame, weight=1)

        try:
            # Screenshot-style sizing: don't force Charts to be enormous by default.
            right_split.paneconfig(charts_frame, minsize=360)
            right_split.paneconfig(right_bottom_split, minsize=220)
        except Exception:
            pass

        try:
            right_bottom_split.paneconfig(trades_frame, minsize=140)
            right_bottom_split.paneconfig(hist_frame, minsize=120)
        except Exception:
            pass

        # Startup defaults to match the screenshot (but never override if user already dragged).
        def _init_right_split_sash_once():
            try:
                if getattr(self, "_did_init_right_split_sash", False):
                    return

                if getattr(self, "_user_moved_right_split", False):
                    self._did_init_right_split_sash = True
                    return

                total = right_split.winfo_height()
                if total <= 2:
                    self.after(10, _init_right_split_sash_once)
                    return

                min_top = 360
                min_bottom = 220
                desired_top = 410  # ~matches screenshot chart pane height
                target = max(min_top, min(total - min_bottom, desired_top))

                right_split.sashpos(0, int(target))
                self._did_init_right_split_sash = True
            except Exception:
                pass

        def _init_right_bottom_split_sash_once():
            try:
                if getattr(self, "_did_init_right_bottom_split_sash", False):
                    return

                if getattr(self, "_user_moved_right_bottom_split", False):
                    self._did_init_right_bottom_split_sash = True
                    return

                total = right_bottom_split.winfo_height()
                if total <= 2:
                    self.after(10, _init_right_bottom_split_sash_once)
                    return

                min_top = 140
                min_bottom = 120
                desired_top = 280  # more space for Current Trades (like screenshot)
                target = max(min_top, min(total - min_bottom, desired_top))

                right_bottom_split.sashpos(0, int(target))
                self._did_init_right_bottom_split_sash = True
            except Exception:
                pass

        self.after_idle(_init_right_split_sash_once)
        self.after_idle(_init_right_bottom_split_sash_once)

        # Initial clamp once everything is laid out
        self.after_idle(lambda: (
            self._schedule_paned_clamp(getattr(self, "_pw_outer", None)),
            self._schedule_paned_clamp(getattr(self, "_pw_left_split", None)),
            self._schedule_paned_clamp(getattr(self, "_pw_right_split", None)),
            self._schedule_paned_clamp(getattr(self, "_pw_right_bottom_split", None)),
        ))

    # Panedwindow helper methods to prevent pane collapse during resize
    def _schedule_paned_clamp(self, pw: ttk.Panedwindow) -> None:
        """
        Debounced clamp so we don't fight the geometry manager mid-resize.

        IMPORTANT: use `after(1, ...)` instead of `after_idle(...)` so it still runs
        while the mouse is held during sash dragging (Tk often doesn't go "idle"
        until after the drag ends, which is exactly when panes can vanish).
        """
        try:
            if not pw or not int(pw.winfo_exists()):
                return
        except Exception:
            return

        key = str(pw)
        if key in self._paned_clamp_after_ids:
            return

        def _run():
            try:
                self._paned_clamp_after_ids.pop(key, None)
            except Exception:
                pass
            self._clamp_panedwindow_sashes(pw)

        try:
            self._paned_clamp_after_ids[key] = self.after(1, _run)
        except Exception:
            pass

    def _clamp_panedwindow_sashes(self, pw: ttk.Panedwindow) -> None:
        """
        Enforces each pane's configured 'minsize' by clamping sash positions.

        NOTE:
        ttk.Panedwindow.paneconfig(pane) typically returns dict values like:
            {"minsize": ("minsize", "minsize", "Minsize", "140"), ...}
        so we MUST pull the last element when it's a tuple/list.
        """
        try:
            if not pw or not int(pw.winfo_exists()):
                return

            panes = list(pw.panes())
            if len(panes) < 2:
                return

            orient = str(pw.cget("orient"))
            total = pw.winfo_height() if orient == "vertical" else pw.winfo_width()
            if total <= 2:
                return

            def _get_minsize(pane_id) -> int:
                try:
                    cfg = pw.paneconfig(pane_id)
                    ms = cfg.get("minsize", 0)

                    # ttk returns tuples like ('minsize','minsize','Minsize','140')
                    if isinstance(ms, (tuple, list)) and ms:
                        ms = ms[-1]

                    # sometimes it's already int/float-like, sometimes it's a string
                    return max(0, int(float(ms)))
                except tk.TclError:
                    # Silently ignore TclError during paneconfig (occurs during widget initialization)
                    return 0
                except (ValueError, TypeError, IndexError) as e:
                    # Debug: log minsize parsing errors
                    if self.settings.get("debug_mode", False):
                        print(f"[HUB DEBUG] Failed to parse minsize for pane {pane_id}: {e}")
                    return 0

            mins: List[int] = [_get_minsize(p) for p in panes]

            # If total space is smaller than sum(mins), we still clamp as best-effort
            # by scaling mins down proportionally but never letting a pane hit 0.
            if sum(mins) >= total:
                # best-effort: keep every pane at least 24px so it can’t disappear
                floor = 24
                mins = [max(floor, m) for m in mins]

                # if even floors don't fit, just stop here (window minsize should prevent this)
                if sum(mins) >= total:
                    return

            # Two-pass clamp so constraints settle even with multiple sashes
            for _ in range(2):
                for i in range(len(panes) - 1):
                    min_pos = sum(mins[: i + 1])
                    max_pos = total - sum(mins[i + 1 :])

                    try:
                        cur = int(pw.sashpos(i))
                    except (tk.TclError, ValueError, TypeError) as e:
                        # Debug: log sash position read errors
                        if self.settings.get("debug_mode", False):
                            print(f"[HUB DEBUG] Failed to read sash position {i}: {e}")
                        continue

                    new = max(min_pos, min(max_pos, cur))
                    if new != cur:
                        try:
                            pw.sashpos(i, new)
                        except tk.TclError as e:
                            # Debug: log sash position set errors (common during rapid resizing)
                            if self.settings.get("debug_mode", False):
                                print(f"[HUB DEBUG] Failed to set sash position {i} to {new}: {e}")

        except (tk.TclError, AttributeError) as e:
            # Debug: log panedwindow clamping errors
            if self.settings.get("debug_mode", False):
                print(f"[HUB DEBUG] Panedwindow sash clamping error: {e}")

    # Subprocess management methods for starting, stopping, and monitoring processes
    def _reader_thread(self, proc: subprocess.Popen, q: "queue.Queue[str]", prefix: str) -> None:
        try:
            # line-buffered text mode
            while True:
                line = proc.stdout.readline() if proc.stdout else ""
                if not line:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.05)
                    continue
                # Strip line and filter out any non-ASCII characters
                line = line.rstrip()
                # Remove BOM and invisible Unicode characters
                line = line.replace('\ufeff', '').replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
                try:
                    line = line.encode('ascii', 'ignore').decode('ascii')
                    # Remove any control characters except printable + newline/tab
                    line = ''.join(ch for ch in line if ch.isprintable() or ch in '\n\t')
                except (UnicodeError, AttributeError) as e:
                    # Debug: log character filtering errors
                    if self.settings.get("debug_mode", False):
                        print(f"[HUB DEBUG] Character filtering failed in {prefix}: {e}")
                q.put(f"{prefix}{line}")
        except Exception as e:
            # Debug: log reader thread errors
            if self.settings.get("debug_mode", False):
                print(f"[HUB DEBUG] Reader thread error for {prefix}: {e}")
        finally:
            q.put(f"{prefix}")  # Empty line before process exited
            q.put(f"{prefix}[Process Exited]")

    def _is_process_already_running(self, script_name: str) -> bool:
        """Check if a process with the given script name is already running"""
        try:
            current_pid = os.getpid()
            script_basename = os.path.basename(script_name)
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    # Skip the current process
                    if proc.info['pid'] == current_pid:
                        continue
                    
                    # Check if it's a Python process
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        cmdline = proc.info.get('cmdline')
                        if cmdline and any(script_basename in arg for arg in cmdline):
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception:
            pass
        return False

    def _start_process(self, p: ProcInfo, log_q: Optional["queue.Queue[str]"] = None, prefix: str = "") -> None:
        if p.proc and p.proc.poll() is None:
            return
        if not os.path.isfile(p.path):
            messagebox.showerror("Missing script", f"Cannot find: {p.path}")
            return

        # Check if process is already running
        if self._is_process_already_running(p.path):
            messagebox.showwarning(
                "Already Running",
                f"{p.name} is already running in another instance.\n\n"
                f"Please close the other instance before starting a new one."
            )
            return

        env = os.environ.copy()
        env["POWERTRADER_HUB_DIR"] = self.hub_dir  # so rhcb writes where GUI reads
        env["PYTHONUNBUFFERED"] = "1"  # Force unbuffered output for real-time logs

        try:
            # Hide console window on Windows if flag is set (can be disabled in Apollo.pyw for debugging)
            hide_console = os.environ.get('POWERTRADER_HIDE_CONSOLE', '1') == '1'
            creation_flags = subprocess.CREATE_NO_WINDOW if (sys.platform == "win32" and hide_console) else 0
            
            # Additional console hiding for Windows using STARTUPINFO
            startupinfo = None
            if sys.platform == "win32" and hide_console:
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            
            p.proc = subprocess.Popen(
                [sys.executable, "-u", p.path],  # -u for unbuffered prints
                cwd=self.project_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
                startupinfo=startupinfo,
            )
            if log_q is not None:
                t = threading.Thread(target=self._reader_thread, args=(p.proc, log_q, prefix), daemon=True)
                t.start()
        except Exception as e:
            messagebox.showerror("Failed to start", f"{p.name} failed to start:\n{e}")

    def _stop_process(self, p: ProcInfo) -> None:
        if not p.proc or p.proc.poll() is not None:
            return
        try:
            p.proc.terminate()
        except Exception:
            pass

    def start_neural(self) -> None:
        # Check training status before allowing manual start
        status_map = self._training_status_map()
        not_trained = [c for c, s in status_map.items() if s in ("NOT TRAINED", "ERROR", "STOPPED")]
        stale_coins = self._get_stale_coins()
        
        all_needs_training = list(set(not_trained + stale_coins))
        
        if all_needs_training:
            messagebox.showwarning(
                "Training Required",
                f"Training is stale or missing for: {', '.join(sorted(all_needs_training))}\n\n"
                f"Use Autopilot to train and start automatically, or train manually first."
            )
            return
        
        # Reset thinker-ready gate file (prevents stale "ready" from a prior run)
        try:
            with open(self.runner_ready_path, "w", encoding="utf-8") as f:
                json.dump({"timestamp": time.time(), "ready": False, "stage": "starting"}, f)
        except Exception:
            pass

        self._start_process(self.proc_neural, log_q=self.runner_log_q, prefix="[THINKER] ")

    def start_trader(self) -> None:
        # Check training status before allowing manual start
        status_map = self._training_status_map()
        not_trained = [c for c, s in status_map.items() if s in ("NOT TRAINED", "ERROR", "STOPPED")]
        stale_coins = self._get_stale_coins()
        
        all_needs_training = list(set(not_trained + stale_coins))
        
        if all_needs_training:
            messagebox.showwarning(
                "Training Required",
                f"Training is stale or missing for: {', '.join(sorted(all_needs_training))}\n\n"
                f"Use Autopilot to train and start automatically, or train manually first."
            )
            return
        
        self._start_process(self.proc_trader, log_q=self.trader_log_q, prefix="[TRADER] ")

    def stop_neural(self) -> None:
        self._stop_process(self.proc_neural)

    def stop_trader(self) -> None:
        self._stop_process(self.proc_trader)

    def toggle_all_scripts(self) -> None:
        neural_running = bool(self.proc_neural.proc and self.proc_neural.proc.poll() is None)
        trader_running = bool(self.proc_trader.proc and self.proc_trader.proc.poll() is None)
        auto_mode_active = getattr(self, "_auto_mode_active", False)

        # If auto mode is active (training phase or running), stop everything
        if auto_mode_active:
            # Stop any trainers that are running as part of auto mode
            pending_coins = getattr(self, "_auto_mode_pending_coins", set())
            for coin in pending_coins:
                lp = self.trainers.get(coin)
                if lp and lp.info.proc and lp.info.proc.poll() is None:
                    try:
                        lp.info.proc.terminate()
                    except Exception:
                        pass
            # Stop thinker/trader and clear auto mode flags
            self.stop_all_scripts()
            return

        # If BOTH are running (but not in auto mode), toggle means "stop trader only" (keep thinker running)
        if neural_running and trader_running:
            self.stop_trader()
            return

        # Otherwise, toggle means "start" (completes autopilot setup)
        # If thinker running: starts trader | If neither running: starts full autopilot
        self.start_all_scripts()
    
    def toggle_auto_start(self) -> None:
        """Toggle auto-start scripts setting and save to gui_settings.json."""
        self.settings["auto_start_scripts"] = bool(self.auto_start_var.get())
        try:
            _safe_write_json(SETTINGS_FILE, self.settings)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save auto-start setting:\n{e}")
    
    def toggle_debug_mode(self) -> None:
        """Toggle debug mode setting and save to gui_settings.json."""
        self.settings["debug_mode"] = bool(self.debug_mode_var.get())
        try:
            _safe_write_json(SETTINGS_FILE, self.settings)
            status = "enabled" if self.debug_mode_var.get() else "disabled"
            messagebox.showinfo(
                "Debug Mode", 
                f"Debug mode {status}.\n\n"
                f"Restart running processes (Thinker/Trader/Trainer) to apply changes."
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save debug mode setting:\n{e}")
    
    def toggle_simulation_mode(self) -> None:
        """Toggle simulation mode setting and save to gui_settings.json."""
        self.settings["simulation_mode"] = bool(self.simulation_mode_var.get())
        try:
            _safe_write_json(SETTINGS_FILE, self.settings)
            status = "enabled" if self.simulation_mode_var.get() else "disabled"
            
            if self.simulation_mode_var.get():
                message = (
                    f"Simulation mode {status}.\n\n"
                    f"⚠️ When enabled, NO REAL TRADES will be executed.\n"
                    f"All trading logic runs normally but orders are simulated.\n\n"
                    f"Restart the Trader process to apply changes."
                )
            else:
                message = (
                    f"Simulation mode {status}.\n\n"
                    f"⚠️ REAL TRADES will now be executed.\n"
                    f"All buy and sell orders will use real funds.\n\n"
                    f"Restart the Trader process to apply changes."
                )
            
            messagebox.showwarning("Simulation Mode", message)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save simulation mode setting:\n{e}")

    def _read_runner_ready(self) -> Dict[str, Any]:
        try:
            if os.path.isfile(self.runner_ready_path):
                with open(self.runner_ready_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {"ready": False}

    def _poll_runner_ready_then_start_trader(self) -> None:
        # Cancelled or already started
        if not bool(getattr(self, "_auto_start_trader_pending", False)):
            return

        # If thinker died, stop waiting
        if not (self.proc_neural.proc and self.proc_neural.proc.poll() is None):
            self._auto_start_trader_pending = False
            return

        st = self._read_runner_ready()
        if bool(st.get("ready", False)):
            self._auto_start_trader_pending = False

            # Start trader if not already running
            if not (self.proc_trader.proc and self.proc_trader.proc.poll() is None):
                self.start_trader()
            return

        # Not ready yet — keep polling
        try:
            self.after(250, self._poll_runner_ready_then_start_trader)
        except Exception:
            pass

    def start_all_scripts(self) -> None:
        """
        Legacy method name preserved for backwards compatibility.
        Simply delegates to start_auto_mode() which implements the full workflow.
        """
        self.start_auto_mode()

    def start_auto_mode(self) -> None:
        """
        Autopilot Mode: One-click intelligent startup of the complete trading system.
        
        Purpose:
        - Eliminate multi-step manual workflow (train → start thinker → start trader)
        - Handle training prerequisites automatically
        - Coordinate startup sequence with proper readiness gates
        - Provide user-friendly progress notifications
        
        Decision Logic:
        1. Check all configured coins for training status:
           - NOT TRAINED: No training data exists yet
           - STALE: Training data older than staleness_days setting
           - ERROR: Training failed or incomplete
           - TRAINED: Ready to trade
        
        2. If ANY coin needs training:
           Phase A (TRAINING): Train all coins that need it in parallel
           Phase B (THINKING): Start thinker, wait for runner_ready.json
           Phase C (TRADING): Start trader
        
        3. If all coins are trained and current:
           Phase B (THINKING): Start thinker immediately
           Phase C (TRADING): Start trader once ready
        
        Workflow Coordination:
        - Sets _auto_mode_active flag to prevent duplicate sessions
        - Tracks _auto_mode_pending_coins set for training completion polling
        - _tick() loop polls training status and advances phases automatically
        - User sees progress via "Auto Mode: TRAINING..." status label
        
        Readiness Gate (Phase B → C):
        - Thinker writes runner_ready.json when first predictions are available
        - Hub polls runner_ready_path every tick
        - Prevents trader from starting with no predictions (would fail immediately)
        
        Error Handling:
        - Validates training completeness before starting thinker
        - Re-checks staleness after training (guards against training failure)
        - Displays user-friendly messages for each phase transition
        
        User Experience:
        - Single button click: "🚀 ENGAGE AUTOPILOT"
        - Progress displayed in status bar: "Auto Mode: TRAINING...", "Auto Mode: THINKING...", etc.
        - Info dialogs explain what's happening at each phase
        - Button changes to "⏹ STOP AUTOPILOT" while active
        """
        # Prevent duplicate Auto Mode sessions
        if getattr(self, "_auto_mode_active", False):
            return
        
        # Check training status
        status_map = self._training_status_map()
        needs_training = [c for c, s in status_map.items() if s in ("NOT TRAINED", "ERROR", "STOPPED")]
        currently_training = [c for c, s in status_map.items() if s == "TRAINING"]
        
        # Check staleness (always respect staleness_days from training_settings.json)
        stale_coins = self._get_stale_coins()
        
        all_needs_training = list(set(needs_training + stale_coins))
        
        if all_needs_training or currently_training:
            # Phase 1: Training required or already in progress
            self._auto_mode_active = True
            self._auto_mode_phase = "TRAINING"
            self._auto_mode_pending_coins = set(all_needs_training + currently_training)
            
            # Show info about what's being trained
            if currently_training and not all_needs_training:
                # Training already in progress, just wait for completion
                messagebox.showinfo(
                    "🚀 Autopilot: Training In Progress",
                    f"Training is already underway for: {', '.join(sorted(currently_training))}\n\n"
                    f"Autopilot will wait for training to complete, then start thinker and trader.\n\n"
                    f"This may take several hours."
                )
            else:
                # Need to start new training
                messagebox.showinfo(
                    "🚀 Autopilot: Training Required",
                    f"Training is needed for: {', '.join(sorted(all_needs_training))}\n\n"
                    f"Autopilot will train these coins first, then start thinker and trader.\n\n"
                    f"This may take several hours."
                )
                
                # Start trainers for coins that need it (don't restart already-training coins)
                for coin in all_needs_training:
                    if coin not in currently_training:
                        self.trainer_coin_var.set(coin)
                        self.start_trainer_for_selected_coin()
            
            # Poll for training completion
            self.after(2000, self._poll_auto_mode_training)
        else:
            # Phase 2: Already trained, go straight to thinker→trader
            self._auto_mode_active = True
            self._auto_mode_phase = "RUNNING"
            self._auto_start_trader_pending = True
            self.start_neural()
            
            # Wait for runner to signal readiness before starting trader
            try:
                self.after(250, self._poll_runner_ready_then_start_trader)
            except Exception:
                pass
            
            # Clear auto mode flag (handed off to existing runner→trader logic)
            self._auto_mode_active = False

    def _poll_auto_mode_training(self) -> None:
        """Poll training status during Auto Mode training phase."""
        if not getattr(self, "_auto_mode_active", False):
            return
        
        if getattr(self, "_auto_mode_phase", "") != "TRAINING":
            return
        
        # Check if all pending coins are done training
        status_map = self._training_status_map()
        pending = getattr(self, "_auto_mode_pending_coins", set())
        
        still_training = [c for c in pending if status_map.get(c) == "TRAINING"]
        errored = [c for c in pending if status_map.get(c) == "ERROR"]
        
        if errored:
            self._auto_mode_active = False
            self._auto_mode_phase = ""
            messagebox.showerror(
                "🚀 Autopilot: Training Failed",
                f"Training failed for: {', '.join(sorted(errored))}\n\n"
                f"Autopilot stopped. Check the training logs for details."
            )
            return
        
        if still_training:
            # Still training, check again in 2 seconds
            self.after(2000, self._poll_auto_mode_training)
        else:
            # All trained! Move to Phase 2: Start thinker→trader
            self._auto_mode_phase = "RUNNING"
            self._auto_start_trader_pending = True
            self.start_neural()
            
            # Wait for runner to signal readiness before starting trader
            try:
                self.after(250, self._poll_runner_ready_then_start_trader)
            except Exception:
                pass
            
            # Clear auto mode flag (handed off to existing runner→trader logic)
            self._auto_mode_active = False

    def _poll_auto_retrain_completion(self) -> None:
        """Poll training status during automatic retraining (when approaching staleness)."""
        if not getattr(self, "_auto_retraining_active", False):
            return
        
        # Check if all pending coins are done training
        status_map = self._training_status_map()
        pending = getattr(self, "_auto_retrain_pending_coins", set())
        
        still_training = [c for c in pending if status_map.get(c) == "TRAINING"]
        errored = [c for c in pending if status_map.get(c) == "ERROR"]
        
        if errored:
            self._auto_retraining_active = False
            self._auto_retrain_pending_coins = set()
            messagebox.showerror(
                "Auto-Retrain Failed",
                f"Auto-retraining failed for: {', '.join(sorted(errored))}\n\n"
                f"System remains stopped. Check training logs and use Autopilot to restart."
            )
            return
        
        if still_training:
            # Still training, check again in 2 seconds
            self.after(2000, self._poll_auto_retrain_completion)
        else:
            # All retrained! Auto-restart using Auto Mode
            self._auto_retraining_active = False
            self._auto_retrain_pending_coins = set()
            
            messagebox.showinfo(
                "Auto-Retrain Complete",
                "Training has been updated successfully.\n\n"
                "System will now restart automatically."
            )
            
            # Use Auto Mode to restart (it will skip training since we just completed it)
            self.start_auto_mode()

    def _compute_file_checksum(self, filepath: str) -> str:
        """Compute SHA256 checksum of a file. Returns empty string on error."""
        try:
            sha256 = hashlib.sha256()
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            # Debug: log checksum computation errors
            if self.settings.get("debug_mode", False):
                print(f"[HUB DEBUG] Checksum computation failed for {filepath}: {e}")
            return ""

    def _get_cached_trading_config(self) -> dict:
        """
        Get trading config with mtime-based caching.
        Only reloads from disk when the file actually changes.
        """
        try:
            mtime = os.path.getmtime(TRADING_SETTINGS_FILE) if os.path.isfile(TRADING_SETTINGS_FILE) else None
        except Exception:
            mtime = None

        # Return cached config if mtime matches
        if self._cached_trading_config_mtime == mtime and self._cached_trading_config is not None:
            return self._cached_trading_config

        # Reload config (file changed or first access)
        config = _load_trading_config()
        self._cached_trading_config = config
        self._cached_trading_config_mtime = mtime

        # Debug: log config reload events
        if self.settings.get("debug_mode", False):
            print(f"[HUB DEBUG] Trading config reloaded (mtime: {mtime})")

        return config

    def _coin_is_trained(self, coin: str) -> bool:
        """Check if coin has ALL required timeframes trained with valid memory files."""
        coin = coin.upper().strip()
        folder = self.coin_folders.get(coin, "")
        if not folder or not os.path.isdir(folder):
            return False

        # If trainer reports it's currently training, stopped, or errored, it's not "trained" yet.
        try:
            st = _safe_read_json(os.path.join(folder, "trainer_status.json"))
            if isinstance(st, dict):
                state = str(st.get("state", "")).upper()
                if state in ("TRAINING", "ERROR", "STOPPED"):
                    return False
        except Exception:
            pass

        # Check timestamp is recent
        stamp_path = os.path.join(folder, "trainer_last_training_time.txt")
        try:
            if not os.path.isfile(stamp_path):
                return False
            with open(stamp_path, "r", encoding="utf-8") as f:
                raw = (f.read() or "").strip()
            ts = float(raw) if raw else 0.0
            if ts <= 0:
                return False
            if (time.time() - ts) > (14 * 24 * 60 * 60):
                return False
        except Exception:
            return False
        
        # Verify ALL required timeframes have valid memory files
        for tf in REQUIRED_THINKER_TIMEFRAMES:
            memory_file = os.path.join(folder, f"memories_{tf}.dat")
            weight_file = os.path.join(folder, f"memory_weights_{tf}.dat")
            threshold_file = os.path.join(folder, f"neural_perfect_threshold_{tf}.dat")
            
            # All three files must exist and be non-empty
            for fpath in [memory_file, weight_file, threshold_file]:
                if not os.path.isfile(fpath):
                    return False
                try:
                    # Threshold files contain single float values (e.g., "5.0", "25.0") and are legitimately small
                    # Memory/weight files contain pattern data and should be larger
                    min_size = 2 if "threshold" in os.path.basename(fpath) else 10
                    if os.path.getsize(fpath) < min_size:
                        return False
                except Exception:
                    return False
        
        return True

    def _get_stale_coins(self) -> List[str]:
        """Returns list of coins whose training is stale (older than staleness_days from training_settings.json)."""
        training_cfg = _load_training_config()
        staleness_days = training_cfg.get("staleness_days", 14)
        max_age_seconds = staleness_days * 24 * 60 * 60
        
        stale = []
        for coin in self.coins:
            folder = self.coin_folders.get(coin, "")
            if not folder:
                continue
            
            stamp_path = os.path.join(folder, "trainer_last_training_time.txt")
            try:
                if not os.path.isfile(stamp_path):
                    stale.append(coin)
                    continue
                
                with open(stamp_path, "r", encoding="utf-8") as f:
                    raw = (f.read() or "").strip()
                ts = float(raw) if raw else 0.0
                
                if ts <= 0 or (time.time() - ts) > max_age_seconds:
                    stale.append(coin)
            except Exception:
                stale.append(coin)
        
        return stale

    def _get_coins_near_stale(self, threshold_hours: float = 2.0) -> List[str]:
        """
        Returns list of coins whose training will become stale within threshold_hours.
        Used for proactive auto-retraining.
        """
        training_cfg = _load_training_config()
        staleness_days = training_cfg.get("staleness_days", 14)
        max_age_seconds = staleness_days * 24 * 60 * 60
        threshold_seconds = threshold_hours * 60 * 60
        
        near_stale = []
        for coin in self.coins:
            folder = self.coin_folders.get(coin, "")
            if not folder:
                continue
            
            stamp_path = os.path.join(folder, "trainer_last_training_time.txt")
            try:
                if not os.path.isfile(stamp_path):
                    # No training file = already stale, not "near stale"
                    continue
                
                with open(stamp_path, "r", encoding="utf-8") as f:
                    raw = (f.read() or "").strip()
                ts = float(raw) if raw else 0.0
                
                if ts <= 0:
                    continue
                
                age_seconds = time.time() - ts
                time_until_stale = max_age_seconds - age_seconds
                
                # If training will become stale within threshold_hours, flag it
                if 0 < time_until_stale <= threshold_seconds:
                    near_stale.append(coin)
            except Exception:
                pass
        
        return near_stale

    def _get_time_until_stale_hours(self, coin: str) -> float:
        """
        Returns hours until training becomes stale for a coin.
        Returns 0.0 if already stale or no training data.
        Returns the calculated hours if still valid.
        """
        training_cfg = _load_training_config()
        staleness_days = training_cfg.get("staleness_days", 14)
        max_age_seconds = staleness_days * 24 * 60 * 60
        
        folder = self.coin_folders.get(coin, "")
        if not folder:
            return 0.0
        
        stamp_path = os.path.join(folder, "trainer_last_training_time.txt")
        try:
            if not os.path.isfile(stamp_path):
                return 0.0
            
            with open(stamp_path, "r", encoding="utf-8") as f:
                raw = (f.read() or "").strip()
            ts = float(raw) if raw else 0.0
            
            if ts <= 0:
                return 0.0
            
            age_seconds = time.time() - ts
            time_until_stale_seconds = max_age_seconds - age_seconds
            
            if time_until_stale_seconds <= 0:
                return 0.0
            
            # Convert to hours
            return time_until_stale_seconds / 3600.0
        except Exception:
            return 0.0

    def _running_trainers(self) -> List[str]:
        running: List[str] = []

        # Trainers launched by this GUI instance
        for c, lp in self.trainers.items():
            try:
                if lp.info.proc and lp.info.proc.poll() is None:
                    running.append(c)
            except Exception:
                pass

        # Trainers launched elsewhere: look at per-coin status file
        for c in self.coins:
            try:
                coin = (c or "").strip().upper()
                folder = self.coin_folders.get(coin, "")
                if not folder or not os.path.isdir(folder):
                    continue

                status_path = os.path.join(folder, "trainer_status.json")
                st = _safe_read_json(status_path)

                if isinstance(st, dict) and str(st.get("state", "")).upper() == "TRAINING":
                    # Verify the process is actually running by checking for pt_trainer.py process
                    try:
                        found_process = False
                        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                            try:
                                cmdline = proc.info.get('cmdline') or []
                                if any('pt_trainer.py' in str(arg) and coin in str(arg) for arg in cmdline):
                                    found_process = True
                                    break
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                continue
                        
                        # If no process found, mark as STOPPED
                        if not found_process:
                            try:
                                st["state"] = "STOPPED"
                                st["timestamp"] = int(time.time())
                                with open(status_path, "w", encoding="utf-8") as f:
                                    json.dump(st, f)
                            except Exception:
                                pass
                            continue
                    except Exception:
                        pass
                    
                    stamp_path = os.path.join(folder, "trainer_last_training_time.txt")

                    try:
                        if os.path.isfile(stamp_path) and os.path.isfile(status_path):
                            if os.path.getmtime(stamp_path) >= os.path.getmtime(status_path):
                                continue
                    except Exception:
                        pass

                    running.append(coin)
            except Exception:
                pass

        # de-dupe while preserving order
        out: List[str] = []
        seen = set()
        for c in running:
            cc = (c or "").strip().upper()
            if cc and cc not in seen:
                seen.add(cc)
                out.append(cc)
        return out

    def _training_status_map(self) -> Dict[str, str]:
        """
        Returns {coin: "TRAINED" | "TRAINING" | "ERROR" | "NOT TRAINED"}.
        """
        running = set(self._running_trainers())
        out: Dict[str, str] = {}
        for c in self.coins:
            if c in running:
                out[c] = "TRAINING"
            elif self._coin_is_trained(c):
                out[c] = "TRAINED"
            else:
                # Check if there's an error or stopped state
                try:
                    folder = self.coin_folders.get(c, "")
                    if folder:
                        st = _safe_read_json(os.path.join(folder, "trainer_status.json"))
                        if isinstance(st, dict):
                            state = str(st.get("state", "")).upper()
                            if state == "ERROR":
                                out[c] = "ERROR"
                                continue
                            elif state == "STOPPED":
                                out[c] = "STOPPED"
                                continue
                except Exception:
                    pass
                out[c] = "NOT TRAINED"
        return out

    def train_selected_coin(self) -> None:
        coin = (getattr(self, 'train_coin_var', self.trainer_coin_var).get() or "").strip().upper()

        if not coin:
            return
        # Reuse the trainers pane runner — start trainer for selected coin
        self.start_trainer_for_selected_coin()

    def train_all_coins(self) -> None:
        # Start trainers for every coin (in parallel)
        for c in self.coins:
            self.trainer_coin_var.set(c)
            self.start_trainer_for_selected_coin()

    def start_trainer_for_selected_coin(self) -> None:
        coin = (self.trainer_coin_var.get() or "").strip().upper()
        if not coin:
            return

        # Stop the Thinker before any training starts (training modifies artifacts the thinker reads)
        self.stop_neural()

        # --- IMPORTANT ---
        # All coins use their own folder (BTC uses BTC/, ETH uses ETH/, etc.)
        coin_cwd = self.coin_folders.get(coin, self.project_dir)

        # Use the trainer script that lives INSIDE that coin's folder so outputs land in the right place.
        trainer_name = os.path.basename(str(self.settings.get("script_neural_trainer", "pt_trainer.py")))

        # If coin folder doesn't exist yet, create it and copy the trainer script.
        # Also check version and force copy if mismatch or unreadable.

        if coin not in self.coin_folders:
            try:
                if not os.path.isdir(coin_cwd):
                    os.makedirs(coin_cwd, exist_ok=True)
            except Exception:
                pass

        # Always ensure coin trainer is up to date with root trainer
        src_main_trainer = os.path.join(self.project_dir, trainer_name)
        dst_trainer_path = os.path.join(coin_cwd, trainer_name)

        # Copy from root if: 1) coin trainer missing, OR 2) checksum mismatch
        should_copy = False
        if not os.path.isfile(dst_trainer_path):
            should_copy = True
        elif os.path.isfile(src_main_trainer):
            # Check if checksums match
            root_checksum = self._compute_file_checksum(src_main_trainer)
            coin_checksum = self._compute_file_checksum(dst_trainer_path)
            if root_checksum != coin_checksum:
                should_copy = True

        if should_copy and os.path.isfile(src_main_trainer):
            try:
                shutil.copy2(src_main_trainer, dst_trainer_path)
            except Exception:
                pass

        trainer_path = os.path.join(coin_cwd, trainer_name)

        if not os.path.isfile(trainer_path):
            messagebox.showerror(
                "Missing trainer",
                f"Cannot find trainer for {coin} at:\n{trainer_path}"
            )
            return

        if coin in self.trainers and self.trainers[coin].info.proc and self.trainers[coin].info.proc.poll() is None:
            return

        try:
            patterns = [
                "trainer_last_training_time.txt",
                "trainer_status.json",
                "trainer_last_start_time.txt",
                "killer.txt",
                "bounce_accuracy.txt",
                "memories_*.dat",
                "memory_weights_*.dat",
                "neural_perfect_threshold_*.dat",
            ]

            deleted = 0
            for pat in patterns:
                for fp in glob.glob(os.path.join(coin_cwd, pat)):
                    try:
                        os.remove(fp)
                        deleted += 1
                    except Exception:
                        pass

            if deleted:
                try:
                    self.status.config(text=f"Deleted {deleted} training file(s) for {coin} before training")
                except Exception:
                    pass
        except Exception:
            pass

        q: "queue.Queue[str]" = queue.Queue()
        info = ProcInfo(name=f"Trainer-{coin}", path=trainer_path)

        env = os.environ.copy()
        env["POWERTRADER_HUB_DIR"] = self.hub_dir
        env["PYTHONUNBUFFERED"] = "1"  # Force unbuffered output for real-time logs

        try:
            # Hide console window on Windows if flag is set (can be disabled in Apollo.pyw for debugging)
            hide_console = os.environ.get('POWERTRADER_HIDE_CONSOLE', '1') == '1'
            creation_flags = subprocess.CREATE_NO_WINDOW if (sys.platform == "win32" and hide_console) else 0
            
            # Additional console hiding for Windows using STARTUPINFO
            startupinfo = None
            if sys.platform == "win32" and hide_console:
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            
            # Pass `coin` argument so trainer processes the correct market
            info.proc = subprocess.Popen(
                [sys.executable, "-u", info.path, coin],
                cwd=coin_cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
                startupinfo=startupinfo,
            )
            t = threading.Thread(target=self._reader_thread, args=(info.proc, q, f"[{coin}] "), daemon=True)
            t.start()

            self.trainers[coin] = LogProc(info=info, log_q=q, thread=t, is_trainer=True, coin=coin)
        except Exception as e:
            messagebox.showerror("Failed to start", f"Trainer for {coin} failed to start:\n{e}")

    def stop_trainer_for_selected_coin(self) -> None:
        coin = (self.trainer_coin_var.get() or "").strip().upper()
        lp = self.trainers.get(coin)
        if not lp or not lp.info.proc or lp.info.proc.poll() is not None:
            return
        try:
            lp.info.proc.terminate()
        except Exception:
            pass

    def stop_all_scripts(self) -> None:
        # Cancel any pending "wait for runner then start trader"
        self._auto_start_trader_pending = False
        
        # Clear auto-mode state (in case user stops during auto-mode training)
        self._auto_mode_active = False
        self._auto_mode_phase = ""
        
        # Note: Don't clear _auto_retraining_active here - it's managed by the auto-retrain flow
        # and needs to persist through stop/restart cycle

        self.stop_neural()
        self.stop_trader()

        # Also reset the thinker-ready gate file (best-effort)
        try:
            with open(self.runner_ready_path, "w", encoding="utf-8") as f:
                json.dump({"timestamp": time.time(), "ready": False, "stage": "stopped"}, f)
        except Exception:
            pass

    def _on_timeframe_changed(self, event) -> None:
        """
        Immediate redraw when the user changes a timeframe in any CandleChart.
        Avoids waiting for the chart_refresh_seconds throttle in _tick().
        """
        try:
            chart = getattr(event, "widget", None)
            if not isinstance(chart, CandleChart):
                return

            coin = getattr(chart, "coin", None)
            if not coin:
                return

            self.coin_folders = build_coin_folders(self.settings["main_neural_dir"], self.coins)

            pos = self._last_positions.get(coin, {}) if isinstance(self._last_positions, dict) else {}
            buy_px = pos.get("current_buy_price", None)
            sell_px = pos.get("current_sell_price", None)
            trail_line = pos.get("trail_line", None)
            dca_line_price = pos.get("dca_line_price", None)

            chart.refresh(
                self.coin_folders,
                current_buy_price=buy_px,
                current_sell_price=sell_px,
                trail_line=trail_line,
                dca_line_price=dca_line_price,
            )

            # Keep the periodic refresh behavior consistent (prevents an immediate full refresh right after this).
            self._last_chart_refresh = time.time()
        except Exception:
            pass

    # UI refresh methods for updating logs, charts, and status displays
    def _drain_queue_to_text(self, q: "queue.Queue[str]", txt: tk.Text, user_scrolled_away: bool, max_lines: int = 2500) -> None:
        """Drain log queue to text widget with smart auto-scroll.
        
        Args:
            q: Queue containing log messages
            txt: Text widget to display messages
            user_scrolled_away: Flag indicating if user has scrolled away from bottom
            max_lines: Maximum number of lines to keep in widget
        """
        try:
            changed = False
            while True:
                line = q.get_nowait()
                txt.insert("end", line + "\n")
                changed = True
        except queue.Empty:
            pass
        except Exception:
            pass

        if changed:
            # trim very old lines
            try:
                current = int(txt.index("end-1c").split(".")[0])
                if current > max_lines:
                    txt.delete("1.0", f"{current - max_lines}.0")
            except Exception:
                pass
            
            # Only auto-scroll if user hasn't scrolled away from bottom
            if not user_scrolled_away:
                try:
                    # Smoother scroll: use yview_moveto instead of see()
                    txt.update_idletasks()
                    txt.yview_moveto(1.0)
                except Exception:
                    txt.see("end")

    def _auto_start_thinker_if_trained(self) -> None:
        """
        Auto-start thinker on Hub startup if at least one coin is trained.
        
        This provides a better user experience: if any training is complete,
        the thinker starts automatically and begins generating predictions immediately
        for those coins. Untrained coins will show warnings and be skipped.
        The user can then use the Autopilot button to start the trader when ready.
        
        The thinker can still be manually stopped via the Scripts menu if needed.
        """
        try:
            # Don't auto-start if user manually started something already
            neural_running = bool(self.proc_neural.proc and self.proc_neural.proc.poll() is None)
            trader_running = bool(self.proc_trader.proc and self.proc_trader.proc.poll() is None)
            if neural_running or trader_running:
                return
            
            # Check if at least one coin is trained and current
            status_map = self._training_status_map()
            trained_coins = [c for c, s in status_map.items() if s == "TRAINED"]
            stale_coins = self._get_stale_coins()
            
            # Remove stale coins from trained list
            current_trained = [c for c in trained_coins if c not in stale_coins]
            
            # If at least one coin is trained and current, start thinker automatically
            if current_trained:
                # Reset thinker-ready gate file
                try:
                    with open(self.runner_ready_path, "w", encoding="utf-8") as f:
                        json.dump({"timestamp": time.time(), "ready": False, "stage": "starting"}, f)
                except Exception:
                    pass
                
                self._start_process(self.proc_neural, log_q=self.runner_log_q, prefix="[THINKER] ")
        except Exception as e:
            # Don't show error dialog on startup - just log if debug mode
            if self.settings.get("debug_mode", False):
                print(f"[HUB DEBUG] Auto-start thinker check failed: {e}")
    
    def _check_first_time_setup(self) -> None:
        """Check if this is first-time setup and open settings dialog if needed."""
        try:
            needs_setup = False
            
            # Check if gui_settings.json exists or is minimal (default config)
            gui_settings_path = os.path.join(self.project_dir, "gui_settings.json")
            if not os.path.isfile(gui_settings_path):
                needs_setup = True
                if self.settings.get("debug_mode", False):
                    print("[HUB DEBUG] gui_settings.json not found - opening setup dialog")
            else:
                # Check if it's a minimal/default config (likely first run)
                try:
                    with open(gui_settings_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        # If file is empty or very small, consider it uninitialized
                        if len(content) < 50:
                            needs_setup = True
                            if self.settings.get("debug_mode", False):
                                print("[HUB DEBUG] gui_settings.json is minimal - opening setup dialog")
                except Exception:
                    pass
            
            # Check if API keys are configured
            key_path = os.path.join(self.project_dir, "rh_key.enc")
            secret_path = os.path.join(self.project_dir, "rh_secret.enc")
            
            if not os.path.isfile(key_path) or not os.path.isfile(secret_path):
                needs_setup = True
                if self.settings.get("debug_mode", False):
                    print("[HUB DEBUG] API keys not found - opening setup dialog")
            
            # Open settings dialog if setup is needed
            if needs_setup:
                if self.settings.get("debug_mode", False):
                    print("[HUB DEBUG] First-time setup needed - opening settings dialog")
                self.open_settings_dialog()
        except Exception as e:
            # Silently fail - this is a convenience feature
            if self.settings.get("debug_mode", False):
                print(f"[HUB DEBUG] _check_first_time_setup failed: {e}")
    
    def _fetch_initial_account_info(self) -> None:
        """
        Fetch Robinhood account data on Hub startup (before trader is started).
        
        Purpose:
        - Display account balance immediately without waiting for trader
        - Provide instant feedback on API key configuration
        - Populate account chart with initial data point
        
        Process:
        1. Check for encrypted API keys (rh_key.enc, rh_secret.enc)
        2. Decrypt keys using Windows DPAPI
        3. Create Ed25519 signature for Robinhood API authentication
        4. Fetch account data (buying power + holdings)
        5. Calculate total account value (cash + crypto holdings at bid price)
        6. Write to trader_status.json (for UI labels)
        7. Append to account_value_history.jsonl (for chart)
        8. Display success/error message in Trader log
        
        Authentication:
        - Uses Ed25519 signature authentication (Robinhood Crypto API requirement)
        - Timestamp must be Unix seconds (not milliseconds)
        - Signature covers: api_key + timestamp + path + method + body
        
        Error Handling:
        - Silently skips if API keys not configured (first-time user experience)
        - Displays user-friendly messages in Trader log for API failures
        - Gracefully degrades if NaCl library unavailable
        
        UI Updates:
        - Resets trader_status.json mtime cache to force immediate UI refresh
        - Triggers account chart redraw
        - Displays success message with account breakdown in Trader log
        """
        print("[HUB] Attempting to fetch initial account info...")
        try:
            # Check if API keys exist
            key_path = os.path.join(self.project_dir, "rh_key.enc")
            secret_path = os.path.join(self.project_dir, "rh_secret.enc")
            
            if not os.path.isfile(key_path) or not os.path.isfile(secret_path):
                print("[HUB] API keys not found, skipping initial account fetch")
                # Write a message to the trader log
                try:
                    if hasattr(self, 'trader_text') and self.trader_text.winfo_exists():
                        msg = "ℹ Account info not loaded - Robinhood API keys not configured.\n"
                        msg += "Go to Settings → Robinhood API to set up your API keys.\n\n"
                        self.trader_text.insert("end", msg)
                except Exception as e:
                    print(f"[HUB] Could not write to trader_text: {e}")
                return
            
            # Read and decrypt API keys
            print("[HUB] Reading and decrypting API keys...")
            try:
                with open(key_path, 'rb') as f:
                    api_key = _decrypt_with_dpapi(f.read()).strip()
                with open(secret_path, 'rb') as f:
                    private_key_b64 = _decrypt_with_dpapi(f.read()).strip()
                print("[HUB] API keys decrypted successfully")
            except Exception as e:
                print(f"[HUB] Failed to decrypt API keys: {e}")
                return
            
            if not api_key or not private_key_b64:
                print("[HUB] API keys are empty after decryption")
                return
            
            # Import nacl for signing (lazy import to avoid dependency if not needed)
            try:
                from nacl.signing import SigningKey
                print("[HUB] NaCl library imported successfully")
            except ImportError:
                print("[HUB] NaCl library not available, skipping account fetch")
                return
            
            # Initialize signing key
            try:
                private_key_seed = base64.b64decode(private_key_b64)
                private_key = SigningKey(private_key_seed)
                print("[HUB] Signing key initialized successfully")
            except Exception as e:
                print(f"[HUB] Failed to initialize signing key: {e}")
                return
            
            # Make API request to get account info
            base_url = "https://trading.robinhood.com"
            path = "/api/v1/crypto/trading/accounts/"
            timestamp = int(time.time())  # Use seconds, not milliseconds (matches test code)
            body = ""
            method = "GET"
            
            message_to_sign = f"{api_key}{timestamp}{path}{method}{body}"
            signed = private_key.sign(message_to_sign.encode("utf-8"))
            
            headers = {
                "x-api-key": api_key,
                "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
                "x-timestamp": str(timestamp),
            }
            
            # Fetch account data
            print(f"[HUB] Making API request to {base_url + path}...")
            try:
                response = requests.get(base_url + path, headers=headers, timeout=10)
                print(f"[HUB] API response status: {response.status_code}")
                if response.status_code == 200:
                    account_data = response.json()
                    
                    # Write to trader_status.json so the UI can display it
                    # API returns account data directly (not in a "results" array)
                    if account_data and "buying_power" in account_data:
                        buying_power = float(account_data.get("buying_power", 0))
                        
                        # Try to fetch holdings for more accurate account value
                        holdings_sell_value = 0.0
                        holdings_buy_value = 0.0
                        total_account_value = buying_power
                        
                        try:
                            # Fetch holdings
                            print("[HUB] Fetching holdings...")
                            holdings_path = "/api/v1/crypto/trading/holdings/"
                            timestamp_h = int(time.time())  # Use seconds
                            message_h = f"{api_key}{timestamp_h}{holdings_path}{method}{body}"
                            signed_h = private_key.sign(message_h.encode("utf-8"))
                            headers_h = {
                                "x-api-key": api_key,
                                "x-signature": base64.b64encode(signed_h.signature).decode("utf-8"),
                                "x-timestamp": str(timestamp_h),
                            }
                            holdings_resp = requests.get(base_url + holdings_path, headers=headers_h, timeout=10)
                            
                            if holdings_resp.status_code == 200:
                                holdings_data = holdings_resp.json()
                                holdings_list = holdings_data.get("results", []) if isinstance(holdings_data, dict) else []
                                
                                # Get symbols for price lookup
                                symbols = [h["asset_code"] + "-USD" for h in holdings_list if h.get("asset_code") != "USDC"]
                                
                                if symbols:
                                    # Fetch trading pairs to get current prices
                                    pairs_path = "/api/v1/crypto/trading/trading_pairs/"
                                    timestamp_p = int(time.time())  # Use seconds
                                    message_p = f"{api_key}{timestamp_p}{pairs_path}{method}{body}"
                                    signed_p = private_key.sign(message_p.encode("utf-8"))
                                    headers_p = {
                                        "x-api-key": api_key,
                                        "x-signature": base64.b64encode(signed_p.signature).decode("utf-8"),
                                        "x-timestamp": str(timestamp_p),
                                    }
                                    pairs_resp = requests.get(base_url + pairs_path, headers=headers_p, timeout=10)
                                    
                                    if pairs_resp.status_code == 200:
                                        pairs_data = pairs_resp.json()
                                        prices = {}
                                        pairs_list = pairs_data.get("results", []) if isinstance(pairs_data, dict) else []
                                        for pair in pairs_list:
                                            sym = pair.get("symbol")
                                            prices[sym] = {
                                                "bid": float(pair.get("bid_inclusive_of_sell_spread", 0)),
                                                "ask": float(pair.get("ask_inclusive_of_buy_spread", 0))
                                            }
                                        
                                        # Calculate holdings value
                                        for holding in holdings_list:
                                            asset = holding.get("asset_code")
                                            if asset == "USDC":
                                                continue
                                            qty = float(holding.get("total_quantity", 0))
                                            if qty <= 0:
                                                continue
                                            sym = f"{asset}-USD"
                                            if sym in prices:
                                                holdings_buy_value += qty * prices[sym]["ask"]
                                                holdings_sell_value += qty * prices[sym]["bid"]
                                        
                                        total_account_value = buying_power + holdings_sell_value
                        except Exception as e_holdings:
                            # If holdings fetch fails, just use buying_power
                            if self.settings.get("debug_mode", False):
                                print(f"[HUB DEBUG] Holdings fetch failed (using buying_power only): {e_holdings}")
                        
                        # Calculate percent in trade
                        percent_in_trade = (holdings_sell_value / total_account_value * 100) if total_account_value > 0 else 0.0
                        # Create a minimal trader_status.json with account info
                        status = {
                            "timestamp": time.time(),
                            "simulation_mode": bool(self.settings.get("simulation_mode", False)),
                            "account": {
                                "total_account_value": total_account_value,
                                "buying_power": buying_power,
                                "holdings_sell_value": holdings_sell_value,
                                "holdings_buy_value": holdings_buy_value,
                                "percent_in_trade": percent_in_trade,
                            },
                            "positions": {},
                            "_source": "hub_initial_fetch"
                        }
                        
                        # Write atomically
                        tmp_path = self.trader_status_path + ".tmp"
                        with open(tmp_path, "w", encoding="utf-8") as f:
                            json.dump(status, f, indent=2)
                        os.replace(tmp_path, self.trader_status_path)
                        
                        # Also write to account value history for the chart
                        try:
                            history_entry = {
                                "ts": time.time(),
                                "total_account_value": total_account_value
                            }
                            with open(self.account_value_history_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(history_entry) + "\n")
                            # Track last written value for duplicate checking in periodic fetches
                            self._last_written_account_value = total_account_value
                        except Exception:
                            pass
                        
                        # Reset mtime cache and refresh UI
                        try:
                            self._last_trader_status_mtime = None
                        except Exception:
                            pass
                        
                        # Force an immediate UI refresh to display the account info
                        try:
                            self._refresh_trader_status()
                        except Exception:
                            pass
                        
                        # Refresh the account chart to show the new data point
                        try:
                            if hasattr(self, 'account_chart'):
                                self.account_chart.refresh()
                        except Exception:
                            pass
                        
                        # Write success message to trader log
                        try:
                            if hasattr(self, 'trader_text') and self.trader_text.winfo_exists():
                                msg = f"✓ Account info loaded: ${total_account_value:.2f} (${buying_power:.2f} buying power, ${holdings_sell_value:.2f} in holdings)\n\n"
                                self.trader_text.insert("end", msg)
                        except Exception:
                            pass
                        
                        # Success - return to avoid error messages below
                        return
                    else:
                        # account_data doesn't have expected structure
                        try:
                            if hasattr(self, 'trader_text') and self.trader_text.winfo_exists():
                                msg = "⚠ Account info fetch failed: unexpected API response structure\n\n"
                                self.trader_text.insert("end", msg)
                        except Exception:
                            pass
                else:
                    # API request failed
                    print(f"[HUB] API request failed with status {response.status_code}")
                    try:
                        if hasattr(self, 'trader_text') and self.trader_text.winfo_exists():
                            msg = f"⚠ Account info fetch failed (HTTP {response.status_code})\n"
                            msg += "Check your API keys in Settings → Robinhood API\n\n"
                            self.trader_text.insert("end", msg)
                    except Exception as e:
                        print(f"[HUB] Could not write error to trader_text: {e}")
            except requests.Timeout:
                print("[HUB] API request timed out")
                try:
                    if hasattr(self, 'trader_text') and self.trader_text.winfo_exists():
                        msg = "⚠ Account info fetch timed out\n"
                        msg += "Check your internet connection\n\n"
                        self.trader_text.insert("end", msg)
                except Exception as e:
                    print(f"[HUB] Could not write error to trader_text: {e}")
            except Exception as e:
                print(f"[HUB] API request failed: {e}")
                try:
                    if hasattr(self, 'trader_text') and self.trader_text.winfo_exists():
                        msg = f"⚠ Account info fetch error: {str(e)[:100]}\n\n"
                        self.trader_text.insert("end", msg)
                except Exception as e2:
                    print(f"[HUB] Could not write error to trader_text: {e2}")
        except Exception as e:
            # Log all outer exceptions
            print(f"[HUB] _fetch_initial_account_info failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_account_data_silent(self) -> None:
        """
        Background account data refresh (silent, no user notifications).
        
        Purpose:
        - Keep account chart and labels current when trader is stopped
        - Provide real-time account value updates during non-trading hours
        - Enable account monitoring without starting the trader
        
        Behavior:
        - Called every 10 seconds (default) by _tick() loop when trader is NOT running
        - Skips automatically when trader starts (trader handles updates at higher frequency)
        - All errors handled silently (no console output, no UI messages)
        - Only writes new data points if account value changed by > $0.01 (prevents flat redundant lines)
        
        Data Flow:
        1. Fetch account data from Robinhood API (same as _fetch_initial_account_info)
        2. Calculate total account value (buying power + holdings)
        3. Compare to last written value (_last_written_account_value)
        4. If changed significantly (> $0.01), write to:
           - trader_status.json (account labels auto-refresh via mtime cache)
           - account_value_history.jsonl (chart auto-updates via mtime cache)
        5. Reset mtime cache to trigger UI refresh on next tick
        
        Integration with Trader:
        - Hub writes "_source": "hub_periodic_fetch" to trader_status.json
        - Trader writes "_source": "trader_update" to trader_status.json
        - Both sources are compatible (same JSON schema)
        - No file conflicts: Hub checks trader_running flag before writing
        
        Performance:
        - Minimal API calls: Respects configurable interval (default 10s)
        - Duplicate filtering: Only writes when value changes
        - Silent failures: Network issues don't spam console or dialogs
        """
        try:
            # Check if API keys exist
            key_path = os.path.join(self.project_dir, "rh_key.enc")
            secret_path = os.path.join(self.project_dir, "rh_secret.enc")
            
            if not os.path.isfile(key_path) or not os.path.isfile(secret_path):
                return  # No keys, skip silently
            
            # Read and decrypt API keys
            try:
                with open(key_path, 'rb') as f:
                    api_key = _decrypt_with_dpapi(f.read()).strip()
                with open(secret_path, 'rb') as f:
                    private_key_b64 = _decrypt_with_dpapi(f.read()).strip()
            except Exception:
                return  # Decryption failed, skip silently
            
            if not api_key or not private_key_b64:
                return  # Empty keys, skip silently
            
            # Import nacl for signing
            try:
                from nacl.signing import SigningKey
            except ImportError:
                return  # NaCl not available, skip silently
            
            # Initialize signing key
            try:
                private_key_seed = base64.b64decode(private_key_b64)
                private_key = SigningKey(private_key_seed)
            except Exception:
                return  # Key initialization failed, skip silently
            
            # Make API request
            base_url = "https://trading.robinhood.com"
            path = "/api/v1/crypto/trading/accounts/"
            timestamp = int(time.time())
            body = ""
            method = "GET"
            
            message_to_sign = f"{api_key}{timestamp}{path}{method}{body}"
            signed = private_key.sign(message_to_sign.encode("utf-8"))
            
            headers = {
                "x-api-key": api_key,
                "x-signature": base64.b64encode(signed.signature).decode("utf-8"),
                "x-timestamp": str(timestamp),
            }
            
            try:
                response = requests.get(base_url + path, headers=headers, timeout=10)
                if response.status_code == 200:
                    account_data = response.json()
                    
                    if account_data and "buying_power" in account_data:
                        buying_power = float(account_data.get("buying_power", 0))
                        holdings_sell_value = 0.0
                        holdings_buy_value = 0.0
                        total_account_value = buying_power
                        
                        # Try to fetch holdings for more accurate account value
                        try:
                            holdings_path = "/api/v1/crypto/trading/holdings/"
                            timestamp_h = int(time.time())
                            message_h = f"{api_key}{timestamp_h}{holdings_path}{method}{body}"
                            signed_h = private_key.sign(message_h.encode("utf-8"))
                            headers_h = {
                                "x-api-key": api_key,
                                "x-signature": base64.b64encode(signed_h.signature).decode("utf-8"),
                                "x-timestamp": str(timestamp_h),
                            }
                            holdings_resp = requests.get(base_url + holdings_path, headers=headers_h, timeout=10)
                            
                            if holdings_resp.status_code == 200:
                                holdings_data = holdings_resp.json()
                                holdings_list = holdings_data.get("results", []) if isinstance(holdings_data, dict) else []
                                
                                # Get symbols for price lookup
                                symbols = [h["asset_code"] + "-USD" for h in holdings_list if h.get("asset_code") != "USDC"]
                                
                                if symbols:
                                    # Fetch trading pairs to get current prices
                                    pairs_path = "/api/v1/crypto/trading/trading_pairs/"
                                    timestamp_p = int(time.time())
                                    message_p = f"{api_key}{timestamp_p}{pairs_path}{method}{body}"
                                    signed_p = private_key.sign(message_p.encode("utf-8"))
                                    headers_p = {
                                        "x-api-key": api_key,
                                        "x-signature": base64.b64encode(signed_p.signature).decode("utf-8"),
                                        "x-timestamp": str(timestamp_p),
                                    }
                                    pairs_resp = requests.get(base_url + pairs_path, headers=headers_p, timeout=10)
                                    
                                    if pairs_resp.status_code == 200:
                                        pairs_data = pairs_resp.json()
                                        prices = {}
                                        pairs_list = pairs_data.get("results", []) if isinstance(pairs_data, dict) else []
                                        for pair in pairs_list:
                                            sym = pair.get("symbol")
                                            prices[sym] = {
                                                "bid": float(pair.get("bid_inclusive_of_sell_spread", 0)),
                                                "ask": float(pair.get("ask_inclusive_of_buy_spread", 0))
                                            }
                                        
                                        # Calculate holdings value
                                        for holding in holdings_list:
                                            asset = holding.get("asset_code")
                                            if asset == "USDC":
                                                continue
                                            qty = float(holding.get("total_quantity", 0))
                                            if qty <= 0:
                                                continue
                                            sym = f"{asset}-USD"
                                            if sym in prices:
                                                holdings_buy_value += qty * prices[sym]["ask"]
                                                holdings_sell_value += qty * prices[sym]["bid"]
                                        
                                        total_account_value = buying_power + holdings_sell_value
                        except Exception:
                            pass  # Holdings fetch failed, use buying_power only
                        
                        # Calculate percent in trade
                        percent_in_trade = (holdings_sell_value / total_account_value * 100) if total_account_value > 0 else 0.0
                        
                        # Create trader_status.json entry
                        status = {
                            "timestamp": time.time(),
                            "simulation_mode": bool(self.settings.get("simulation_mode", False)),
                            "account": {
                                "total_account_value": total_account_value,
                                "buying_power": buying_power,
                                "holdings_sell_value": holdings_sell_value,
                                "holdings_buy_value": holdings_buy_value,
                                "percent_in_trade": percent_in_trade,
                            },
                            "positions": {},
                            "_source": "hub_periodic_fetch"
                        }
                        
                        # Write atomically
                        tmp_path = self.trader_status_path + ".tmp"
                        with open(tmp_path, "w", encoding="utf-8") as f:
                            json.dump(status, f, indent=2)
                        os.replace(tmp_path, self.trader_status_path)
                        
                        # Also write to account value history for the chart
                        # Only write if value changed significantly (> $0.01) to avoid duplicate flat lines
                        try:
                            last_written_value = getattr(self, "_last_written_account_value", None)
                            value_changed = (last_written_value is None or 
                                           abs(total_account_value - last_written_value) > 0.01)
                            
                            if value_changed:
                                history_entry = {
                                    "ts": time.time(),
                                    "total_account_value": total_account_value
                                }
                                with open(self.account_value_history_path, "a", encoding="utf-8") as f:
                                    f.write(json.dumps(history_entry) + "\n")
                                self._last_written_account_value = total_account_value
                        except Exception:
                            pass
                        
                        # Reset mtime cache and refresh UI (no blocking UI operations)
                        try:
                            self._last_trader_status_mtime = None
                        except Exception:
                            pass
                        
                        # Chart will refresh on next tick cycle (already throttled)
            except requests.Timeout:
                pass  # Timeout, skip silently
            except Exception:
                pass  # Other error, skip silently
        except Exception:
            pass  # Outer exception, skip silently
    
    def _display_startup_accuracy(self) -> None:
        """Display bounce accuracy and signal accuracy results for all coins on Hub startup."""
        try:
            trained_coins_found = 0
            
            # Only check files if coins are configured
            if self.coins:
                for coin in self.coins:
                    coin_folder = self.coin_folders.get(coin)
                    if not coin_folder:
                        continue
                    
                    bounce_accuracy_file = os.path.join(coin_folder, "bounce_accuracy.txt")
                    signal_accuracy_file = os.path.join(coin_folder, "signal_accuracy.txt")
                    
                    # Initialize history for this coin if not present
                    if coin not in self.trainer_log_history:
                        self.trainer_log_history[coin] = []
                    
                    if not os.path.isfile(bounce_accuracy_file):
                        # No training data yet - add a helpful message
                        msg = f"Training Status ({coin})\n"
                        msg += f"Status: Not yet trained\n\n"
                        msg += f"ℹ This coin has not been trained yet.\n"
                        msg += f"Click 'Start Trainer' to begin training for {coin}.\n\n"
                        
                        # Add to history
                        for line in msg.split("\n"):
                            if line or msg.endswith("\n"):
                                self.trainer_log_history[coin].append(line)
                        
                        # Display if this coin is currently selected
                        current_coin = (self.trainer_coin_var.get() or "").strip().upper()
                        if coin == current_coin:
                            self.trainer_text.insert("end", msg)
                            self.trainer_text.see("end")
                            # Reset scroll flag when loading initial status
                            self._trainer_log_user_scrolled_away = False
                        
                        continue
                    
                    try:
                        # Read bounce accuracy file
                        with open(bounce_accuracy_file, 'r', encoding='utf-8') as f:
                            bounce_lines = f.read().strip().split('\n')
                        
                        if len(bounce_lines) < 2:
                            continue
                        
                        # Parse bounce accuracy content
                        timestamp = bounce_lines[0].replace('Last Updated: ', '').strip()
                        bounce_average = bounce_lines[1].replace('Average: ', '').strip()
                        
                        # Parse bounce accuracy per timeframe
                        bounce_tf_dict = {}
                        suspicious_accuracy = False
                        for line in bounce_lines[2:]:
                            if ':' in line:
                                parts = line.split(':')
                                tf = parts[0].strip()
                                value = parts[1].strip()
                                bounce_tf_dict[tf] = value
                                # Check if any timeframe has >= 99% accuracy
                                try:
                                    accuracy_value = float(value.replace('%', ''))
                                    if accuracy_value >= 99.0:
                                        suspicious_accuracy = True
                                except:
                                    pass
                        
                        # Try to read signal accuracy file
                        signal_average = None
                        signal_tf_dict = {}
                        if os.path.isfile(signal_accuracy_file):
                            try:
                                with open(signal_accuracy_file, 'r', encoding='utf-8') as f:
                                    signal_lines = f.read().strip().split('\n')
                                
                                if len(signal_lines) >= 2:
                                    signal_average = signal_lines[1].replace('Average: ', '').strip()
                                    
                                    # Parse signal accuracy per timeframe
                                    for line in signal_lines[2:]:
                                        if ':' in line:
                                            parts = line.split(':')
                                            tf = parts[0].strip()
                                            value = parts[1].strip()
                                            signal_tf_dict[tf] = value
                            except:
                                pass  # Signal accuracy not critical, skip if unavailable
                        
                        # Use the same training validation logic as the training window
                        if self._coin_is_trained(coin):
                            trained_coins_found += 1
                        
                        # Format and display
                        msg = f"Training Accuracy Results ({coin})\n"
                        msg += f"Last trained: {timestamp}\n\n"
                        msg += f"Average Limit-Breach Accuracy: {bounce_average}\n"
                        if signal_average:
                            msg += f"Average Signal Accuracy:       {signal_average}\n"
                        msg += f"\nPer Timeframe:\n"
                        
                        # Display per-timeframe results
                        for tf in sorted(bounce_tf_dict.keys()):
                            bounce_val = bounce_tf_dict[tf]
                            if signal_tf_dict and tf in signal_tf_dict:
                                signal_val = signal_tf_dict[tf]
                                msg += f"  {tf:8} | Limit: {bounce_val:>6} | Signal: {signal_val:>6}\n"
                            else:
                                msg += f"  {tf:8} | Limit: {bounce_val:>6}\n"
                        msg += "\n"
                        
                        # Add warning if suspicious accuracy detected
                        if suspicious_accuracy:
                            msg += f"⚠ WARNING: Accuracy of 100% detected. This may indicate incomplete training.\n"
                            msg += f"⚠ Please verify that training completed properly and memories were saved.\n"
                            msg += f"\n\n"
                        
                        # Display error if coin is not fully trained
                        if not self._coin_is_trained(coin):
                            msg += f"❌ ERROR: Training incomplete - missing or invalid model files.\n"
                            msg += f"❌ Please retrain this coin to complete the training process.\n"
                            msg += f"\n\n"
                        
                        # Split message into lines and add to history (coin already initialized above)
                        for line in msg.split("\n"):
                            if line or msg.endswith("\n"):  # Preserve empty lines if they were in the original
                                self.trainer_log_history[coin].append(line)
                        
                        # Keep history manageable
                        if len(self.trainer_log_history[coin]) > 2500:
                            self.trainer_log_history[coin] = self.trainer_log_history[coin][-2500:]
                        
                        # Insert into trainer log if this coin is currently selected
                        current_coin = (self.trainer_coin_var.get() or "").strip().upper()
                        if coin == current_coin:
                            self.trainer_text.insert("end", msg)
                            self.trainer_text.see("end")
                            # Reset scroll flag when loading initial status
                            self._trainer_log_user_scrolled_away = False
                    
                    except Exception as e:
                        # Silently skip files that can't be read
                        pass
            
            # If no trained coins were found, display a helpful message in Thinker log
            if trained_coins_found == 0:
                msg = (
                    "⚠ No trained coins detected\n\n"
                    "The Thinker requires trained AI models to generate trading signals.\n"
                    "Please train at least one coin before starting the Thinker:\n\n"
                    "  1. Select a coin (BTC, ETH, etc.)\n"
                    "  2. Click 'Train' or 'Train All'\n"
                    "  3. Wait for training to complete (may take several hours)\n"
                    "  4. Once training finishes, the Thinker can be started\n\n"
                    "Training creates AI memory patterns that predict price movements.\n"
                )
                
                try:
                    self.runner_text.insert("end", msg)
                    self.runner_text.see("end")
                    # Reset scroll flag when loading initial message
                    self._runner_log_user_scrolled_away = False
                except Exception:
                    pass
        except Exception as e:
            if self.settings.get("debug_mode", False):
                import traceback
                traceback.print_exc()
            else:
                # If not in debug mode, at least print the error
                print(f"Error checking trained coins: {e}")

    def _tick(self) -> None:
        """
        Main UI update loop - refreshes all Hub components periodically.
        
        Execution Frequency:
        - Runs every 1 second (default, configurable via "ui_refresh_seconds")
        - Reschedules itself at the end of each cycle
        
        Primary Responsibilities:
        1. Monitor subprocess status (trainer/thinker/trader running state)
        2. Check for training data staleness and trigger auto-retrain
        3. Fetch periodic account data when trader is stopped
        4. Update status labels (Thinker/Trader running indicators)
        5. Update button states based on training completeness
        6. Refresh data-driven panels (trader status, PnL, trade history)
        7. Refresh charts (throttled to separate configurable interval)
        8. Drain log queues into UI text widgets
        9. Poll for Autopilot/auto-retrain workflow completion
        
        Performance Optimizations:
        - Mtime caching: Only redraws UI when underlying data files change
        - Chart throttling: Charts refresh at lower frequency (default 10s)
        - Subprocess polling: Uses poll() for zero-overhead process checking
        - Queue draining: Non-blocking get_nowait() prevents thread contention
        
        Staleness Monitoring:
        - RUNTIME CHECK: Immediately stops everything if training becomes stale during trading
        - AUTO-RETRAIN: Proactively retrains coins approaching staleness (24hr warning)
        - Batch retraining: Retrains all coins near staleness in one operation
        
        Autopilot Integration:
        - Coordinates multi-phase workflow: train → start thinker → start trader
        - Polls for completion of each phase before advancing
        - Handles auto-start of trader once thinker reports ready
        """
        
        # Subprocess Status Detection
        # Check if processes are alive by polling (non-blocking, no overhead)
        neural_running = bool(self.proc_neural.proc and self.proc_neural.proc.poll() is None)
        trader_running = bool(self.proc_trader.proc and self.proc_trader.proc.poll() is None)

        # --- RUNTIME STALENESS CHECK ---
        # If thinker or trader is running, check if training has become stale
        # If so, stop everything and notify the user
        if neural_running or trader_running:
            stale_coins = self._get_stale_coins()
            if stale_coins:
                # Training has become stale during runtime - stop everything
                self.stop_all_scripts()
                messagebox.showwarning(
                    "Training Data Stale",
                    f"Training data has become stale for: {', '.join(sorted(stale_coins))}\n\n"
                    f"Thinker and Trader have been stopped.\n\n"
                    f"Use Autopilot to retrain and restart."
                )

        # --- AUTO-RETRAIN CHECK (proactive retraining before staleness) ---
        # Check once per minute (throttle to avoid excessive checks)
        now = time.time()
        last_check = getattr(self, "_last_auto_retrain_check", 0)
        
        if (now - last_check) >= 60:  # Check every 60 seconds
            self._last_auto_retrain_check = now
            
            # Only proceed if setting is enabled and not already retraining
            training_cfg = _load_training_config()
            auto_train_enabled = training_cfg.get("auto_train_when_stale", False)
            auto_retraining = getattr(self, "_auto_retraining_active", False)
            
            if auto_train_enabled and not auto_retraining and (neural_running or trader_running):
                # Check if any coins have gone stale
                stale_coins = self._get_stale_coins()
                
                if stale_coins:
                    # At least one coin is stale - batch retrain everything within 24 hours of staleness
                    near_stale = self._get_coins_near_stale(threshold_hours=24.0)
                    
                    # Combine stale + near-stale for batch retraining
                    all_to_retrain = list(set(stale_coins + near_stale))
                    
                    # Training staleness reached - stop, retrain, restart
                    self._auto_retraining_active = True
                    self._auto_retrain_pending_coins = set(all_to_retrain)
                    
                    # Stop everything gracefully
                    self.stop_all_scripts()
                    
                    # Show notification
                    stale_list = ', '.join(sorted(stale_coins))
                    if near_stale:
                        batch_list = ', '.join(sorted(all_to_retrain))
                        messagebox.showinfo(
                            "Auto-Retrain Starting",
                            f"Training data is stale for: {stale_list}\n\n"
                            f"Batch retraining all coins within 24hrs of staleness: {batch_list}\n\n"
                            f"System will restart automatically when complete.\n\n"
                            f"Estimated time: several hours per coin."
                        )
                    else:
                        messagebox.showinfo(
                            "Auto-Retrain Starting",
                            f"Training data is stale for: {stale_list}\n\n"
                            f"System will auto-retrain and restart when complete.\n\n"
                            f"Estimated time: several hours per coin."
                        )
                    
                    # Start training for all coins that need it
                    for coin in all_to_retrain:
                        self.trainer_coin_var.set(coin)
                        self.start_trainer_for_selected_coin()
                    
                    # Poll for completion
                    self.after(2000, self._poll_auto_retrain_completion)

        # --- PERIODIC ACCOUNT DATA REFRESH ---
        # Fetch fresh account data at regular intervals to keep chart/labels current
        # Skip if trader is running (trader handles account updates at higher frequency)
        if not trader_running:
            last_account_fetch = getattr(self, "_last_account_fetch_time", 0)
            # Match chart refresh frequency for uniform x-axis updates (default 10 seconds)
            chart_refresh_sec = float(self.settings.get("chart_refresh_seconds", 10.0))
            account_fetch_interval = float(self.settings.get("account_refresh_seconds", chart_refresh_sec))
            
            if (now - last_account_fetch) >= account_fetch_interval:
                self._last_account_fetch_time = now
                # Fetch in background (non-blocking, silent)
                try:
                    self._update_account_data_silent()
                except Exception as e:
                    if self.settings.get("debug_mode", False):
                        print(f"[HUB DEBUG] Periodic account fetch failed: {e}")

        # Determine status for thinker and trader (STOPPED, WAITING, or RUNNING)
        auto_mode_active = getattr(self, "_auto_mode_active", False)
        auto_mode_phase = getattr(self, "_auto_mode_phase", "")
        
        # Show WAITING when autopilot is engaged but processes aren't running yet
        # Show RUNNING when processes are active
        # Show STOPPED otherwise
        if neural_running:
            neural_status = "RUNNING"
        elif auto_mode_active or (auto_mode_phase in ("TRAINING", "RUNNING") and self._auto_start_trader_pending):
            neural_status = "WAITING"
        else:
            neural_status = "STOPPED"
        
        if trader_running:
            trader_status = "RUNNING"
        elif auto_mode_active or (auto_mode_phase in ("TRAINING", "RUNNING") and self._auto_start_trader_pending):
            trader_status = "WAITING"
        else:
            trader_status = "STOPPED"
        
        self.lbl_neural.config(text=f"Thinker: {neural_status}")
        self.lbl_trader.config(text=f"Trader: {trader_status}")

        # Start All is now a toggle (Start/Stop)
        # Show STOP when auto mode is active (training/thinking/trading) OR when both thinker and trader are running
        try:
            if hasattr(self, "btn_toggle_all") and self.btn_toggle_all:
                auto_mode_active = getattr(self, "_auto_mode_active", False)
                if auto_mode_active or (neural_running and trader_running):
                    self.btn_toggle_all.config(text="⏹ STOP AUTOPILOT")
                else:
                    self.btn_toggle_all.config(text="🚀 ENGAGE AUTOPILOT")
        except Exception:
            pass

        # --- Auto Mode button state ---
        status_map = self._training_status_map()
        all_trained = all(v == "TRAINED" for v in status_map.values()) if status_map else False

        # Auto Mode is always available when idle (it handles training automatically)
        # Only disable during active auto-retrain operations
        auto_retraining = getattr(self, "_auto_retraining_active", False)
        can_toggle_all = not auto_retraining

        try:
            self.btn_toggle_all.configure(state=("normal" if can_toggle_all else "disabled"))
        except Exception:
            pass

        # Training overview + per-coin list
        try:
            training_running = [c for c, s in status_map.items() if s == "TRAINING"]
            not_trained = [c for c, s in status_map.items() if s in ("NOT TRAINED", "ERROR", "STOPPED")]

            if training_running:
                self.lbl_training_overview.config(text=f"Training: RUNNING ({', '.join(training_running)})")
            elif not_trained:
                self.lbl_training_overview.config(text=f"Training: REQUIRED ({len(not_trained)} need training)")
            else:
                self.lbl_training_overview.config(text="Training: READY (all trained)")

            # show each coin status (ONLY redraw the list if it actually changed)
            # Include time-until-stale for trained coins
            # Build signature with time-until-stale for TRAINED coins
            sig_data = []
            for c in self.coins:
                st = status_map.get(c, "N/A")
                if st == "TRAINED":
                    hours_remaining = self._get_time_until_stale_hours(c)
                    # Round to nearest hour for signature to avoid constant redraws
                    sig_data.append((c, st, int(hours_remaining)))
                else:
                    sig_data.append((c, st, None))
            
            sig = tuple(sig_data)
            if getattr(self, "_last_training_sig", None) != sig:
                self._last_training_sig = sig
                self.training_list.delete(0, "end")
                for c, st, hours in sig:
                    # For trained coins, show time until stale
                    if st == "TRAINED" and hours is not None and hours > 0:
                        self.training_list.insert("end", f"{c}: {st.upper()} (T-{hours} HRS)")
                    else:
                        self.training_list.insert("end", f"{c}: {st.upper()}")

            # show gating hint (Auto Mode handles train->runner->ready->trader sequence)
            auto_mode_active = getattr(self, "_auto_mode_active", False)
            auto_mode_phase = getattr(self, "_auto_mode_phase", "")
            
            # Determine the current workflow state and what's next
            if auto_mode_active and auto_mode_phase == "TRAINING":
                pending = getattr(self, "_auto_mode_pending_coins", set())
                self.lbl_flow_hint.config(text=f"Flow: TRAIN ({len(pending)} coins) → THINK → TRADE")
            elif training_running:
                # Training is actively running
                self.lbl_flow_hint.config(text=f"Flow: TRAIN ⧗ THINK → TRADE")
            elif not all_trained:
                # Training needed before anything else
                self.lbl_flow_hint.config(text="Flow: TRAIN → THINK → TRADE")
            elif self._auto_start_trader_pending:
                # Training done, thinker starting
                self.lbl_flow_hint.config(text="Flow: TRAIN ✓ THINK ⧗ TRADE")
            elif neural_running and trader_running:
                # Everything is running
                self.lbl_flow_hint.config(text="Flow: TRAIN ✓ THINK ✓ TRADE ✓")
            elif neural_running:
                # Only neural/thinker is running
                self.lbl_flow_hint.config(text="Flow: TRAIN ✓ THINK ✓ TRADE")
            elif trader_running:
                # Only trader is running (unusual but possible)
                self.lbl_flow_hint.config(text="Flow: TRAIN ✓ THINK → TRADE ⟳")
            else:
                # All trained, ready to start
                self.lbl_flow_hint.config(text="Flow: THINK ✓ THINK → TRADE")
        except Exception:
            pass

        # neural overview bars (mtime-cached inside)
        self._refresh_neural_overview()

        # trader status -> current trades table (now mtime-cached inside)
        self._refresh_trader_status()

        # pnl ledger -> realized profit (now mtime-cached inside)
        self._refresh_pnl()

        # trade history (now mtime-cached inside)
        self._refresh_trade_history()

        # charts (throttle)
        now = time.time()
        if (now - self._last_chart_refresh) >= float(self.settings.get("chart_refresh_seconds", 10.0)):
            # account value chart (internally mtime-cached already)
            try:
                if self.account_chart:
                    self.account_chart.refresh()
            except Exception:
                pass

            # Only rebuild coin_folders when inputs change (avoids directory scans every refresh)
            try:
                cf_sig = (self.settings.get("main_neural_dir"), tuple(self.coins))
                if getattr(self, "_coin_folders_sig", None) != cf_sig:
                    self._coin_folders_sig = cf_sig
                    self.coin_folders = build_coin_folders(self.settings["main_neural_dir"], self.coins)
            except Exception:
                try:
                    self.coin_folders = build_coin_folders(self.settings["main_neural_dir"], self.coins)
                except Exception:
                    pass

            # Refresh ONLY the currently visible coin tab (prevents O(N_coins) network/plot stalls)
            selected_tab = None

            # Primary: our custom chart pages (multi-row tab buttons)
            try:
                selected_tab = getattr(self, "_current_chart_page", None)
            except Exception:
                selected_tab = None

            # Fallback: old notebook-based UI (if it exists)
            if not selected_tab:
                try:
                    if hasattr(self, "nb") and self.nb:
                        selected_tab = self.nb.tab(self.nb.select(), "text")
                except Exception:
                    selected_tab = None

            if selected_tab and str(selected_tab).strip().upper() != "ACCOUNT":
                coin = str(selected_tab).strip().upper()
                chart = self.charts.get(coin)
                if chart:
                    pos = self._last_positions.get(coin, {}) if isinstance(self._last_positions, dict) else {}
                    buy_px = pos.get("current_buy_price", None)
                    sell_px = pos.get("current_sell_price", None)
                    trail_line = pos.get("trail_line", None)
                    dca_line_price = pos.get("dca_line_price", None)
                    try:
                        chart.refresh(
                            self.coin_folders,
                            current_buy_price=buy_px,
                            current_sell_price=sell_px,
                            trail_line=trail_line,
                            dca_line_price=dca_line_price,
                        )
                    except Exception:
                        pass

            self._last_chart_refresh = now

        # drain logs into panes
        self._drain_queue_to_text(self.runner_log_q, self.runner_text, self._runner_log_user_scrolled_away)
        self._drain_queue_to_text(self.trader_log_q, self.trader_text, self._trader_log_user_scrolled_away)

        # trainer logs: show selected trainer output
        try:
            sel = (self.trainer_coin_var.get() or "").strip().upper()
            running = [c for c, lp in self.trainers.items() if lp.info.proc and lp.info.proc.poll() is None]
            self.trainer_status_lbl.config(text=f"running: {', '.join(running)}" if running else "(no trainers running)")

            lp = self.trainers.get(sel)
            if lp:
                # Initialize history for this coin if not present
                if sel not in self.trainer_log_history:
                    self.trainer_log_history[sel] = []
                
                # Drain new messages and add to history
                try:
                    while True:
                        line = lp.log_q.get_nowait()
                        self.trainer_log_history[sel].append(line)
                        # Keep history manageable (last 2500 lines)
                        if len(self.trainer_log_history[sel]) > 2500:
                            self.trainer_log_history[sel] = self.trainer_log_history[sel][-2500:]
                except queue.Empty:
                    pass
                
                # Display to text widget (only if this coin is currently selected)
                if self.trainer_log_history[sel]:
                    # Check if we need to update the display
                    current_line_count = int(self.trainer_text.index("end-1c").split(".")[0])
                    history_line_count = len(self.trainer_log_history[sel])
                    
                    # If mismatch, refresh the entire display
                    if current_line_count != history_line_count:
                        self.trainer_text.delete("1.0", "end")
                        self.trainer_text.insert("end", "\n".join(self.trainer_log_history[sel]))
                        if not self.trainer_log_history[sel][-1].endswith("\n"):
                            self.trainer_text.insert("end", "\n")
                        
                        # Only auto-scroll if user hasn't scrolled away from bottom
                        if not self._trainer_log_user_scrolled_away:
                            self.trainer_text.see("end")
        except Exception:
            pass

        # Auto-Switch to Priority Coin
        # Check if a coin is approaching a trigger and switch chart view automatically
        self._check_auto_switch()

        self.status.config(text=f"{_now_str()}  |  v{VERSION}  |  {self.settings['main_neural_dir']}")
        self.after(int(float(self.settings.get("ui_refresh_seconds", 1.0)) * 1000), self._tick)

    def _check_auto_switch(self) -> None:
        """
        Check if auto-switch should activate based on coin proximity to trading actions.
        Monitors four conditions:
        1. New buy entry signal (if slots available)
        2. DCA trigger (existing position approaching next DCA level)
        3. Stop loss trigger (existing position approaching stop loss)
        4. Take profit trigger (existing position approaching trailing stop)
        
        Priority when multiple coins qualify:
        1. Closest by % distance to trigger
        2. Action type: stop loss > DCA > new entry > take profit
        3. Largest position value
        
        Guards:
        - Only runs if auto-switch enabled in GUI settings
        - Respects manual override timer (2 minutes after user selection)
        - Only switches if current chart is different from priority
        """
        try:
            # Load auto-switch settings from GUI settings
            auto_switch_cfg = self.settings.get("auto_switch", {})
            
            if not auto_switch_cfg.get("enabled", False):
                self._hide_priority_alert()
                return
            
            threshold = auto_switch_cfg.get("threshold_pct", 2.0)
            
            # Check manual override timer
            import time
            now = time.time()
            time_since_manual = now - self._last_manual_chart_switch
            
            if time_since_manual < self._manual_override_duration:
                self._hide_priority_alert()
                return
            
            # Load trader status for position data
            trader_data = _safe_read_json(self.trader_status_path)
            if not trader_data:
                self._hide_priority_alert()
                return
            
            positions = trader_data.get("positions", {})
            if not positions:
                self._hide_priority_alert()
                return
            
            # Load trading config for thresholds and limits
            trading_cfg = self._get_cached_trading_config()
            long_signal_min = trading_cfg.get("entry_signals", {}).get("long_signal_min", 4)
            stop_loss_pct = trading_cfg.get("profit_margin", {}).get("stop_loss_pct", -40.0)
            max_concurrent = trading_cfg.get("position_sizing", {}).get("max_concurrent_positions", 3)
            
            # Count active positions (slots used)
            active_positions = sum(1 for pos in positions.values() if float(pos.get("quantity", 0.0)) > 0)
            slots_available = max_concurrent - active_positions
            
            # Evaluate each coin for proximity to trading actions
            candidates = []
            
            for coin in self.coins:
                pos = positions.get(coin)
                if not pos:
                    continue
                
                current_price = float(pos.get("current_sell_price", 0.0) or 0.0)
                if current_price <= 0:
                    continue
                
                quantity = float(pos.get("quantity", 0.0))
                has_position = quantity > 0
                
                # Condition 1: New buy entry (if signal strong and slots available)
                if not has_position and slots_available > 0:
                    entry_distance = self._calculate_entry_distance(coin, current_price, long_signal_min)
                    if entry_distance is not None and entry_distance <= threshold:
                        candidates.append({
                            "coin": coin,
                            "distance": entry_distance,
                            "reason": "New Entry",
                            "priority": 3,  # new entry priority
                            "position_value": 0.0
                        })
                
                # Conditions 2-4 only apply to existing positions
                if has_position:
                    avg_cost = float(pos.get("avg_cost_basis", 0.0))
                    position_value = float(pos.get("value_usd", 0.0))
                    
                    if avg_cost <= 0:
                        continue
                    
                    # Condition 2: DCA trigger
                    dca_line = float(pos.get("dca_line_price", 0.0))
                    if dca_line > 0:
                        dca_distance = abs(current_price - dca_line) / current_price * 100.0
                        if dca_distance <= threshold:
                            candidates.append({
                                "coin": coin,
                                "distance": dca_distance,
                                "reason": "DCA Trigger",
                                "priority": 2,  # DCA priority
                                "position_value": position_value
                            })
                    
                    # Condition 3: Stop loss
                    stop_loss_price = avg_cost * (1.0 + stop_loss_pct / 100.0)
                    stop_distance = abs(current_price - stop_loss_price) / current_price * 100.0
                    if stop_distance <= threshold and current_price <= stop_loss_price * 1.05:  # approaching from above
                        candidates.append({
                            "coin": coin,
                            "distance": stop_distance,
                            "reason": "Stop Loss",
                            "priority": 1,  # highest priority
                            "position_value": position_value
                        })
                    
                    # Condition 4: Take profit (trailing stop)
                    trail_active = pos.get("trail_active", False)
                    trail_line = float(pos.get("trail_line", 0.0))
                    if trail_active and trail_line > 0:
                        trail_distance = abs(current_price - trail_line) / current_price * 100.0
                        if trail_distance <= threshold:
                            candidates.append({
                                "coin": coin,
                                "distance": trail_distance,
                                "reason": "Take Profit",
                                "priority": 4,  # lowest priority
                                "position_value": position_value
                            })
            
            # No candidates within threshold
            if not candidates:
                self._hide_priority_alert()
                return
            
            # Sort by priority: closest distance, then action type, then largest position
            candidates.sort(key=lambda x: (x["distance"], x["priority"], -x["position_value"]))
            
            # Select top priority coin
            priority_coin = candidates[0]["coin"]
            priority_distance = candidates[0]["distance"]
            priority_reason = candidates[0]["reason"]
            
            # Show alert banner
            self._show_priority_alert(priority_coin, priority_distance, priority_reason)
            
            # Check if already on this chart
            current_chart = getattr(self, "_current_chart_page", "ACCOUNT")
            if current_chart == priority_coin:
                return
            
            # Switch to priority coin
            if hasattr(self, "_show_chart_page"):
                self._show_chart_page(priority_coin, is_auto_switch=True)
                
                if self.settings.get("debug_mode", False):
                    print(f"[HUB AUTO-SWITCH] Switched to {priority_coin} ({priority_distance:.2f}% from {priority_reason})")
        except Exception:
            self._hide_priority_alert()
            pass  # Silently fail - don't crash tick loop
    
    def _calculate_entry_distance(self, coin: str, current_price: float, long_signal_min: int) -> Optional[float]:
        """
        Calculate percentage distance from current price to buy entry trigger.
        Returns None if no valid entry signal exists.
        
        Entry occurs when:
        1. Signal strength >= long_signal_min (default 4)
        2. Price is at or below predicted low level
        
        Args:
            coin: Coin symbol (e.g., "BTC")
            current_price: Current market price
            long_signal_min: Minimum signal strength to trigger entry
        
        Returns:
            Percentage distance to entry trigger, or None if not applicable
        """
        try:
            # Read signal strength
            folder = self.coin_folders.get(coin, "")
            if not folder:
                return None
            
            signal_path = os.path.join(folder, "long_dca_signal.txt")
            try:
                with open(signal_path, "r", encoding="utf-8") as f:
                    signal = int(float((f.read() or "0").strip()))
            except Exception:
                signal = 0
            
            # Check if signal is strong enough
            if signal < long_signal_min:
                return None
            
            # Read predicted low prices (buy entry levels)
            low_path = os.path.join(folder, "low_bound_prices.txt")
            low_levels = read_price_levels_from_html(low_path)
            
            if not low_levels:
                return None
            
            # Use the lowest price level (most immediate/closest timeframe)
            # These are the levels where buys would trigger
            entry_price = min(price for _, price in low_levels)
            
            if entry_price <= 0:
                return None
            
            # Calculate distance from current price to entry level
            # Positive distance means price needs to drop to reach entry
            distance_pct = abs(current_price - entry_price) / current_price * 100.0
            
            return distance_pct
        except Exception:
            return None
    
    def _show_priority_alert(self, coin: str, distance_pct: float, reason: str) -> None:
        """
        Display the priority alert in the current chart's top bar.
        
        Args:
            coin: Coin symbol (e.g., "BTC")
            distance_pct: Percentage distance to trigger
            reason: Trigger type (e.g., "Long Entry", "Long Exit")
        """
        try:
            # Format alert text with emoji and details
            alert_text = f"🔔 Priority: {coin} within {distance_pct:.1f}% of {reason}"
            
            # Update all chart priority labels (current chart will be visible)
            current_chart = self._current_chart_page
            if current_chart == "ACCOUNT":
                chart = self.account_chart
            else:
                chart = self.charts.get(current_chart)
            
            if chart and hasattr(chart, "priority_alert_label"):
                chart.priority_alert_label.config(text=alert_text)
        except Exception:
            pass
    
    def _hide_priority_alert(self) -> None:
        """Clear the priority alert from all charts."""
        try:
            # Clear priority alert from all coin charts
            for chart in self.charts.values():
                if hasattr(chart, "priority_alert_label"):
                    chart.priority_alert_label.config(text="")
            
            # Clear from account chart too
            if hasattr(self, "account_chart") and hasattr(self.account_chart, "priority_alert_label"):
                self.account_chart.priority_alert_label.config(text="")
        except Exception:
            pass

    def _refresh_trader_status(self) -> None:
        # mtime cache: rebuilding the whole tree every tick is expensive with many rows
        try:
            mtime = os.path.getmtime(self.trader_status_path)
        except Exception:
            mtime = None

        if getattr(self, "_last_trader_status_mtime", object()) == mtime:
            return
        self._last_trader_status_mtime = mtime

        data = _safe_read_json(self.trader_status_path)
        if not data:
            self.lbl_last_status.config(text="Last status: N/A (no trader_status.json yet)")
            
            # Hide simulation banner when no data
            try:
                self.lbl_simulation_banner.pack_forget()
            except Exception:
                pass

            # account summary (right-side status area)
            try:
                self.lbl_acct_total_value.config(text="Total Account Value: N/A")
                self.lbl_acct_holdings_value.config(text="Holdings Value: N/A")
                self.lbl_acct_buying_power.config(text="Buying Power: N/A")
                self.lbl_acct_percent_in_trade.config(text="Percent In Trade: N/A")

                # DCA affordability
                self.lbl_acct_dca_spread.config(text="DCA Levels (spread): N/A")
                self.lbl_acct_dca_single.config(text="DCA Levels (single): N/A")
            except Exception:
                pass

            # clear tree (once; subsequent ticks are mtime-short-circuited)
            for iid in self.trades_tree.get_children():
                self.trades_tree.delete(iid)
            return

        ts = data.get("timestamp")
        try:
            if isinstance(ts, (int, float)):
                self.lbl_last_status.config(text=f"Last status: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}")
            else:
                self.lbl_last_status.config(text="Last status: (unknown timestamp)")
        except Exception:
            self.lbl_last_status.config(text="Last status: (timestamp parse error)")
        
        # Update simulation mode banner - show/hide as needed
        try:
            sim_mode = data.get("simulation_mode", False)
            if self.settings.get("debug_mode", False):
                print(f"[HUB DEBUG] Simulation mode status: {sim_mode}")
            if sim_mode:
                self.lbl_simulation_banner.config(text="⚠️ SIMULATION MODE ⚠️")
                if not self.lbl_simulation_banner.winfo_ismapped():
                    self.lbl_simulation_banner.pack(anchor="w", padx=6, pady=(4, 4), before=self.lbl_acct_total_value)
                    if self.settings.get("debug_mode", False):
                        print("[HUB DEBUG] Simulation banner displayed")
            else:
                self.lbl_simulation_banner.pack_forget()
        except Exception as e:
            # Log error if debug mode enabled
            if self.settings.get("debug_mode", False):
                print(f"[HUB DEBUG] Failed to update simulation banner: {e}")

        # --- account summary (same info the trader prints above current trades) ---
        acct = data.get("account", {}) or {}
        try:
            total_val = float(acct.get("total_account_value", 0.0) or 0.0)
            self.lbl_acct_total_value.config(
                text=f"Total Account Value: {_fmt_money(acct.get('total_account_value', None))}"
            )
            self.lbl_acct_holdings_value.config(
                text=f"Holdings Value: {_fmt_money(acct.get('holdings_sell_value', None))}"
            )
            self.lbl_acct_buying_power.config(
                text=f"Buying Power: {_fmt_money(acct.get('buying_power', None))}"
            )

            pit = acct.get("percent_in_trade", None)
            try:
                pit_txt = f"{float(pit):.2f}%"
            except Exception:
                pit_txt = "N/A"
            self.lbl_acct_percent_in_trade.config(text=f"Percent In Trade: {pit_txt}")

            # DCA affordability calculation using trading settings to determine available DCA levels
            coins = getattr(self, "coins", None) or []
            n = len(coins) if len(coins) > 0 else 1
            
            # Load trading config (mtime-cached to avoid repeated file reads)
            trading_cfg = self._get_cached_trading_config()
            alloc_pct = trading_cfg.get("position_sizing", {}).get("initial_allocation_pct", 0.005)
            min_alloc = trading_cfg.get("position_sizing", {}).get("min_allocation_usd", 0.5)
            multiplier = trading_cfg.get("dca", {}).get("position_multiplier", 2.0)
            max_concurrent = trading_cfg.get("position_sizing", {}).get("max_concurrent_positions", 3)
            dca_levels_configured = trading_cfg.get("dca", {}).get("levels", [-2.5, -5.0, -10.0, -20.0])
            total_dca_levels = len(dca_levels_configured)
            
            # Use max_concurrent_positions instead of total coins for spread calculation
            # since not all coins can be active simultaneously
            n_spread = min(n, max_concurrent) if max_concurrent > 0 else n

            spread_levels = 0
            single_levels = 0

            if total_val > 0.0:
                # Spread across max concurrent positions (not all coins)
                alloc_spread = total_val * (alloc_pct / n_spread)
                if alloc_spread < min_alloc:
                    alloc_spread = min_alloc

                required = alloc_spread * n  # initial buys for all coins
                while spread_levels < total_dca_levels and required > 0.0 and (required * multiplier) <= (total_val + 1e-9):
                    required *= multiplier
                    spread_levels += 1

                # All DCA into a single coin
                alloc_single = total_val * alloc_pct
                if alloc_single < min_alloc:
                    alloc_single = min_alloc

                required = alloc_single  # initial buy for one coin
                while single_levels < total_dca_levels and required > 0.0 and (required * multiplier) <= (total_val + 1e-9):
                    required *= multiplier
                    single_levels += 1

            # Calculate percentage of total DCA levels that can be afforded
            spread_pct = int((spread_levels / total_dca_levels * 100)) if total_dca_levels > 0 else 0
            single_pct = int((single_levels / total_dca_levels * 100)) if total_dca_levels > 0 else 0

            # Show labels with count and percentage
            self.lbl_acct_dca_spread.config(text=f"DCA Levels (spread): {spread_levels} ({spread_pct}%)")
            self.lbl_acct_dca_single.config(text=f"DCA Levels (single): {single_levels} ({single_pct}%)")

        except Exception:
            pass

        positions = data.get("positions", {}) or {}
        self._last_positions = positions

        # --- precompute per-coin DCA count in rolling 24h (and after last SELL for that coin) ---
        dca_24h_by_coin: Dict[str, int] = {}
        try:
            now = time.time()
            window_floor = now - (24 * 3600)

            trades = _read_trade_history_jsonl(self.trade_history_path) if self.trade_history_path else []

            last_sell_ts: Dict[str, float] = {}
            for tr in trades:
                sym = str(tr.get("symbol", "")).upper().strip()
                base = sym.split("-")[0].strip() if sym else ""
                if not base:
                    continue

                side = str(tr.get("side", "")).lower().strip()
                if side != "sell":
                    continue

                try:
                    tsf = float(tr.get("ts", 0))
                except Exception:
                    continue

                prev = float(last_sell_ts.get(base, 0.0))
                if tsf > prev:
                    last_sell_ts[base] = tsf

            for tr in trades:
                sym = str(tr.get("symbol", "")).upper().strip()
                base = sym.split("-")[0].strip() if sym else ""
                if not base:
                    continue

                side = str(tr.get("side", "")).lower().strip()
                if side != "buy":
                    continue

                tag = str(tr.get("tag") or "").upper().strip()
                if tag != "DCA":
                    continue

                try:
                    tsf = float(tr.get("ts", 0))
                except Exception:
                    continue

                start_ts = max(window_floor, float(last_sell_ts.get(base, 0.0)))
                if tsf >= start_ts:
                    dca_24h_by_coin[base] = int(dca_24h_by_coin.get(base, 0)) + 1
        except Exception:
            dca_24h_by_coin = {}

        # rebuild tree (only when file changes)
        for iid in self.trades_tree.get_children():
            self.trades_tree.delete(iid)

        for sym, pos in positions.items():
            coin = sym
            qty = pos.get("quantity", 0.0)

            # Hide "not in trade" rows (0 qty), but keep them in _last_positions for chart overlays
            try:
                if float(qty) <= 0.0:
                    continue
            except Exception:
                continue

            value = pos.get("value_usd", 0.0)
            avg_cost = pos.get("avg_cost_basis", 0.0)

            buy_price = pos.get("current_buy_price", 0.0)
            buy_pnl = pos.get("gain_loss_pct_buy", 0.0)

            sell_price = pos.get("current_sell_price", 0.0)
            sell_pnl = pos.get("gain_loss_pct_sell", 0.0)

            dca_stages = pos.get("dca_triggered_stages", 0)
            dca_24h = int(dca_24h_by_coin.get(str(coin).upper().strip(), 0))
            next_dca = pos.get("next_dca_display", "")

            trail_line = pos.get("trail_line", 0.0)

            self.trades_tree.insert(
                "",
                "end",
                values=(
                    coin,
                    f"{qty:.8f}".rstrip("0").rstrip("."),
                    _fmt_money(value),       # position value (USD)
                    _fmt_price(avg_cost),    # per-unit price (USD) -> dynamic decimals
                    _fmt_price(buy_price),
                    _fmt_pct(buy_pnl),
                    _fmt_price(sell_price),
                    _fmt_pct(sell_pnl),
                    dca_stages,
                    dca_24h,
                    next_dca,
                    _fmt_price(trail_line),  # trail line is a price level
                ),
            )

    def _refresh_pnl(self) -> None:
        # mtime cache: avoid reading/parsing every tick
        try:
            mtime = os.path.getmtime(self.pnl_ledger_path)
        except Exception:
            mtime = None

        if getattr(self, "_last_pnl_mtime", object()) == mtime:
            return
        self._last_pnl_mtime = mtime

        data = _safe_read_json(self.pnl_ledger_path)
        if not data:
            self.lbl_pnl.config(text="Total realized: N/A")
            return
        total = float(data.get("total_realized_profit_usd", 0.0))
        self.lbl_pnl.config(text=f"Total realized: {_fmt_money(total)}")

    def _refresh_trade_history(self) -> None:
        # mtime cache: avoid reading/parsing/rebuilding the list every tick
        try:
            mtime = os.path.getmtime(self.trade_history_path)
        except Exception:
            mtime = None

        if getattr(self, "_last_trade_history_mtime", object()) == mtime:
            return
        self._last_trade_history_mtime = mtime

        if not os.path.isfile(self.trade_history_path):
            self.hist_list.delete(0, "end")
            self.hist_list.insert("end", "(no trade history yet)")
            return

        # show last N lines
        try:
            with open(self.trade_history_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            return

        lines = lines[-250:]  # cap for UI
        self.hist_list.delete(0, "end")
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ts = obj.get("ts", None)
                tss = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if isinstance(ts, (int, float)) else "?"
                side = str(obj.get("side", "")).upper()
                tag = str(obj.get("tag", "") or "").upper()

                sym = obj.get("symbol", "")
                qty = obj.get("qty", "")
                px = obj.get("price", None)
                pnl = obj.get("realized_profit_usd", None)

                pnl_pct = obj.get("pnl_pct", None)

                px_txt = _fmt_price(px) if px is not None else "N/A"

                action = side
                if tag:
                    action = f"{side}/{tag}"

                txt = f"{tss} | {action:10s} {sym:5s} | qty={qty} | px={px_txt}"

                # Show the exact trade-time PnL%:
                # - DCA buys: show the BUY-side PnL (how far below avg cost it was when it bought)
                # - sells: show the SELL-side PnL (how far above/below avg cost it sold)
                show_trade_pnl_pct = None
                if side == "SELL":
                    show_trade_pnl_pct = pnl_pct
                elif side == "BUY" and tag == "DCA":
                    show_trade_pnl_pct = pnl_pct

                if show_trade_pnl_pct is not None:
                    try:
                        txt += f" | pnl@trade={_fmt_pct(float(show_trade_pnl_pct))}"
                    except Exception:
                        txt += f" | pnl@trade={show_trade_pnl_pct}"

                if pnl is not None:
                    try:
                        txt += f" | realized={float(pnl):+.2f}"
                    except Exception:
                        txt += f" | realized={pnl}"

                self.hist_list.insert("end", txt)
            except Exception:
                self.hist_list.insert("end", line)

    def _refresh_coin_dependent_ui(self, prev_coins: List[str]) -> None:
        """
        After settings change: refresh every coin-driven UI element:
          - Training dropdown (Train coin)
          - Trainers tab dropdown (Coin)
          - Chart tabs (Notebook): add/remove tabs to match current coin list
          - Neural overview tiles (new): add/remove tiles to match current coin list
        """
        # Rebuild dependent pieces
        self.coins = [c.upper().strip() for c in (self.settings.get("coins") or []) if c.strip()]
        self.coin_folders = build_coin_folders(self.settings.get("main_neural_dir") or self.project_dir, self.coins)

        # Refresh coin dropdowns (they don't auto-update)
        try:
            # Training pane dropdown
            if hasattr(self, "train_coin_combo") and self.train_coin_combo.winfo_exists():
                self.train_coin_combo["values"] = self.coins
                cur = (self.train_coin_var.get() or "").strip().upper() if hasattr(self, "train_coin_var") else ""
                if self.coins and cur not in self.coins:
                    self.train_coin_var.set(self.coins[0])

            # Trainers tab dropdown
            if hasattr(self, "trainer_coin_combo") and self.trainer_coin_combo.winfo_exists():
                self.trainer_coin_combo["values"] = self.coins
                cur = (self.trainer_coin_var.get() or "").strip().upper() if hasattr(self, "trainer_coin_var") else ""
                if self.coins and cur not in self.coins:
                    self.trainer_coin_var.set(self.coins[0])

            # Keep both selectors aligned if both exist
            if hasattr(self, "train_coin_var") and hasattr(self, "trainer_coin_var"):
                if self.train_coin_var.get():
                    self.trainer_coin_var.set(self.train_coin_var.get())
        except Exception:
            pass

        # Rebuild neural overview tiles (if the widget exists)
        try:
            if hasattr(self, "neural_wrap") and self.neural_wrap.winfo_exists():
                self._rebuild_neural_overview()
                self._refresh_neural_overview()
        except Exception:
            pass

        # Rebuild chart tabs if the coin list changed
        try:
            prev_set = set([str(c).strip().upper() for c in (prev_coins or []) if str(c).strip()])
            if prev_set != set(self.coins):
                self._rebuild_coin_chart_tabs()
        except Exception:
            pass

    def _rebuild_neural_overview(self) -> None:
        """
        Recreate the coin tiles in the left-side Neural Signals box to match self.coins.
        Uses WrapFrame so it automatically breaks into multiple rows.
        Adds hover highlighting and click-to-open chart.
        """
        if not hasattr(self, "neural_wrap") or self.neural_wrap is None:
            return

        # Clear old tiles
        try:
            if hasattr(self.neural_wrap, "clear"):
                self.neural_wrap.clear(destroy_widgets=True)
            else:
                for ch in list(self.neural_wrap.winfo_children()):
                    ch.destroy()
        except Exception:
            pass

        self.neural_tiles = {}

        for coin in (self.coins or []):
            tile = NeuralSignalTile(self.neural_wrap, coin)

            # --- Hover highlighting (real, visible) ---
            def _on_enter(_e=None, t=tile):
                try:
                    t.set_hover(True)
                except Exception:
                    pass

            def _on_leave(_e=None, t=tile):
                # Avoid flicker: when moving between child widgets, ignore "leave" if pointer is still inside tile.
                try:
                    x = t.winfo_pointerx()
                    y = t.winfo_pointery()
                    w = t.winfo_containing(x, y)
                    while w is not None:
                        if w == t:
                            return
                        w = getattr(w, "master", None)
                except Exception:
                    pass

                try:
                    t.set_hover(False)
                except Exception:
                    pass

            tile.bind("<Enter>", _on_enter, add="+")
            tile.bind("<Leave>", _on_leave, add="+")
            try:
                for w in tile.winfo_children():
                    w.bind("<Enter>", _on_enter, add="+")
                    w.bind("<Leave>", _on_leave, add="+")
            except Exception:
                pass

            # --- Click: open chart page ---
            def _open_coin_chart(_e=None, c=coin):
                try:
                    fn = getattr(self, "_show_chart_page", None)
                    if callable(fn):
                        fn(str(c).strip().upper())
                except Exception:
                    pass

            tile.bind("<Button-1>", _open_coin_chart, add="+")
            try:
                for w in tile.winfo_children():
                    w.bind("<Button-1>", _open_coin_chart, add="+")
            except Exception:
                pass

            self.neural_wrap.add(tile, padx=(0, 6), pady=(0, 6))
            self.neural_tiles[coin] = tile

        # Layout and scrollbar refresh
        try:
            self.neural_wrap._schedule_reflow()
        except Exception:
            pass

        # No dynamic height adjustment needed - let content size naturally

        try:
            fn = getattr(self, "_update_neural_overview_scrollbars", None)
            if callable(fn):
                self.after_idle(fn)
        except Exception:
            pass

    def _refresh_neural_overview(self) -> None:
        """
        Update each coin tile with long/short neural signals.
        Uses mtime caching so it's cheap to call every UI tick.
        """
        if not hasattr(self, "neural_tiles"):
            return

        # Keep coin_folders aligned with current settings/coins
        try:
            sig = (str(self.settings.get("main_neural_dir") or ""), tuple(self.coins or []))
            if getattr(self, "_coin_folders_sig", None) != sig:
                self._coin_folders_sig = sig
                self.coin_folders = build_coin_folders(self.settings.get("main_neural_dir") or self.project_dir, self.coins)
        except Exception:
            pass

        if not hasattr(self, "_neural_overview_cache"):
            self._neural_overview_cache = {}  # path -> (mtime, value)

        def _cached(path: str, loader, default: Any) -> Any:
            """
            Mtime-cached file loader. Returns the loaded value only.
            Mtime tracking is internal to the cache.
            """
            try:
                mtime = os.path.getmtime(path)
            except Exception:
                return default

            hit = self._neural_overview_cache.get(path)
            if hit and hit[0] == mtime:
                return hit[1]

            v = loader(path)
            self._neural_overview_cache[path] = (mtime, v)
            return v

        def _load_short_from_memory_json(path: str) -> int:
            try:
                obj = _safe_read_json(path) or {}
                return int(float(obj.get("short_dca_signal", 0)))
            except Exception:
                return 0

        latest_ts = None

        for coin, tile in list(self.neural_tiles.items()):
            folder = ""
            try:
                folder = (self.coin_folders or {}).get(coin, "")
            except Exception:
                folder = ""

            if not folder or not os.path.isdir(folder):
                tile.set_values(0, 0)
                continue

            long_sig = 0
            short_sig = 0
            mt_candidates: List[float] = []

            # Long signal
            long_path = os.path.join(folder, "long_dca_signal.txt")
            if os.path.isfile(long_path):
                long_sig = _cached(long_path, read_int_from_file, 0)
                try:
                    mt_candidates.append(float(os.path.getmtime(long_path)))
                except Exception:
                    pass

            # Short signal (prefer txt; fallback to memory.json)
            short_txt = os.path.join(folder, "short_dca_signal.txt")
            if os.path.isfile(short_txt):
                short_sig = _cached(short_txt, read_int_from_file, 0)
                try:
                    mt_candidates.append(float(os.path.getmtime(short_txt)))
                except Exception:
                    pass
            else:
                mem = os.path.join(folder, "memory.json")
                if os.path.isfile(mem):
                    short_sig = _cached(mem, _load_short_from_memory_json, 0)
                    try:
                        mt_candidates.append(float(os.path.getmtime(mem)))
                    except Exception:
                        pass

            tile.set_values(long_sig, short_sig)
            
            # Update marker lines if config changed
            try:
                tile.update_marker_lines()
            except Exception:
                pass

            if mt_candidates:
                mx = max(mt_candidates)
                latest_ts = mx if (latest_ts is None or mx > latest_ts) else latest_ts

        # Update "Last:" label
        try:
            if hasattr(self, "lbl_neural_overview_last") and self.lbl_neural_overview_last.winfo_exists():
                if latest_ts:
                    self.lbl_neural_overview_last.config(
                        text=f"Last status: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(latest_ts)))}"
                    )
                else:
                    self.lbl_neural_overview_last.config(text="Last status: N/A")
        except Exception:
            pass

    def _rebuild_coin_chart_tabs(self) -> None:
        """
        Ensure the Charts multi-row tab bar + pages match self.coins.
        Keeps the ACCOUNT page intact and preserves the currently selected page when possible.
        """
        charts_frame = getattr(self, "_charts_frame", None)
        if charts_frame is None or (hasattr(charts_frame, "winfo_exists") and not charts_frame.winfo_exists()):
            return

        # Remember selected page (coin or ACCOUNT)
        selected = getattr(self, "_current_chart_page", "ACCOUNT")
        if selected not in (["ACCOUNT"] + list(self.coins)):
            selected = "ACCOUNT"

        # Destroy existing tab bar + pages container (clean rebuild)
        try:
            if hasattr(self, "chart_tabs_bar") and self.chart_tabs_bar.winfo_exists():
                self.chart_tabs_bar.destroy()
        except Exception:
            pass

        try:
            if hasattr(self, "chart_pages_container") and self.chart_pages_container.winfo_exists():
                self.chart_pages_container.destroy()
        except Exception:
            pass

        # Recreate
        self.chart_tabs_bar = WrapFrame(charts_frame)
        self.chart_tabs_bar.pack(fill="x", padx=6, pady=(6, 0))

        self.chart_pages_container = ttk.Frame(charts_frame)
        self.chart_pages_container.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self._chart_tab_buttons = {}
        self.chart_pages = {}
        self._current_chart_page = selected

        def _show_page(name: str) -> None:
            self._current_chart_page = name
            for f in self.chart_pages.values():
                try:
                    f.pack_forget()
                except Exception:
                    pass
            f = self.chart_pages.get(name)
            if f is not None:
                f.pack(fill="both", expand=True)

            for txt, b in self._chart_tab_buttons.items():
                try:
                    b.configure(style=("ChartTabSelected.TButton" if txt == name else "ChartTab.TButton"))
                except Exception:
                    pass

        self._show_chart_page = _show_page

        # ACCOUNT page
        acct_page = ttk.Frame(self.chart_pages_container)
        self.chart_pages["ACCOUNT"] = acct_page

        acct_btn = ttk.Button(
            self.chart_tabs_bar,
            text="ACCOUNT",
            style="ChartTab.TButton",
            command=lambda: self._show_chart_page("ACCOUNT"),
        )
        self.chart_tabs_bar.add(acct_btn, padx=(0, 6), pady=(0, 6))
        self._chart_tab_buttons["ACCOUNT"] = acct_btn

        self.account_chart = AccountValueChart(
            acct_page,
            self.account_value_history_path,
            self.trade_history_path,
        )
        self.account_chart.pack(fill="both", expand=True)

        # Coin pages
        self.charts = {}
        for coin in self.coins:
            page = ttk.Frame(self.chart_pages_container)
            self.chart_pages[coin] = page

            btn = ttk.Button(
                self.chart_tabs_bar,
                text=coin,
                style="ChartTab.TButton",
                command=lambda c=coin: self._show_chart_page(c),
            )
            self.chart_tabs_bar.add(btn, padx=(0, 6), pady=(0, 6))
            self._chart_tab_buttons[coin] = btn

            chart = CandleChart(page, self.fetcher, coin, self._settings_getter, self.trade_history_path)
            chart.pack(fill="both", expand=True)
            self.charts[coin] = chart

        # Restore selection
        self._show_chart_page(selected)

    # Settings dialog methods for GUI-based configuration management
    def open_settings_dialog(self) -> None:
        # Prevent duplicate dialogs - bring existing one to front if already open
        if hasattr(self, "_settings_dialog_win") and self._settings_dialog_win and self._settings_dialog_win.winfo_exists():
            self._settings_dialog_win.lift()
            self._settings_dialog_win.focus_force()
            return

        win = tk.Toplevel(self)
        win.title("Settings")
        # Big enough for the bottom buttons on most screens + still scrolls if someone resizes smaller.
        win.geometry("860x680")
        win.minsize(760, 560)
        win.configure(bg=DARK_BG)
        win.transient(self)  # Keep dialog on top of parent
        
        self._settings_dialog_win = win
        
        def on_close():
            self._settings_dialog_win = None
            win.destroy()
        
        win.protocol("WM_DELETE_WINDOW", on_close)

        # Scrollable settings content (auto-hides the scrollbar if everything fits),
        # using the same pattern as the Neural Levels scrollbar.
        viewport = ttk.Frame(win)
        viewport.pack(fill="both", expand=True, padx=12, pady=12)
        viewport.grid_rowconfigure(0, weight=1)
        viewport.grid_columnconfigure(0, weight=1)

        settings_canvas = tk.Canvas(
            viewport,
            bg=DARK_BG,
            highlightthickness=1,
            highlightbackground=DARK_BORDER,
            bd=0,
        )
        settings_canvas.grid(row=0, column=0, sticky="nsew")

        settings_scroll = ttk.Scrollbar(
            viewport,
            orient="vertical",
            command=settings_canvas.yview,
        )
        settings_scroll.grid(row=0, column=1, sticky="ns")

        settings_canvas.configure(yscrollcommand=settings_scroll.set)

        frm = ttk.Frame(settings_canvas)
        settings_window = settings_canvas.create_window((0, 0), window=frm, anchor="nw")

        def _update_settings_scrollbars(event=None) -> None:
            """Update scrollregion + hide/show the scrollbar depending on overflow."""
            try:
                c = settings_canvas
                win_id = settings_window

                c.update_idletasks()
                bbox = c.bbox(win_id)
                if not bbox:
                    settings_scroll.grid_remove()
                    return

                c.configure(scrollregion=bbox)
                content_h = int(bbox[3] - bbox[1])
                view_h = int(c.winfo_height())

                if content_h > (view_h + 1):
                    settings_scroll.grid()
                else:
                    settings_scroll.grid_remove()
                    try:
                        c.yview_moveto(0)
                    except Exception:
                        pass
            except Exception:
                pass

        def _on_settings_canvas_configure(e) -> None:
            # Keep the inner frame exactly the canvas width so wrapping is correct.
            try:
                settings_canvas.itemconfigure(settings_window, width=int(e.width))
            except Exception:
                pass
            _update_settings_scrollbars()

        settings_canvas.bind("<Configure>", _on_settings_canvas_configure, add="+")
        frm.bind("<Configure>", _update_settings_scrollbars, add="+")

        # Mousewheel scrolling when the mouse is over the settings window.
        def _wheel(e):
            try:
                if settings_scroll.winfo_ismapped():
                    settings_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
            except Exception:
                pass

        def _bind_mousewheel_recursive(widget):
            """Recursively bind mouse wheel to widget and all children so it works everywhere."""
            try:
                widget.bind("<MouseWheel>", _wheel, add="+")
                widget.bind("<Button-4>", lambda _e: settings_canvas.yview_scroll(-3, "units"), add="+")
                widget.bind("<Button-5>", lambda _e: settings_canvas.yview_scroll(3, "units"), add="+")
                for child in widget.winfo_children():
                    _bind_mousewheel_recursive(child)
            except Exception:
                pass

        settings_canvas.bind("<MouseWheel>", _wheel, add="+")  # Windows / Mac
        settings_canvas.bind("<Button-4>", lambda _e: settings_canvas.yview_scroll(-3, "units"), add="+")  # Linux
        settings_canvas.bind("<Button-5>", lambda _e: settings_canvas.yview_scroll(3, "units"), add="+")   # Linux
        _bind_mousewheel_recursive(frm)  # Bind to all child widgets

        # Make the entry column expand
        frm.columnconfigure(0, weight=0)  # labels
        frm.columnconfigure(1, weight=1)  # entries
        frm.columnconfigure(2, weight=0)  # browse buttons

        def add_row(parent, r: int, label: str, var: tk.Variable, browse: Optional[str] = None):
            """
            browse: "dir" to attach a directory chooser, else None.
            """
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=(0, 10), pady=6)

            ent = ttk.Entry(parent, textvariable=var)
            ent.grid(row=r, column=1, sticky="ew", pady=6)

            if browse == "dir":
                def do_browse():
                    picked = filedialog.askdirectory()
                    if picked:
                        var.set(picked)
                ttk.Button(parent, text="Browse", command=do_browse).grid(row=r, column=2, sticky="e", padx=(10, 0), pady=6)
            else:
                # keep column alignment consistent
                ttk.Label(parent, text="").grid(row=r, column=2, sticky="e", padx=(10, 0), pady=6)

        main_dir_var = tk.StringVar(value=self.settings["main_neural_dir"])
        coins_var = tk.StringVar(value=",".join(self.settings["coins"]))
        hub_dir_var = tk.StringVar(value=self.settings.get("hub_data_dir", ""))

        neural_script_var = tk.StringVar(value=self.settings["script_neural_runner2"])
        trainer_script_var = tk.StringVar(value=self.settings.get("script_neural_trainer", "pt_trainer.py"))
        trader_script_var = tk.StringVar(value=self.settings["script_trader"])

        ui_refresh_var = tk.StringVar(value=str(self.settings["ui_refresh_seconds"]))
        chart_refresh_var = tk.StringVar(value=str(self.settings["chart_refresh_seconds"]))
        candles_limit_var = tk.StringVar(value=str(self.settings["candles_limit"]))

        r = 0
        
        # Basic Settings Section
        basic_frame = ttk.LabelFrame(frm, text=" Basic Settings ", padding=15)
        basic_frame.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(0, 15)); r += 1
        basic_frame.columnconfigure(0, weight=0)
        basic_frame.columnconfigure(1, weight=1)
        basic_frame.columnconfigure(2, weight=0)
        
        basic_r = 0
        add_row(basic_frame, basic_r, "Main neural folder:", main_dir_var, browse="dir"); basic_r += 1
        add_row(basic_frame, basic_r, "Coins (comma):", coins_var); basic_r += 1
        add_row(basic_frame, basic_r, "Hub data dir (optional):", hub_dir_var, browse="dir"); basic_r += 1

        # Script Paths Section
        script_frame = ttk.LabelFrame(frm, text=" Script Paths ", padding=15)
        script_frame.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(0, 15)); r += 1
        script_frame.columnconfigure(0, weight=0)
        script_frame.columnconfigure(1, weight=1)
        script_frame.columnconfigure(2, weight=0)
        
        script_r = 0
        add_row(script_frame, script_r, "pt_thinker.py path:", neural_script_var); script_r += 1
        add_row(script_frame, script_r, "pt_trainer.py path:", trainer_script_var); script_r += 1
        add_row(script_frame, script_r, "pt_trader.py path:", trader_script_var); script_r += 1

        # --- Robinhood API setup (writes rh_key.enc + rh_secret.enc using Windows DPAPI encryption) ---
        def _api_paths() -> Tuple[str, str]:
            key_path = os.path.join(self.project_dir, "rh_key.enc")
            secret_path = os.path.join(self.project_dir, "rh_secret.enc")
            return key_path, secret_path

        def _read_api_files() -> Tuple[str, str]:
            """Read and decrypt API keys."""
            key_path, secret_path = _api_paths()
            
            # Read encrypted files
            try:
                with open(key_path, "rb") as f:
                    k = _decrypt_with_dpapi(f.read()).strip()
            except Exception:
                k = ""
            try:
                with open(secret_path, "rb") as f:
                    s = _decrypt_with_dpapi(f.read()).strip()
            except Exception:
                s = ""
            return k, s

        api_status_var = tk.StringVar(value="")

        def _refresh_api_status() -> None:
            key_path, secret_path = _api_paths()
            k, s = _read_api_files()

            missing = []
            if not k:
                missing.append("rh_key.enc (API Key)")
            if not s:
                missing.append("rh_secret.enc (PRIVATE key)")

            if missing:
                api_status_var.set("Not configured ❌ (missing " + ", ".join(missing) + ")")
            else:
                api_status_var.set("Configured ✅ (credentials found)")

        def _open_api_folder() -> None:
            """Open the folder where rh_key.txt / rh_secret.txt live."""
            try:
                folder = os.path.abspath(self.project_dir)
                if os.name == "nt":
                    os.startfile(folder)  # type: ignore[attr-defined]
                    return
                if sys.platform == "darwin":
                    subprocess.Popen(["open", folder])
                    return
                subprocess.Popen(["xdg-open", folder])
            except Exception as e:
                messagebox.showerror("Couldn't open folder", f"Tried to open:\n{self.project_dir}\n\nError:\n{e}")

        def _clear_api_files() -> None:
            """Delete rh_key.enc / rh_secret.enc (with a big confirmation)."""
            key_path, secret_path = _api_paths()
            if not messagebox.askyesno(
                "Delete API credentials?",
                "This will delete:\n"
                f"  {key_path}\n"
                f"  {secret_path}\n\n"
                "After deleting, the trader can NOT authenticate until you run the setup wizard again.\n\n"
                "Are you sure you want to delete these files?"
            ):
                return

            try:
                if os.path.isfile(key_path):
                    os.remove(key_path)
                if os.path.isfile(secret_path):
                    os.remove(secret_path)
            except Exception as e:
                messagebox.showerror("Delete failed", f"Couldn't delete the files:\n\n{e}")
                return

            _refresh_api_status()
            messagebox.showinfo("Deleted", "Deleted rh_key.enc and rh_secret.enc (encrypted API keys).")

        def _open_robinhood_api_wizard() -> None:
            """
            Beginner-friendly wizard that creates + stores Robinhood Crypto Trading API credentials.

            What we store:
              - rh_key.enc    = your Robinhood *API Key* (encrypted with Windows DPAPI)
              - rh_secret.enc = your *PRIVATE key* (encrypted, treat like a password — never share it)
            """
            import webbrowser
            import base64
            import platform
            from datetime import datetime
            import time

            # Friendly dependency errors (laymen-proof)
            try:
                from cryptography.hazmat.primitives.asymmetric import ed25519
                from cryptography.hazmat.primitives import serialization
            except Exception:
                messagebox.showerror(
                    "Missing dependency",
                    "The 'cryptography' package is required for Robinhood API setup.\n\n"
                    "Fix: open a Command Prompt / Terminal in this folder and run:\n"
                    "  pip install cryptography\n\n"
                    "Then re-open this Setup Wizard."
                )
                return

            try:
                import requests # for the 'Test credentials' button
            except Exception:
                requests = None

            wiz = tk.Toplevel(win)
            wiz.title("Robinhood API Setup")
            # Big enough to show the bottom buttons, but still scrolls if the window is resized smaller.
            wiz.geometry("980x720")
            wiz.minsize(860, 620)
            wiz.configure(bg=DARK_BG)

            # Scrollable content area (same pattern as the Neural Levels scrollbar).
            viewport = ttk.Frame(wiz)
            viewport.pack(fill="both", expand=True, padx=12, pady=12)
            viewport.grid_rowconfigure(0, weight=1)
            viewport.grid_columnconfigure(0, weight=1)

            wiz_canvas = tk.Canvas(
                viewport,
                bg=DARK_BG,
                highlightthickness=1,
                highlightbackground=DARK_BORDER,
                bd=0,
            )
            wiz_canvas.grid(row=0, column=0, sticky="nsew")

            wiz_scroll = ttk.Scrollbar(viewport, orient="vertical", command=wiz_canvas.yview)
            wiz_scroll.grid(row=0, column=1, sticky="ns")
            wiz_canvas.configure(yscrollcommand=wiz_scroll.set)

            container = ttk.Frame(wiz_canvas)
            wiz_window = wiz_canvas.create_window((0, 0), window=container, anchor="nw")
            container.columnconfigure(0, weight=1)

            def _update_wiz_scrollbars(event=None) -> None:
                """Update scrollregion + hide/show the scrollbar depending on overflow."""
                try:
                    c = wiz_canvas
                    win_id = wiz_window

                    c.update_idletasks()
                    bbox = c.bbox(win_id)
                    if not bbox:
                        wiz_scroll.grid_remove()
                        return

                    c.configure(scrollregion=bbox)
                    content_h = int(bbox[3] - bbox[1])
                    view_h = int(c.winfo_height())

                    if content_h > (view_h + 1):
                        wiz_scroll.grid()
                    else:
                        wiz_scroll.grid_remove()
                        try:
                            c.yview_moveto(0)
                        except Exception:
                            pass
                except Exception:
                    pass

            def _on_wiz_canvas_configure(e) -> None:
                # Keep the inner frame exactly the canvas width so labels wrap nicely.
                try:
                    wiz_canvas.itemconfigure(wiz_window, width=int(e.width))
                except Exception:
                    pass
                _update_wiz_scrollbars()

            wiz_canvas.bind("<Configure>", _on_wiz_canvas_configure, add="+")
            container.bind("<Configure>", _update_wiz_scrollbars, add="+")

            def _wheel(e):
                try:
                    if wiz_scroll.winfo_ismapped():
                        wiz_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
                except Exception:
                    pass

            wiz_canvas.bind("<Enter>", lambda _e: wiz_canvas.focus_set(), add="+")
            wiz_canvas.bind("<MouseWheel>", _wheel, add="+")  # Windows / Mac
            wiz_canvas.bind("<Button-4>", lambda _e: wiz_canvas.yview_scroll(-3, "units"), add="+")  # Linux
            wiz_canvas.bind("<Button-5>", lambda _e: wiz_canvas.yview_scroll(3, "units"), add="+")   # Linux

            key_path, secret_path = _api_paths()

            # Load any existing credentials so users can update without re-generating keys.
            existing_api_key, existing_private_b64 = _read_api_files()
            private_b64_state = {"value": (existing_private_b64 or "").strip()}

            # Helper functions for file management and clipboard operations
            def _open_in_file_manager(path: str) -> None:
                try:
                    p = os.path.abspath(path)
                    if os.name == "nt":
                        os.startfile(p)  # type: ignore[attr-defined]
                        return
                    if sys.platform == "darwin":
                        subprocess.Popen(["open", p])
                        return
                    subprocess.Popen(["xdg-open", p])
                except Exception as e:
                    messagebox.showerror("Couldn't open folder", f"Tried to open:\n{path}\n\nError:\n{e}")

            def _copy_to_clipboard(txt: str, title: str = "Copied") -> None:
                try:
                    wiz.clipboard_clear()
                    wiz.clipboard_append(txt)
                    messagebox.showinfo(title, "Copied to clipboard.")
                except Exception:
                    pass

            def _mask_path(p: str) -> str:
                try:
                    return os.path.abspath(p)
                except Exception:
                    return p

            # Instructional text and UI layout for Robinhood API setup wizard
            intro_frame = ttk.Frame(container)
            intro_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
            intro_frame.columnconfigure(0, weight=1)
            
            intro_text = (
                "This trader uses Robinhood's Crypto Trading API credentials.\n\n"
                "You only do this once. When finished, pt_trader.py can authenticate automatically."
            )
            ttk.Label(intro_frame, text=intro_text, justify="left", foreground=DARK_FG).grid(row=0, column=0, sticky="w")
            
            steps_text = (
                "✅ What you will do in this window:\n"
                "  1) Generate a Public Key + Private Key (Ed25519).\n"
                "  2) Copy the PUBLIC key and paste it into Robinhood to create an API credential.\n"
                "  3) Robinhood will show you an API Key (usually starts with 'rh...'). Copy it.\n"
                "  4) Paste that API Key back here and click Save."
            )
            ttk.Label(intro_frame, text=steps_text, justify="left", foreground=DARK_FG, font=("TkDefaultFont", 9)).grid(row=1, column=0, sticky="w", pady=(10, 0))
            
            robinhood_text = (
                "🧭 EXACTLY where to paste the Public Key on Robinhood (desktop web is best):\n"
                "  A) Log in to Robinhood on a computer.\n"
                "  B) Click Account (top-right) → Settings.\n"
                "  C) Click Crypto.\n"
                "  D) Scroll down to API Trading and click + Add Key (or Add key).\n"
                "  E) Paste the Public Key into the Public key field.\n"
                "  F) Give it any name (example: ApolloTrader).\n"
                "  G) Permissions: this TRADER needs READ + TRADE. (READ-only cannot place orders.)\n"
                "  H) Click Save. Robinhood shows your API Key — copy it right away (it may only show once)."
            )
            ttk.Label(intro_frame, text=robinhood_text, justify="left", foreground=DARK_FG, font=("TkDefaultFont", 9)).grid(row=2, column=0, sticky="w", pady=(10, 0))
            
            mobile_note = "📱 Mobile note: if you can't find API Trading in the app, use robinhood.com in a browser."
            ttk.Label(intro_frame, text=mobile_note, justify="left", foreground="orange", font=("TkDefaultFont", 9, "italic")).grid(row=3, column=0, sticky="w", pady=(10, 0))
            
            files_text = (
                "This wizard will save two encrypted files in the same folder as pt_hub.py:\n"
                "  • rh_key.enc    (your API Key, encrypted)\n"
                "  • rh_secret.enc (your PRIVATE key in base64, encrypted)"
            )
            ttk.Label(intro_frame, text=files_text, justify="left", foreground=DARK_FG, font=("TkDefaultFont", 9)).grid(row=4, column=0, sticky="w", pady=(10, 0))
            
            security_warning = "⚠ Keep rh_secret.enc SECRET like a password! It grants full access to trade your crypto."
            ttk.Label(intro_frame, text=security_warning, justify="left", foreground="orange", font=("TkDefaultFont", 9, "bold")).grid(row=5, column=0, sticky="w", pady=(10, 0))

            top_btns = ttk.Frame(container)
            top_btns.grid(row=1, column=0, sticky="ew", pady=(0, 10))
            top_btns.columnconfigure(0, weight=1)

            def open_robinhood_page():
                # Robinhood entry point. User will still need to click into Settings → Crypto → API Trading.
                webbrowser.open("https://robinhood.com/account/crypto")

            ttk.Button(top_btns, text="Open Robinhood API Credentials page (Crypto)", command=open_robinhood_page).pack(side="left")
            ttk.Button(top_btns, text="Open Robinhood Crypto Trading API docs", command=lambda: webbrowser.open("https://docs.robinhood.com/crypto/trading/")).pack(side="left", padx=8)
            ttk.Button(top_btns, text="Open Folder With rh_key.enc / rh_secret.enc", command=lambda: _open_in_file_manager(self.project_dir)).pack(side="left", padx=8)

            # Step 1: Key generation UI for creating Ed25519 keypair
            step1 = ttk.LabelFrame(container, text="Step 1 — Generate your keys (click once)")
            step1.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
            step1.columnconfigure(0, weight=1)

            ttk.Label(step1, text="Public Key (this is what you paste into Robinhood):").grid(row=0, column=0, sticky="w", padx=10, pady=(8, 0))

            pub_box = tk.Text(step1, height=4, wrap="none")
            pub_box.grid(row=1, column=0, sticky="nsew", padx=10, pady=(6, 10))
            pub_box.configure(bg=DARK_PANEL, fg=DARK_FG, insertbackground=DARK_FG)

            def _render_public_from_private_b64(priv_b64: str) -> str:
                """Return Robinhood-compatible Public Key: base64(raw_ed25519_public_key_32_bytes)."""
                try:
                    raw = base64.b64decode(priv_b64)

                    # Accept either:
                    #   - 32 bytes: Ed25519 seed
                    #   - 64 bytes: NaCl/tweetnacl secretKey (seed + public)
                    if len(raw) == 64:
                        seed = raw[:32]
                    elif len(raw) == 32:
                        seed = raw
                    else:
                        return ""

                    pk = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
                    pub_raw = pk.public_key().public_bytes(
                        encoding=serialization.Encoding.Raw,
                        format=serialization.PublicFormat.Raw,
                    )
                    return base64.b64encode(pub_raw).decode("utf-8")
                except Exception:
                    return ""

            def _set_pub_text(txt: str) -> None:
                try:
                    pub_box.delete("1.0", "end")
                    pub_box.insert("1.0", txt or "")
                except Exception:
                    pass

            # If already configured before, show the public key again (derived from stored private key)
            if private_b64_state["value"]:
                _set_pub_text(_render_public_from_private_b64(private_b64_state["value"]))

            def generate_keys():
                # Generate an Ed25519 keypair (Robinhood expects base64 raw public key bytes)
                priv = ed25519.Ed25519PrivateKey.generate()
                pub = priv.public_key()

                seed = priv.private_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PrivateFormat.Raw,
                    encryption_algorithm=serialization.NoEncryption(),
                )
                pub_raw = pub.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw,
                )

                # Store PRIVATE key as base64(seed32) because pt_thinker.py uses nacl.signing.SigningKey(seed)
                # and it requires exactly 32 bytes.
                private_b64_state["value"] = base64.b64encode(seed).decode("utf-8")

                # Show what you paste into Robinhood: base64(raw public key)
                _set_pub_text(base64.b64encode(pub_raw).decode("utf-8"))

                messagebox.showinfo(
                    "Step 1 complete",
                    "Public/Private keys generated.\n\n"
                    "Next (Robinhood):\n"
                    "  1) Click 'Copy Public Key' in this window\n"
                    "  2) On Robinhood (desktop web): Account → Settings → Crypto\n"
                    "  3) Scroll to 'API Trading' → click '+ Add Key'\n"
                    "  4) Paste the Public Key (base64) into the 'Public key' field\n"
                    "  5) Enable permissions READ + TRADE (this trader needs both), then Save\n"
                    "  6) Robinhood shows an API Key (usually starts with 'rh...') — copy it right away\n\n"
                    "Then come back here and paste that API Key into the 'API Key' box."
                )

            def copy_public_key():
                txt = (pub_box.get("1.0", "end") or "").strip()
                if not txt:
                    messagebox.showwarning("Nothing to copy", "Click 'Generate Keys' first.")
                    return
                _copy_to_clipboard(txt, title="Public Key copied")

            step1_btns = ttk.Frame(step1)
            step1_btns.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 10))
            ttk.Button(step1_btns, text="Generate Keys", command=generate_keys).pack(side="left")
            ttk.Button(step1_btns, text="Copy Public Key", command=copy_public_key).pack(side="left", padx=8)

            # Step 2: API key input UI for Robinhood-provided credentials
            step2 = ttk.LabelFrame(container, text="Step 2 — Paste your Robinhood API Key here")
            step2.grid(row=3, column=0, sticky="nsew", pady=(0, 10))
            step2.columnconfigure(0, weight=1)

            step2_help = (
                "In Robinhood, after you add the Public Key, Robinhood will show an API Key.\n"
                "Paste that API Key below. (It often starts with 'rh.'.)"
            )
            ttk.Label(step2, text=step2_help, justify="left", foreground=DARK_FG, font=("TkDefaultFont", 9)).grid(row=0, column=0, sticky="w", padx=10, pady=(8, 0))

            api_key_var = tk.StringVar(value=existing_api_key or "")
            api_ent = ttk.Entry(step2, textvariable=api_key_var)
            api_ent.grid(row=1, column=0, sticky="ew", padx=10, pady=(6, 10))

            def _test_credentials() -> None:
                api_key = (api_key_var.get() or "").strip()
                priv_b64 = (private_b64_state.get("value") or "").strip()

                if not requests:
                    messagebox.showerror(
                        "Missing dependency",
                        "The 'requests' package is required for the Test button.\n\n"
                        "Fix: pip install requests\n\n"
                        "(You can still Save without testing.)"
                    )
                    return

                if not priv_b64:
                    messagebox.showerror("Missing private key", "Step 1: click 'Generate Keys' first.")
                    return
                if not api_key:
                    messagebox.showerror("Missing API key", "Paste the API key from Robinhood into Step 2 first.")
                    return

                # Safe test: market-data endpoint (no trading)
                base_url = "https://trading.robinhood.com"
                path = "/api/v1/crypto/marketdata/best_bid_ask/?symbol=BTC-USD"
                method = "GET"
                body = ""
                ts = int(time.time())
                msg = f"{api_key}{ts}{path}{method}{body}".encode("utf-8")

                try:
                    raw = base64.b64decode(priv_b64)

                    # Accept either:
                    #   - 32 bytes: Ed25519 seed
                    #   - 64 bytes: NaCl/tweetnacl secretKey (seed + public)
                    if len(raw) == 64:
                        seed = raw[:32]
                    elif len(raw) == 32:
                        seed = raw
                    else:
                        raise ValueError(f"Unexpected private key length: {len(raw)} bytes (expected 32 or 64)")

                    pk = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
                    sig_b64 = base64.b64encode(pk.sign(msg)).decode("utf-8")
                except Exception as e:
                    messagebox.showerror("Bad private key", f"Couldn't use your private key (rh_secret.enc).\n\nError:\n{e}")
                    return

                headers = {
                    "x-api-key": api_key,
                    "x-timestamp": str(ts),
                    "x-signature": sig_b64,
                    "Content-Type": "application/json",
                }

                try:
                    resp = requests.get(f"{base_url}{path}", headers=headers, timeout=10)
                    if resp.status_code >= 400:
                        # Give layman-friendly hints for common failures
                        hint = ""
                        if resp.status_code in (401, 403):
                            hint = (
                                "\n\nCommon fixes:\n"
                                "  • Make sure you pasted the API Key (not the public key).\n"
                                "  • In Robinhood, ensure the key has permissions READ + TRADE.\n"
                                "  • If you just created the key, wait 30–60 seconds and try again.\n"
                            )
                        messagebox.showerror("Test failed", f"Robinhood returned HTTP {resp.status_code}.\n\n{resp.text}{hint}")
                        return

                    data = resp.json()
                    # Try to show something reassuring
                    ask = None
                    try:
                        if data.get("results"):
                            ask = data["results"][0].get("ask_inclusive_of_buy_spread")
                    except Exception:
                        pass

                    messagebox.showinfo(
                        "Test successful",
                        "✅ Your API Key + Private Key worked!\n\n"
                        "Robinhood responded successfully.\n"
                        f"BTC-USD ask (example): {ask if ask is not None else 'received'}\n\n"
                        "Next: click Save."
                    )
                except Exception as e:
                    messagebox.showerror("Test failed", f"Couldn't reach Robinhood.\n\nError:\n{e}")

            step2_btns = ttk.Frame(step2)
            step2_btns.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 10))
            ttk.Button(step2_btns, text="Test Credentials (safe, no trading)", command=_test_credentials).pack(side="left")

            # Step 3: Credential persistence UI for saving encrypted keys to disk
            step3 = ttk.LabelFrame(container, text="Step 3 — Save to files (required)")
            step3.grid(row=4, column=0, sticky="nsew")
            step3.columnconfigure(0, weight=1)

            ttk.Label(
                step3,
                text="⚠ These encrypted files grant full trading access. Keep them secure!",
                foreground="orange",
                font=("TkDefaultFont", 9, "bold")
            ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 8))

            ack_var = tk.BooleanVar(value=False)
            ack = ttk.Checkbutton(
                step3,
                text="I understand rh_secret.enc is PRIVATE (encrypted for my Windows account) and I will not share it.",
                variable=ack_var,
            )
            ack.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 6))

            save_btns = ttk.Frame(step3)
            save_btns.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 12))

            def do_save():
                api_key = (api_key_var.get() or "").strip()
                priv_b64 = (private_b64_state.get("value") or "").strip()

                if not priv_b64:
                    messagebox.showerror("Missing private key", "Step 1: click 'Generate Keys' first.")
                    return

                # Normalize private key so pt_thinker.py can load it:
                # - Accept 32 bytes (seed) OR 64 bytes (seed+pub) from older hub versions
                # - Save ONLY base64(seed32) to rh_secret.txt
                try:
                    raw = base64.b64decode(priv_b64)
                    if len(raw) == 64:
                        raw = raw[:32]
                        priv_b64 = base64.b64encode(raw).decode("utf-8")
                        private_b64_state["value"] = priv_b64  # keep UI state consistent
                    elif len(raw) != 32:
                        messagebox.showerror(
                            "Bad private key",
                            f"Your private key decodes to {len(raw)} bytes, but it must be 32 bytes.\n\n"
                            "Click 'Generate Keys' again to create a fresh keypair."
                        )
                        return
                except Exception as e:
                    messagebox.showerror(
                        "Bad private key",
                        f"Couldn't decode the private key as base64.\n\nError:\n{e}"
                    )
                    return

                if not api_key:
                    messagebox.showerror("Missing API key", "Step 2: paste your API key from Robinhood first.")
                    return
                if not bool(ack_var.get()):
                    messagebox.showwarning(
                        "Please confirm",
                        "For safety, please check the box confirming you understand rh_secret.txt is private."
                    )
                    return

                # Small sanity warning (don’t block, just help)
                if len(api_key) < 10:
                    if not messagebox.askyesno(
                        "API key looks short",
                        "That API key looks unusually short. Are you sure you pasted the API Key from Robinhood?"
                    ):
                        return

                # Back up existing files (so user can undo mistakes)
                try:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    if os.path.isfile(key_path):
                        shutil.copy2(key_path, f"{key_path}.bak_{ts}")
                    if os.path.isfile(secret_path):
                        shutil.copy2(secret_path, f"{secret_path}.bak_{ts}")
                except Exception:
                    pass

                try:
                    # Encrypt and save using Windows DPAPI
                    with open(key_path, "wb") as f:
                        f.write(_encrypt_with_dpapi(api_key))
                    with open(secret_path, "wb") as f:
                        f.write(_encrypt_with_dpapi(priv_b64))
                except Exception as e:
                    messagebox.showerror("Save failed", f"Couldn't write the encrypted credential files.\n\nError:\n{e}")
                    return

                _refresh_api_status()
                messagebox.showinfo(
                    "Saved",
                    "✅ Saved!\n\n"
                    "The trader will automatically read these files next time it starts:\n"
                    f"  API Key → {_mask_path(key_path)}\n"
                    f"  Private Key → {_mask_path(secret_path)}\n\n"
                    "Next steps:\n"
                    "  1) Close this window\n"
                    "  2) Start the trader (pt_trader.py)\n"
                    "If something fails, come back here and click 'Test Credentials'."
                )
                wiz.destroy()

            ttk.Button(save_btns, text="Save", command=do_save).pack(side="left")
            ttk.Button(save_btns, text="Close", command=wiz.destroy).pack(side="left", padx=8)

        # Robinhood API Section
        api_frame = ttk.LabelFrame(frm, text=" Robinhood API ", padding=15)
        api_frame.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(0, 15)); r += 1
        api_frame.columnconfigure(0, weight=1)

        api_row = ttk.Frame(api_frame)
        api_row.pack(fill="x")
        api_row.columnconfigure(0, weight=1)

        ttk.Label(api_row, textvariable=api_status_var).grid(row=0, column=0, sticky="w")
        ttk.Button(api_row, text="Setup Wizard", command=_open_robinhood_api_wizard).grid(row=0, column=1, sticky="e", padx=(10, 0))
        ttk.Button(api_row, text="Open Folder", command=_open_api_folder).grid(row=0, column=2, sticky="e", padx=(8, 0))
        ttk.Button(api_row, text="Clear", command=_clear_api_files).grid(row=0, column=3, sticky="e", padx=(8, 0))

        _refresh_api_status()

        # Display Settings Section
        display_frame = ttk.LabelFrame(frm, text=" Display Settings ", padding=15)
        display_frame.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(0, 15)); r += 1
        display_frame.columnconfigure(0, weight=0)
        display_frame.columnconfigure(1, weight=1)
        display_frame.columnconfigure(2, weight=0)
        
        display_r = 0
        add_row(display_frame, display_r, "UI refresh seconds:", ui_refresh_var); display_r += 1
        add_row(display_frame, display_r, "Chart refresh seconds:", chart_refresh_var); display_r += 1
        add_row(display_frame, display_r, "Candles limit:", candles_limit_var); display_r += 1
        
        # Theme path (if empty, uses default)
        theme_path_var = tk.StringVar(value=self.settings.get("theme_path", ""))
        add_row(display_frame, display_r, "Theme path:", theme_path_var); display_r += 1
        ttk.Label(
            display_frame,
            text="Leave empty for default theme. Relative paths resolved from project directory.",
            foreground=DARK_MUTED,
            font=("TkDefaultFont", 8)
        ).grid(row=display_r, column=0, columnspan=2, sticky="w", pady=(0, 6)); display_r += 1
        
        # Auto-Switch section
        ttk.Label(display_frame, text="").grid(row=display_r, column=0, sticky="w"); display_r += 1  # spacing
        ttk.Label(
            display_frame,
            text="Auto-Switch to Priority Coin:",
            font=("TkDefaultFont", 9, "bold")
        ).grid(row=display_r, column=0, columnspan=2, sticky="w"); display_r += 1
        
        ttk.Label(
            display_frame,
            text="Automatically switches chart view to coins approaching buy/exit triggers.",
            foreground=DARK_MUTED,
            font=("TkDefaultFont", 8)
        ).grid(row=display_r, column=0, columnspan=2, sticky="w"); display_r += 1
        
        autoswitch_enabled_var = tk.BooleanVar(value=self.settings.get("auto_switch", {}).get("enabled", False))
        ttk.Checkbutton(
            display_frame,
            text="Enable Auto-Switch",
            variable=autoswitch_enabled_var
        ).grid(row=display_r, column=0, columnspan=2, sticky="w", pady=6); display_r += 1
        
        ttk.Label(display_frame, text="Threshold (%):").grid(row=display_r, column=0, sticky="w", padx=(0, 10))
        autoswitch_threshold_var = tk.StringVar(value=str(self.settings.get("auto_switch", {}).get("threshold_pct", 2.0)))
        ttk.Entry(display_frame, textvariable=autoswitch_threshold_var, width=15).grid(row=display_r, column=1, sticky="w"); display_r += 1
        
        ttk.Label(
            display_frame,
            text="Switches when within this % of trigger (e.g., 2.0 for 2%)",
            foreground=DARK_MUTED,
            font=("TkDefaultFont", 8)
        ).grid(row=display_r, column=0, columnspan=2, sticky="w"); display_r += 1
        
        ttk.Label(
            display_frame,
            text="⚠ Manual chart selection pauses auto-switch for 2 minutes",
            foreground="orange",
            font=("TkDefaultFont", 8, "bold")
        ).grid(row=display_r, column=0, columnspan=2, sticky="w", pady=(5, 0)); display_r += 1

        btns = ttk.Frame(frm)
        btns.grid(row=r, column=0, columnspan=3, sticky="ew", pady=14)
        btns.columnconfigure(0, weight=1)

        def save():
            try:
                # Track coins before changes so we can detect newly added coins
                prev_coins = set([str(c).strip().upper() for c in (self.settings.get("coins") or []) if str(c).strip()])

                self.settings["main_neural_dir"] = main_dir_var.get().strip()
                self.settings["coins"] = [c.strip().upper() for c in coins_var.get().split(",") if c.strip()]
                self.settings["hub_data_dir"] = hub_dir_var.get().strip()
                self.settings["script_neural_runner2"] = neural_script_var.get().strip()
                self.settings["script_neural_trainer"] = trainer_script_var.get().strip()
                self.settings["script_trader"] = trader_script_var.get().strip()

                self.settings["ui_refresh_seconds"] = float(ui_refresh_var.get().strip())
                self.settings["chart_refresh_seconds"] = float(chart_refresh_var.get().strip())
                self.settings["candles_limit"] = int(float(candles_limit_var.get().strip()))
                
                # Handle theme path
                prev_theme_path = self.settings.get("theme_path", "")
                new_theme_path = theme_path_var.get().strip()
                self.settings["theme_path"] = new_theme_path
                
                # Handle auto-switch settings
                autoswitch_enabled = autoswitch_enabled_var.get()
                autoswitch_threshold = float(autoswitch_threshold_var.get().strip())
                
                # Validate auto-switch threshold
                if not (0 < autoswitch_threshold <= 100):
                    messagebox.showerror("Validation Error", "Auto-switch threshold must be between 0 and 100%")
                    return
                
                self.settings["auto_switch"] = {
                    "enabled": autoswitch_enabled,
                    "threshold_pct": autoswitch_threshold
                }
                
                self._save_settings()

                # If new coin(s) were added and their training folder doesn't exist yet,
                # create the folder and copy neural_trainer.py from MAIN folder into it RIGHT AFTER saving settings.
                try:
                    new_coins = [c.strip().upper() for c in (self.settings.get("coins") or []) if c.strip()]
                    added = [c for c in new_coins if c and c not in prev_coins]

                    main_dir = self.settings.get("main_neural_dir") or self.project_dir
                    trainer_name = os.path.basename(str(self.settings.get("script_neural_trainer", "neural_trainer.py")))

                    # Source trainer: MAIN folder (preferred location)
                    src_main_trainer = os.path.join(main_dir, trainer_name)
                    src_trainer_path = src_main_trainer if os.path.isfile(src_main_trainer) else str(self.settings.get("script_neural_trainer", trainer_name))

                    for coin in added:
                        coin_dir = os.path.join(main_dir, coin)
                        if not os.path.isdir(coin_dir):
                            os.makedirs(coin_dir, exist_ok=True)

                        dst_trainer_path = os.path.join(coin_dir, trainer_name)
                        if (not os.path.isfile(dst_trainer_path)) and os.path.isfile(src_trainer_path):
                            shutil.copy2(src_trainer_path, dst_trainer_path)
                except Exception:
                    pass

                # Refresh all coin-driven UI (dropdowns + chart tabs)
                self._refresh_coin_dependent_ui(prev_coins)

                # Notify user about restart if theme changed
                if new_theme_path != prev_theme_path:
                    messagebox.showinfo(
                        "Saved", 
                        "Settings saved.\n\nTheme setting changed — restart ApolloTrader for full effect."
                    )
                else:
                    messagebox.showinfo("Saved", "Settings saved.")
                on_close()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save settings:\n{e}")

        def reset_hub_defaults():
            try:
                from copy import deepcopy
                defaults = deepcopy(DEFAULT_SETTINGS)
                
                # Reset Basic Settings
                main_dir_var.set(defaults.get("main_neural_dir", ""))
                coins_var.set(",".join(defaults.get("coins", [])))
                hub_dir_var.set(defaults.get("hub_data_dir", ""))
                
                # Reset Script Paths
                neural_script_var.set(defaults.get("script_neural_runner2", "pt_thinker.py"))
                trainer_script_var.set(defaults.get("script_neural_trainer", "pt_trainer.py"))
                trader_script_var.set(defaults.get("script_trader", "pt_trader.py"))
                
                # Reset Display Settings
                ui_refresh_var.set(str(defaults.get("ui_refresh_seconds", 1.0)))
                chart_refresh_var.set(str(defaults.get("chart_refresh_seconds", 10.0)))
                candles_limit_var.set(str(defaults.get("candles_limit", 120)))
                theme_path_var.set(defaults.get("theme_path", ""))
                
                # Save to file
                self.settings.update(defaults)
                self._save_settings()
                messagebox.showinfo("Reset", "Hub settings have been reset to defaults.")
                on_close()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to reset settings:\n{e}")

        ttk.Button(btns, text="Save", command=save).pack(side="left")
        ttk.Button(btns, text="Cancel", command=on_close).pack(side="left", padx=8)
        ttk.Button(btns, text="Reset to Defaults", command=reset_hub_defaults).pack(side="left")

    # Configuration file editor dialogs for trading and training settings
    def open_trading_settings_dialog(self) -> None:
        """Open GUI dialog for editing trading_settings.json."""
        # Prevent duplicate dialogs - bring existing one to front if already open
        if hasattr(self, "_trading_settings_win") and self._trading_settings_win and self._trading_settings_win.winfo_exists():
            self._trading_settings_win.lift()
            self._trading_settings_win.focus_force()
            return
        
        config_path = os.path.join(self.project_dir, TRADING_SETTINGS_FILE)
        
        # Load current config
        cfg = _load_trading_config()

        win = tk.Toplevel(self)
        win.title("Trading Settings")
        win.geometry("860x680")
        win.minsize(760, 560)
        win.configure(bg=DARK_BG)
        win.transient(self)  # Keep dialog on top of parent
        
        self._trading_settings_win = win
        
        def on_close():
            self._trading_settings_win = None
            win.destroy()
        
        win.protocol("WM_DELETE_WINDOW", on_close)

        # Scrollable content
        viewport = ttk.Frame(win)
        viewport.pack(fill="both", expand=True, padx=12, pady=12)
        viewport.grid_rowconfigure(0, weight=1)
        viewport.grid_columnconfigure(0, weight=1)

        settings_canvas = tk.Canvas(
            viewport,
            bg=DARK_BG,
            highlightthickness=1,
            highlightbackground=DARK_BORDER,
            bd=0,
        )
        settings_canvas.grid(row=0, column=0, sticky="nsew")

        settings_scroll = ttk.Scrollbar(
            viewport,
            orient="vertical",
            command=settings_canvas.yview,
        )
        settings_scroll.grid(row=0, column=1, sticky="ns")
        settings_canvas.configure(yscrollcommand=settings_scroll.set)

        frm = ttk.Frame(settings_canvas, padding=(0, 0, 13, 0))
        settings_window = settings_canvas.create_window((0, 0), window=frm, anchor="nw")

        def _update_scrollbars(event=None) -> None:
            try:
                c = settings_canvas
                win_id = settings_window

                c.update_idletasks()
                bbox = c.bbox(win_id)
                if not bbox:
                    settings_scroll.grid_remove()
                    return

                c.configure(scrollregion=bbox)
                content_h = int(bbox[3] - bbox[1])
                view_h = int(c.winfo_height())

                if content_h > (view_h + 1):
                    settings_scroll.grid()
                else:
                    settings_scroll.grid_remove()
                    try:
                        c.yview_moveto(0)
                    except Exception:
                        pass
            except Exception:
                pass

        def _on_canvas_configure(e) -> None:
            try:
                settings_canvas.itemconfigure(settings_window, width=int(e.width))
            except Exception:
                pass
            _update_scrollbars()

        settings_canvas.bind("<Configure>", _on_canvas_configure, add="+")
        frm.bind("<Configure>", _update_scrollbars, add="+")

        def _wheel(e):
            try:
                if settings_scroll.winfo_ismapped():
                    settings_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
            except Exception:
                pass

        def _bind_mousewheel_recursive(widget):
            """Recursively bind mouse wheel to widget and all children so it works everywhere."""
            try:
                widget.bind("<MouseWheel>", _wheel, add="+")
                widget.bind("<Button-4>", lambda _e: settings_canvas.yview_scroll(-3, "units"), add="+")
                widget.bind("<Button-5>", lambda _e: settings_canvas.yview_scroll(3, "units"), add="+")
                for child in widget.winfo_children():
                    _bind_mousewheel_recursive(child)
            except Exception:
                pass

        settings_canvas.bind("<MouseWheel>", _wheel, add="+")
        settings_canvas.bind("<Button-4>", lambda _e: settings_canvas.yview_scroll(-3, "units"), add="+")
        settings_canvas.bind("<Button-5>", lambda _e: settings_canvas.yview_scroll(3, "units"), add="+")
        _bind_mousewheel_recursive(frm)  # Bind to all child widgets

        frm.columnconfigure(0, weight=0)
        frm.columnconfigure(1, weight=1)

        r = 0

        # DCA (Dollar Cost Averaging) Section
        dca_frame = ttk.LabelFrame(frm, text=" DCA (Dollar Cost Averaging) ", padding=15)
        dca_frame.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(0, 15)); r += 1
        dca_frame.columnconfigure(0, weight=0)
        dca_frame.columnconfigure(1, weight=1)

        ttk.Label(
            dca_frame,
            text="Buy more when price drops to reduce average cost. Each level increases position size.",
            foreground=DARK_FG,
            wraplength=550
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        dca_levels = cfg.get("dca", {}).get("levels", [-2.5, -5.0, -10.0, -20.0])
        dca_levels_str = ", ".join(str(l) for l in dca_levels)

        dca_r = 1
        ttk.Label(dca_frame, text="DCA Levels (%):").grid(row=dca_r, column=0, sticky="w", padx=(0, 10), pady=6)
        dca_levels_var = tk.StringVar(value=dca_levels_str)
        ttk.Entry(dca_frame, textvariable=dca_levels_var, width=50).grid(row=dca_r, column=1, sticky="w", pady=6)
        dca_r += 1
        ttk.Label(dca_frame, text="Comma-separated negative percentages (e.g., -2.5, -5, -10, -20, -40)", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=dca_r, column=0, columnspan=2, sticky="w", pady=(0, 10))
        dca_r += 1

        ttk.Label(
            dca_frame,
            text="⚠ Levels must be negative and increasingly negative. Each must be unique. Max 20 levels.",
            foreground="orange",
            font=("TkDefaultFont", 8, "bold")
        ).grid(row=dca_r, column=0, columnspan=2, sticky="w", pady=(0, 15)); dca_r += 1

        ttk.Label(dca_frame, text="Max DCA buys per window:").grid(row=dca_r, column=0, sticky="w", padx=(0, 10), pady=6)
        max_buys_var = tk.StringVar(value=str(cfg.get("dca", {}).get("max_buys_per_window", 2)))
        ttk.Entry(dca_frame, textvariable=max_buys_var, width=15).grid(row=dca_r, column=1, sticky="w", pady=6)
        dca_r += 1
        ttk.Label(dca_frame, text="Limits additional DCA purchases within time window", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=dca_r, column=0, columnspan=2, sticky="w", pady=(0, 10))
        dca_r += 1

        ttk.Label(dca_frame, text="Window (hours):").grid(row=dca_r, column=0, sticky="w", padx=(0, 10), pady=6)
        window_hours_var = tk.StringVar(value=str(cfg.get("dca", {}).get("window_hours", 24)))
        ttk.Entry(dca_frame, textvariable=window_hours_var, width=15).grid(row=dca_r, column=1, sticky="w", pady=6)
        dca_r += 1
        ttk.Label(dca_frame, text="Time window for counting DCA buys", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=dca_r, column=0, columnspan=2, sticky="w", pady=(0, 10))
        dca_r += 1

        ttk.Label(dca_frame, text="Position multiplier:").grid(row=dca_r, column=0, sticky="w", padx=(0, 10), pady=6)
        multiplier_var = tk.StringVar(value=str(cfg.get("dca", {}).get("position_multiplier", 2.0)))
        ttk.Entry(dca_frame, textvariable=multiplier_var, width=15).grid(row=dca_r, column=1, sticky="w", pady=6)
        dca_r += 1
        ttk.Label(dca_frame, text="Each DCA level buys this multiple of previous level", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=dca_r, column=0, columnspan=2, sticky="w", pady=(0, 0))

        # Profit Margin Section
        profit_frame = ttk.LabelFrame(frm, text=" Profit Margin ", padding=15)
        profit_frame.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(0, 15)); r += 1
        profit_frame.columnconfigure(0, weight=0)
        profit_frame.columnconfigure(1, weight=1)

        ttk.Label(
            profit_frame,
            text="Trailing stop strategy: exits when price falls specified % below peak since entry.",
            foreground=DARK_FG,
            wraplength=550
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        
        ttk.Label(profit_frame, text="Trailing gap (%):").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=6)
        trailing_gap_var = tk.StringVar(value=str(cfg.get("profit_margin", {}).get("trailing_gap_pct", 0.5)))
        ttk.Entry(profit_frame, textvariable=trailing_gap_var, width=15).grid(row=1, column=1, sticky="w", pady=6)
        ttk.Label(profit_frame, text="Distance below peak to trigger sell", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(profit_frame, text="Target without DCA (%):").grid(row=3, column=0, sticky="w", padx=(0, 10), pady=6)
        target_no_dca_var = tk.StringVar(value=str(cfg.get("profit_margin", {}).get("target_no_dca_pct", 5.0)))
        ttk.Entry(profit_frame, textvariable=target_no_dca_var, width=15).grid(row=3, column=1, sticky="w", pady=6)
        ttk.Label(profit_frame, text="Profit target for initial entry only", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=4, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(profit_frame, text="Target with DCA (%):").grid(row=5, column=0, sticky="w", padx=(0, 10), pady=6)
        target_with_dca_var = tk.StringVar(value=str(cfg.get("profit_margin", {}).get("target_with_dca_pct", 2.5)))
        ttk.Entry(profit_frame, textvariable=target_with_dca_var, width=15).grid(row=5, column=1, sticky="w", pady=6)
        ttk.Label(profit_frame, text="Profit target when DCA has triggered", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=6, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(profit_frame, text="Stop-loss (%):").grid(row=7, column=0, sticky="w", padx=(0, 10), pady=6)
        stop_loss_var = tk.StringVar(value=str(cfg.get("profit_margin", {}).get("stop_loss_pct", -40.0)))
        ttk.Entry(profit_frame, textvariable=stop_loss_var, width=15).grid(row=7, column=1, sticky="w", pady=6)
        ttk.Label(profit_frame, text="Emergency exit if loss exceeds this % (negative value)", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=8, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(
            profit_frame,
            text="⚠ Stop-loss is a safety net for catastrophic drops. Set deep enough to allow DCA stages.",
            foreground="orange",
            font=("TkDefaultFont", 8, "bold")
        ).grid(row=9, column=0, columnspan=2, sticky="w", pady=(5, 0))

        # Entry Signals Section
        entry_frame = ttk.LabelFrame(frm, text=" Entry Signals ", padding=15)
        entry_frame.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(0, 15)); r += 1
        entry_frame.columnconfigure(0, weight=0)
        entry_frame.columnconfigure(1, weight=1)
        entry_frame.columnconfigure(2, weight=0)

        ttk.Label(
            entry_frame,
            text="AI generates 0-7 signals. Higher numbers = stronger prediction. Minimum threshold triggers trades.",
            foreground=DARK_FG,
            wraplength=550
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        
        ttk.Label(entry_frame, text="Long signal minimum (1-7):").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=6)
        long_min_var = tk.StringVar(value=str(cfg.get("entry_signals", {}).get("long_signal_min", 4)))
        ttk.Entry(entry_frame, textvariable=long_min_var, width=15).grid(row=1, column=1, sticky="w", pady=6)
        ttk.Label(entry_frame, text="AI signal strength required to enter buy", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(entry_frame, text="Short signal maximum (0-7):").grid(row=3, column=0, sticky="w", padx=(0, 10), pady=6)
        short_max_var = tk.StringVar(value=str(cfg.get("entry_signals", {}).get("short_signal_max", 0)))
        ttk.Entry(entry_frame, textvariable=short_max_var, width=15).grid(row=3, column=1, sticky="w", pady=6)
        ttk.Label(entry_frame, text="Reserved for future short trading (unused)", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=4, column=0, columnspan=2, sticky="w", pady=(0, 0))

        # Position Sizing Section
        position_frame = ttk.LabelFrame(frm, text=" Position Sizing ", padding=15)
        position_frame.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(0, 15)); r += 1
        position_frame.columnconfigure(0, weight=0)
        position_frame.columnconfigure(1, weight=1)
        position_frame.columnconfigure(2, weight=0)


        ttk.Label(
            position_frame,
            text="Determines how much to invest per trade based on account balance and risk management.",
            foreground=DARK_FG,
            wraplength=550
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        
        ttk.Label(position_frame, text="Initial allocation (%):").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=6)
        allocation_var = tk.StringVar(value=str(cfg.get("position_sizing", {}).get("initial_allocation_pct", 0.01) * 100))
        ttk.Entry(position_frame, textvariable=allocation_var, width=15).grid(row=1, column=1, sticky="w", pady=6)
        ttk.Label(position_frame, text="Percentage of account for first buy (e.g., 1 for 1%)", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(position_frame, text="Minimum allocation ($):").grid(row=3, column=0, sticky="w", padx=(0, 10), pady=6)
        min_alloc_var = tk.StringVar(value=str(cfg.get("position_sizing", {}).get("min_allocation_usd", 0.5)))
        ttk.Entry(position_frame, textvariable=min_alloc_var, width=15).grid(row=3, column=1, sticky="w", pady=6)
        ttk.Label(position_frame, text="Floor for position size in dollars", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=4, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(position_frame, text="Max concurrent positions:").grid(row=5, column=0, sticky="w", padx=(0, 10), pady=6)
        max_positions_var = tk.StringVar(value=str(cfg.get("position_sizing", {}).get("max_concurrent_positions", 3)))
        ttk.Entry(position_frame, textvariable=max_positions_var, width=15).grid(row=5, column=1, sticky="w", pady=6)
        ttk.Label(position_frame, text="Maximum number of coins to hold simultaneously", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=6, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(
            position_frame,
            text="⚠ Limits exposure by preventing too many positions at once. Slots open when positions exit.",
            foreground="orange",
            font=("TkDefaultFont", 8, "bold")
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(5, 0))

        # Timing Section
        timing_frame = ttk.LabelFrame(frm, text=" Timing ", padding=15)
        timing_frame.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(0, 15)); r += 1
        timing_frame.columnconfigure(0, weight=0)
        timing_frame.columnconfigure(1, weight=1)

        ttk.Label(
            timing_frame,
            text="Controls how frequently the trading bot checks for opportunities and cooldown periods.",
            foreground=DARK_FG,
            wraplength=550
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        
        ttk.Label(timing_frame, text="Main loop delay (seconds):").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=6)
        loop_delay_var = tk.StringVar(value=str(cfg.get("timing", {}).get("main_loop_delay_seconds", 0.5)))
        ttk.Entry(timing_frame, textvariable=loop_delay_var, width=15).grid(row=1, column=1, sticky="w", pady=6)
        ttk.Label(timing_frame, text="Pause between trade checks", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(timing_frame, text="Post-trade delay (seconds):").grid(row=3, column=0, sticky="w", padx=(0, 10), pady=6)
        post_trade_var = tk.StringVar(value=str(cfg.get("timing", {}).get("post_trade_delay_seconds", 30)))
        ttk.Entry(timing_frame, textvariable=post_trade_var, width=15).grid(row=3, column=1, sticky="w", pady=6)
        ttk.Label(timing_frame, text="Cooldown after executing trade", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=4, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(
            timing_frame,
            text="Note: Lower delays = more responsive but higher API usage",
            foreground=DARK_FG,
            font=("TkDefaultFont", 8, "italic")
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(5, 0))

        # Buttons
        btns = ttk.Frame(frm)
        btns.grid(row=r, column=0, columnspan=2, sticky="ew", pady=14)
        btns.columnconfigure(0, weight=1)

        def save():
            try:
                # Parse and validate all fields
                dca_input = dca_levels_var.get().strip()
                if not dca_input:
                    messagebox.showerror("Validation Error", "DCA levels cannot be empty")
                    return
                dca_lvls = [float(x.strip()) for x in dca_input.split(',') if x.strip()]
                if not dca_lvls:
                    messagebox.showerror("Validation Error", "DCA levels cannot be empty")
                    return
                if len(dca_lvls) > 20:
                    messagebox.showerror("Validation Error", f"Maximum 20 DCA levels allowed (you entered {len(dca_lvls)})")
                    return
                
                max_buys = int(max_buys_var.get())
                window_hrs = int(window_hours_var.get())
                multiplier = float(multiplier_var.get())
                
                trailing_gap = float(trailing_gap_var.get())
                target_no_dca = float(target_no_dca_var.get())
                target_with_dca = float(target_with_dca_var.get())
                stop_loss = float(stop_loss_var.get())
                
                long_min = int(long_min_var.get())
                short_max = int(short_max_var.get())
                
                alloc_pct = float(allocation_var.get()) / 100.0  # Convert from percentage to decimal
                min_alloc = float(min_alloc_var.get())
                max_positions = int(max_positions_var.get())
                
                loop_delay = float(loop_delay_var.get())
                post_trade = float(post_trade_var.get())

                # Validation
                # Check for duplicate levels
                if len(dca_lvls) != len(set(dca_lvls)):
                    # Find duplicates for error message
                    seen = set()
                    duplicates = set()
                    for lvl in dca_lvls:
                        if lvl in seen:
                            duplicates.add(lvl)
                        seen.add(lvl)
                    
                    # Debug: log duplicate detection
                    if self.settings.get("debug_mode", False):
                        print(f"[HUB DEBUG] Duplicate DCA levels detected: {sorted(duplicates)}")
                    
                    messagebox.showerror(
                        "Validation Error",
                        f"Duplicate DCA levels detected: {sorted(duplicates)}\n\n"
                        f"Each DCA level must be unique."
                    )
                    return
                
                # DCA levels must be negative and in descending order (more negative)
                for i, lvl in enumerate(dca_lvls):
                    if lvl >= 0:
                        messagebox.showerror("Validation Error", f"DCA Level {i+1} must be negative (e.g., -2.5)")
                        return
                    if i > 0 and lvl >= dca_lvls[i-1]:
                        messagebox.showerror("Validation Error", f"DCA Level {i+1} ({lvl}%) must be more negative than Level {i} ({dca_lvls[i-1]}%)")
                        return

                if max_buys < 1:
                    messagebox.showerror("Validation Error", "Max DCA buys per window must be at least 1")
                    messagebox.showerror("Validation Error", "Window hours must be at least 1")
                    return

                if multiplier <= 0:
                    messagebox.showerror("Validation Error", "Position multiplier must be greater than 0")
                    return

                # Percentages must be 0-100
                if not (0 < trailing_gap <= 100):
                    messagebox.showerror("Validation Error", "Trailing gap must be between 0 and 100%")
                    return
                if not (0 < target_no_dca <= 100):
                    messagebox.showerror("Validation Error", "Target without DCA must be between 0 and 100%")
                    return
                if not (0 < target_with_dca <= 100):
                    messagebox.showerror("Validation Error", "Target with DCA must be between 0 and 100%")
                    return
                if not (-100 <= stop_loss < 0):
                    messagebox.showerror("Validation Error", "Stop-loss must be negative and between -100 and 0% (e.g., -50)")
                    return

                # Entry signals
                if not (1 <= long_min <= 7):
                    messagebox.showerror("Validation Error", "Long signal minimum must be between 1 and 7")
                    return
                if not (0 <= short_max <= 7):
                    messagebox.showerror("Validation Error", "Short signal maximum must be between 0 and 7")
                    return

                # Position sizing
                if not (0 < alloc_pct <= 100):
                    messagebox.showerror("Validation Error", "Initial allocation % must be between 0 and 100")
                    return
                if min_alloc < 0:
                    messagebox.showerror("Validation Error", "Minimum allocation must be non-negative")
                    return
                if max_positions < 1:
                    messagebox.showerror("Validation Error", "Max concurrent positions must be at least 1")
                    return

                # Timing
                if loop_delay <= 0:
                    messagebox.showerror("Validation Error", "Main loop delay must be greater than 0 seconds")
                    return
                if post_trade < 0:
                    messagebox.showerror("Validation Error", "Post-trade delay must be non-negative")
                    return

                # Build config from all fields
                new_cfg = {
                    "dca": {
                        "levels": dca_lvls,
                        "max_buys_per_window": max_buys,
                        "window_hours": window_hrs,
                        "position_multiplier": multiplier
                    },
                    "profit_margin": {
                        "trailing_gap_pct": trailing_gap,
                        "target_no_dca_pct": target_no_dca,
                        "target_with_dca_pct": target_with_dca,
                        "stop_loss_pct": stop_loss
                    },
                    "entry_signals": {
                        "long_signal_min": long_min,
                        "short_signal_max": short_max
                    },
                    "position_sizing": {
                        "initial_allocation_pct": alloc_pct,
                        "min_allocation_usd": min_alloc,
                        "max_concurrent_positions": max_positions
                    },
                    "timing": {
                        "main_loop_delay_seconds": loop_delay,
                        "post_trade_delay_seconds": post_trade
                    }
                }

                _safe_write_json(config_path, new_cfg)
                messagebox.showinfo("Saved", "Trading settings saved.")
                on_close()

            except ValueError as e:
                messagebox.showerror("Invalid Input", f"Please enter valid numbers in all fields.\n\n{e}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save settings:\n{e}")

        def reset_trading_defaults():
            try:
                from copy import deepcopy
                defaults = deepcopy(DEFAULT_TRADING_CONFIG)
                
                # Reset DCA fields
                dca = defaults.get("dca", {})
                levels = dca.get("levels", [])
                dca_levels_var.set(", ".join(str(l) for l in levels))
                max_buys_var.set(str(dca.get("max_buys_per_window", 2)))
                window_hours_var.set(str(dca.get("window_hours", 24)))
                multiplier_var.set(str(dca.get("position_multiplier", 2.0)))
                
                # Reset Profit Margin fields
                profit = defaults.get("profit_margin", {})
                trailing_gap_var.set(str(profit.get("trailing_gap_pct", 0.5)))
                target_no_dca_var.set(str(profit.get("target_no_dca_pct", 5.0)))
                target_with_dca_var.set(str(profit.get("target_with_dca_pct", 2.5)))
                stop_loss_var.set(str(profit.get("stop_loss_pct", -40.0)))
                
                # Reset Entry Signals fields
                entry = defaults.get("entry_signals", {})
                long_min_var.set(str(entry.get("long_signal_min", 4)))
                short_max_var.set(str(entry.get("short_signal_max", 0)))
                
                # Reset Position Sizing fields
                position = defaults.get("position_sizing", {})
                allocation_var.set(str(position.get("initial_allocation_pct", 0.01) * 100))
                min_alloc_var.set(str(position.get("min_allocation_usd", 0.5)))
                max_positions_var.set(str(position.get("max_concurrent_positions", 3)))
                
                # Reset Timing fields
                timing = defaults.get("timing", {})
                loop_delay_var.set(str(timing.get("main_loop_delay_seconds", 0.5)))
                post_trade_var.set(str(timing.get("post_trade_delay_seconds", 30)))
                
                # Save to file
                _safe_write_json(config_path, defaults)
                messagebox.showinfo("Reset", "Trading settings have been reset to defaults.")
                on_close()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to reset settings:\n{e}")

        ttk.Button(btns, text="Apply", command=save).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Cancel", command=on_close).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Reset to Defaults", command=reset_trading_defaults).pack(side="left")

    def open_training_settings_dialog(self) -> None:
        """Open GUI dialog for editing training_settings.json."""
        # Prevent duplicate dialogs - bring existing one to front if already open
        if hasattr(self, "_training_settings_win") and self._training_settings_win and self._training_settings_win.winfo_exists():
            self._training_settings_win.lift()
            self._training_settings_win.focus_force()
            return
        
        config_path = os.path.join(self.project_dir, TRAINING_SETTINGS_FILE)
        
        # Load current config
        cfg = _load_training_config()

        win = tk.Toplevel(self)
        win.title("Training Settings")
        win.geometry("700x700")
        win.minsize(650, 600)
        win.configure(bg=DARK_BG)
        win.transient(self)  # Keep dialog on top of parent
        
        self._training_settings_win = win
        
        def on_close():
            self._training_settings_win = None
            win.destroy()
        
        win.protocol("WM_DELETE_WINDOW", on_close)

        # Scrollable content (matching Trading Settings style)
        viewport = ttk.Frame(win)
        viewport.pack(fill="both", expand=True, padx=12, pady=12)
        viewport.grid_rowconfigure(0, weight=1)
        viewport.grid_columnconfigure(0, weight=1)

        settings_canvas = tk.Canvas(
            viewport,
            bg=DARK_BG,
            highlightthickness=1,
            highlightbackground=DARK_BORDER,
            bd=0,
        )
        settings_canvas.grid(row=0, column=0, sticky="nsew")

        settings_scroll = ttk.Scrollbar(
            viewport,
            orient="vertical",
            command=settings_canvas.yview,
        )
        settings_scroll.grid(row=0, column=1, sticky="ns")
        settings_canvas.configure(yscrollcommand=settings_scroll.set)

        frm = ttk.Frame(settings_canvas, padding=(0, 0, 13, 0))
        settings_window = settings_canvas.create_window((0, 0), window=frm, anchor="nw")

        def _update_scrollbars(event=None) -> None:
            try:
                c = settings_canvas
                win_id = settings_window

                c.update_idletasks()
                bbox = c.bbox(win_id)
                if not bbox:
                    settings_scroll.grid_remove()
                    return

                c.configure(scrollregion=bbox)
                content_h = int(bbox[3] - bbox[1])
                view_h = int(c.winfo_height())

                if content_h > (view_h + 1):
                    settings_scroll.grid()
                else:
                    settings_scroll.grid_remove()
                    try:
                        c.yview_moveto(0)
                    except Exception:
                        pass
            except Exception:
                pass

        def _on_canvas_configure(e) -> None:
            try:
                settings_canvas.itemconfigure(settings_window, width=int(e.width))
            except Exception:
                pass
            _update_scrollbars()

        settings_canvas.bind("<Configure>", _on_canvas_configure, add="+")
        frm.bind("<Configure>", _update_scrollbars, add="+")

        def _wheel(e):
            try:
                if settings_scroll.winfo_ismapped():
                    settings_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
            except Exception:
                pass

        def _bind_mousewheel_recursive(widget):
            """Recursively bind mouse wheel to widget and all children so it works everywhere."""
            try:
                widget.bind("<MouseWheel>", _wheel, add="+")
                widget.bind("<Button-4>", lambda _e: settings_canvas.yview_scroll(-3, "units"), add="+")
                widget.bind("<Button-5>", lambda _e: settings_canvas.yview_scroll(3, "units"), add="+")
                for child in widget.winfo_children():
                    _bind_mousewheel_recursive(child)
            except Exception:
                pass

        settings_canvas.bind("<MouseWheel>", _wheel, add="+")
        settings_canvas.bind("<Button-4>", lambda _e: settings_canvas.yview_scroll(-3, "units"), add="+")
        settings_canvas.bind("<Button-5>", lambda _e: settings_canvas.yview_scroll(3, "units"), add="+")
        _bind_mousewheel_recursive(frm)  # Bind to all child widgets

        frm.columnconfigure(0, weight=0)
        frm.columnconfigure(1, weight=1)

        r = 0

        # General Settings Section
        general_frame = ttk.LabelFrame(frm, text=" General Settings ", padding=15)
        general_frame.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(0, 15)); r += 1
        general_frame.columnconfigure(0, weight=0)
        general_frame.columnconfigure(1, weight=1)

        # Staleness threshold
        ttk.Label(general_frame, text="Staleness threshold (days):").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=6)
        staleness_var = tk.StringVar(value=str(cfg.get("staleness_days", 14)))
        ttk.Entry(general_frame, textvariable=staleness_var, width=15).grid(row=0, column=1, sticky="w", pady=6)

        ttk.Label(
            general_frame,
            text="Number of days before AI training is considered stale and should be retrained.",
            foreground=DARK_FG,
            wraplength=500
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 10))

        # Auto-train checkbox
        auto_train_var = tk.BooleanVar(value=bool(cfg.get("auto_train_when_stale", False)))
        chk = ttk.Checkbutton(general_frame, text="Automatically retrain when stale", variable=auto_train_var)
        chk.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 0))

        # Learning Settings Section
        learning_frame = ttk.LabelFrame(frm, text=" Learning Settings ", padding=15)
        learning_frame.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(0, 15)); r += 1
        learning_frame.columnconfigure(0, weight=0)
        learning_frame.columnconfigure(1, weight=1)

        # Pattern size (moved from Pattern Quality)
        ttk.Label(learning_frame, text="Pattern size (candles):").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=6)
        pattern_size_var = tk.StringVar(value=str(cfg.get("pattern_size", 4)))
        ttk.Entry(learning_frame, textvariable=pattern_size_var, width=15).grid(row=0, column=1, sticky="w", pady=6)

        ttk.Label(
            learning_frame,
            text="Number of candles per pattern (2-5 range; 3=balanced, 5=specific). ⚠ Changing requires full retrain!",
            foreground="orange",
            font=("TkDefaultFont", 8, "bold")
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 15))

        ttk.Label(
            learning_frame,
            text="PID controller adaptively adjusts similarity threshold for optimal pattern matching:",
            foreground=DARK_FG,
            wraplength=550
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 10))

        # Kp (Proportional gain)
        ttk.Label(learning_frame, text="Kp (Proportional):").grid(row=3, column=0, sticky="w", padx=(0, 10), pady=6)
        pid_kp_var = tk.StringVar(value=str(cfg.get("pid_kp", 0.5)))
        ttk.Entry(learning_frame, textvariable=pid_kp_var, width=15).grid(row=3, column=1, sticky="w", pady=6)

        # Ki (Integral gain)
        ttk.Label(learning_frame, text="Ki (Integral):").grid(row=4, column=0, sticky="w", padx=(0, 10), pady=6)
        pid_ki_var = tk.StringVar(value=str(cfg.get("pid_ki", 0.005)))
        ttk.Entry(learning_frame, textvariable=pid_ki_var, width=15).grid(row=4, column=1, sticky="w", pady=6)

        # Kd (Derivative gain)
        ttk.Label(learning_frame, text="Kd (Derivative):").grid(row=5, column=0, sticky="w", padx=(0, 10), pady=6)
        pid_kd_var = tk.StringVar(value=str(cfg.get("pid_kd", 0.2)))
        ttk.Entry(learning_frame, textvariable=pid_kd_var, width=15).grid(row=5, column=1, sticky="w", pady=6)

        # Integral limit
        ttk.Label(learning_frame, text="Integral limit:").grid(row=6, column=0, sticky="w", padx=(0, 10), pady=6)
        pid_limit_var = tk.StringVar(value=str(cfg.get("pid_integral_limit", 20)))
        ttk.Entry(learning_frame, textvariable=pid_limit_var, width=15).grid(row=6, column=1, sticky="w", pady=6)

        ttk.Label(
            learning_frame,
            text="PID gains: Kp=response speed, Ki=steady-state correction, Kd=damping. Leave defaults unless tuning.",
            foreground=DARK_FG,
            font=("TkDefaultFont", 8)
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(0, 10))

        # Minimum threshold
        ttk.Label(learning_frame, text="Min threshold (%):").grid(row=8, column=0, sticky="w", padx=(0, 10), pady=6)
        min_threshold_var = tk.StringVar(value=str(cfg.get("min_threshold", 5.0)))
        ttk.Entry(learning_frame, textvariable=min_threshold_var, width=15).grid(row=8, column=1, sticky="w", pady=6)

        ttk.Label(
            learning_frame,
            text="Lower limit for threshold adjustment (prevents too strict matching)",
            foreground=DARK_FG,
            font=("TkDefaultFont", 8)
        ).grid(row=9, column=0, columnspan=2, sticky="w", pady=(0, 10))

        # Maximum threshold
        ttk.Label(learning_frame, text="Max threshold (%):").grid(row=10, column=0, sticky="w", padx=(0, 10), pady=6)
        max_threshold_var = tk.StringVar(value=str(cfg.get("max_threshold", 20.0)))
        ttk.Entry(learning_frame, textvariable=max_threshold_var, width=15).grid(row=10, column=1, sticky="w", pady=6)

        ttk.Label(
            learning_frame,
            text="Upper limit for threshold adjustment (prevents too loose matching)",
            foreground=DARK_FG,
            font=("TkDefaultFont", 8)
        ).grid(row=11, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(
            learning_frame,
            text="Note: Initial and target thresholds will be the average of min and max",
            foreground=DARK_FG,
            font=("TkDefaultFont", 8, "italic")
        ).grid(row=12, column=0, columnspan=2, sticky="w", pady=(0, 0))

        # Advanced Weight System Section
        advanced_frame = ttk.LabelFrame(frm, text=" Advanced Weight System ", padding=15)
        advanced_frame.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(0, 15)); r += 1
        advanced_frame.columnconfigure(0, weight=0)
        advanced_frame.columnconfigure(1, weight=1)

        ttk.Label(
            advanced_frame,
            text="⚠ Research-backed optimizations for high bounce accuracy. Default values are tuned. Only modify if you understand learning crypto dynamics.",
            foreground="orange",
            font=("TkDefaultFont", 9, "bold"),
            wraplength=550
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 15))

        # Error-Proportional Scaling
        ttk.Label(advanced_frame, text="Error-Proportional Scaling", font=("TkDefaultFont", 9, "bold")).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))
        
        ttk.Label(advanced_frame, text="Base weight step:").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=6)
        weight_base_step_var = tk.StringVar(value=str(cfg.get("weight_base_step", 0.25)))
        ttk.Entry(advanced_frame, textvariable=weight_base_step_var, width=15).grid(row=2, column=1, sticky="w", pady=6)
        ttk.Label(advanced_frame, text="Base adjustment step size (0.1-0.5 range)", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(advanced_frame, text="Max step multiplier:").grid(row=4, column=0, sticky="w", padx=(0, 10), pady=6)
        weight_step_cap_var = tk.StringVar(value=str(cfg.get("weight_step_cap_multiplier", 2.0)))
        ttk.Entry(advanced_frame, textvariable=weight_step_cap_var, width=15).grid(row=4, column=1, sticky="w", pady=6)
        ttk.Label(advanced_frame, text="Cap on error-based scaling (1.5-3.0 range, higher = faster large error correction)", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 15))

        # Volatility-Adaptive Thresholds
        ttk.Label(advanced_frame, text="Volatility-Adaptive Thresholds", font=("TkDefaultFont", 9, "bold")).grid(row=6, column=0, columnspan=2, sticky="w", pady=(0, 8))
        
        ttk.Label(advanced_frame, text="Base threshold multiplier:").grid(row=7, column=0, sticky="w", padx=(0, 10), pady=6)
        weight_threshold_base_var = tk.StringVar(value=str(cfg.get("weight_threshold_base", 0.1)))
        ttk.Entry(advanced_frame, textvariable=weight_threshold_base_var, width=15).grid(row=7, column=1, sticky="w", pady=6)
        ttk.Label(advanced_frame, text="Starting point for adaptive thresholds (0.05-0.15 range)", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=8, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(advanced_frame, text="Min threshold multiplier:").grid(row=9, column=0, sticky="w", padx=(0, 10), pady=6)
        weight_threshold_min_var = tk.StringVar(value=str(cfg.get("weight_threshold_min", 0.03)))
        ttk.Entry(advanced_frame, textvariable=weight_threshold_min_var, width=15).grid(row=9, column=1, sticky="w", pady=6)
        ttk.Label(advanced_frame, text="Floor for adaptive thresholds in calm markets (0.01-0.05 range)", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=10, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(advanced_frame, text="Max threshold multiplier:").grid(row=11, column=0, sticky="w", padx=(0, 10), pady=6)
        weight_threshold_max_var = tk.StringVar(value=str(cfg.get("weight_threshold_max", 0.2)))
        ttk.Entry(advanced_frame, textvariable=weight_threshold_max_var, width=15).grid(row=11, column=1, sticky="w", pady=6)
        ttk.Label(advanced_frame, text="Ceiling for adaptive thresholds in volatile markets (0.15-0.3 range)", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=12, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(advanced_frame, text="Volatility smoothing:").grid(row=13, column=0, sticky="w", padx=(0, 10), pady=6)
        volatility_ewma_var = tk.StringVar(value=str(cfg.get("volatility_ewma_decay", 0.9)))
        ttk.Entry(advanced_frame, textvariable=volatility_ewma_var, width=15).grid(row=13, column=1, sticky="w", pady=6)
        ttk.Label(advanced_frame, text="EWMA decay for volatility tracking (0.8-0.95 range, higher = smoother)", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=14, column=0, columnspan=2, sticky="w", pady=(0, 15))

        # Temporal Decay
        ttk.Label(advanced_frame, text="Temporal Decay", font=("TkDefaultFont", 9, "bold")).grid(row=15, column=0, columnspan=2, sticky="w", pady=(0, 8))
        
        ttk.Label(advanced_frame, text="Weight decay rate:").grid(row=16, column=0, sticky="w", padx=(0, 10), pady=6)
        weight_decay_rate_var = tk.StringVar(value=str(cfg.get("weight_decay_rate", 0.001)))
        ttk.Entry(advanced_frame, textvariable=weight_decay_rate_var, width=15).grid(row=16, column=1, sticky="w", pady=6)
        ttk.Label(advanced_frame, text="Rate weights decay toward baseline per validation (0.0001-0.01 range)", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=17, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(advanced_frame, text="Weight decay target:").grid(row=18, column=0, sticky="w", padx=(0, 10), pady=6)
        weight_decay_target_var = tk.StringVar(value=str(cfg.get("weight_decay_target", 1.0)))
        ttk.Entry(advanced_frame, textvariable=weight_decay_target_var, width=15).grid(row=18, column=1, sticky="w", pady=6)
        ttk.Label(advanced_frame, text="Target weight for decay (usually 1.0 = neutral baseline)", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=19, column=0, columnspan=2, sticky="w", pady=(0, 15))

        # Pruning sigma level (moved from Pattern Quality)
        ttk.Label(advanced_frame, text="Pruning sigma level:").grid(row=20, column=0, sticky="w", padx=(0, 10), pady=6)
        pruning_sigma_var = tk.StringVar(value=str(cfg.get("pruning_sigma_level", 2.0)))
        ttk.Entry(advanced_frame, textvariable=pruning_sigma_var, width=15).grid(row=20, column=1, sticky="w", pady=6)

        ttk.Label(
            advanced_frame,
            text="Patterns with weight < (mean - sigma × std_dev) are pruned. Higher = more aggressive pruning (1.5-3.0 range)",
            foreground=DARK_FG,
            font=("TkDefaultFont", 8)
        ).grid(row=21, column=0, columnspan=2, sticky="w", pady=(0, 15))

        # Age-Based Pruning
        ttk.Label(advanced_frame, text="Age-Based Pruning", font=("TkDefaultFont", 9, "bold")).grid(row=22, column=0, columnspan=2, sticky="w", pady=(0, 8))
        
        age_pruning_enabled_var = tk.BooleanVar(value=bool(cfg.get("age_pruning_enabled", True)))
        ttk.Checkbutton(advanced_frame, text="Enable age-based pruning", variable=age_pruning_enabled_var).grid(row=23, column=0, columnspan=2, sticky="w", pady=(0, 10))
        
        ttk.Label(advanced_frame, text="Age pruning percentile:").grid(row=24, column=0, sticky="w", padx=(0, 10), pady=6)
        age_pruning_percentile_var = tk.StringVar(value=str(cfg.get("age_pruning_percentile", 0.10)))
        ttk.Entry(advanced_frame, textvariable=age_pruning_percentile_var, width=15).grid(row=24, column=1, sticky="w", pady=6)
        ttk.Label(advanced_frame, text="Fraction of oldest patterns to consider for removal (0.05-0.20 range)", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=25, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(advanced_frame, text="Age pruning weight limit:").grid(row=26, column=0, sticky="w", padx=(0, 10), pady=6)
        age_pruning_weight_var = tk.StringVar(value=str(cfg.get("age_pruning_weight_limit", 1.0)))
        ttk.Entry(advanced_frame, textvariable=age_pruning_weight_var, width=15).grid(row=26, column=1, sticky="w", pady=6)
        ttk.Label(advanced_frame, text="Only prune old patterns below this weight (0.5-1.5 range)", foreground=DARK_FG, font=("TkDefaultFont", 8)).grid(row=27, column=0, columnspan=2, sticky="w", pady=(0, 0))

        # Timeframes Section with Checkboxes
        timeframes_frame = ttk.LabelFrame(frm, text=" Allowed Timeframes ", padding=15)
        timeframes_frame.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(0, 15)); r += 1
        timeframes_frame.columnconfigure(0, weight=1)

        # Check current configuration status
        current_timeframes = cfg.get("timeframes", [])
        has_all_required = set(current_timeframes) == set(REQUIRED_THINKER_TIMEFRAMES)
        
        if has_all_required:
            status_text = "✓ All 7 required timeframes configured"
            status_color = "green"
        else:
            status_text = f"⚠ Custom configuration: {', '.join(current_timeframes) if current_timeframes else 'none'}"
            status_color = "orange"
        
        # Status indicator
        status_label = ttk.Label(
            timeframes_frame,
            text=status_text,
            foreground=status_color,
            font=("TkDefaultFont", 9, "bold")
        )
        status_label.grid(row=0, column=0, sticky="w", pady=(0, 5))
        
        # Timeframes are locked to the 7 required by Thinker
        # Users can manually edit training_settings.json for debugging purposes
        ttk.Label(
            timeframes_frame,
            text=f"Required for Thinker: {', '.join(REQUIRED_THINKER_TIMEFRAMES)}\n\n"
                 "These are the 7 timeframes required by the Thinker for predictions and neural levels.\n"
                 "Advanced users can manually edit training_settings.json to change this for debugging.",
            wraplength=550,
            justify="left"
        ).grid(row=1, column=0, sticky="w", pady=5)

        # Delete training data button
        def delete_training_data():
            """Delete all training data files for all coins (fresh start)."""
            coins_list = ', '.join(self.coins) if self.coins else 'No coins configured'
            confirm = messagebox.askyesno(
                "Delete Training Data",
                f"This will delete ALL training data for all coins:\n\n{coins_list}\n\n"
                "This includes:\n"
                "• All pattern memories\n"
                "• All learned weights\n"
                "• Threshold convergence data\n"
                "• Training timestamps\n\n"
                "You will need to retrain from scratch.\n\n"
                "Are you sure you want to continue?",
                icon='warning'
            )
            
            if not confirm:
                return
            
            deleted_count = 0
            error_count = 0
            
            for coin in self.coins:
                folder = self.coin_folders.get(coin, "")
                if not folder or not os.path.isdir(folder):
                    continue
                
                # Training files to delete (per timeframe)
                for tf in REQUIRED_THINKER_TIMEFRAMES:
                    files_to_delete = [
                        f"memories_{tf}.dat",
                        f"memory_weights_{tf}.dat",
                        f"memory_weights_high_{tf}.dat",
                        f"memory_weights_low_{tf}.dat",
                        f"neural_perfect_threshold_{tf}.dat"
                    ]
                    
                    for filename in files_to_delete:
                        filepath = os.path.join(folder, filename)
                        try:
                            if os.path.exists(filepath):
                                os.remove(filepath)
                                deleted_count += 1
                        except Exception:
                            error_count += 1
                
                # Delete training status files
                status_files = [
                    "trainer_last_training_time.txt",
                    "trainer_status.json"
                ]
                
                for filename in status_files:
                    filepath = os.path.join(folder, filename)
                    try:
                        if os.path.exists(filepath):
                            os.remove(filepath)
                            deleted_count += 1
                    except Exception:
                        error_count += 1
            
            if error_count > 0:
                messagebox.showwarning(
                    "Partial Success",
                    f"Deleted {deleted_count} files.\n\n"
                    f"Failed to delete {error_count} files (may be in use or locked)."
                )
            else:
                messagebox.showinfo(
                    "Success",
                    f"Successfully deleted {deleted_count} training data files.\n\n"
                    "All coins will need to be retrained before trading."
                )
        
        delete_btn_frame = ttk.Frame(timeframes_frame)
        delete_btn_frame.grid(row=2, column=0, sticky="w", pady=(10, 0))
        
        ttk.Button(
            delete_btn_frame,
            text="Delete All Training Data",
            command=delete_training_data,
            style="Danger.TButton"  # Red/warning style if available
        ).pack(side="left", padx=(0, 5))
        
        ttk.Label(
            delete_btn_frame,
            text="(Clears all learned patterns for fresh retrain)",
            foreground="gray",
            font=("TkDefaultFont", 8)
        ).pack(side="left")

        # Buttons at bottom
        btns_frame = ttk.Frame(frm)
        btns_frame.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(10, 0)); r += 1

        def save():
            try:
                # Validate all numeric inputs
                staleness = int(staleness_var.get())
                if staleness < 1:
                    messagebox.showerror("Validation Error", "Staleness threshold must be at least 1 day")
                    return
                
                # PID Controller parameters
                pid_kp = float(pid_kp_var.get())
                pid_ki = float(pid_ki_var.get())
                pid_kd = float(pid_kd_var.get())
                pid_limit = float(pid_limit_var.get())
                
                if pid_kp <= 0 or pid_ki <= 0 or pid_kd <= 0:
                    messagebox.showerror("Validation Error", "PID gains (Kp, Ki, Kd) must be greater than 0")
                    return
                if pid_limit <= 0:
                    messagebox.showerror("Validation Error", "PID integral limit must be greater than 0")
                    return
                
                min_threshold = float(min_threshold_var.get())
                max_threshold = float(max_threshold_var.get())
                if min_threshold < 0 or min_threshold > 100:
                    messagebox.showerror("Validation Error", "Min threshold must be between 0 and 100")
                    return
                if max_threshold < 0 or max_threshold > 100:
                    messagebox.showerror("Validation Error", "Max threshold must be between 0 and 100")
                    return
                if min_threshold >= max_threshold:
                    messagebox.showerror("Validation Error", "Min threshold must be less than max threshold")
                    return
                
                pruning_sigma = float(pruning_sigma_var.get())
                if pruning_sigma < 0:
                    messagebox.showerror("Validation Error", "Pruning sigma level must be non-negative")
                    return
                
                pattern_size = int(pattern_size_var.get())
                if pattern_size < 2 or pattern_size > 10:
                    messagebox.showerror("Validation Error", "Pattern size must be between 2 and 10 candles")
                    return
                
                # Validate Phase 1-4 parameters
                weight_base_step = float(weight_base_step_var.get())
                weight_step_cap = float(weight_step_cap_var.get())
                weight_threshold_base = float(weight_threshold_base_var.get())
                weight_threshold_min = float(weight_threshold_min_var.get())
                weight_threshold_max = float(weight_threshold_max_var.get())
                volatility_ewma = float(volatility_ewma_var.get())
                weight_decay_rate = float(weight_decay_rate_var.get())
                weight_decay_target = float(weight_decay_target_var.get())
                age_pruning_percentile = float(age_pruning_percentile_var.get())
                age_pruning_weight = float(age_pruning_weight_var.get())
                
                # Validate ranges
                if weight_base_step < 0.05 or weight_base_step > 1.0:
                    messagebox.showerror("Validation Error", "Base weight step must be between 0.05 and 1.0")
                    return
                if weight_step_cap < 1.0 or weight_step_cap > 5.0:
                    messagebox.showerror("Validation Error", "Max step multiplier must be between 1.0 and 5.0")
                    return
                if weight_threshold_min >= weight_threshold_max:
                    messagebox.showerror("Validation Error", "Min threshold multiplier must be less than max")
                    return
                if volatility_ewma < 0.5 or volatility_ewma > 0.99:
                    messagebox.showerror("Validation Error", "Volatility smoothing must be between 0.5 and 0.99")
                    return
                if weight_decay_rate < 0.0 or weight_decay_rate > 0.1:
                    messagebox.showerror("Validation Error", "Weight decay rate must be between 0.0 and 0.1")
                    return
                if age_pruning_percentile < 0.01 or age_pruning_percentile > 0.5:
                    messagebox.showerror("Validation Error", "Age pruning percentile must be between 0.01 and 0.5")
                    return
                
                # Read existing config to preserve timeframes and check for pattern_size change
                existing_cfg = _safe_read_json(config_path) if os.path.isfile(config_path) else {}
                timeframes = existing_cfg.get("timeframes", REQUIRED_THINKER_TIMEFRAMES)
                old_pattern_size = existing_cfg.get("pattern_size", 4)
                
                # Warn if pattern_size changed (requires retraining)
                if pattern_size != old_pattern_size:
                    confirm = messagebox.askyesno(
                        "Pattern Size Changed",
                        f"Pattern size changed from {old_pattern_size} to {pattern_size} candles.\n\n"
                        "This requires retraining ALL coins from scratch (existing patterns incompatible).\n\n"
                        "Continue?",
                        icon='warning'
                    )
                    if not confirm:
                        return
                
                new_cfg = {
                    "staleness_days": staleness,
                    "auto_train_when_stale": bool(auto_train_var.get()),
                    "pruning_sigma_level": pruning_sigma,
                    "pid_kp": pid_kp,
                    "pid_ki": pid_ki,
                    "pid_kd": pid_kd,
                    "pid_integral_limit": pid_limit,
                    "min_threshold": min_threshold,
                    "max_threshold": max_threshold,
                    "pattern_size": pattern_size,
                    "timeframes": timeframes,
                    "weight_base_step": weight_base_step,
                    "weight_step_cap_multiplier": weight_step_cap,
                    "weight_threshold_base": weight_threshold_base,
                    "weight_threshold_min": weight_threshold_min,
                    "weight_threshold_max": weight_threshold_max,
                    "volatility_ewma_decay": volatility_ewma,
                    "weight_decay_rate": weight_decay_rate,
                    "weight_decay_target": weight_decay_target,
                    "age_pruning_enabled": bool(age_pruning_enabled_var.get()),
                    "age_pruning_percentile": age_pruning_percentile,
                    "age_pruning_weight_limit": age_pruning_weight
                }
                _safe_write_json(config_path, new_cfg)
                messagebox.showinfo(
                    "Saved",
                    "Training settings saved.\n\nStop and restart any active training for changes to take effect."
                )
                on_close()
            except ValueError as e:
                messagebox.showerror("Invalid Input", f"Please enter valid numbers for all fields.\n\n{e}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save settings:\n{e}")

        def reset_to_defaults():
            try:
                from copy import deepcopy
                defaults = deepcopy(DEFAULT_TRAINING_CONFIG)
                
                # Reset all fields to defaults
                staleness_var.set(str(defaults.get("staleness_days", 14)))
                auto_train_var.set(bool(defaults.get("auto_train_when_stale", False)))
                pid_kp_var.set(str(defaults.get("pid_kp", 0.5)))
                pid_ki_var.set(str(defaults.get("pid_ki", 0.01)))
                pid_kd_var.set(str(defaults.get("pid_kd", 0.02)))
                pid_limit_var.set(str(defaults.get("pid_integral_limit", 0.5)))
                min_threshold_var.set(str(defaults.get("min_threshold", 5.0)))
                max_threshold_var.set(str(defaults.get("max_threshold", 25.0)))
                pruning_sigma_var.set(str(defaults.get("pruning_sigma_level", 2.0)))
                pattern_size_var.set(str(defaults.get("pattern_size", 4)))
                weight_base_step_var.set(str(defaults.get("weight_base_step", 0.25)))
                weight_step_cap_var.set(str(defaults.get("weight_step_cap_multiplier", 2.0)))
                weight_threshold_base_var.set(str(defaults.get("weight_threshold_base", 0.1)))
                weight_threshold_min_var.set(str(defaults.get("weight_threshold_min", 0.03)))
                weight_threshold_max_var.set(str(defaults.get("weight_threshold_max", 0.2)))
                volatility_ewma_var.set(str(defaults.get("volatility_ewma_decay", 0.9)))
                weight_decay_rate_var.set(str(defaults.get("weight_decay_rate", 0.001)))
                weight_decay_target_var.set(str(defaults.get("weight_decay_target", 1.0)))
                age_pruning_enabled_var.set(bool(defaults.get("age_pruning_enabled", True)))
                age_pruning_percentile_var.set(str(defaults.get("age_pruning_percentile", 0.10)))
                age_pruning_weight_var.set(str(defaults.get("age_pruning_weight_limit", 1.0)))
                
                # Save to file (includes resetting timeframes to required 7)
                _safe_write_json(config_path, defaults)
                messagebox.showinfo(
                    "Reset",
                    "Training settings have been reset to defaults.\n\nStop and restart any active training for changes to take effect."
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to reset settings:\n{e}")

        ttk.Button(btns_frame, text="Apply", command=save).pack(side="left", padx=(0, 8))
        ttk.Button(btns_frame, text="Cancel", command=on_close).pack(side="left", padx=(0, 8))
        ttk.Button(btns_frame, text="Reset to Defaults", command=reset_to_defaults).pack(side="left")

    def _open_trading_config(self) -> None:
        """Open trading_settings.json in default text editor. Creates with defaults if missing."""
        config_path = os.path.join(self.project_dir, TRADING_SETTINGS_FILE)
        
        # Create file with defaults if it doesn't exist
        if not os.path.isfile(config_path):
            try:
                _safe_write_json(config_path, DEFAULT_TRADING_CONFIG)
            except Exception as e:
                messagebox.showerror(
                    "Error Creating Settings",
                    f"Could not create {TRADING_SETTINGS_FILE}:\n\n{e}"
                )
                return
        
        # Open in default text editor
        try:
            if sys.platform == "win32":
                os.startfile(config_path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", config_path])
            else:
                subprocess.Popen(["xdg-open", config_path])
        except Exception as e:
            messagebox.showerror(
                "Error Opening File",
                f"Could not open {TRADING_SETTINGS_FILE} in text editor:\n\n{e}\n\n"
                f"File location: {config_path}"
            )

    def _open_training_config(self) -> None:
        """Open training_settings.json in default text editor. Creates with defaults if missing."""
        config_path = os.path.join(self.project_dir, TRAINING_SETTINGS_FILE)
        
        # Create file with defaults if it doesn't exist
        if not os.path.isfile(config_path):
            try:
                _safe_write_json(config_path, DEFAULT_TRAINING_CONFIG)
            except Exception as e:
                messagebox.showerror(
                    "Error Creating Settings",
                    f"Could not create {TRAINING_SETTINGS_FILE}:\n\n{e}"
                )
                return
        
        # Open in default text editor
        try:
            if sys.platform == "win32":
                os.startfile(config_path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", config_path])
            else:
                subprocess.Popen(["xdg-open", config_path])
        except Exception as e:
            messagebox.showerror(
                "Error Opening File",
                f"Could not open {TRAINING_SETTINGS_FILE} in text editor:\n\n{e}\n\n"
                f"File location: {config_path}"
            )

    # Window close handler for cleanup on application exit
    def _on_close(self) -> None:
        # Don’t force kill; just stop if running (you can change this later)
        try:
            self.stop_all_scripts()
        except Exception:
            pass
        self.destroy()

if __name__ == "__main__":
    app = ApolloHub()
    app.mainloop()
