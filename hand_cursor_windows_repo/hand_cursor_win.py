#!/usr/bin/env python3
"""
Local Hand + Head/Mouth Controlled Cursor for Windows 10/11 (Stages A - F v2.0)
===============================================================================
Windows port of https://github.com/MasalMavzer10/Hand-Cursor-Mac

Controls (Decoupled v2.0 Architecture for Zero Gesture Confusion):
------------------------------------------------------------------
1. CURSOR MOVE : Move your hand inside the active box (uses a stabilized
                 70% Index MCP Knuckle + 30% Index Tip anchor, 1-Euro velocity
                 smoothing, and a 1.35 fine-aim power curve so your cursor
                 stays rock-solid over small Windows buttons).
2. LEFT CLICK  : Open your mouth briefly (tracked via MediaPipe FaceMesh inner
                 lip aspect ratio MAR >= 0.26 with hysteresis & cooldown — your
                 hand stays completely still while clicking!).
3. DRAG & DROP : Pinch thumb + index finger and hold (>180ms) to drag; release
                 to drop (100% dedicated to dragging so click and drag never
                 conflict).
4. SCROLL UP/DN: Tilt/move your head UP to scroll up, or DOWN to scroll down
                 (tracked via FaceMesh nose pitch relative to your neutral head
                 position, with a comfort deadzone).
5. PAUSE       : Press SPACE / P (or global Ctrl+Shift+P) to pause immediately.

Windows Native Highlights:
--------------------------
- Zero-dependency native Win32 pointer backend via `ctypes.windll.user32`
  (`SetCursorPos` + `mouse_event` with `MOUSEEVENTF_WHEEL`), plus automatic
  Per-Monitor DPI Awareness (`shcore.SetProcessDpiAwareness` / `SetProcessDPIAware`)
  so 125% / 150% / 200% Windows display scaling maps 1:1 to physical pixels.
- Fallback support for `pynput` if desired.
- Fast webcam startup on Windows using DirectShow (`cv2.CAP_DSHOW`) with
  automatic fallback to default Media Foundation (`cv2.CAP_ANY`).

Quick Start on Windows (PowerShell or Command Prompt):
------------------------------------------------------
1. Install dependencies:
   pip install --no-compile opencv-python mediapipe numpy pynput

2. Test safely in Preview-Only mode first (no system mouse movement):
   python hand_cursor_win.py --preview-only

3. Run with full Windows Desktop Control:
   python hand_cursor_win.py

4. Run headless automated self-test:
   python hand_cursor_win.py --self-test

Keyboard Shortcuts (inside preview window, or Ctrl+Shift+P globally):
---------------------------------------------------------------------
- SPACE or P : Pause / Resume tracking immediately
- N          : Re-center neutral head position for scrolling
- D          : Toggle live Windows Desktop Control vs. Preview-Only mode (3s countdown)
- C          : Start 4-point comfortable hand workspace Calibration wizard
- A          : Cycle cursor anchor point (Stabilized Blend -> Knuckle -> Index Tip)
- M          : Toggle horizontal camera mirroring
- H          : Cycle preferred handedness (Any -> Right -> Left)
- S          : Toggle live settings trackbars window
- R          : Reset settings and calibration to defaults
- Q or ESC   : Safely release any active drag and quit
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Windows Native Win32 API (ctypes) & Optional pynput Fallback
# ---------------------------------------------------------------------------
IS_WINDOWS = sys.platform.startswith("win")
WIN32_AVAILABLE = False
PYNPUT_AVAILABLE = False

# Win32 mouse_event constants
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000
WHEEL_DELTA = 120

_user32 = None
if IS_WINDOWS:
    try:
        # Enable Per-Monitor DPI awareness on Windows 8.1/10/11 so GetSystemMetrics
        # and SetCursorPos operate in true physical screen coordinates.
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        _user32 = ctypes.windll.user32
        WIN32_AVAILABLE = True
    except Exception:
        _user32 = None
        WIN32_AVAILABLE = False

try:
    from pynput import keyboard as pynput_keyboard  # type: ignore
    from pynput import mouse as pynput_mouse  # type: ignore

    PYNPUT_AVAILABLE = True
except ImportError:
    pynput_keyboard = None  # type: ignore
    pynput_mouse = None  # type: ignore

# ---------------------------------------------------------------------------
# MediaPipe Hands & FaceMesh Import
# ---------------------------------------------------------------------------
MP_AVAILABLE = False
try:
    import mediapipe as mp  # type: ignore

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        MP_AVAILABLE = True
except ImportError:
    mp = None  # type: ignore


CONFIG_PATH = Path.home() / ".hand_cursor_win_config.json"


# ===========================================================================
# Stage F: Configuration & Persistent Settings
# ===========================================================================
@dataclass
class CursorConfig:
    camera_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    mirror: bool = True
    handedness: str = "Any"  # "Any", "Right", "Left"

    # Cursor anchor mode: "blend" (70% knuckle + 30% tip), "knuckle" (lm 5), "tip" (lm 8)
    anchor_mode: str = "blend"

    # Stage B: Active hand region margins (normalized)
    roi_left: float = 0.18
    roi_right: float = 0.82
    roi_top: float = 0.18
    roi_bottom: float = 0.82

    # Stage B: Stabilized 1-Euro / Velocity-Adaptive Smoothing & Precision Curve
    base_smoothing: float = 0.16     # Heavy smoothing at low speed for rock-solid aim
    max_smoothing: float = 0.78      # Fast response at high hand speed
    cursor_speed: float = 1.15       # Overall range multiplier
    precision_exponent: float = 1.35 # > 1.0 dampens micro-tremors & accelerates larger moves
    jitter_deadzone_px: float = 2.5  # Ignore sub-pixel tremors

    # Mouth-Open Click Settings (MediaPipe FaceMesh MAR)
    mouth_click_enabled: bool = True
    mouth_enter_thresh: float = 0.26  # Lip opening / mouth width ratio to trigger click
    mouth_exit_thresh: float = 0.16   # Lip ratio below which mouth is considered closed
    mouth_debounce_s: float = 0.045   # Brief 45ms stability check before firing click
    click_cooldown_s: float = 0.32    # Cooldown between consecutive mouth clicks

    # Stage D: Thumb-Index Pinch & Hold (Dedicated 100% to Drag & Drop)
    pinch_enter_thresh: float = 0.26  # Palm-normalized thumb-index distance to start pinch
    pinch_exit_thresh: float = 0.38   # Distance ratio above which drag releases
    drag_hold_delay_s: float = 0.18   # Hold pinch briefly to engage drag
    lost_hand_drag_grace_s: float = 0.18

    # Stage E: Head Movement (Pitch Up/Down) Scroll Settings
    head_scroll_enabled: bool = True
    head_deadzone_norm: float = 0.035  # Comfort neutral zone around neutral head pitch
    head_scroll_speed: float = 32.0    # Scroll speed multiplier when head tilts up/down

    # Stage C & F: Safety & startup
    startup_countdown_s: float = 3.0
    min_detection_confidence: float = 0.65
    min_tracking_confidence: float = 0.65

    def save(self, path: Path = CONFIG_PATH) -> None:
        try:
            path.write_text(json.dumps(asdict(self), indent=2))
        except OSError as exc:
            print(f"[warn] Could not save config to {path}: {exc}")

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "CursorConfig":
        cfg = cls()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for key, val in data.items():
                    if hasattr(cfg, key):
                        setattr(cfg, key, val)
            except Exception as exc:
                print(f"[warn] Could not load config from {path}, using defaults: {exc}")
        return cfg


# ===========================================================================
# Stage C: Windows Pointer & Event Controller (Win32 ctypes + pynput fallback)
# ===========================================================================
class WinPointerController:
    """
    Wraps Windows native Win32 user32 API (preferred on Windows) or pynput
    (fallback) to move the system pointer, click, drag, and scroll with
    fail-safe state cleanup.
    """

    def __init__(self, preview_only: bool = False) -> None:
        self.preview_only = preview_only
        self.backend_name = "Preview-Only"
        self._mouse_ctrl = None
        self.is_dragging = False
        self.last_pos: Tuple[float, float] = (0.0, 0.0)
        self.screen_w, self.screen_h = self._detect_screen_size()

        if not preview_only:
            if WIN32_AVAILABLE and _user32 is not None:
                self.backend_name = "Windows Win32 API (user32)"
            elif PYNPUT_AVAILABLE:
                self.backend_name = "pynput MouseController"
                self._mouse_ctrl = pynput_mouse.Controller()
            else:
                self.backend_name = "Preview-Only (run on Windows or install pynput)"
                self.preview_only = True

    def _detect_screen_size(self) -> Tuple[int, int]:
        if WIN32_AVAILABLE and _user32 is not None:
            try:
                w = int(_user32.GetSystemMetrics(0))  # SM_CXSCREEN
                h = int(_user32.GetSystemMetrics(1))  # SM_CYSCREEN
                if w > 0 and h > 0:
                    return w, h
            except Exception:
                pass
        return 1920, 1080

    def move(self, x: float, y: float) -> None:
        x = float(max(0, min(self.screen_w - 1, x)))
        y = float(max(0, min(self.screen_h - 1, y)))
        self.last_pos = (x, y)

        if self.preview_only:
            return

        ix, iy = int(round(x)), int(round(y))
        if WIN32_AVAILABLE and _user32 is not None:
            _user32.SetCursorPos(ix, iy)
        elif self._mouse_ctrl is not None:
            self._mouse_ctrl.position = (ix, iy)

    def click(self, x: Optional[float] = None, y: Optional[float] = None) -> None:
        if x is None or y is None:
            x, y = self.last_pos
        x = float(max(0, min(self.screen_w - 1, x)))
        y = float(max(0, min(self.screen_h - 1, y)))
        self.last_pos = (x, y)

        if self.preview_only:
            return

        ix, iy = int(round(x)), int(round(y))
        if WIN32_AVAILABLE and _user32 is not None:
            _user32.SetCursorPos(ix, iy)
            _user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            _user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        elif self._mouse_ctrl is not None:
            self._mouse_ctrl.position = (ix, iy)
            self._mouse_ctrl.click(pynput_mouse.Button.left, 1)

    def start_drag(self, x: Optional[float] = None, y: Optional[float] = None) -> None:
        if self.is_dragging:
            return
        if x is None or y is None:
            x, y = self.last_pos
        self.is_dragging = True

        if self.preview_only:
            return

        ix, iy = int(round(x)), int(round(y))
        if WIN32_AVAILABLE and _user32 is not None:
            _user32.SetCursorPos(ix, iy)
            _user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        elif self._mouse_ctrl is not None:
            self._mouse_ctrl.press(pynput_mouse.Button.left)

    def end_drag(self, x: Optional[float] = None, y: Optional[float] = None) -> None:
        if not self.is_dragging:
            return
        if x is None or y is None:
            x, y = self.last_pos
        self.is_dragging = False

        if self.preview_only:
            return

        ix, iy = int(round(x)), int(round(y))
        if WIN32_AVAILABLE and _user32 is not None:
            _user32.SetCursorPos(ix, iy)
            _user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        elif self._mouse_ctrl is not None:
            self._mouse_ctrl.release(pynput_mouse.Button.left)

    def scroll(self, dy_pixels: float) -> None:
        if self.preview_only:
            return
        ticks = int(round(dy_pixels))
        if ticks == 0:
            return

        if WIN32_AVAILABLE and _user32 is not None:
            # Scale smooth pixel delta to Win32 wheel units (WHEEL_DELTA = 120 per notch)
            wheel_amount = int(round(dy_pixels * (WHEEL_DELTA / 12.0)))
            if wheel_amount != 0:
                _user32.mouse_event(
                    MOUSEEVENTF_WHEEL,
                    0,
                    0,
                    ctypes.c_ulong(wheel_amount & 0xFFFFFFFF).value,
                    0,
                )
        elif self._mouse_ctrl is not None:
            self._mouse_ctrl.scroll(0, int(round(ticks / 10.0)))

    def safe_release(self) -> None:
        """Guarantees any held mouse button is released on pause, hand loss, or exit."""
        if self.is_dragging:
            self.end_drag()


# ===========================================================================
# Observations & Gesture Processor (Hand + FaceMesh Head/Mouth)
# ===========================================================================
@dataclass
class HandObservation:
    """Normalized 21-point hand landmarks in (x, y, z) where x, y are in [0, 1]."""

    landmarks: List[Tuple[float, float, float]]
    handedness: str  # "Right" or "Left"
    score: float


@dataclass
class FaceObservation:
    """
    Extracted face metrics from MediaPipe FaceMesh:
    - nose_xy: normalized (x, y) of nose tip (landmark 1)
    - pitch_ratio: vertical head pitch metric (positive = head tilted UP, negative = DOWN)
    - mouth_mar: inner lip vertical opening / mouth width ratio (landmarks 13, 14 / 78, 308)
    - mouth_center_xy: normalized (x, y) of mouth center
    """

    nose_xy: Tuple[float, float]
    pitch_ratio: float
    mouth_mar: float
    mouth_center_xy: Tuple[float, float]

    @classmethod
    def from_facemesh_landmarks(
        cls, lm: List[Tuple[float, float, float]]
    ) -> "FaceObservation":
        # Nose tip: 1; Forehead: 10; Chin: 152; Left/Right eye outer: 33, 263
        nose_x, nose_y, _ = lm[1]
        forehead_y = lm[10][1]
        chin_y = lm[152][1]
        eye_mid_y = 0.5 * (lm[33][1] + lm[263][1])
        face_h = max(0.08, abs(chin_y - forehead_y))

        # When head tilts UP, nose moves closer to eye_mid_y (smaller nose_y in image coords)
        # Normal nose offset below eye midpoint is ~0.36 of face_h
        raw_nose_offset = (nose_y - eye_mid_y) / face_h
        pitch_ratio = 0.36 - raw_nose_offset  # Positive = UP, Negative = DOWN

        # Mouth Aspect Ratio (MAR): inner upper lip (13) to inner lower lip (14)
        # divided by mouth corner width (78 to 308)
        up_x, up_y, _ = lm[13]
        lo_x, lo_y, _ = lm[14]
        left_x, left_y, _ = lm[78]
        right_x, right_y, _ = lm[308]

        lip_open = math.hypot(lo_x - up_x, lo_y - up_y)
        mouth_w = max(0.03, math.hypot(right_x - left_x, right_y - left_y))
        mar = lip_open / mouth_w
        mouth_cx = 0.5 * (up_x + lo_x)
        mouth_cy = 0.5 * (up_y + lo_y)

        return cls(
            nose_xy=(nose_x, nose_y),
            pitch_ratio=pitch_ratio,
            mouth_mar=mar,
            mouth_center_xy=(mouth_cx, mouth_cy),
        )


@dataclass
class GestureOutput:
    mode: str  # "PAUSED", "COUNTDOWN", "NO HAND", "MOVE", "CLICK", "DRAG", "SCROLL", "CALIBRATING"
    screen_x: float
    screen_y: float
    norm_x: float
    norm_y: float
    pinch_ratio: float
    mouth_mar: float
    head_pitch_delta: float
    is_pinched: bool
    is_mouth_open: bool
    is_dragging: bool
    is_scrolling: bool
    scroll_delta: float
    clicked_this_frame: bool
    status_detail: str


class HandHeadGestureProcessor:
    """
    Combines:
    - Hand Tracking -> Stabilized Cursor Movement + Pinch-Hold Drag
    - Face Tracking -> Mouth-Open Click + Head-Pitch Up/Down Scroll
    """

    def __init__(self, config: CursorConfig, pointer: WinPointerController) -> None:
        self.cfg = config
        self.pointer = pointer

        self.paused: bool = False
        self.desktop_enabled: bool = not pointer.preview_only
        self.countdown_start: Optional[float] = (
            time.monotonic() if self.desktop_enabled else None
        )

        # Smoothed cursor coordinates
        self.smooth_x: float = pointer.screen_w / 2.0
        self.smooth_y: float = pointer.screen_h / 2.0
        self._has_smoothed_pos: bool = False

        # Stage D: Pinch & Hold Drag state (Dedicated to Drag)
        self.is_pinched: bool = False
        self.pinch_start_time: float = 0.0
        self.last_hand_seen_time: float = 0.0

        # Mouth-Open Click state
        self.mouth_open_state: bool = False
        self.mouth_open_start: Optional[float] = None
        self.mouth_click_fired_this_open: bool = False
        self.last_click_time: float = 0.0

        # Stage E: Head Pitch Scroll state
        self.neutral_head_pitch: Optional[float] = None
        self._pitch_calib_samples: List[float] = []

        # Stage F: 4-Corner Hand Calibration state
        self.calibrating: bool = False
        self.calib_step: int = 0
        self.calib_points: List[Tuple[float, float]] = []
        self.calib_labels = [
            "TOP-LEFT corner",
            "TOP-RIGHT corner",
            "BOTTOM-RIGHT corner",
            "BOTTOM-LEFT corner",
        ]

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        if self.paused:
            self.pointer.safe_release()
            self._reset_gesture_states()

    def recenter_head_pitch(self, current_pitch: Optional[float] = None) -> None:
        if current_pitch is not None:
            self.neutral_head_pitch = current_pitch
        else:
            self.neutral_head_pitch = None
            self._pitch_calib_samples = []

    def cycle_anchor_mode(self) -> str:
        order = ["blend", "knuckle", "tip"]
        idx = (
            (order.index(self.cfg.anchor_mode) + 1) % len(order)
            if self.cfg.anchor_mode in order
            else 0
        )
        self.cfg.anchor_mode = order[idx]
        self.cfg.save()
        return self.cfg.anchor_mode

    def toggle_desktop_control(self) -> None:
        self.desktop_enabled = not self.desktop_enabled
        self.pointer.preview_only = not self.desktop_enabled
        if self.desktop_enabled:
            self.countdown_start = time.monotonic()
        else:
            self.countdown_start = None
            self.pointer.safe_release()

    def start_calibration(self) -> None:
        self.pointer.safe_release()
        self._reset_gesture_states()
        self.calibrating = True
        self.calib_step = 0
        self.calib_points = []

    def record_calibration_point(self, raw_ix: float, raw_iy: float) -> Optional[str]:
        if not self.calibrating:
            return None
        self.calib_points.append((raw_ix, raw_iy))
        self.calib_step += 1
        if self.calib_step >= 4:
            xs = [p[0] for p in self.calib_points]
            ys = [p[1] for p in self.calib_points]
            self.cfg.roi_left = round(max(0.02, min(0.42, min(xs))), 3)
            self.cfg.roi_right = round(min(0.98, max(0.58, max(xs))), 3)
            self.cfg.roi_top = round(max(0.02, min(0.42, min(ys))), 3)
            self.cfg.roi_bottom = round(min(0.98, max(0.58, max(ys))), 3)
            self.cfg.save()
            self.calibrating = False
            return "Calibration saved!"
        return f"Captured point {self.calib_step}/4"

    def _reset_gesture_states(self) -> None:
        self.is_pinched = False
        self.mouth_open_state = False
        self.mouth_open_start = None
        self.mouth_click_fired_this_open = False
        self.pointer.safe_release()

    # -----------------------------------------------------------------------
    # Stage B: Stabilized Hand Anchor + Precision Acceleration Curve
    # -----------------------------------------------------------------------
    def get_hand_anchor(self, lm: List[Tuple[float, float, float]]) -> Tuple[float, float]:
        """
        Computes a rock-solid tracking point on the hand.
        - 'blend' (default): 70% Index MCP Knuckle (5) + 30% Index Tip (8).
          Prevents cursor jumping when fingers bend or pinch!
        - 'knuckle': 100% Index MCP Knuckle (5) for maximum stability.
        - 'tip': 100% Index Fingertip (8).
        """
        if self.cfg.anchor_mode == "knuckle":
            return lm[5][0], lm[5][1]
        if self.cfg.anchor_mode == "tip":
            return lm[8][0], lm[8][1]
        return 0.70 * lm[5][0] + 0.30 * lm[8][0], 0.70 * lm[5][1] + 0.30 * lm[8][1]

    def map_to_screen(self, raw_x: float, raw_y: float) -> Tuple[float, float, float, float]:
        w_roi = max(0.10, self.cfg.roi_right - self.cfg.roi_left)
        h_roi = max(0.10, self.cfg.roi_bottom - self.cfg.roi_top)

        nx = (raw_x - self.cfg.roi_left) / w_roi
        ny = (raw_y - self.cfg.roi_top) / h_roi

        # Centered coordinates in [-1, 1]
        cx = (nx - 0.5) * 2.0
        cy = (ny - 0.5) * 2.0

        # True precision curve: exponent > 1.0 makes small movements near center
        # gentler/steadier, while larger movements smoothly reach corners!
        exp = max(1.0, self.cfg.precision_exponent)
        cx_curved = math.copysign(abs(cx) ** exp, cx)
        cy_curved = math.copysign(abs(cy) ** exp, cy)

        nx_out = max(0.0, min(1.0, 0.5 + (cx_curved * self.cfg.cursor_speed) / 2.0))
        ny_out = max(0.0, min(1.0, 0.5 + (cy_curved * self.cfg.cursor_speed) / 2.0))

        target_x = nx_out * (self.pointer.screen_w - 1)
        target_y = ny_out * (self.pointer.screen_h - 1)

        if not self._has_smoothed_pos:
            self.smooth_x = target_x
            self.smooth_y = target_y
            self._has_smoothed_pos = True
        else:
            dx = target_x - self.smooth_x
            dy = target_y - self.smooth_y
            dist = math.hypot(dx, dy)

            if dist >= self.cfg.jitter_deadzone_px:
                # 1-Euro style velocity-adaptive alpha:
                # Slow movements -> base_smoothing (0.16) for rock-solid precision
                # Fast movements -> scales up to max_smoothing (0.78) for zero lag
                vel_norm = min(1.0, dist / 110.0)
                alpha = self.cfg.base_smoothing + vel_norm * (
                    self.cfg.max_smoothing - self.cfg.base_smoothing
                )
                self.smooth_x += dx * alpha
                self.smooth_y += dy * alpha

        return (
            self.smooth_x,
            self.smooth_y,
            self.smooth_x / max(1, self.pointer.screen_w - 1),
            self.smooth_y / max(1, self.pointer.screen_h - 1),
        )

    @staticmethod
    def compute_pinch_ratio(lm: List[Tuple[float, float, float]]) -> float:
        tx, ty, _ = lm[4]
        ix, iy, _ = lm[8]
        wx, wy, _ = lm[0]
        mx, my, _ = lm[9]
        pinch_dist = math.hypot(ix - tx, iy - ty)
        palm_len = max(0.04, math.hypot(mx - wx, my - wy))
        return pinch_dist / palm_len

    # -----------------------------------------------------------------------
    # Main Per-Frame Processing Step (Hand + Face)
    # -----------------------------------------------------------------------
    def process(
        self,
        hand: Optional[HandObservation],
        face: Optional[FaceObservation] = None,
        now: Optional[float] = None,
    ) -> GestureOutput:
        if now is None:
            now = time.monotonic()

        # 1. Check Pause
        if self.paused:
            return GestureOutput(
                mode="PAUSED",
                screen_x=self.smooth_x,
                screen_y=self.smooth_y,
                norm_x=self.smooth_x / max(1, self.pointer.screen_w),
                norm_y=self.smooth_y / max(1, self.pointer.screen_h),
                pinch_ratio=1.0,
                mouth_mar=0.0,
                head_pitch_delta=0.0,
                is_pinched=False,
                is_mouth_open=False,
                is_dragging=False,
                is_scrolling=False,
                scroll_delta=0.0,
                clicked_this_frame=False,
                status_detail="Tracking paused — safe to use physical mouse & keyboard",
            )

        # 2. Check Startup Countdown
        in_countdown = False
        countdown_rem = 0.0
        if self.countdown_start is not None:
            elapsed = now - self.countdown_start
            if elapsed < self.cfg.startup_countdown_s:
                in_countdown = True
                countdown_rem = self.cfg.startup_countdown_s - elapsed
            else:
                self.countdown_start = None

        # 3. Evaluate Face: Head-Pitch Scroll & Mouth-Open Click
        mouth_mar = 0.0
        head_pitch_delta = 0.0
        scroll_delta = 0.0
        is_scrolling = False
        clicked_this_frame = False

        if face is not None:
            mouth_mar = face.mouth_mar

            # Calibrate neutral head pitch automatically over initial frames
            if self.neutral_head_pitch is None:
                self._pitch_calib_samples.append(face.pitch_ratio)
                if len(self._pitch_calib_samples) >= 10:
                    self.neutral_head_pitch = sum(self._pitch_calib_samples) / len(
                        self._pitch_calib_samples
                    )
                else:
                    self.neutral_head_pitch = face.pitch_ratio

            head_pitch_delta = face.pitch_ratio - self.neutral_head_pitch

            if not in_countdown and not self.calibrating:
                # Mouth-Open Click with hysteresis & debounce
                if self.cfg.mouth_click_enabled and not self.pointer.is_dragging:
                    if not self.mouth_open_state:
                        if mouth_mar >= self.cfg.mouth_enter_thresh:
                            if self.mouth_open_start is None:
                                self.mouth_open_start = now
                            elif (now - self.mouth_open_start) >= self.cfg.mouth_debounce_s:
                                self.mouth_open_state = True
                                if (now - self.last_click_time) >= self.cfg.click_cooldown_s:
                                    self.last_click_time = now
                                    self.mouth_click_fired_this_open = True
                                    clicked_this_frame = True
                                    self.pointer.click(self.smooth_x, self.smooth_y)
                        else:
                            self.mouth_open_start = None
                    else:
                        if mouth_mar <= self.cfg.mouth_exit_thresh:
                            self.mouth_open_state = False
                            self.mouth_open_start = None
                            self.mouth_click_fired_this_open = False

                # Head Pitch Up/Down Scroll (when not dragging)
                if self.cfg.head_scroll_enabled and not self.pointer.is_dragging:
                    if abs(head_pitch_delta) > self.cfg.head_deadzone_norm:
                        is_scrolling = True
                        eff = math.copysign(
                            abs(head_pitch_delta) - self.cfg.head_deadzone_norm,
                            head_pitch_delta,
                        )
                        scroll_delta = eff * self.cfg.head_scroll_speed * 10.0
                        self.pointer.scroll(scroll_delta)

        # 4. Evaluate Hand: Cursor Movement & Pinch-Hold Drag
        if hand is None or (
            self.cfg.handedness != "Any"
            and hand.handedness.lower() != self.cfg.handedness.lower()
        ):
            if (
                self.pointer.is_dragging
                and (now - self.last_hand_seen_time) <= self.cfg.lost_hand_drag_grace_s
            ):
                return GestureOutput(
                    mode="DRAG",
                    screen_x=self.smooth_x,
                    screen_y=self.smooth_y,
                    norm_x=self.smooth_x / max(1, self.pointer.screen_w),
                    norm_y=self.smooth_y / max(1, self.pointer.screen_h),
                    pinch_ratio=self.cfg.pinch_enter_thresh,
                    mouth_mar=mouth_mar,
                    head_pitch_delta=head_pitch_delta,
                    is_pinched=True,
                    is_mouth_open=self.mouth_open_state,
                    is_dragging=True,
                    is_scrolling=False,
                    scroll_delta=0.0,
                    clicked_this_frame=False,
                    status_detail="Holding drag across brief hand tracking gap...",
                )

            self.is_pinched = False
            self.pointer.safe_release()

            # Even if hand is down, head scroll or mouth click at current cursor position works!
            if clicked_this_frame:
                mode_no_hand = "CLICK"
                detail_no_hand = "Mouth Click triggered (hand resting)"
            elif is_scrolling:
                direction = "UP" if scroll_delta > 0 else "DOWN"
                mode_no_hand = "SCROLL"
                detail_no_hand = f"Head Scroll {direction} ({scroll_delta:+.1f})"
            else:
                mode_no_hand = "NO HAND"
                detail_no_hand = "Hand not detected — pointer movement stopped"

            return GestureOutput(
                mode=mode_no_hand,
                screen_x=self.smooth_x,
                screen_y=self.smooth_y,
                norm_x=self.smooth_x / max(1, self.pointer.screen_w),
                norm_y=self.smooth_y / max(1, self.pointer.screen_h),
                pinch_ratio=1.0,
                mouth_mar=mouth_mar,
                head_pitch_delta=head_pitch_delta,
                is_pinched=False,
                is_mouth_open=self.mouth_open_state,
                is_dragging=False,
                is_scrolling=is_scrolling,
                scroll_delta=scroll_delta,
                clicked_this_frame=clicked_this_frame,
                status_detail=detail_no_hand,
            )

        self.last_hand_seen_time = now
        lm = hand.landmarks
        ax, ay = self.get_hand_anchor(lm)
        pinch_ratio = self.compute_pinch_ratio(lm)

        # 5. Calibration Wizard
        if self.calibrating:
            sx, sy, nx, ny = self.map_to_screen(ax, ay)
            was_pinched = self.is_pinched
            if not self.is_pinched and (
                pinch_ratio < self.cfg.pinch_enter_thresh or clicked_this_frame
            ):
                self.is_pinched = True
            elif self.is_pinched and pinch_ratio > self.cfg.pinch_exit_thresh:
                self.is_pinched = False
                if was_pinched:
                    self.record_calibration_point(ax, ay)

            step_idx = min(self.calib_step, 3)
            return GestureOutput(
                mode="CALIBRATING",
                screen_x=sx,
                screen_y=sy,
                norm_x=nx,
                norm_y=ny,
                pinch_ratio=pinch_ratio,
                mouth_mar=mouth_mar,
                head_pitch_delta=head_pitch_delta,
                is_pinched=self.is_pinched,
                is_mouth_open=self.mouth_open_state,
                is_dragging=False,
                is_scrolling=False,
                scroll_delta=0.0,
                clicked_this_frame=False,
                status_detail=f"Point at {self.calib_labels[step_idx]} & Pinch/Mouth-Click ({self.calib_step + 1}/4)",
            )

        sx, sy, nx, ny = self.map_to_screen(ax, ay)

        if in_countdown:
            return GestureOutput(
                mode="COUNTDOWN",
                screen_x=sx,
                screen_y=sy,
                norm_x=nx,
                norm_y=ny,
                pinch_ratio=pinch_ratio,
                mouth_mar=mouth_mar,
                head_pitch_delta=head_pitch_delta,
                is_pinched=False,
                is_mouth_open=False,
                is_dragging=False,
                is_scrolling=False,
                scroll_delta=0.0,
                clicked_this_frame=False,
                status_detail=f"Desktop control begins in {countdown_rem:.1f}s...",
            )

        # 6. Dedicated Pinch & Hold Drag (never triggers accidental click)
        if not self.is_pinched:
            if pinch_ratio < self.cfg.pinch_enter_thresh:
                self.is_pinched = True
                self.pinch_start_time = now
        else:
            if pinch_ratio > self.cfg.pinch_exit_thresh:
                self.is_pinched = False
                if self.pointer.is_dragging:
                    self.pointer.end_drag(sx, sy)
                    return GestureOutput(
                        mode="MOVE",
                        screen_x=sx,
                        screen_y=sy,
                        norm_x=nx,
                        norm_y=ny,
                        pinch_ratio=pinch_ratio,
                        mouth_mar=mouth_mar,
                        head_pitch_delta=head_pitch_delta,
                        is_pinched=False,
                        is_mouth_open=self.mouth_open_state,
                        is_dragging=False,
                        is_scrolling=is_scrolling,
                        scroll_delta=scroll_delta,
                        clicked_this_frame=False,
                        status_detail="Drag released (Drop)",
                    )
            else:
                if (
                    not self.pointer.is_dragging
                    and (now - self.pinch_start_time) >= self.cfg.drag_hold_delay_s
                ):
                    self.pointer.start_drag(sx, sy)

        self.pointer.move(sx, sy)

        if self.pointer.is_dragging:
            return GestureOutput(
                mode="DRAG",
                screen_x=sx,
                screen_y=sy,
                norm_x=nx,
                norm_y=ny,
                pinch_ratio=pinch_ratio,
                mouth_mar=mouth_mar,
                head_pitch_delta=head_pitch_delta,
                is_pinched=True,
                is_mouth_open=self.mouth_open_state,
                is_dragging=True,
                is_scrolling=False,
                scroll_delta=0.0,
                clicked_this_frame=False,
                status_detail="Dragging with Pinch — release pinch to drop",
            )

        if clicked_this_frame or self.mouth_open_state:
            return GestureOutput(
                mode="CLICK",
                screen_x=sx,
                screen_y=sy,
                norm_x=nx,
                norm_y=ny,
                pinch_ratio=pinch_ratio,
                mouth_mar=mouth_mar,
                head_pitch_delta=head_pitch_delta,
                is_pinched=self.is_pinched,
                is_mouth_open=True,
                is_dragging=False,
                is_scrolling=is_scrolling,
                scroll_delta=scroll_delta,
                clicked_this_frame=clicked_this_frame,
                status_detail="Mouth-Open Click triggered!",
            )

        if is_scrolling:
            direction = "UP" if scroll_delta > 0 else "DOWN"
            return GestureOutput(
                mode="SCROLL",
                screen_x=sx,
                screen_y=sy,
                norm_x=nx,
                norm_y=ny,
                pinch_ratio=pinch_ratio,
                mouth_mar=mouth_mar,
                head_pitch_delta=head_pitch_delta,
                is_pinched=self.is_pinched,
                is_mouth_open=False,
                is_dragging=False,
                is_scrolling=True,
                scroll_delta=scroll_delta,
                clicked_this_frame=False,
                status_detail=f"Head Tilt Scroll {direction} (speed: {scroll_delta:+.1f})",
            )

        detail = (
            "Holding pinch — keep holding to start drag"
            if self.is_pinched
            else f"Moving cursor ({self.cfg.anchor_mode} anchor) | Open mouth to Click | Tilt head to Scroll"
        )
        return GestureOutput(
            mode="MOVE",
            screen_x=sx,
            screen_y=sy,
            norm_x=nx,
            norm_y=ny,
            pinch_ratio=pinch_ratio,
            mouth_mar=mouth_mar,
            head_pitch_delta=head_pitch_delta,
            is_pinched=self.is_pinched,
            is_mouth_open=False,
            is_dragging=False,
            is_scrolling=False,
            scroll_delta=0.0,
            clicked_this_frame=False,
            status_detail=detail,
        )


# ===========================================================================
# OpenCV Camera HUD, Landmarks & On-Screen Controls
# ===========================================================================
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

MODE_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "MOVE": (230, 200, 40),       # Cyan-Teal
    "CLICK": (80, 230, 100),      # Bright Green (Mouth Click)
    "DRAG": (50, 150, 255),       # Vibrant Orange (Pinch Drag)
    "SCROLL": (250, 130, 180),    # Purple-Pink (Head Scroll)
    "PAUSED": (90, 90, 235),      # Red
    "NO HAND": (110, 110, 140),   # Muted Gray-Red
    "COUNTDOWN": (60, 210, 255),  # Amber
    "CALIBRATING": (255, 210, 60),
}


class HandCursorHUD:
    """Renders camera preview, hand & face overlays, virtual cursor, and buttons."""

    def __init__(self, processor: HandHeadGestureProcessor) -> None:
        self.proc = processor
        self.buttons: List[Tuple[str, Tuple[int, int, int, int]]] = []
        self.settings_open: bool = False
        self.settings_win: str = "Hand + Head Cursor Settings (Windows)"
        self.last_face: Optional[FaceObservation] = None

    def toggle_settings_window(self) -> None:
        cfg = self.proc.cfg
        if not self.settings_open:
            cv2.namedWindow(self.settings_win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.settings_win, 440, 340)
            cv2.createTrackbar(
                "Smooth %",
                self.settings_win,
                int(cfg.base_smoothing * 100),
                70,
                lambda v: setattr(cfg, "base_smoothing", max(0.05, v / 100.0)),
            )
            cv2.createTrackbar(
                "Speed %",
                self.settings_win,
                int(cfg.cursor_speed * 100),
                220,
                lambda v: setattr(cfg, "cursor_speed", max(0.60, v / 100.0)),
            )
            cv2.createTrackbar(
                "Mouth Clk%",
                self.settings_win,
                int(cfg.mouth_enter_thresh * 100),
                55,
                lambda v: setattr(cfg, "mouth_enter_thresh", max(0.15, v / 100.0)),
            )
            cv2.createTrackbar(
                "Pinch Drag%",
                self.settings_win,
                int(cfg.pinch_enter_thresh * 100),
                48,
                lambda v: setattr(cfg, "pinch_enter_thresh", max(0.14, v / 100.0)),
            )
            cv2.createTrackbar(
                "Head Scrl",
                self.settings_win,
                int(cfg.head_scroll_speed),
                80,
                lambda v: setattr(cfg, "head_scroll_speed", float(max(5, v))),
            )
            self.settings_open = True
        else:
            try:
                cv2.destroyWindow(self.settings_win)
            except Exception:
                pass
            cfg.save()
            self.settings_open = False

    def handle_mouse_event(
        self, event: int, x: int, y: int, flags: int, param: object
    ) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for name, (bx1, by1, bx2, by2) in self.buttons:
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                if name == "pause":
                    self.proc.toggle_pause()
                elif name == "desktop":
                    self.proc.toggle_desktop_control()
                elif name == "recenter_head":
                    pitch = self.last_face.pitch_ratio if self.last_face else None
                    self.proc.recenter_head_pitch(pitch)
                elif name == "anchor":
                    self.proc.cycle_anchor_mode()
                elif name == "calibrate":
                    if self.proc.calibrating:
                        self.proc.calibrating = False
                    else:
                        self.proc.start_calibration()
                elif name == "settings":
                    self.toggle_settings_window()

    def draw(
        self,
        frame: np.ndarray,
        hand: Optional[HandObservation],
        face: Optional[FaceObservation],
        out: GestureOutput,
        fps: float,
    ) -> np.ndarray:
        self.last_face = face
        h, w = frame.shape[:2]
        cfg = self.proc.cfg
        color = MODE_COLORS_BGR.get(out.mode, (200, 200, 200))

        # 1. Draw Active Hand Tracking Rectangle
        rx1, ry1 = int(cfg.roi_left * w), int(cfg.roi_top * h)
        rx2, ry2 = int(cfg.roi_right * w), int(cfg.roi_bottom * h)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (170, 175, 185), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Hand Tracking Zone (Anchor: {cfg.anchor_mode})",
            (rx1 + 8, max(72, ry1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (210, 215, 225),
            1,
            cv2.LINE_AA,
        )

        # 2. Draw Face Overlay: Nose Pitch Scroll Band & Mouth Click Indicator
        if face is not None:
            nx_px = int(face.nose_xy[0] * w)
            ny_px = int(face.nose_xy[1] * h)
            dz_px = int(cfg.head_deadzone_norm * h * 0.45)
            neutral_y = int(ny_px + out.head_pitch_delta * h * 0.45)

            # Neutral head pitch band
            cv2.rectangle(
                frame,
                (nx_px - 45, neutral_y - dz_px),
                (nx_px + 45, neutral_y + dz_px),
                (230, 130, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.circle(
                frame,
                (nx_px, ny_px),
                7,
                (250, 140, 250) if out.is_scrolling else (230, 230, 240),
                -1,
                cv2.LINE_AA,
            )

            # Mouth Click circle
            mx_px = int(face.mouth_center_xy[0] * w)
            my_px = int(face.mouth_center_xy[1] * h)
            mouth_col = (80, 240, 120) if out.is_mouth_open else (160, 210, 160)
            cv2.circle(
                frame,
                (mx_px, my_px),
                14 if out.is_mouth_open else 9,
                mouth_col,
                2,
                cv2.LINE_AA,
            )

        # 3. Draw Hand Skeleton & Stabilized Anchor Point
        if hand is not None:
            pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in hand.landmarks]
            for i, j in HAND_CONNECTIONS:
                cv2.line(frame, pts[i], pts[j], (180, 220, 180), 2, cv2.LINE_AA)
            for idx, pt in enumerate(pts):
                radius = 5 if idx in (4, 5, 8) else 3
                cv2.circle(frame, pt, radius, (255, 255, 255), -1, cv2.LINE_AA)

            ax, ay = self.proc.get_hand_anchor(hand.landmarks)
            apx, apy = int(ax * w), int(ay * h)
            cv2.circle(frame, (apx, apy), 12, color, 2, cv2.LINE_AA)
            cv2.circle(frame, (apx, apy), 4, color, -1, cv2.LINE_AA)

            if out.is_pinched or out.is_dragging:
                cv2.line(frame, pts[4], pts[8], (50, 150, 255), 3, cv2.LINE_AA)

        # 4. Draw Virtual Cursor Preview Crosshair
        pv_x = int(out.norm_x * (w - 1))
        pv_y = int(out.norm_y * (h - 1))
        cv2.circle(frame, (pv_x, pv_y), 14, color, 2, cv2.LINE_AA)
        cv2.line(frame, (pv_x - 20, pv_y), (pv_x + 20, pv_y), color, 1, cv2.LINE_AA)
        cv2.line(frame, (pv_x, pv_y - 20), (pv_x, pv_y + 20), color, 1, cv2.LINE_AA)

        # 5. Top Status Banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 58), (22, 24, 30), -1)
        cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)

        cv2.rectangle(frame, (14, 10), (165, 48), color, -1)
        cv2.putText(
            frame,
            out.mode,
            (26, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (18, 18, 22),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            out.status_detail,
            (180, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (240, 240, 245),
            1,
            cv2.LINE_AA,
        )
        sub_info = (
            f"Mouth MAR: {out.mouth_mar:.2f} (Click>{cfg.mouth_enter_thresh:.2f}) | "
            f"Head Pitch: {out.head_pitch_delta:+.3f} | "
            f"Pinch Drag: {out.pinch_ratio:.2f} (<{cfg.pinch_enter_thresh:.2f}) | "
            f"{fps:.0f} FPS | {self.proc.pointer.backend_name}"
        )
        cv2.putText(
            frame,
            sub_info,
            (180, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (175, 185, 195),
            1,
            cv2.LINE_AA,
        )

        # 6. Bottom Interactive Control Bar
        self.buttons = []
        bar_y1 = h - 46
        cv2.rectangle(frame, (0, bar_y1), (w, h), (22, 24, 30), -1)

        btn_specs = [
            ("pause", "RESUME (Space)" if self.proc.paused else "PAUSE (Space)"),
            ("desktop", f"Desktop: {'ON' if self.proc.desktop_enabled else 'PREVIEW'} (D)"),
            ("recenter_head", "Center Head (N)"),
            ("anchor", f"Anchor: {cfg.anchor_mode} (A)"),
            ("calibrate", "Calibrate Hand (C)"),
            ("settings", "Sliders (S)"),
        ]
        bx = 12
        for key, label in btn_specs:
            bw = max(110, len(label) * 8 + 16)
            by1, by2 = bar_y1 + 8, h - 8
            bx2 = bx + bw
            cv2.rectangle(frame, (bx, by1), (bx2, by2), (55, 62, 78), -1)
            cv2.rectangle(frame, (bx, by1), (bx2, by2), (110, 120, 145), 1)
            cv2.putText(
                frame,
                label,
                (bx + 8, by1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.41,
                (235, 240, 245),
                1,
                cv2.LINE_AA,
            )
            self.buttons.append((key, (bx, by1, bx2, by2)))
            bx = bx2 + 8

        return frame


# ===========================================================================
# Self-Test Suite (Verifies Hand Move, Mouth Click, Pinch Drag & Head Scroll)
# ===========================================================================
def run_self_test() -> int:
    print("Running Upgraded Windows Hand + Head/Mouth Cursor Self-Test...")
    cfg = CursorConfig()
    pointer = WinPointerController(preview_only=True)
    proc = HandHeadGestureProcessor(cfg, pointer)
    proc.countdown_start = None

    def make_hand(
        ix: float = 0.5, iy: float = 0.5, pinch_dist: float = 0.12
    ) -> HandObservation:
        lm = [(0.5, 0.7, 0.0) for _ in range(21)]
        lm[0] = (0.5, 0.85, 0.0)
        lm[9] = (0.5, 0.65, 0.0)  # palm_len = 0.20
        lm[5] = (ix, iy, 0.0)     # Index MCP knuckle
        lm[8] = (ix, iy, 0.0)     # Index tip
        lm[4] = (ix + pinch_dist, iy, 0.0)
        return HandObservation(landmarks=lm, handedness="Right", score=0.98)

    def make_face(pitch: float = 0.0, mar: float = 0.08) -> FaceObservation:
        return FaceObservation(
            nose_xy=(0.5, 0.5 - pitch),
            pitch_ratio=pitch,
            mouth_mar=mar,
            mouth_center_xy=(0.5, 0.62),
        )

    t = 100.0
    # 1. Missing hand safety
    out = proc.process(None, make_face(0.0, 0.05), now=t)
    assert out.mode == "NO HAND"
    print("  [PASS] Stage A: Missing hand stops pointer movement safely")

    # 2. Stabilized Cursor Move to all corners
    for _ in range(20):
        t += 0.02
        out_br = proc.process(
            make_hand(ix=cfg.roi_right, iy=cfg.roi_bottom), make_face(0.0, 0.05), now=t
        )
    assert out_br.mode == "MOVE" and out_br.screen_x > pointer.screen_w * 0.85
    print("  [PASS] Stage B & C: Stabilized anchor maps smoothly to Windows screen corners")

    # 3. Mouth-Open Click (Decoupled from Pinch!)
    t += 0.5
    proc.process(make_hand(), make_face(pitch=0.0, mar=0.34), now=t)
    t += cfg.mouth_debounce_s + 0.02
    out_mclick = proc.process(make_hand(), make_face(pitch=0.0, mar=0.34), now=t)
    assert out_mclick.clicked_this_frame and out_mclick.mode == "CLICK"
    # Close mouth
    t += 0.1
    proc.process(make_hand(), make_face(pitch=0.0, mar=0.08), now=t)
    print("  [PASS] Mouth Click: Opening mouth triggers crisp single click without hand movement")

    # 4. Pinch & Hold Dedicated to Drag (Never triggers stray click on quick pinch)
    t += 0.5
    proc.process(make_hand(ix=0.45, iy=0.45, pinch_dist=0.02), make_face(), now=t)
    t += cfg.drag_hold_delay_s + 0.05
    out_drag = proc.process(make_hand(ix=0.55, iy=0.55, pinch_dist=0.02), make_face(), now=t)
    assert out_drag.mode == "DRAG" and out_drag.is_dragging
    t += 0.1
    out_drop = proc.process(make_hand(ix=0.55, iy=0.55, pinch_dist=0.12), make_face(), now=t)
    assert not out_drop.is_dragging and out_drop.mode == "MOVE"
    print("  [PASS] Pinch Drag: Pinch & hold drags and drops cleanly without click confusion")

    # 5. Head Movement (Pitch Up / Down) Scrolling
    t += 0.2
    out_scroll_up = proc.process(make_hand(), make_face(pitch=0.09, mar=0.08), now=t)
    assert out_scroll_up.mode == "SCROLL" and out_scroll_up.scroll_delta > 0
    out_scroll_dn = proc.process(make_hand(), make_face(pitch=-0.09, mar=0.08), now=t + 0.05)
    assert out_scroll_dn.mode == "SCROLL" and out_scroll_dn.scroll_delta < 0
    print("  [PASS] Head Scroll: Tilting head UP scrolls up, tilting head DOWN scrolls down")

    print("All upgraded Windows Hand + Head/Mouth self-tests passed!")
    return 0


def _open_windows_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    """
    Opens webcam using DirectShow (cv2.CAP_DSHOW) first on Windows to avoid
    slow MSMF initialization, falling back to default VideoCapture if needed.
    """
    if IS_WINDOWS and hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return cap
        cap.release()

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


# ===========================================================================
# Main Application Loop
# ===========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local Hand + Head/Mouth Cursor for Windows 10/11"
    )
    parser.add_argument("--camera", type=int, default=None, help="Camera index (default: 0)")
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Start in safe Preview-Only mode without moving the system cursor",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run automated headless verification and exit",
    )
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    if not MP_AVAILABLE:
        print(
            "[error] MediaPipe is not installed in this Python environment.\n"
            "Install required packages on Windows with:\n"
            "  pip install --no-compile opencv-python mediapipe numpy pynput"
        )
        return 1

    cfg = CursorConfig.load()
    if args.camera is not None:
        cfg.camera_index = args.camera

    pointer = WinPointerController(preview_only=args.preview_only)
    processor = HandHeadGestureProcessor(cfg, pointer)
    hud = HandCursorHUD(processor)

    atexit.register(pointer.safe_release)
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    global_listener = None
    if PYNPUT_AVAILABLE and pynput_keyboard is not None:
        try:
            global_listener = pynput_keyboard.GlobalHotKeys(
                {"<ctrl>+<shift>+p": processor.toggle_pause}
            )
            global_listener.start()
        except Exception:
            global_listener = None

    cap = _open_windows_camera(cfg.camera_index, cfg.frame_width, cfg.frame_height)

    if not cap.isOpened():
        print(
            f"[error] Could not open camera index {cfg.camera_index}. "
            "Check Windows Settings -> Privacy & security -> Camera and ensure "
            "'Let desktop apps access your camera' is enabled."
        )
        return 1

    win_name = "Local Hand + Head/Mouth Cursor (Windows)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, hud.handle_mouse_event)

    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    prev_time = time.monotonic()
    fps = 30.0

    print(f"Started {win_name} | Backend: {pointer.backend_name}")
    print(
        "Controls: [SPACE/P or Ctrl+Shift+P] Pause | [N] Center Head | [A] Anchor Mode | "
        "[D] Desktop Control | [C] Calibrate | [S] Sliders | [Q] Quit"
    )

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        ) as hands, mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        ) as face_mesh:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    pointer.safe_release()
                    print("[error] Camera frame lost — stopping pointer actions.")
                    break

                if cfg.mirror:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                hand_results = hands.process(rgb)
                face_results = face_mesh.process(rgb)
                rgb.flags.writeable = True

                hand_obs: Optional[HandObservation] = None
                if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                    hand_lms = hand_results.multi_hand_landmarks[0]
                    hand_info = hand_results.multi_handedness[0].classification[0]
                    pts = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]
                    hand_obs = HandObservation(
                        landmarks=pts,
                        handedness=hand_info.label,
                        score=float(hand_info.score),
                    )

                face_obs: Optional[FaceObservation] = None
                if face_results.multi_face_landmarks:
                    face_lms = face_results.multi_face_landmarks[0]
                    fpts = [(lm.x, lm.y, lm.z) for lm in face_lms.landmark]
                    face_obs = FaceObservation.from_facemesh_landmarks(fpts)

                now = time.monotonic()
                dt = max(1e-4, now - prev_time)
                prev_time = now
                fps = 0.88 * fps + 0.12 * (1.0 / dt)

                out = processor.process(hand_obs, face_obs, now=now)
                annotated = hud.draw(frame, hand_obs, face_obs, out, fps)
                cv2.imshow(win_name, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                elif key in (ord(" "), ord("p"), ord("P")):
                    processor.toggle_pause()
                elif key in (ord("n"), ord("N")):
                    pitch = face_obs.pitch_ratio if face_obs else None
                    processor.recenter_head_pitch(pitch)
                elif key in (ord("a"), ord("A")):
                    processor.cycle_anchor_mode()
                elif key in (ord("d"), ord("D")):
                    processor.toggle_desktop_control()
                elif key in (ord("c"), ord("C")):
                    if processor.calibrating:
                        processor.calibrating = False
                    else:
                        processor.start_calibration()
                elif key in (ord("m"), ord("M")):
                    cfg.mirror = not cfg.mirror
                    cfg.save()
                elif key in (ord("h"), ord("H")):
                    order = ["Any", "Right", "Left"]
                    cfg.handedness = order[(order.index(cfg.handedness) + 1) % len(order)]
                    cfg.save()
                elif key in (ord("s"), ord("S")):
                    hud.toggle_settings_window()
                elif key in (ord("r"), ord("R")):
                    cfg = CursorConfig()
                    cfg.save()
                    processor.cfg = cfg
    finally:
        pointer.safe_release()
        cap.release()
        cv2.destroyAllWindows()
        if global_listener is not None:
            global_listener.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
