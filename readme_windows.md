# Local Hand + Head/Mouth Controlled Cursor Studio for Windows 10/11 (v2.0)

> **A high-precision, low-latency, 100% local computer vision mouse controller for Windows 10/11 and modern browsers.** Move your cursor with your hand, click effortlessly by opening your mouth, scroll naturally with subtle head tilts, and pinch to drag—with zero cloud dependencies and zero video frames sent over the network.

Built as the native Windows counterpart to [MasalMavzer10/Hand-Cursor-Mac](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FMasalMavzer10%2FHand-Cursor-Mac).

## Overview & Highlights

Modern touchless pointer tracking systems typically suffer from two major flaws:

1. **Tremor Disruption on Clicks**: Pinching fingers together inevitably shakes the hand, moving the pointer off small buttons or link targets right as the click registers.
2. **Gesture Ambiguity**: Systems frequently confuse a quick pinch (click) with a held pinch (drag), or mistake finger position transitions for scrolling.

**Cursor Studio v2.0 for Windows solves this by decoupling biometric controls across independent anatomical channels**:

* **Cursor Aiming**: Steered by your hand using a **stabilized knuckle anchor** (`70% Index MCP + 30% Index Tip`) with velocity-adaptive **1-Euro smoothing** and a micro-precision fine-aim power curve (`1.35`).
* **Left Click**: Triggered by **opening your mouth briefly** (Mouth Aspect Ratio via MediaPipe FaceMesh). Because clicking is decoupled from your hands, your hand stays **completely still** while clicking microscopic targets.
* **Scrolling**: Controlled by **subtle head tilt pitch** (tilting head UP scrolls up; tilting DOWN scrolls down) with a customizable neutral deadband and 1-key recentering (`N`).
* **Drag & Drop**: Thumb-and-index pinch is dedicated **100% to dragging and dropping**, eliminating click-vs-drag collisions.
* **Instant Pause**: Hit `Space`, `P`, or global `Ctrl+Shift+P` anytime to safely yield control back to your physical mouse or keyboard.

## Windows-Specific Engineering Architecture

### 1. Zero-Dependency Win32 Native Pointer Bridge (`ctypes.windll.user32`)
Instead of macOS Quartz `CGEventPost`, `WinPointerController` hooks directly into the Windows `user32.dll` and `shcore.dll` system APIs via Python's built-in `ctypes`:
* **Per-Monitor High-DPI Awareness**: Calls `ctypes.windll.shcore.SetProcessDpiAwareness(2)` (with fallback to `user32.SetProcessDPIAware()`) on startup so Windows 125%, 150%, or 200% display scaling maps 1:1 to true physical pixels via `GetSystemMetrics(0)` and `GetSystemMetrics(1)`.
* **Low-Latency Cursor & Click Injection**: Uses `user32.SetCursorPos(x, y)` and `user32.mouse_event` (`MOUSEEVENTF_LEFTDOWN`, `MOUSEEVENTF_LEFTUP`) with `pynput` as an automatic fallback.
* **Smooth Win32 Wheel Scrolling**: Converts continuous head-pitch scroll deltas into scaled `MOUSEEVENTF_WHEEL` increments (`WHEEL_DELTA = 120`).

### 2. Instant DirectShow Webcam Initialization (`cv2.CAP_DSHOW`)
On Windows 10 and 11, OpenCV's default Microsoft Media Foundation (`CAP_MSMF`) backend can take 5–15 seconds to initialize a webcam. `hand_cursor_win.py` automatically opens the camera with **DirectShow (`cv2.CAP_DSHOW`)** first for near-instant startup, falling back to `cv2.CAP_ANY` if needed.

### 3. Global Windows Hotkey (`Ctrl+Shift+P`)
Registers `<ctrl>+<shift>+p` via `pynput.keyboard.GlobalHotKeys` so you can pause or resume tracking from any active Windows application.

## Decoupled Gesture Control Scheme

| Action | Biometric Trigger | Technical Thresholds | Behavioral Guarantee |
| :--- | :--- | :--- | :--- |
| **Cursor Move** | Hand movement in active ROI | 70% Knuckle (`lm[5]`) + 30% Tip (`lm[8]`), 1-Euro smoothing (`0.16` to `0.78`) | Cursor remains steady when fingers bend or twitch |
| **Left Click** | Open mouth briefly | MAR >= `0.26` enter, <= `0.16` reset, `45 ms` debounce, `320 ms` cooldown | Fingers stay motionless; zero target slipping |
| **Drag & Drop** | Pinch thumb + index & hold | Distance ratio < `0.26` hold > `180 ms`; release > `0.38` | 100% dedicated to drag; never fires accidental click |
| **Scroll Up** | Tilt head upward | Head pitch delta > `+0.035` above calibrated neutral | Smooth continuous scrolling at adjustable speed |
| **Scroll Down** | Tilt head downward | Head pitch delta < `-0.035` below calibrated neutral | Reversible scrolling; press `N` to re-center |
| **Pause / Resume** | Keyboard shortcut | `Space`, `P`, or global `Ctrl+Shift+P` | Instantly releases active drag and freezes pointer |
| **4-Pt Calibrate** | Calibration wizard | `C` key or HUD button | Sets custom comfortable boundaries to your reach |

## Repository Structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml              # Automated GitHub Actions test workflow (Windows + Ubuntu)
├── .gitignore                  # Clean ignores (.venv, __pycache__, config)
├── LICENSE                     # Permissive MIT License
├── README.md                   # Comprehensive Windows documentation & guide
├── hand_cursor_win.py          # Standalone Windows Python application (Stages A-F v2.0)
├── index.html                  # Interactive Web Studio (GitHub Pages ready)
├── requirements.txt            # Python dependencies for Windows
└── run_windows.bat             # 1-Click Windows double-clickable batch launcher
```

## Windows 10/11 Local Quickstart

### Method A: One-Click Batch Runner (Easiest)

1. Clone or extract this repository on your Windows PC.
2. In Windows File Explorer, double-click **`run_windows.bat`**.
   *(It automatically creates a `.venv` virtual environment, installs dependencies with `--no-compile`, and launches `hand_cursor_win.py`).*

### Method B: PowerShell / Command Prompt Setup

```powershell
# 1. Create and activate a clean virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
$env:PYTHONIOENCODING="utf-8"
python -m pip install --upgrade pip
pip install --no-compile -r requirements.txt

# 3. Safe Preview Mode (tests tracking without moving your Windows mouse)
python hand_cursor_win.py --preview-only

# 4. Full Windows Desktop Control Mode
python hand_cursor_win.py
```

> **Windows Camera Permission Note**: Ensure camera access is enabled under **Settings -> Privacy & security -> Camera -> Let desktop apps access your camera**.

## Automated CI Verification

Run the built-in headless self-test suite anytime without a webcam:

```powershell
python hand_cursor_win.py --self-test
```

## License

Distributed under the **MIT License**.
