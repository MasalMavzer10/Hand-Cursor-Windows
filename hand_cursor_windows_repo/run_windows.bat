@echo off
setlocal
cd /d "%~dp0"
echo ============================================================================
echo   Local Hand + Head/Mouth Controlled Cursor Studio (Windows 10/11 v2.0)
echo ============================================================================

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python 3 was not found in your PATH.
    echo Please install Python 3.9, 3.10, or 3.11 from https://www.python.org/downloads/windows/
    echo and check "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating clean Python virtual environment ^(.venv^)...
    python -m venv .venv
)

echo [2/3] Activating virtual environment and verifying dependencies...
call .venv\Scripts\activate.bat
set PYTHONIOENCODING=utf-8
python -m pip install --upgrade pip --quiet
pip install --no-compile -r requirements.txt

echo [3/3] Launching Hand + Head/Mouth Cursor for Windows...
python hand_cursor_win.py
pause
