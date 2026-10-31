@echo off
REM Double-click to start the live demo page (for recording the 2-minute video).
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
echo Starting the ZentroMart voice demo... (first start can take a minute)
python -m pip install -q gradio >nul 2>&1
python app.py
echo.
echo The demo stopped. If you see an error above, copy it to Claude.
pause
