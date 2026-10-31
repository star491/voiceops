@echo off
REM One-click run that fits Groq's free daily limits. Safe to run again: finished work is kept.
REM Order: voice run -> questions (Step 2) -> re-score -> conversations -> re-score.  Log: run_log.txt
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
echo ==== %date% %time% start ==== >> run_log.txt

echo [1/6] Checking packages...
python -m pip install -q -r requirements.txt >> run_log.txt 2>&1

echo [2/6] Voice run on the test audio (Gnani STT -^> agent -^> Gnani TTS)...
if not exist test_audio\PQ0002.wav python make_test_audio.py >> run_log.txt 2>&1
python run.py --input test_audio --output outputs --resume >> run_log.txt 2>&1

echo [3/6] Practice questions: Step 2 with the final agent (Step 1 results are kept)...
python eval.py --only questions --resume >> run_log.txt 2>&1

echo [4/6] Re-scoring with the fixed judge (small model, separate quota)...
python rejudge.py >> run_log.txt 2>&1

echo [5/6] Practice conversations (may need more than one day on the free tier)...
python eval.py --only conversations --resume >> run_log.txt 2>&1

echo [6/6] Re-scoring new conversation turns...
python rejudge.py >> run_log.txt 2>&1

echo ==== %date% %time% done ==== >> run_log.txt
findstr /c:"STOPPED" run_log.txt >nul && echo. && echo Note: the free daily limit was reached at some point. Run this again tomorrow to continue.
echo.
echo All done for now. Tell Claude "finished".
pause
