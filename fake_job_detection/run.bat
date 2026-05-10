@echo off
echo.
echo ============================================================
echo   TrueHire AI - Fake Job Detection Platform
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/3] Installing dependencies...
pip install -r requirements.txt --quiet

echo [2/3] Preparing ML data and training models...
if not exist "ml\models\random_forest.pkl" (
    python ml\prepare_data.py
    python ml\train.py
) else (
    echo      Models already trained - skipping.
)

echo [3/3] Starting Flask server...
echo.
echo   Open browser: http://localhost:5000
echo   Press CTRL+C to stop
echo.
echo ============================================================
echo.
python app_ml.py
pause
