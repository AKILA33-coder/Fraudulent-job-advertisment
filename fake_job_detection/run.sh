#!/bin/bash
echo ""
echo "============================================================"
echo "  TrueHire AI - Fake Job Detection Platform"
echo "============================================================"
echo ""

cd "$(dirname "$0")"

echo "[1/3] Installing dependencies..."
pip install -r requirements.txt --quiet

echo "[2/3] Preparing ML data and training models..."
if [ ! -f "ml/models/random_forest.pkl" ]; then
    python3 ml/prepare_data.py
    python3 ml/train.py
else
    echo "      Models already trained - skipping."
fi

echo "[3/3] Starting Flask server..."
echo ""
echo "  Open browser: http://localhost:5000"
echo "  Press CTRL+C to stop"
echo ""
echo "============================================================"
echo ""
python3 app_ml.py
