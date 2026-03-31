#!/bin/bash
# ==============================================================================
# ToxiPred — Application Launcher
# ==============================================================================
# Launches the Streamlit web application.
#
# Usage:
#   bash scripts/run_app.sh
#   or:
#   chmod +x scripts/run_app.sh && ./scripts/run_app.sh
# ==============================================================================

set -e

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  🧬 ToxiPred — Starting Application"
echo "=========================================="

# Check if model exists
if [ ! -f "models/xgboost_model.joblib" ]; then
    echo ""
    echo "⚠️  No trained model found."
    echo "   Run the training pipeline first:"
    echo "   python scripts/run_pipeline.py"
    echo ""
    echo "Starting app anyway (predictions will be unavailable)..."
    echo ""
fi

# Launch Streamlit
echo "Starting Streamlit server..."
streamlit run src/app/streamlit_app.py \
    --server.port=8501 \
    --server.headless=true \
    --browser.gatherUsageStats=false
