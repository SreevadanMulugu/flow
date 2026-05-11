#!/bin/bash
# Flow setup for Linux (NVIDIA GPU or CPU)
# Usage: bash scripts/setup_linux.sh

set -e
NEMO_VENV="$HOME/.flow/nemo_env"
echo "Setting up Flow on Linux..."
mkdir -p "$HOME/.flow"

# Detect GPU
HAS_CUDA=false
if nvidia-smi &>/dev/null; then
    HAS_CUDA=true
    CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}')
    echo "NVIDIA GPU detected (CUDA $CUDA_VER)"
else
    echo "No NVIDIA GPU — using ONNX CPU INT8 backend"
fi

# Find Python 3.10+
PY=$(which python3.10 2>/dev/null || which python3.11 2>/dev/null || which python3 2>/dev/null)
PY_VER=$("$PY" -c "import sys; print(sys.version_info[:2])")
echo "Using Python: $PY ($PY_VER)"

# Install NeMo venv (needed for ONNX export + CUDA streaming)
echo ""
echo "Creating NeMo venv at $NEMO_VENV ..."
"$PY" -m venv "$NEMO_VENV"
"$NEMO_VENV/bin/pip" install --upgrade pip --quiet
"$NEMO_VENV/bin/pip" install youtokentome --no-build-isolation --quiet

if $HAS_CUDA; then
    "$NEMO_VENV/bin/pip" install "nemo_toolkit[asr]==2.7.3" --quiet
    "$NEMO_VENV/bin/pip" install onnxruntime-gpu soundfile sounddevice numpy --quiet
    echo ""
    echo "Exporting Parakeet to ONNX INT8 (~5 min, one-time)..."
    "$NEMO_VENV/bin/python" scripts/export_onnx.py --quantize
else
    "$NEMO_VENV/bin/pip" install onnxruntime soundfile sounddevice numpy fastapi uvicorn websockets sentencepiece --quiet
    echo ""
    echo "No NVIDIA GPU — ONNX CPU INT8 backend will be used (~10x RTFx)"
    echo "To export ONNX model, run on a machine with NeMo:"
    echo "  python scripts/export_onnx.py --quantize"
fi

echo ""
echo "Setup complete. Run the app:"
echo "  npm install && npm run tauri dev"
