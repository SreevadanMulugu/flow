#!/bin/bash
# Flow setup for Linux (NVIDIA GPU or CPU)
# Usage: bash scripts/setup_linux.sh

set -e
echo "Setting up Flow on Linux..."

# Detect GPU
HAS_CUDA=false
if nvidia-smi &>/dev/null; then
    HAS_CUDA=true
    CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}')
    echo "NVIDIA GPU detected (CUDA $CUDA_VER)"
else
    echo "No NVIDIA GPU detected — using ONNX CPU INT8 backend"
fi

# Python venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

if $HAS_CUDA; then
    pip install -r requirements/linux_cuda.txt

    echo ""
    echo "Installing NeMo for true streaming (conformer_stream_step on CUDA)..."
    pip install youtokentome --no-build-isolation
    pip install nemo_toolkit[asr]==2.7.3

    echo ""
    echo "Exporting Parakeet to ONNX with INT8 quantization..."
    echo "(~5 min, one-time — downloads 2.4GB NeMo model then exports)"
    python3 scripts/export_onnx.py --quantize
else
    pip install onnxruntime numpy sounddevice soundfile fastapi uvicorn websockets sentencepiece
    echo ""
    echo "Exporting Parakeet to ONNX with INT8 quantization..."
    echo "(requires NeMo on another machine, or download pre-exported model)"
    echo "See: https://github.com/SreevadanMulugu/flow#onnx-model"
fi

echo ""
echo "Done! Run Flow:"
echo "  npm run tauri dev"
