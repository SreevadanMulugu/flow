#!/bin/bash
# Flow setup for macOS (Apple Silicon and Intel)
# Usage: bash scripts/setup_mac.sh

set -e
ARCH=$(uname -m)
NEMO_VENV="$HOME/.flow/nemo_env"   # consistent location on all machines

echo "Setting up Flow on macOS ($ARCH)..."
mkdir -p "$HOME/.flow"

# Backend selection
if [ "$ARCH" = "arm64" ]; then
    echo ""
    echo "Apple Silicon detected — checking for CoreML backend (~110x RTFx)..."

    if python3 -c "import coremltools" 2>/dev/null; then
        echo "  coremltools found — CoreML (ANE) backend will be used"
        echo "  Model downloads automatically on first launch (~800MB)"
    else
        echo "  coremltools not available — installing NeMo MPS backend (~37x RTFx)"
        _install_nemo
    fi
else
    echo ""
    echo "Intel Mac — installing NeMo CPU backend..."
    _install_nemo
fi

echo ""
echo "Setup complete. Run the app:"
echo "  npm install && npm run tauri dev"
echo ""
echo "On first launch the model downloads automatically."

_install_nemo() {
    # NeMo requires Python 3.10+
    PY310=$(which python3.10 2>/dev/null || which python3.11 2>/dev/null || echo "")
    if [ -z "$PY310" ]; then
        echo "  Python 3.10+ not found. Install with:"
        echo "    brew install python@3.10"
        exit 1
    fi

    echo "  Creating NeMo venv at $NEMO_VENV ..."
    "$PY310" -m venv "$NEMO_VENV"
    "$NEMO_VENV/bin/pip" install --upgrade pip --quiet
    "$NEMO_VENV/bin/pip" install youtokentome --no-build-isolation --quiet
    "$NEMO_VENV/bin/pip" install "nemo_toolkit[asr]==2.7.3" --quiet
    "$NEMO_VENV/bin/pip" install soundfile sounddevice numpy --quiet
    echo "  NeMo installed at $NEMO_VENV"
}
