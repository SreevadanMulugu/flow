#!/bin/bash
# Flow setup for macOS (Apple Silicon and Intel)
# Usage: bash scripts/setup_mac.sh

set -e
ARCH=$(uname -m)
echo "Setting up Flow on macOS ($ARCH)..."

# 1. Check Python
PY=$(python3 --version 2>&1)
echo "Python: $PY"

# 2. Create venv
python3 -m venv .venv
source .venv/bin/activate

# 3. Base dependencies
pip install --upgrade pip
pip install -r requirements/mac.txt

# 4. Backend selection
if [ "$ARCH" = "arm64" ]; then
    echo ""
    echo "Apple Silicon detected — choosing backend:"
    echo "  Trying CoreML (ANE, ~110x RTFx)..."

    if python3 -c "import coremltools" 2>/dev/null; then
        echo "  ✓ coremltools available — CoreML backend will be used"
        echo "  Model will download (~800MB) on first run from HuggingFace"
    else
        echo "  coremltools not found — falling back to NeMo MPS (~37x RTFx)"
        echo "  Installing NeMo in Python 3.10 venv..."
        _install_nemo
    fi
else
    echo ""
    echo "Intel Mac detected — installing NeMo MPS backend..."
    _install_nemo
fi

echo ""
echo "Done! Run Flow:"
echo "  npm run tauri dev"
echo ""
echo "First launch: Parakeet model downloads automatically (~800MB CoreML or ~2.4GB NeMo)"
}

_install_nemo() {
    # NeMo requires Python 3.10
    if python3.10 --version &>/dev/null; then
        python3.10 -m venv /tmp/nemo_env
        /tmp/nemo_env/bin/pip install --upgrade pip
        /tmp/nemo_env/bin/pip install youtokentome --no-build-isolation
        /tmp/nemo_env/bin/pip install nemo_toolkit[asr]==2.7.3
        /tmp/nemo_env/bin/pip install soundfile sounddevice numpy
        echo "  ✓ NeMo installed in /tmp/nemo_env"
    else
        echo "  Python 3.10 not found — install it: brew install python@3.10"
        exit 1
    fi
}

main "$@"
