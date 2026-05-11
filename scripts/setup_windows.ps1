# Flow setup for Windows
# Run in PowerShell as Administrator: .\scripts\setup_windows.ps1

$NEMO_VENV = "$env:USERPROFILE\.flow\nemo_env"
Write-Host "Setting up Flow on Windows..."
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.flow" | Out-Null

# Detect GPU
$hasCuda = $false
try {
    & nvidia-smi 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $hasCuda = $true
        Write-Host "NVIDIA GPU detected — will use ONNX CUDA backend"
    }
} catch {}
if (-not $hasCuda) {
    Write-Host "No NVIDIA GPU — will use ONNX DirectML or CPU backend"
}

# Find Python 3.10+
$py = Get-Command python3.10 -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command python3 -ErrorAction SilentlyContinue }
if (-not $py) { $py = Get-Command python  -ErrorAction SilentlyContinue }
if (-not $py) {
    Write-Host "Python not found. Install from https://python.org (3.10 or 3.11 recommended)"
    exit 1
}
$PYTHON = $py.Source
Write-Host "Using Python: $PYTHON"

# Create venv at consistent location
Write-Host "Creating venv at $NEMO_VENV ..."
& $PYTHON -m venv $NEMO_VENV
$PIP = "$NEMO_VENV\Scripts\pip.exe"

& $PIP install --upgrade pip --quiet

if ($hasCuda) {
    & $PIP install onnxruntime-gpu soundfile sounddevice numpy fastapi uvicorn websockets sentencepiece --quiet
    Write-Host "ONNX Runtime with CUDA installed"
} else {
    & $PIP install onnxruntime soundfile sounddevice numpy fastapi uvicorn websockets sentencepiece --quiet
    Write-Host "ONNX Runtime CPU installed"
}

Write-Host ""
Write-Host "To get the quantized model, run on a Linux/Mac machine with NeMo:"
Write-Host "  python scripts/export_onnx.py --quantize"
Write-Host "Then copy ~/.flow/models/parakeet-onnx/ to this machine's same path."
Write-Host ""
Write-Host "Setup complete. Run the app:"
Write-Host "  npm install; npm run tauri dev"
