# Flow setup for Windows
# Run in PowerShell: .\scripts\setup_windows.ps1

Write-Host "Setting up Flow on Windows..."

# Detect GPU
$hasCuda = $false
try {
    $nvsmi = & nvidia-smi 2>$null
    if ($LASTEXITCODE -eq 0) {
        $hasCuda = $true
        Write-Host "NVIDIA GPU detected"
    }
} catch {}

# Python venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip

if ($hasCuda) {
    pip install -r requirements/windows.txt
    Write-Host "CUDA ONNX Runtime installed — will use GPU acceleration"
} else {
    pip install onnxruntime numpy sounddevice soundfile fastapi uvicorn websockets sentencepiece
    Write-Host "CPU ONNX Runtime installed"
}

Write-Host ""
Write-Host "Download pre-exported ONNX model (no NeMo needed on Windows):"
Write-Host "  See: https://github.com/SreevadanMulugu/flow#onnx-model"
Write-Host ""
Write-Host "Or export from NeMo on Linux/Mac, then copy ~/.flow/models/parakeet-onnx/"
Write-Host ""
Write-Host "Done! Run Flow:"
Write-Host "  npm run tauri dev"
