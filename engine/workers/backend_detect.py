"""
Detect the best available inference backend for Parakeet TDT on this machine.

Priority order (highest RTFx first):
  1. CoreML  — Apple Silicon (ANE + GPU)  ~110x RTFx
  2. CUDA    — NVIDIA GPU (NeMo streaming) ~60-100x RTFx
  3. MPS     — Apple Silicon via PyTorch  ~37x RTFx
  4. ONNX    — any GPU via ONNX Runtime   ~20-60x RTFx
  5. CPU     — ONNX INT8 quantized        ~8-15x RTFx
"""
import platform, subprocess, sys

SYSTEM = platform.system()          # "Darwin" | "Linux" | "Windows"
MACHINE = platform.machine()        # "arm64" | "x86_64" | "AMD64"

def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=3)
        return result.returncode == 0
    except Exception:
        return False

def _has_mps() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False

def _has_coreml() -> bool:
    if SYSTEM != "Darwin":
        return False
    try:
        import coremltools  # noqa
        return MACHINE == "arm64"
    except ImportError:
        return False

def _has_onnxruntime() -> bool:
    try:
        import onnxruntime  # noqa
        return True
    except ImportError:
        return False

def _onnx_best_ep() -> str:
    """Return the best ONNX Runtime execution provider available."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        for ep in ("CUDAExecutionProvider", "ROCMExecutionProvider",
                   "CoreMLExecutionProvider", "DirectMLExecutionProvider"):
            if ep in providers:
                return ep
        return "CPUExecutionProvider"
    except Exception:
        return "CPUExecutionProvider"

def detect() -> dict:
    """
    Returns a dict describing the best backend:
      {
        "backend": "coreml" | "cuda" | "mps" | "onnx" | "cpu",
        "ep": "CUDAExecutionProvider" | ...,   # for onnx only
        "rtfx_est": 80,                        # rough estimate
        "reason": "Apple Silicon ANE via CoreML",
      }
    """
    if _has_coreml():
        return {
            "backend": "coreml",
            "ep": None,
            "rtfx_est": 110,
            "reason": "Apple Silicon ANE via CoreML",
        }

    if _has_cuda():
        return {
            "backend": "cuda",
            "ep": "CUDAExecutionProvider",
            "rtfx_est": 70,
            "reason": "NVIDIA CUDA (NeMo + ONNX Runtime)",
        }

    if _has_mps():
        return {
            "backend": "mps",
            "ep": None,
            "rtfx_est": 37,
            "reason": "Apple Silicon MPS via PyTorch",
        }

    if _has_onnxruntime():
        ep = _onnx_best_ep()
        rtfx = {
            "CUDAExecutionProvider": 60,
            "ROCMExecutionProvider": 35,
            "CoreMLExecutionProvider": 80,
            "DirectMLExecutionProvider": 25,
            "CPUExecutionProvider": 10,
        }.get(ep, 10)
        return {
            "backend": "onnx",
            "ep": ep,
            "rtfx_est": rtfx,
            "reason": f"ONNX Runtime ({ep})",
        }

    return {
        "backend": "cpu",
        "ep": "CPUExecutionProvider",
        "rtfx_est": 8,
        "reason": "CPU fallback (install onnxruntime for better performance)",
    }

if __name__ == "__main__":
    import json
    print(json.dumps(detect(), indent=2))
