#!/usr/bin/env python3
"""
Parakeet TDT 0.6B v3 — auto-routing worker.

Detects the best inference backend for this machine and runs it:
  CoreML  → Apple Silicon (ANE + GPU)    ~110x RTFx
  CUDA    → NVIDIA GPU (NeMo streaming)  ~60-100x RTFx
  MPS     → Apple Silicon PyTorch        ~37x RTFx
  ONNX    → any GPU / CPU INT8           ~10-60x RTFx

Protocol (stdin → stdout):
  IN:  4-byte LE uint32 (num float32 samples) + raw float32 bytes
  OUT: newline-delimited JSON: {"text": "...", "rtfx": 1.2, "error": null}
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

# Allow imports from sibling directory
sys.path.insert(0, os.path.dirname(__file__))
from backend_detect import detect

def _log(msg):
    print(f"[parakeet_worker] {msg}", file=sys.stderr, flush=True)

def main():
    info = detect()
    backend  = info["backend"]
    ep       = info.get("ep")
    rtfx_est = info["rtfx_est"]
    reason   = info["reason"]

    _log(f"Backend: {backend} — {reason} (est. {rtfx_est}x RTFx)")

    if backend == "coreml":
        import backend_coreml
        backend_coreml.main()

    elif backend == "cuda":
        # NeMo with true conformer_stream_step streaming on CUDA
        import backend_nemo_cuda
        backend_nemo_cuda.main()

    elif backend == "mps":
        # PyTorch MPS — current proven path on Apple Silicon
        import backend_nemo_mps
        backend_nemo_mps.main()

    elif backend in ("onnx", "cpu"):
        os.environ["ONNX_EP"] = ep or "CPUExecutionProvider"
        import backend_onnx
        backend_onnx.main(ep)

    else:
        _log(f"Unknown backend '{backend}' — falling back to ONNX CPU")
        import backend_onnx
        backend_onnx.main("CPUExecutionProvider")

if __name__ == "__main__":
    main()
