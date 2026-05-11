"""
CoreML backend for Parakeet TDT on Apple Silicon.
Uses FluidInference/parakeet-tdt-0.6b-v3-coreml from HuggingFace.
Routes through ANE (Apple Neural Engine) — ~110x RTFx on M4 Pro.

Model is downloaded once to ~/.flow/models/parakeet-coreml/ and cached.
"""
import os, sys, time, json, struct, tempfile
import numpy as np

MODEL_REPO = "FluidInference/parakeet-tdt-0.6b-v3-coreml"
MODEL_DIR  = os.path.expanduser("~/.flow/models/parakeet-coreml")

def _log(msg):
    print(f"[coreml] {msg}", file=sys.stderr, flush=True)

def _download_model():
    """Download CoreML model from HuggingFace if not cached."""
    if os.path.exists(MODEL_DIR):
        contents = os.listdir(MODEL_DIR)
        if any(f.endswith(".mlpackage") or f.endswith(".mlmodel") for f in contents):
            return MODEL_DIR
    os.makedirs(MODEL_DIR, exist_ok=True)
    _log(f"Downloading CoreML model from {MODEL_REPO}...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=MODEL_REPO, local_dir=MODEL_DIR,
                          ignore_patterns=["*.git*", "*.md"])
        _log("Download complete.")
    except Exception as e:
        _log(f"Download failed: {e}")
        raise
    return MODEL_DIR

def load():
    """Load the CoreML model. Returns a callable predict function."""
    import coremltools as ct

    model_dir = _download_model()

    # Find .mlpackage or .mlmodel
    mlmodel_path = None
    for root, dirs, files in os.walk(model_dir):
        for d in dirs:
            if d.endswith(".mlpackage"):
                mlmodel_path = os.path.join(root, d)
                break
        if mlmodel_path:
            break
        for f in files:
            if f.endswith(".mlmodel"):
                mlmodel_path = os.path.join(root, f)
                break
        if mlmodel_path:
            break

    if not mlmodel_path:
        raise FileNotFoundError(f"No .mlpackage or .mlmodel found in {model_dir}")

    _log(f"Loading {mlmodel_path}...")
    model = ct.models.MLModel(mlmodel_path,
                               compute_units=ct.ComputeUnit.ALL)  # ANE + GPU + CPU
    _log("CoreML model loaded.")
    return model

def transcribe(model, audio: np.ndarray, sample_rate: int = 16000) -> str:
    """
    Transcribe float32 audio array. Returns transcript string.
    Input: mono float32 at 16kHz.
    """
    import coremltools as ct

    # CoreML expects specific input format — check model spec
    spec = model.get_spec()
    input_names = [inp.name for inp in spec.description.input]

    # Standard Parakeet CoreML input: "audio" as float32 array
    inputs = {}
    if "audio" in input_names:
        inputs["audio"] = audio.astype(np.float32)
    elif "input_features" in input_names:
        # Whisper-style mel spectrogram input — compute it
        inputs["input_features"] = _compute_mel(audio, sample_rate)
    else:
        # Try first input with raw audio
        inputs[input_names[0]] = audio.astype(np.float32)

    try:
        result = model.predict(inputs)
        # Output is usually "text" or "transcript"
        for key in ("text", "transcript", "output", list(result.keys())[0]):
            if key in result:
                return str(result[key]).strip()
    except Exception as e:
        _log(f"Predict error: {e}")
        return ""
    return ""

def _compute_mel(audio: np.ndarray, sr: int) -> np.ndarray:
    """Fallback: compute log-mel spectrogram for Whisper-style models."""
    try:
        import librosa
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, n_fft=400, hop_length=160)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        return log_mel[np.newaxis].astype(np.float32)  # [1, 80, T]
    except ImportError:
        raise ImportError("Install librosa: pip install librosa")


# ── Worker main loop (same protocol as parakeet_worker.py) ───────────────────

def main():
    _log("Loading CoreML model...")
    t0 = time.time()
    model = load()
    _log(f"Model loaded in {time.time()-t0:.1f}s")

    # Warmup
    warm = (np.random.randn(16000 * 3) * 0.15).astype(np.float32)
    transcribe(model, warm)
    transcribe(model, warm)
    _log("Warmed up — ready")

    import sys
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    stdin  = sys.stdin.buffer
    stdout = sys.stdout

    while True:
        header = stdin.read(4)
        if len(header) < 4:
            break
        n_samples = struct.unpack("<I", header)[0]
        if n_samples == 0:
            continue
        raw = stdin.read(n_samples * 4)
        if len(raw) < n_samples * 4:
            break

        audio    = np.frombuffer(raw, dtype=np.float32).copy()
        duration = len(audio) / 16000.0
        t1       = time.time()
        text     = transcribe(model, audio)
        elapsed  = time.time() - t1
        rtfx     = round(duration / elapsed, 1) if elapsed > 0 else 0

        stdout.write(json.dumps({"text": text, "rtfx": rtfx, "error": None}) + "\n")
        stdout.flush()

if __name__ == "__main__":
    main()
