#!/usr/bin/env python3
"""
Parakeet TDT 0.6B v3 worker process.
Runs in the NeMo Python 3.10 venv as a persistent subprocess.

Protocol (stdin → stdout):
  IN:  4-byte LE uint32 (num float32 samples) + raw float32 bytes
  OUT: newline-delimited JSON: {"text": "...", "rtfx": 1.2, "error": null}

Flow.py spawns this once, keeps it alive, sends audio chunks, reads results.
"""
import sys, os, struct, time, json, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import soundfile as sf
import tempfile
import torch

def _log(msg):
    print(f"[parakeet_worker] {msg}", file=sys.stderr, flush=True)

LANG = os.environ.get("PARAKEET_LANG", "").strip()   # e.g. "en", "" = auto-detect

def _transcribe(model, wav_path):
    """Transcribe wav_path; use manifest with lang field when LANG is set."""
    if LANG:
        mf = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        json.dump({"audio_filepath": wav_path, "lang": LANG, "duration": 600}, mf)
        mf.write("\n")
        mf.close()
        try:
            return model.transcribe([mf.name], verbose=False, timestamps=False,
                                    use_lhotse=False)
        finally:
            try: os.unlink(mf.name)
            except Exception: pass
    else:
        return model.transcribe([wav_path], verbose=False, timestamps=False)

def main():
    _log(f"loading model... (lang={'auto' if not LANG else LANG})")
    t0 = time.time()

    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")

    # Use MPS (Metal GPU) if available, else CPU
    if torch.backends.mps.is_available():
        model = model.to("mps")
        device = "mps"
    else:
        device = "cpu"
    model.eval()

    _log(f"model loaded in {time.time()-t0:.1f}s on {device}")

    # Warmup — run TWO passes with speech-shaped audio so MPS compiles its
    # kernels for real input dimensions. Silence (zeros) doesn't fully warm MPS.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        warmup_path = f.name
    sr = 16000
    warm_sig = (np.random.randn(sr * 3) * 0.15).astype(np.float32)
    sf.write(warmup_path, warm_sig, sr)
    _transcribe(model, warmup_path)   # pass 1 — kernel compile
    _transcribe(model, warmup_path)   # pass 2 — steady state
    os.unlink(warmup_path)
    _log("warmed up — ready")

    # Signal readiness to parent
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    stdin  = sys.stdin.buffer
    stdout = sys.stdout

    while True:
        # Read header: 4-byte LE uint32 = number of float32 samples
        header = stdin.read(4)
        if len(header) < 4:
            break  # parent closed pipe — exit cleanly

        n_samples = struct.unpack("<I", header)[0]
        if n_samples == 0:
            continue

        # Read audio bytes
        raw = stdin.read(n_samples * 4)
        if len(raw) < n_samples * 4:
            break

        audio = np.frombuffer(raw, dtype=np.float32).copy()
        duration = len(audio) / 16000.0

        # Write to temp wav and transcribe
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            sf.write(tmp_path, audio, 16000)
            t1 = time.time()
            out = _transcribe(model, tmp_path)
            elapsed = time.time() - t1
            text = out[0].text if hasattr(out[0], "text") else str(out[0])
            rtfx = round(duration / elapsed, 1) if elapsed > 0 else 0
            result = {"text": text.strip(), "rtfx": rtfx, "error": None}
        except Exception as e:
            result = {"text": "", "rtfx": 0, "error": str(e)}
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        stdout.write(json.dumps(result) + "\n")
        stdout.flush()

if __name__ == "__main__":
    main()
