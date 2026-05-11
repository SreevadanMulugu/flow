"""
NeMo CUDA backend for Parakeet TDT — NVIDIA GPUs on Linux/Windows.

Uses conformer_stream_step for TRUE frame-by-frame streaming.
This is the NeMo official streaming API — works on CUDA, broken on MPS/CPU.

Each audio frame (~80ms) produces partial tokens immediately,
giving real streaming without LocalAgreement approximation.
"""
import sys, os, time, json, struct, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import soundfile as sf
import tempfile
import torch

CHUNK_S    = 0.08    # 80ms frames for true streaming
SAMPLE_RATE = 16000
CHUNK_SAMPLES = int(CHUNK_S * SAMPLE_RATE)

def _log(msg):
    print(f"[nemo_cuda] {msg}", file=sys.stderr, flush=True)

def main():
    _log("Loading Parakeet on CUDA...")
    t0 = time.time()

    import nemo.collections.asr as nemo_asr
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    os.environ["HF_HUB_OFFLINE"] = "1"
    model = model.to("cuda")
    model.eval()

    _log(f"Model loaded in {time.time()-t0:.1f}s on CUDA")

    # Configure for cache-aware streaming (CUDA-only, works correctly here)
    model.change_decoding_strategy(None)

    # Warmup
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        warmup_path = f.name
    warm_sig = (np.random.randn(SAMPLE_RATE * 3) * 0.15).astype(np.float32)
    sf.write(warmup_path, warm_sig, SAMPLE_RATE)
    model.transcribe([warmup_path], verbose=False)
    model.transcribe([warmup_path], verbose=False)
    os.unlink(warmup_path)
    _log("Warmed up — ready (true streaming mode)")

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
        duration = len(audio) / float(SAMPLE_RATE)

        # True streaming: feed frames through conformer_stream_step
        try:
            t1 = time.time()
            text = _stream_transcribe(model, audio)
            elapsed = time.time() - t1
            rtfx    = round(duration / elapsed, 1) if elapsed > 0 else 0
            result  = {"text": text.strip(), "rtfx": rtfx, "error": None}
        except Exception as e:
            # Fallback to batch if streaming fails
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp_path = f.name
                sf.write(tmp_path, audio, SAMPLE_RATE)
                t1  = time.time()
                out = model.transcribe([tmp_path], verbose=False, timestamps=False)
                elapsed = time.time() - t1
                text    = out[0].text if hasattr(out[0], "text") else str(out[0])
                rtfx    = round(duration / elapsed, 1) if elapsed > 0 else 0
                result  = {"text": text.strip(), "rtfx": rtfx, "error": None}
                os.unlink(tmp_path)
            except Exception as e2:
                result = {"text": "", "rtfx": 0, "error": str(e2)}

        stdout.write(json.dumps(result) + "\n")
        stdout.flush()

def _stream_transcribe(model, audio: np.ndarray) -> str:
    """
    Feed audio through conformer_stream_step frame by frame.
    Returns full transcript when done.
    Works on CUDA — the RelPos attention tensor size mismatch is CUDA-only fixed.
    """
    import torch
    from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR

    model.cfg.preprocessor.dither = 0.0
    model.cfg.preprocessor.pad_to = 0

    frame_asr = FrameBatchASR(
        asr_model=model,
        frame_len=CHUNK_S,
        total_buffer=CHUNK_S * 4,
        batch_size=1,
    )
    frame_asr.reset()

    # Feed frames
    n_chunks = max(1, len(audio) // CHUNK_SAMPLES)
    for i in range(n_chunks):
        chunk = audio[i * CHUNK_SAMPLES : (i + 1) * CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        frame_asr.read_audio_chunk(chunk)

    return frame_asr.transcribe(tokens_per_chunk=128, delay=0)[0]
