"""
ONNX Runtime backend for Parakeet TDT.
Runs on Windows and Linux with any GPU (CUDA, ROCm, DirectML) or CPU.

Model must be exported first via: python scripts/export_onnx.py
Quantized INT8 model is ~600MB vs 2.4GB FP32.

Auto-selects the best Execution Provider:
  CUDA → ROCm → DirectML → CPU
"""
import os, sys, time, json, struct, tempfile
import numpy as np

MODEL_DIR   = os.path.expanduser("~/.flow/models/parakeet-onnx")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.onnx")
DECODER_PATH = os.path.join(MODEL_DIR, "decoder_joint.onnx")

def _log(msg):
    print(f"[onnx] {msg}", file=sys.stderr, flush=True)

def _best_providers(ep_hint: str = None):
    import onnxruntime as ort
    available = ort.get_available_providers()
    priority  = [
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "DirectMLExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    if ep_hint and ep_hint in available:
        rest = [p for p in priority if p != ep_hint and p in available]
        return [ep_hint] + rest
    return [p for p in priority if p in available]

def load(ep_hint: str = None):
    """Load encoder and decoder ONNX sessions."""
    import onnxruntime as ort

    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(
            f"ONNX model not found at {MODEL_DIR}.\n"
            "Run:  python scripts/export_onnx.py\n"
            "This converts the NeMo model to ONNX (one-time, ~5 min)."
        )

    providers = _best_providers(ep_hint)
    _log(f"Using providers: {providers}")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = os.cpu_count()

    encoder = ort.InferenceSession(ENCODER_PATH, sess_opts, providers=providers)
    decoder = ort.InferenceSession(DECODER_PATH, sess_opts, providers=providers)

    _log("ONNX sessions loaded.")
    return encoder, decoder

def _mel_features(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Compute log-mel spectrogram expected by Parakeet encoder."""
    # 80-dim log-mel, 25ms window, 10ms hop (standard Parakeet config)
    n_fft     = 512
    hop       = 160    # 10ms at 16kHz
    n_mels    = 80
    win_len   = 400    # 25ms

    # STFT
    pad = n_fft // 2
    audio_padded = np.pad(audio, pad, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(
        audio_padded, win_len)[::hop]
    window = np.hanning(win_len).astype(np.float32)
    windowed = frames * window
    fft = np.fft.rfft(windowed, n=n_fft)
    power = (np.abs(fft) ** 2).astype(np.float32)

    # Mel filterbank
    mel_filters = _mel_filterbank(sr, n_fft, n_mels)
    mel = np.dot(power, mel_filters.T)
    log_mel = np.log(np.maximum(mel, 1e-10))

    # [1, n_mels, T] with length as second output
    feat = log_mel.T[np.newaxis].astype(np.float32)
    length = np.array([feat.shape[2]], dtype=np.int64)
    return feat, length

def _mel_filterbank(sr, n_fft, n_mels):
    low_hz, high_hz = 0.0, sr / 2.0
    def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel): return 700 * (10 ** (mel / 2595) - 1)
    mel_pts = np.linspace(hz_to_mel(low_hz), hz_to_mel(high_hz), n_mels + 2)
    hz_pts  = mel_to_hz(mel_pts)
    bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb      = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        fb[m-1, bins[m-1]:bins[m]]   = (np.arange(bins[m-1], bins[m])   - bins[m-1]) / (bins[m] - bins[m-1])
        fb[m-1, bins[m]:bins[m+1]]   = (bins[m+1] - np.arange(bins[m], bins[m+1]))   / (bins[m+1] - bins[m])
    return fb

def transcribe(encoder, decoder, audio: np.ndarray) -> str:
    """Run encoder → greedy decoder → return transcript string."""
    feat, feat_len = _mel_features(audio)

    # Encoder
    enc_inputs  = {encoder.get_inputs()[0].name: feat,
                   encoder.get_inputs()[1].name: feat_len}
    enc_outputs = encoder.run(None, enc_inputs)
    enc_out, enc_len = enc_outputs[0], enc_outputs[1]

    # Greedy decoder (step-by-step RNNT/TDT beam search)
    tokens = []
    blank_id = decoder.get_outputs()[0].shape[-1] - 1   # last token = blank

    # Initial hidden state (zeros)
    h_shape = [s if isinstance(s, int) and s > 0 else 1
               for s in decoder.get_inputs()[2].shape]
    h = np.zeros(h_shape, dtype=np.float32)

    prev_token = np.array([[0]], dtype=np.int64)   # SOS token
    T = enc_out.shape[1]

    for t in range(T):
        frame = enc_out[:, t:t+1, :]   # [1, 1, D]
        dec_inputs = {
            decoder.get_inputs()[0].name: frame,
            decoder.get_inputs()[1].name: prev_token,
            decoder.get_inputs()[2].name: h,
        }
        dec_out = decoder.run(None, dec_inputs)
        logits, h_new = dec_out[0], dec_out[1]
        token_id = int(np.argmax(logits[0, 0]))
        if token_id != blank_id:
            tokens.append(token_id)
            prev_token = np.array([[token_id]], dtype=np.int64)
        h = h_new

    # Decode tokens — requires tokenizer from NeMo model config
    # Fallback: return raw token IDs (replace with SentencePiece decode)
    return _decode_tokens(tokens)

def _decode_tokens(tokens: list) -> str:
    """Decode token IDs using cached SentencePiece vocab."""
    vocab_path = os.path.join(MODEL_DIR, "tokenizer.model")
    if not os.path.exists(vocab_path):
        return f"<{len(tokens)} tokens — run export_onnx.py to include tokenizer>"
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load(vocab_path)
        return sp.Decode(tokens)
    except Exception:
        return " ".join(str(t) for t in tokens)


# ── Worker main loop ──────────────────────────────────────────────────────────

def main(ep_hint: str = None):
    _log("Loading ONNX model...")
    t0 = time.time()
    encoder, decoder = load(ep_hint)
    _log(f"Loaded in {time.time()-t0:.1f}s")

    warm = (np.random.randn(16000 * 3) * 0.15).astype(np.float32)
    transcribe(encoder, decoder, warm)
    transcribe(encoder, decoder, warm)
    _log("Warmed up — ready")

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
        try:
            text    = transcribe(encoder, decoder, audio)
            elapsed = time.time() - t1
            rtfx    = round(duration / elapsed, 1) if elapsed > 0 else 0
            result  = {"text": text, "rtfx": rtfx, "error": None}
        except Exception as e:
            result = {"text": "", "rtfx": 0, "error": str(e)}

        stdout.write(json.dumps(result) + "\n")
        stdout.flush()

if __name__ == "__main__":
    ep = os.environ.get("ONNX_EP")
    main(ep)
