# Flow — Offline Speech-to-Text Dictation

Fully offline, low-latency dictation app. Press a hotkey → speak → words appear at your cursor. No cloud. No subscription. Runs entirely on-device using NVIDIA Parakeet TDT 0.6B v3.

Auto-detects the best inference backend for your hardware — from Apple Neural Engine to NVIDIA CUDA to quantized CPU.

---

## Performance by Platform

| Platform | Backend | RTFx | Notes |
|---|---|---|---|
| Apple Silicon (M1–M4) | CoreML (ANE) | ~110x | Recommended |
| Apple Silicon (M1–M4) | PyTorch MPS | ~37x | Fallback if CoreML unavailable |
| NVIDIA GPU (Linux/Windows) | NeMo CUDA + ONNX | ~60–100x | True frame-by-frame streaming |
| AMD GPU (Linux) | ONNX ROCm | ~35x | |
| Intel/AMD GPU (Windows) | ONNX DirectML | ~20–30x | |
| Any CPU | ONNX INT8 | ~8–15x | Quantized, works everywhere |

RTFx = real-time factor. 10x = a 10s clip transcribed in 1s.

---

## Features

- **Global dictation** — works in any app: email, Slack, VS Code, browsers
- **Hotkey control** — Alt+Space to start/stop
- **Three modes** — Dictation (types at cursor), Meeting (continuous transcript), Capture (voice notes)
- **Real-time streaming** — words appear as you speak, not after silence
- **Multilingual** — auto-detects language per utterance, 25 languages, script-locks to prevent mid-sentence flipping
- **Sentence-aware turn detection** — commits on acoustic silence and linguistic sentence boundaries
- **Custom vocabulary** — heard-as → should-be replacements, auto-learned
- **Transcript history** — searchable local log
- **Mini bar** — 220×52 px floating overlay

---

## Architecture

```
┌──────────────────────────────┐
│   Tauri Shell (Rust)         │  Window management, system tray,
│   src-tauri/                 │  sidecar spawn, hotkey pass-through
└──────────────┬───────────────┘
               │ spawns
┌──────────────▼───────────────┐
│   flow_server (Python)       │  FastAPI + WebSocket server
│   engine/flow_server.py      │  Audio capture, streaming, history
└──────────────┬───────────────┘
               │ spawns
┌──────────────▼───────────────┐
│   parakeet_worker            │  Auto-routing inference worker
│   engine/workers/            │
│   ├── backend_detect.py      │  Hardware fingerprint at startup
│   ├── backend_coreml.py      │  Apple ANE via CoreML (~110x RTFx)
│   ├── backend_nemo_cuda.py   │  NVIDIA true streaming (~70x RTFx)
│   ├── backend_nemo_mps.py    │  Apple MPS via PyTorch (~37x RTFx)
│   └── backend_onnx.py        │  ONNX Runtime quantized (~10–60x)
└──────────────────────────────┘
```

**Streaming pipeline:**
- Audio captured at 16kHz, transcribed every 0.5s (rolling window)
- Script family (Latin/Cyrillic/etc.) detected on first result, locked per utterance
- Sentence-ending punctuation halves the silence wait for faster commits
- NVIDIA CUDA: true `conformer_stream_step` streaming (NeMo official API)
- Mac/CPU: LocalAgreement-2 sliding window approximation

---

## Installation

### macOS — Apple Silicon (recommended)

```bash
# Prerequisites
brew install python@3.10 node rust

git clone https://github.com/SreevadanMulugu/flow.git
cd flow
bash scripts/setup_mac.sh
npm install
npm run tauri dev
```

First launch downloads the CoreML model (~800MB) automatically. Subsequent launches are instant.

### macOS — Intel

Same steps. Uses NeMo MPS or CPU backend (~8–15x RTFx).

### Linux — NVIDIA GPU

```bash
# Prerequisites: CUDA 12.1+, Python 3.10+, Node.js 18+, Rust
git clone https://github.com/SreevadanMulugu/flow.git
cd flow
bash scripts/setup_linux.sh   # installs NeMo, exports ONNX INT8 (~5 min)
npm install
npm run tauri dev
```

Uses NeMo `conformer_stream_step` for true token-by-token streaming on CUDA.

### Linux — CPU only

```bash
bash scripts/setup_linux.sh   # detects no GPU, installs ONNX CPU INT8
```

### Windows

```powershell
git clone https://github.com/SreevadanMulugu/flow.git
cd flow
.\scripts\setup_windows.ps1
npm install
npm run tauri dev
```

ONNX Runtime auto-selects: CUDA EP → DirectML EP → CPU EP.

---

## ONNX Model Export (Linux/Mac with NeMo)

Export the quantized model once — it then works everywhere:

```bash
# INT8 quantized (~600MB, recommended)
python scripts/export_onnx.py --quantize

# FP32 (~2.4GB, highest accuracy)
python scripts/export_onnx.py
```

Model saved to `~/.flow/models/parakeet-onnx/` and auto-detected on all platforms.

---

## Configuration

`~/.flow/config.json`:

```json
{
  "model": "parakeet-tdt-0.6b-v3",
  "language": "",        
  "hotkey": "alt+space",
  "mode": "dictation",
  "active_app_context": true,
  "regex_cleanup": true,
  "spell_check": true
}
```

`language`: `""` = auto-detect per utterance, `"en"` = force English, `"uk"` = force Ukrainian, etc.

---

## Model

**NVIDIA Parakeet TDT 0.6B v3** — [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- 600M parameters, FastConformer encoder + Token Duration Transducer decoder
- 85,000 hours training data across 25 languages
- Word-level timestamps at 40ms granularity
- Outperforms Whisper large-v3 on English benchmarks

CoreML (Apple Silicon): [FluidInference/parakeet-tdt-0.6b-v3-coreml](https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v3-coreml)

---

## Hardware Requirements

| | Minimum | Recommended |
|---|---|---|
| RAM | 4GB | 8GB+ |
| Storage | 1GB (ONNX INT8) | 3GB (NeMo FP32) |
| GPU | None (CPU works) | Apple M-series or NVIDIA RTX |
| OS | macOS 13+, Ubuntu 22.04+, Windows 11 | macOS 14+ Apple Silicon |

---

## Development

```bash
# Check which backend your machine will use
python engine/workers/backend_detect.py

# Run server only (browser at http://localhost:PORT)
python engine/flow_server.py

# Full Tauri dev
npm run tauri dev
```

---

## License

MIT
