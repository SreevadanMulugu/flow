# Flow — Offline Speech-to-Text Dictation

Flow is a fully offline, low-latency speech-to-text dictation app that types into any application on your Mac. Press a hotkey to start, speak, release to stop — your words appear at the cursor instantly.

No cloud. No subscription. Runs entirely on-device.

---

## Features

- **Global dictation** — works in any app: email, Slack, VS Code, terminals, browsers
- **Hotkey control** — press Alt+Space to start/stop recording
- **Three modes**
  - *Dictation* — inserts text at cursor in the focused app
  - *Meeting* — continuous transcript capture
  - *Capture* — records thoughts without inserting text (double-tap hotkey)
- **Voice commands**
  - `"scratch that"` — undo the last dictation
  - `"polish that"` — rewrite with local LLM (optional)
- **Model tiers** — Fast (tiny.en), Balanced (distil-medium.en), Best (distil-large-v3)
- **Custom vocabulary** — heard-as → should-be replacements, auto-learned after 2 corrections
- **File & URL transcription** — drop audio/video files or paste a YouTube URL
- **Transcript history** — searchable local log of every dictation, tagged by source app
- **Captured thoughts** — separate vault for voice memos
- **Real-time metrics** — live RTFx (real-time factor) and word count per session
- **Mini bar** — 220×52 px floating window for always-on-top access

---

## Architecture

```
┌─────────────────────────────┐
│   Tauri Shell (Rust)        │  Window management, system tray,
│   src-tauri/                │  hotkey pass-through, sidecar spawn
└──────────────┬──────────────┘
               │  spawns
┌──────────────▼──────────────┐
│   flow_server (Python)      │  HTTP + WebSocket server on localhost
│   (bundled binary)          │  Whisper inference, audio capture,
│                             │  vocabulary, history, text injection
└──────────────┬──────────────┘
               │  HTTP / WS
┌──────────────▼──────────────┐
│   UI  (HTML / JS / CSS)     │  Main window + floating mini bar
│   ui/                       │  Vanilla JS, zero JS framework deps
└─────────────────────────────┘
```

**How it works end-to-end:**

1. Tauri starts the bundled `flow_server` sidecar and reads its port from `~/.flow/port`
2. The main window and mini bar both load from the Python HTTP server
3. JavaScript calls `/api/<method>` (POST/JSON) for commands
4. Real-time word streaming and state updates come over a WebSocket (`/ws`)
5. If WebSocket is unavailable, the frontend falls back to 800 ms HTTP polling

---

## Project Structure

```
flow/
├── ui/                        # Frontend — static HTML/JS/CSS
│   ├── index.html             # Main app + 5-step onboarding flow
│   ├── mini_bar.html          # Floating 220×52 control bar
│   ├── app.js                 # All UI logic (~850 lines, vanilla JS)
│   ├── style.css              # Glass-morphism design system
│   ├── bg.jpeg                # Background image
│   ├── lucide.min.js          # Icons (bundled, offline)
│   ├── InterVariable.woff2    # Inter font (bundled)
│   └── InterVariable-Italic.woff2
│
├── src-tauri/                 # Tauri Rust shell
│   ├── src/
│   │   ├── main.rs            # Entry point
│   │   └── lib.rs             # Window setup, tray, sidecar spawning
│   ├── capabilities/
│   │   └── default.json       # Tauri permission grants
│   ├── icons/                 # App icons (all sizes)
│   ├── tauri.conf.json        # App identifier, bundle config
│   ├── Cargo.toml             # Rust dependencies
│   ├── Cargo.lock
│   └── build.rs
│
├── package.json               # Tauri CLI npm wrapper
└── README.md
```

> **Note:** `src-tauri/binaries/` (the compiled `flow_server` Python executable) is excluded from git. Build it separately — see [Building the sidecar](#building-the-sidecar).

---

## Getting Started

### Prerequisites

- [Rust](https://rustup.rs/) (stable)
- [Node.js](https://nodejs.org/) ≥ 18
- [Tauri CLI v2](https://tauri.app/start/prerequisites/)

### Run in development

```bash
# Install Tauri CLI
npm install

# Place a dev build of flow_server into src-tauri/binaries/
# (see Building the sidecar below)

# Start dev mode
npm run dev
```

### Build for production

```bash
npm run build
```

The `.app` bundle will be in `src-tauri/target/release/bundle/macos/`.

---

## Building the Sidecar

The Python backend (`flow_server`) is distributed as a standalone executable bundled inside the Tauri app. It is **not** included in this repository.

To build it:

1. Have the Python source for `flow_server` available
2. Build a standalone binary with [PyInstaller](https://pyinstaller.org/):
   ```bash
   pyinstaller --onefile flow_server.py -n flow_server
   ```
3. Rename and place the output to match Tauri's expected triple:
   ```bash
   # macOS Apple Silicon
   cp dist/flow_server src-tauri/binaries/flow_server-aarch64-apple-darwin
   
   # macOS Intel
   cp dist/flow_server src-tauri/binaries/flow_server-x86_64-apple-darwin
   ```

The binary is loaded at runtime via the [Tauri sidecar API](https://tauri.app/develop/sidecar/).

---

## API Reference

The `flow_server` exposes a local HTTP + WebSocket API that the UI communicates with.

### HTTP Endpoints (`POST /api/<method>`)

| Method | Args | Returns | Description |
|---|---|---|---|
| `get_config` | — | config object | Settings + onboarding status |
| `check_permissions` | — | `{mic, accessibility}` | Check macOS permission grants |
| `trigger_mic_prompt` | — | — | Show system mic dialog |
| `trigger_accessibility_prompt` | — | — | Show accessibility dialog |
| `run_benchmark` | — | `{score}` | CPU benchmark for auto-tuning workers |
| `finish_onboarding` | — | — | Mark onboarding complete |
| `toggle_recording` | — | — | Start / stop microphone capture |
| `set_mode` | `mode: "dictation"\|"meeting"\|"capture"` | — | Switch recording mode |
| `update_config` | `{model, num_workers, threads_per_worker, hotkey, …}` | — | Save settings |
| `get_state` | — | state object | Current recording state + transcriptions |
| `get_stats` | — | `{words, time_str}` | Daily usage stats |
| `pop_toast` | — | toast string | Speed hint or low-confidence tip |
| `list_mics` | — | `[{index, name}]` | Available input devices |
| `set_mic` | `device: int\|"default"` | — | Switch input device |
| `set_launch_at_login` | `enabled: bool` | — | Auto-start on login |
| `check_model_cached` | `modelId: string` | `{cached, size_mb}` | Check if model is downloaded |
| `download_model` | `modelId: string` | — | Start model download |
| `get_download_progress` | — | `{pct, done, error}` | Model download status |
| `add_vocab` | `wrong: string, right: string` | — | Add custom replacement |
| `get_vocab` | — | `{heard: should}` | All vocabulary replacements |
| `get_history` | `limit?: int` | `[{ts, app, text}]` | Transcript history |
| `search_history` | `q: string` | `[{ts, app, text}]` | Search transcripts |
| `get_thoughts` | `limit?: int` | `[{ts, text}]` | Captured thoughts |
| `transcribe_file` | `path: string` | `{ok, text, duration, rtfx, language}` | Transcribe audio/video file |
| `transcribe_url` | `url: string` | same as above | Transcribe from URL |
| `pick_file_via_dialog` | — | `{path}` | Open file picker |
| `set_language` | `code: string` | — | Lock detection language (e.g. `"en"`) |
| `dismiss_lang` | — | — | Dismiss language notification |
| `detach_to_mini` | — | — | Switch to mini bar view |
| `accept_suggestion` | `idx: int` | — | Accept vocabulary suggestion |
| `reject_suggestion` | `idx: int` | — | Reject vocabulary suggestion |
| `get_suggestions` | — | `[{wrong, right}]` | Pending vocab suggestions |

### WebSocket (`/ws`)

Two event types:

```jsonc
// Full state update
{
  "type": "state",
  "data": {
    "recording": false,
    "mode": "dictation",
    "status": "Ready",
    "detected_lang": "en",
    "last_rtfx": 4.2,
    "transcriptions": [{ "time": "14:03", "app": "Slack", "text": "..." }],
    "speed_toast": null,
    "low_conf_tip": false
  }
}

// Word stream (during active recording)
{ "type": "words", "words": ["Hello", "world"] }
```

---

## Configuration

Settings are stored in `~/.flow/config`. The following fields are user-configurable:

| Key | Type | Default | Description |
|---|---|---|---|
| `model` | string | `"distil-medium.en"` | Whisper model ID |
| `num_workers` | int | auto | Parallel inference workers |
| `threads_per_worker` | int | auto | CPU threads per worker |
| `hotkey` | string | `"alt+space"` | Global recording hotkey |
| `regex_cleanup` | bool | `true` | Apply regex post-processing |
| `spell_check` | bool | `false` | Spell-check transcriptions |
| `llm_cleanup` | bool | `false` | Rewrite with local LLM |
| `sound_feedback` | bool | `true` | Audible start/stop feedback |
| `active_app_context` | bool | `true` | Tag transcriptions with source app |

---

## Whisper Models

| ID | Size | Speed | Quality |
|---|---|---|---|
| `tiny.en` | ~75 MB | Fastest | Good for clear speech |
| `distil-medium.en` | ~400 MB | Balanced | Recommended for most users |
| `distil-large-v3` | ~750 MB | Slower | Best accuracy |

Models are downloaded on first use and cached locally.

---

## Onboarding

First launch runs a 5-step guided setup:

1. **Welcome** — overview of Flow
2. **Permissions** — grant Microphone and Accessibility access
3. **Benchmark** — auto-tunes `num_workers` and `threads_per_worker` to your hardware
4. **Practice** — live test with the microphone
5. **Done** — launches the main app

---

## Tech Stack

| Layer | Technology |
|---|---|
| Desktop shell | Tauri v2 (Rust) |
| Frontend | Vanilla HTML / JS / CSS |
| Icons | Lucide (bundled) |
| Fonts | Inter Variable (bundled) |
| Speech engine | Whisper (via Python sidecar) |
| IPC | HTTP + WebSocket (localhost) |

---

## License

MIT
