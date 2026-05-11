#!/usr/bin/env python3
"""
Flow — FastAPI backend server
==============================
Replaces pywebview bridge. Exposes all engine methods as:
  POST /api/<method>   — JSON body {args: [...]}  → JSON result
  WS   /ws            — real-time push (words, state, toasts)
  GET  /              — serves flow_ui/index.html
  GET  /<path>        — serves flow_ui/ static files

Tauri shell (or any browser) connects to http://localhost:PORT.
"""

import os, sys, time, json, threading, asyncio, logging
from pathlib import Path
from datetime import datetime
from typing import Any

# Offline-first: no HuggingFace network requests unless explicitly downloading
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.path.insert(0, str(Path(__file__).parent))
from flow import (
    DictationEngine, LLMCleanup, Speller, HotkeyManager,
    LFM_MODEL_FILE, CONFIG_FILE, VOCAB_FILE, SUGGESTIONS_FILE, HISTORY_FILE,
    load_config, save_config, load_vocab, save_vocab,
    load_suggestions, save_suggestions, append_history, load_thoughts,
    clean_text, get_active_app, list_mics, set_launch_at_login,
    IS_MAC, IS_WIN, IS_LINUX,
)

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("flow")

# ─────────────────────────────────────────────────────────────────────────────
# ENGINE SINGLETON  (same logic as FlowApi, now framework-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

class FlowEngine:
    def __init__(self):
        self.cfg         = load_config()
        self.vocab       = load_vocab()
        self.suggestions = load_suggestions()

        self.llm     = LLMCleanup(LFM_MODEL_FILE)
        self.speller = Speller()
        self.speller.add_user_words(list(self.vocab.get("replacements", {}).values()))

        self.engine: DictationEngine = None
        self.hotkey: HotkeyManager   = None
        self._hotkey_proc    = None
        self._last_hotkey_ts = 0.0

        self._state = {
            "recording":        False,
            "status":           "Starting…",
            "last_rtfx":        0.0,
            "transcriptions":   [],
            "mode":             "dictation",
            "detected_lang":    None,
            "streaming_text":   "",
            "word_queue":       [],
            "words_today":      0,
            "seconds_saved":    0.0,
            "speed_toast":      None,
            "low_conf_tip":     False,
            "is_first_dictation": True,
        }
        self._state_lock = threading.Lock()

        # WebSocket connections — push updates to all connected clients
        self._ws_clients: list[WebSocket] = []
        self._ws_lock = threading.Lock()

        self._download_state  = {"pct": 0, "done": True, "error": None}
        self._download_thread = None

        threading.Thread(target=self._boot, daemon=True).start()

    # ── Boot ─────────────────────────────────────────────────────────────────

    def _boot(self):
        try:
            self.speller.load()
            self.engine = DictationEngine(
                cfg=self.cfg,
                llm=self.llm,
                speller=self.speller,
                on_status=self._set_status,
                on_text=self._on_transcription,
                on_control=self._on_voice_control,
            )
            self.engine.load()
            if self.cfg.get("onboarded"):
                self._start_hotkey()
                self._set_status("Ready")
            else:
                self._set_status("Welcome to Flow")
        except Exception as e:
            self._set_status(f"Boot error: {e}")
            log.exception("boot failed")

    # ── State helpers ─────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        with self._state_lock:
            self._state["status"] = msg
        self._push_state()

    def _push_state(self):
        """Push current state to all WebSocket clients (non-blocking)."""
        with self._state_lock:
            snapshot = dict(self._state)
        self._broadcast({"type": "state", "data": snapshot})

    def _push_words(self, words: list):
        self._broadcast({"type": "words", "words": words})

    def _broadcast(self, msg: dict):
        txt = json.dumps(msg, default=str)
        with self._ws_lock:
            dead = []
            for ws in self._ws_clients:
                try:
                    # Schedule coroutine on the ws's event loop
                    asyncio.run_coroutine_threadsafe(ws.send_text(txt), ws._loop)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._ws_clients.remove(ws)

    # ── Transcription callbacks ───────────────────────────────────────────────

    def _on_transcription(self, text, app, rtfx, logprob):
        is_final = (logprob == 1)
        with self._state_lock:
            self._state["last_rtfx"] = rtfx
            if is_final:
                entries = self._state["transcriptions"]
                if (entries and entries[-1].get("app") == app
                        and self._state["recording"]):
                    entries[-1]["text"] += " " + text
                else:
                    entries.append({
                        "text": text, "app": app, "rtfx": rtfx,
                        "time": datetime.now().strftime("%H:%M:%S"),
                    })
                self._state["transcriptions"] = entries[-100:]
                self._state["word_queue"] = []
                self._state["streaming_text"] = ""

                # Passive value signals
                wc = len(text.split())
                self._state["words_today"] += wc
                typing_s = wc  # ~60 WPM = 1 word/s
                self._state["seconds_saved"] += max(0, typing_s * 0.6)
                if self._state["is_first_dictation"]:
                    self._state["speed_toast"] = (
                        f"{wc} words in seconds. Typing that would've taken ~{wc}s."
                    )
                    self._state["is_first_dictation"] = False
                else:
                    self._state["speed_toast"] = f"{wc} words · ~{wc}s saved"
                if isinstance(logprob, float) and logprob < -0.55:
                    self._state["low_conf_tip"] = True
            else:
                shown = self._state["streaming_text"]
                if not shown:
                    new_words = text.split()
                elif text.startswith(shown):
                    new_words = text[len(shown):].strip().split()
                else:
                    new_words = text.split()
                    self._state["streaming_text"] = ""
                self._state["word_queue"].extend(new_words)
                # Push words immediately via WS
                if new_words:
                    self._push_words(new_words)
                    shown_upd = self._state["streaming_text"]
                    self._state["streaming_text"] = (
                        (shown_upd + " " + " ".join(new_words)).strip()
                    )
        self._push_state()

    def _on_voice_control(self, cmd: str):
        if cmd == "pause":
            self._set_status("Paused — say 'resume Flow' to continue")
        elif cmd == "resume":
            self._set_status("Ready")

    def _on_meeting_text(self, text, speaker, rtfx):
        with self._state_lock:
            self._state["transcriptions"].append({
                "text": text, "app": f"Speaker {speaker}",
                "rtfx": rtfx, "time": datetime.now().strftime("%H:%M:%S"),
                "is_meeting": True,
            })
            self._state["transcriptions"] = self._state["transcriptions"][-200:]
        self._push_state()

    def _learn_correction(self, wrong, right):
        self.vocab.setdefault("replacements", {})[wrong] = right
        save_vocab(self.vocab)
        if self.engine:
            self.engine._cur_vocab = self.vocab

    # ── Hotkey ───────────────────────────────────────────────────────────────

    def _start_hotkey(self):
        hk = self.cfg.get("hotkey", "<alt>+<space>")
        if IS_MAC:
            self._start_hotkey_daemon_mac(hk)
        else:
            try:
                from pynput import keyboard as kb
                if self.hotkey:
                    self.hotkey.stop()
                self.hotkey = HotkeyManager(
                    hotkey=hk,
                    on_press=self._on_hotkey_down,
                    on_release=lambda: None,
                )
                self.hotkey.start()
            except Exception as e:
                log.warning(f"hotkey: {e}")

    def _start_hotkey_daemon_mac(self, hotkey):
        import subprocess
        script = f"""
import sys
from pynput import keyboard as kb
combo = kb.HotKey.parse({repr(hotkey)})
current = set(); active = False
def press(key):
    global active
    try: canonical = listener.canonical(key)
    except Exception: canonical = key
    if canonical in combo:
        current.add(canonical)
        if all(k in current for k in combo) and not active:
            active = True
            sys.stdout.write("press\\n"); sys.stdout.flush()
def release(key):
    global active
    try: canonical = listener.canonical(key)
    except Exception: canonical = key
    if canonical in current:
        current.discard(canonical)
        if active: active = False
listener = kb.Listener(on_press=press, on_release=release)
listener.start(); listener.join()
"""
        if self._hotkey_proc:
            try: self._hotkey_proc.terminate()
            except Exception: pass
        self._hotkey_proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        def _reader():
            while self._hotkey_proc and self._hotkey_proc.poll() is None:
                line = self._hotkey_proc.stdout.readline().decode().strip()
                if line == "press":
                    self._on_hotkey_down()
        threading.Thread(target=_reader, daemon=True, name="hotkey-mac").start()

    def _on_hotkey_down(self):
        now = time.time()
        double_tap = (now - self._last_hotkey_ts) < 0.35
        self._last_hotkey_ts = now
        if not self.engine:
            return
        if double_tap:
            if self.engine.mode == DictationEngine.MODE_CAPTURE:
                self.engine.mode = DictationEngine.MODE_DICTATION
                self._set_status("Dictation mode")
            else:
                self.engine.mode = DictationEngine.MODE_CAPTURE
                self._set_status("Capture mode — thoughts saved, nothing inserted")
            with self._state_lock:
                self._state["mode"] = self.engine.mode
            self._push_state()
            return
        self.engine.toggle(
            vocab=self.vocab,
            vocab_callback=self._learn_correction,
            on_meeting_text=self._on_meeting_text,
        )
        with self._state_lock:
            self._state["recording"] = self.engine.is_recording()
        self._push_state()

    # ── API methods (called by HTTP handlers) ─────────────────────────────────

    def get_state(self):
        with self._state_lock:
            return dict(self._state)

    def get_config(self):
        return dict(self.cfg)

    def update_config(self, new_cfg: dict):
        needs_reload = any(
            k in new_cfg and new_cfg[k] != self.cfg.get(k)
            for k in ("model", "num_workers", "threads_per_worker")
        )
        hotkey_changed = ("hotkey" in new_cfg
                          and new_cfg["hotkey"] != self.cfg.get("hotkey"))
        self.cfg.update(new_cfg)
        save_config(self.cfg)
        if needs_reload:
            threading.Thread(target=self._reload_engine, daemon=True).start()
        if hotkey_changed:
            threading.Thread(target=self._start_hotkey, daemon=True).start()
        return True

    def _reload_engine(self):
        try:
            self._set_status("Reloading model…")
            self.engine = DictationEngine(
                cfg=self.cfg, llm=self.llm, speller=self.speller,
                on_status=self._set_status, on_text=self._on_transcription,
                on_control=self._on_voice_control,
            )
            self.engine.load()
            self._set_status("Ready")
        except Exception as e:
            self._set_status(f"Reload error: {e}")

    def toggle_recording(self):
        if self.engine:
            self.engine.toggle(
                vocab=self.vocab,
                vocab_callback=self._learn_correction,
                on_meeting_text=self._on_meeting_text,
            )
            with self._state_lock:
                self._state["recording"] = self.engine.is_recording()
            self._push_state()
        return True

    def set_mode(self, mode: str):
        if mode not in ("dictation", "meeting", "capture"):
            return False
        if self.engine:
            self.engine.mode = mode
        with self._state_lock:
            self._state["mode"] = mode
            self._state["recording"] = False
        self._push_state()
        return True

    def pop_toast(self):
        with self._state_lock:
            msg = self._state.get("speed_toast")
            tip = self._state.get("low_conf_tip", False)
            self._state["speed_toast"] = None
            self._state["low_conf_tip"] = False
        return {"toast": msg, "low_conf_tip": tip}

    def get_stats(self):
        with self._state_lock:
            w = self._state.get("words_today", 0)
            s = self._state.get("seconds_saved", 0.0)
        mins, secs = int(s // 60), int(s % 60)
        time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        return {"words": w, "seconds_saved": round(s), "time_str": time_str}

    def pop_words(self, max_count=5):
        with self._state_lock:
            q = self._state["word_queue"]
            if not q:
                return []
            count = min(max(1, len(q) // 3 + 1), max_count)
            batch, self._state["word_queue"] = q[:count], q[count:]
            self._state["streaming_text"] = (
                self._state["streaming_text"] + " " + " ".join(batch)
            ).strip()
        return batch

    def check_permissions(self):
        mic = False
        acc = False
        try:
            import sounddevice as sd
            sd.query_devices()
            mic = True
        except Exception:
            pass
        if IS_MAC:
            try:
                import subprocess
                r = subprocess.run(
                    ["osascript", "-e",
                     'tell application "System Events" to get name of first process'],
                    capture_output=True, timeout=2,
                )
                acc = r.returncode == 0
            except Exception:
                pass
        else:
            acc = True
        return {"mic": mic, "accessibility": acc}

    def trigger_mic_prompt(self):
        try:
            import sounddevice as sd
            sd.query_devices(kind="input")
        except Exception:
            pass
        return True

    def trigger_accessibility_prompt(self):
        if IS_MAC:
            import subprocess
            subprocess.run(
                ["osascript", "-e",
                 'tell application "System Events" to get name of first process'],
                capture_output=True, timeout=2,
            )
        return True

    def run_benchmark(self):
        import multiprocessing
        cores = multiprocessing.cpu_count()
        if cores >= 8:
            best, quality, model = (2, 4), "best", "distil-large-v3"
        elif cores >= 4:
            best, quality, model = (2, 3), "balanced", "distil-medium.en"
        else:
            best, quality, model = (1, 2), "fast", "tiny.en"
        self.cfg["num_workers"] = best[0]
        self.cfg["threads_per_worker"] = best[1]
        if "model" not in self.cfg:
            self.cfg["model"] = model
        save_config(self.cfg)
        return {"best": best, "cores": cores, "quality": quality, "model": model}

    def finish_onboarding(self):
        self.cfg["onboarded"] = True
        save_config(self.cfg)
        try:
            self.engine = DictationEngine(
                cfg=self.cfg, llm=self.llm, speller=self.speller,
                on_status=self._set_status, on_text=self._on_transcription,
                on_control=self._on_voice_control,
            )
            self.engine.load()
        except Exception as e:
            log.warning(f"finish_onboarding reload: {e}")
        self._start_hotkey()
        self._set_status("Ready")
        return True

    def get_vocab(self):
        return self.vocab.get("replacements", {})

    def add_vocab(self, wrong: str, right: str):
        self.vocab.setdefault("replacements", {})[wrong.lower()] = right
        save_vocab(self.vocab)
        return True

    def get_suggestions(self):
        return self.suggestions

    def accept_suggestion(self, idx: int):
        if 0 <= idx < len(self.suggestions):
            s = self.suggestions.pop(idx)
            self.vocab.setdefault("replacements", {})[s["wrong"]] = s["right"]
            save_vocab(self.vocab)
            save_suggestions(self.suggestions)
        return True

    def reject_suggestion(self, idx: int):
        if 0 <= idx < len(self.suggestions):
            self.suggestions.pop(idx)
            save_suggestions(self.suggestions)
        return True

    def get_history(self, limit=100):
        try:
            entries = []
            with open(HISTORY_FILE) as f:
                for line in f:
                    try: entries.append(json.loads(line))
                    except Exception: pass
            return entries[-limit:][::-1]
        except FileNotFoundError:
            return []

    def search_history(self, query: str):
        all_h = self.get_history(500)
        q = query.lower()
        return [e for e in all_h if q in e.get("text", "").lower()]

    def get_thoughts(self, limit=100):
        return load_thoughts(limit)

    def set_language(self, lang_code: str):
        self.cfg["language"] = lang_code
        save_config(self.cfg)
        with self._state_lock:
            self._state["detected_lang"] = None
        self._push_state()
        return True

    def dismiss_lang(self):
        with self._state_lock:
            self._state["detected_lang"] = None
        self._push_state()
        return True

    def check_model_cached(self, model_name: str):
        try:
            from faster_whisper.utils import get_assets_path
            local = Path(get_assets_path()) / model_name
            if local.exists():
                size = sum(f.stat().st_size for f in local.rglob("*") if f.is_file())
                return {"cached": True, "size_mb": round(size / 1e6)}
        except Exception:
            pass
        return {"cached": False, "size_mb": 0}

    def download_model(self, model_name: str):
        if getattr(self, "_download_thread", None) and self._download_thread.is_alive():
            return False
        self._download_state = {"model": model_name, "pct": 0, "done": False, "error": None}
        def _do():
            try:
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                self._set_status(f"Downloading {model_name}…")
                from faster_whisper import WhisperModel
                from faster_whisper.utils import get_assets_path
                expected_mb = {"tiny.en":75,"distil-medium.en":394,"distil-large-v3":756}.get(model_name, 800)
                done_ev = threading.Event()
                def _watch():
                    target = Path(get_assets_path()) / model_name
                    while not done_ev.is_set():
                        if target.exists():
                            cur = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
                            self._download_state["pct"] = min(95, int(cur / (expected_mb * 1e6) * 100))
                        time.sleep(0.5)
                threading.Thread(target=_watch, daemon=True).start()
                WhisperModel(model_name, device="cpu", compute_type="int8")
                done_ev.set()
                self._download_state.update({"pct": 100, "done": True})
                self._set_status(f"✓ {model_name} ready")
            except Exception as e:
                self._download_state["error"] = str(e)
                self._set_status(f"Download failed: {e}")
            finally:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
        self._download_thread = threading.Thread(target=_do, daemon=True)
        self._download_thread.start()
        return True

    def get_download_progress(self):
        return dict(self._download_state)

    def list_mics(self):
        return list_mics()

    def set_mic(self, device_index):
        idx = None if device_index == "default" else int(device_index)
        self.cfg["mic_device"] = idx
        save_config(self.cfg)
        return True

    def set_launch_at_login(self, enable: bool):
        result = set_launch_at_login(enable)
        if result:
            self.cfg["launch_at_login"] = enable
            save_config(self.cfg)
        return result

    def pick_file_via_dialog(self):
        # In server mode, file picking is handled by Tauri's dialog API
        # Returns None — frontend uses Tauri's open() dialog directly
        return None

    def transcribe_file(self, path: str):
        import time as _t
        try:
            self._set_status(f"Transcribing {Path(path).name}…")
            t0 = _t.perf_counter()
            if not self.engine or not self.engine._model:
                return {"ok": False, "error": "Engine not ready"}
            import soundfile as sf
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
            if sr != 16000:
                import resampy
                audio = resampy.resample(audio, sr, 16000)
            segs, info = self.engine._model.transcribe(audio, beam_size=1, vad_filter=True)
            text = " ".join(s.text.strip() for s in segs)
            elapsed = _t.perf_counter() - t0
            duration = len(audio) / 16000
            self._set_status("Ready")
            return {
                "ok": True, "text": text,
                "duration": duration, "elapsed": elapsed,
                "rtfx": duration / elapsed if elapsed > 0 else 0,
                "language": info.language,
            }
        except Exception as e:
            self._set_status("Ready")
            return {"ok": False, "error": str(e)}

    def transcribe_url(self, url: str):
        return {"ok": False, "error": "URL transcription not supported in server mode yet"}

    def detach_to_mini(self):
        return True  # handled by Tauri shell

    def show_main_window(self):
        return True  # handled by Tauri shell

    def set_language_code(self, code: str):
        return self.set_language(code)


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

engine = FlowEngine()
app = FastAPI(title="Flow", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UI_DIR = Path(__file__).parent / "flow_ui"

# ── Health check (must be before /{path:path} catchall) ──────────────────────

@app.get("/health")
async def health():
    return {"ok": True, "status": engine._state.get("status", "?")}

# ── Static files ─────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(UI_DIR / "index.html")

@app.get("/{path:path}")
async def static(path: str):
    file = UI_DIR / path
    if file.exists() and file.is_file():
        return FileResponse(file)
    return FileResponse(UI_DIR / "index.html")

# ── API — generic method dispatcher ─────────────────────────────────────────

@app.post("/api/{method}")
async def api_call(method: str, body: dict = {}):
    """
    Universal dispatcher. Body: {"args": [...]} or {"kwargs": {...}}.
    Returns {"result": <value>} or {"error": <msg>}.
    """
    fn = getattr(engine, method, None)
    if fn is None or method.startswith("_"):
        return JSONResponse({"error": f"Unknown method: {method}"}, status_code=404)
    try:
        args   = body.get("args", [])
        kwargs = body.get("kwargs", {})
        # Run sync methods in thread pool to avoid blocking the event loop
        import asyncio
        from fastapi.concurrency import run_in_threadpool
        result = await run_in_threadpool(fn, *args, **kwargs)
        return {"result": result}
    except Exception as e:
        log.exception(f"api_call {method}")
        return JSONResponse({"error": str(e)}, status_code=500)

# ── WebSocket — real-time push ────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    # Attach event loop so the sync broadcast can schedule coroutines
    ws._loop = asyncio.get_event_loop()
    with engine._ws_lock:
        engine._ws_clients.append(ws)
    try:
        # Send current state immediately on connect
        with engine._state_lock:
            snapshot = dict(engine._state)
        await ws.send_text(json.dumps({"type": "state", "data": snapshot}, default=str))
        # Keep connection alive — server pushes; client can send pings
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=30)
                if msg == "ping":
                    await ws.send_text('{"type":"pong"}')
            except asyncio.TimeoutError:
                await ws.send_text('{"type":"ping"}')
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        with engine._ws_lock:
            if ws in engine._ws_clients:
                engine._ws_clients.remove(ws)

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def find_free_port(start=7878, end=7999) -> int:
    import socket
    for port in range(start, end):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start

def write_port_file(port: int):
    """Write port to ~/.flow/port so Tauri shell can find it."""
    from flow import CONFIG_DIR
    (CONFIG_DIR / "port").write_text(str(port))

def _cleanup_all():
    """Kill all child processes so no orphans remain after server exits."""
    try:
        if engine and engine.engine:
            eng = engine.engine
            if hasattr(eng, '_kill_parakeet'):
                eng._kill_parakeet()
            if hasattr(eng, '_mic_proc') and eng._mic_proc:
                try: eng._mic_proc.kill()
                except Exception: pass
    except Exception:
        pass
    # Kill direct children only (not the whole process group — that would kill ourselves)
    try:
        import signal as _sig, subprocess as _sp
        result = _sp.run(["pgrep", "-P", str(os.getpid())], capture_output=True, text=True)
        for pid in result.stdout.split():
            try: os.kill(int(pid), _sig.SIGTERM)
            except Exception: pass
    except Exception:
        pass

import atexit
atexit.register(_cleanup_all)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else find_free_port()
    write_port_file(port)
    print(f"[flow] server starting on http://127.0.0.1:{port}", flush=True)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
