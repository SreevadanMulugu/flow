#!/usr/bin/env python3
"""
Flow — Local Voice Dictation (Cross-Platform)
================================================
Offline, pay-once voice dictation for macOS / Windows / Linux.

Pitch:
  1. Works on the plane — offline always
  2. Pay once, own forever — anti-subscription
  3. Learns your jargon — vocabulary that grows with you
  4. Faster than Wispr — distil-large-v3 + 5-trick stack

Stack:
  • faster-whisper distil-large-v3 int8 (shared model, num_workers)
  • Two-pass router + regex cleanup (Tier 1)
  • Optional Gemma-3-270M Tier 2 LLM cleanup (llama-cpp-python)
  • Silero VAD, Whisper VAD filter
  • Global hotkey (push-to-talk)
  • System-wide cursor insertion (clipboard paste)
  • Vocabulary learning (manual + "scratch that" auto-learn)
  • Cross-platform tray icon (pystray)
  • Onboarding flow (permissions → benchmark → practice)
  • Linear design language

Run: python flow.py
"""

import os, sys, json, time, re, threading, queue, gc, platform
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, List, Tuple

import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext

IS_MAC     = sys.platform == "darwin"
IS_WIN     = sys.platform.startswith("win")
IS_LINUX   = sys.platform.startswith("linux")


# ─────────────────────────────────────────────────────────────────────────────
# LINEAR DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────

class Theme:
    MARKETING_BLACK = "#08090a"
    PANEL_DARK      = "#0f1011"
    LEVEL3          = "#191a1b"
    SECONDARY       = "#28282c"
    TEXT_PRIMARY    = "#f7f8f8"
    TEXT_SECONDARY  = "#d0d6e0"
    TEXT_TERTIARY   = "#8a8f98"
    TEXT_QUATERNARY = "#62666d"
    INDIGO          = "#5e6ad2"
    VIOLET          = "#7170ff"
    VIOLET_HOVER    = "#828fff"
    SUCCESS         = "#27a644"
    EMERALD         = "#10b981"
    ERROR           = "#ef4444"
    WARN            = "#f59e0b"
    BORDER_PRIMARY  = "#23252a"
    BORDER_SECOND   = "#34343a"

    # macOS Tk has font-rendering bugs with custom font names — use Helvetica
    # which is the system fallback and always renders correctly.
    if IS_MAC:
        UI_FONT = "Helvetica"
        MONO    = "Menlo"
    elif IS_WIN:
        UI_FONT = "Segoe UI"
        MONO    = "Consolas"
    else:
        UI_FONT = "TkDefaultFont"
        MONO    = "TkFixedFont"

    FONT_H1    = (UI_FONT, 26, "bold")
    FONT_H2    = (UI_FONT, 18, "bold")
    FONT_H3    = (UI_FONT, 15, "bold")
    FONT_BODY  = (UI_FONT, 13)
    FONT_UI    = (UI_FONT, 12)
    FONT_SMALL = (UI_FONT, 11)
    FONT_MICRO = (UI_FONT, 10)
    FONT_MONO  = (MONO, 11)

    S_XS = 4; S_SM = 8; S_MD = 16; S_LG = 24; S_XL = 32


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG + STORAGE
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_DIR = Path.home() / ".flow"
CONFIG_DIR.mkdir(exist_ok=True)
MODELS_DIR = CONFIG_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

CONFIG_FILE       = CONFIG_DIR / "config.json"
VOCAB_FILE        = CONFIG_DIR / "vocabulary.json"
HISTORY_FILE      = CONFIG_DIR / "history.jsonl"
SUGGESTIONS_FILE  = CONFIG_DIR / "vocab_suggestions.json"
THOUGHTS_FILE     = CONFIG_DIR / "thoughts.jsonl"


def list_mics() -> list:
    """Return [{index, name, channels}] for all input devices."""
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        result = []
        for i, d in enumerate(devs):
            if d["max_input_channels"] > 0:
                result.append({"index": i, "name": d["name"], "channels": d["max_input_channels"]})
        return result
    except Exception:
        return []


def set_launch_at_login(enable: bool) -> bool:
    """Register / unregister Flow to launch automatically on login."""
    try:
        if getattr(sys, "frozen", False):
            prog = [sys.executable]
        else:
            prog = [sys.executable, str(Path(__file__).parent / "flow_app.py")]

        if IS_MAC:
            plist_dir = Path.home() / "Library" / "LaunchAgents"
            plist_dir.mkdir(parents=True, exist_ok=True)
            plist = plist_dir / "com.flow.app.plist"
            if enable:
                args = "".join(f"<string>{a}</string>" for a in prog)
                plist.write_text(
                    f'<?xml version="1.0" encoding="UTF-8"?>\n'
                    f'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
                    f'"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
                    f'<plist version="1.0"><dict>'
                    f'<key>Label</key><string>com.flow.app</string>'
                    f'<key>ProgramArguments</key><array>{args}</array>'
                    f'<key>RunAtLoad</key><true/>'
                    f'<key>KeepAlive</key><false/>'
                    f'</dict></plist>\n'
                )
            else:
                plist.unlink(missing_ok=True)
            return True

        elif IS_WIN:
            import winreg
            reg_key = r"Software\Microsoft\Windows\CurrentVersion\Run"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_key, 0,
                                winreg.KEY_SET_VALUE) as k:
                if enable:
                    cmd = " ".join(f'"{p}"' for p in prog)
                    winreg.SetValueEx(k, "Flow", 0, winreg.REG_SZ, cmd)
                else:
                    try:
                        winreg.DeleteValue(k, "Flow")
                    except FileNotFoundError:
                        pass
            return True

        elif IS_LINUX:
            desktop_dir = Path.home() / ".config" / "autostart"
            desktop_dir.mkdir(parents=True, exist_ok=True)
            desktop = desktop_dir / "flow.desktop"
            if enable:
                exec_cmd = " ".join(prog)
                desktop.write_text(
                    "[Desktop Entry]\nType=Application\nName=Flow\n"
                    f"Exec={exec_cmd}\nHidden=false\n"
                    "X-GNOME-Autostart-enabled=true\n"
                )
            else:
                desktop.unlink(missing_ok=True)
            return True

    except Exception as e:
        print(f"[launch_at_login] {e}")
    return False


def append_thought(text: str):
    entry = {"ts": datetime.now().isoformat(), "text": text}
    with open(THOUGHTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_thoughts(limit: int = 100) -> list:
    if not THOUGHTS_FILE.exists():
        return []
    lines = THOUGHTS_FILE.read_text().splitlines()
    results = []
    for line in reversed(lines[-limit * 2:]):
        try:
            results.append(json.loads(line))
        except Exception:
            pass
    return results[:limit]

DEFAULT_CONFIG = {
    "model": "parakeet-tdt-0.6b-v3",
    "threads_per_worker": 3,
    "num_workers": 2,
    "hotkey": "<alt>+<space>",
    "insert_method": "clipboard_paste",
    "sound_feedback": True,
    "regex_cleanup": True,
    "llm_cleanup": False,
    "llm_model_path": "",                  # Gemma-3-270M-IT GGUF
    "active_app_context": True,
    "onboarded": False,
    "paused_until": 0,
    "benchmark_rtfx": 0.0,
    "language": "en",
    "spell_check": True,                   # Tier 2 SymSpell
    "mic_device": None,                    # None = system default; int = device index
    "launch_at_login": False,
}

# App category → controls clean_text behaviour (punctuation, capitalization, formality)
# "formal"   → full punctuation, sentence caps, grammar fixes (email/docs default)
# "chat"     → no trailing period, contractions kept, lowercase ok
# "code"     → no trailing period, no sentence caps, no grammar expansions
# "terminal" → raw passthrough — zero cleanup
# "notes"    → minimal: fillers only, preserve voice
APP_CATEGORIES = {
    # chat
    "Slack": "chat", "Discord": "chat", "Messages": "chat",
    "WhatsApp": "chat", "Telegram": "chat", "Signal": "chat",
    "Teams": "chat", "Zoom": "chat", "Twitter": "chat",
    "X.com": "chat", "Instagram": "chat",
    # formal
    "Mail": "formal", "Outlook": "formal", "Gmail": "formal",
    "Spark": "formal", "Airmail": "formal", "Mimestream": "formal",
    "Word": "formal", "Pages": "formal", "Docs": "formal",
    # notes
    "Notes": "notes", "Obsidian": "notes", "Notion": "notes",
    "Bear": "notes", "Craft": "notes", "Logseq": "notes",
    "Roam": "notes", "Capacities": "notes",
    # code
    "VS Code": "code", "Code": "code", "Xcode": "code",
    "IntelliJ": "code", "PyCharm": "code", "WebStorm": "code",
    "Cursor": "code", "Vim": "code", "Neovim": "code",
    "Emacs": "code",
    # terminal
    "Terminal": "terminal", "iTerm2": "terminal", "Hyper": "terminal",
    "Warp": "terminal", "Alacritty": "terminal",
}

# LLM hint strings (only used by Tier 3 polish, not Tier 1 cleanup)
PER_APP_STYLES = {
    "chat":     "casual, short sentences, contractions OK",
    "formal":   "professional tone, full punctuation",
    "notes":    "preserve voice, minimal editing",
    "code":     "technical, preserve identifiers",
    "terminal": "terse, command-line style",
}


def get_app_category(app_name: str) -> str:
    if not app_name:
        return "formal"
    for key, cat in APP_CATEGORIES.items():
        if key.lower() in app_name.lower():
            return cat
    return "formal"

def load_json(path, default):
    if path.exists():
        try:
            return {**default, **json.loads(path.read_text())}
        except Exception:
            return dict(default)
    return dict(default)

def save_json(path, data):
    path.write_text(json.dumps(data, indent=2))

def load_config()  -> dict: return load_json(CONFIG_FILE, DEFAULT_CONFIG)
def save_config(c) -> None: save_json(CONFIG_FILE, c)
def load_vocab()   -> dict: return load_json(VOCAB_FILE, {"replacements": {}, "hotwords": []})
def save_vocab(v):          save_json(VOCAB_FILE, v)
def load_suggestions() -> list:
    if SUGGESTIONS_FILE.exists():
        try:   return json.loads(SUGGESTIONS_FILE.read_text())
        except Exception: return []
    return []
def save_suggestions(s: list): SUGGESTIONS_FILE.write_text(json.dumps(s, indent=2))

def append_history(text: str, app: str = ""):
    entry = {"ts": datetime.now().isoformat(), "text": text, "app": app}
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# TIER 1 CLEANUP — Regex + vocabulary
# ─────────────────────────────────────────────────────────────────────────────

VOICE_COMMANDS = {
    "new line":       "\n",
    "new paragraph":  "\n\n",
    "period":         ".",
    "comma":          ",",
    "question mark":  "?",
    "exclamation mark": "!",
    "exclamation":    "!",
    "colon":          ":",
    "semicolon":      ";",
    "open quote":     '"',
    "close quote":    '"',
    "open paren":     "(",
    "close paren":    ")",
    "dash":           " — ",
}

SAFE_FILLERS = re.compile(
    r"\b(um+|uh+|er+|ah+|hmm+|mhm+|mmm+)\b(,?\s*)",
    re.IGNORECASE,
)
REPEATED_WORD = re.compile(r"\b(\w+)(\s+\1\b)+", re.IGNORECASE)

# Common Whisper / speech mistakes (deterministic fixes)
GRAMMAR_FIXES = [
    (re.compile(r"\bi\b"),        "I"),
    (re.compile(r"\bi'm\b"),      "I'm"),
    (re.compile(r"\bi've\b"),     "I've"),
    (re.compile(r"\bi'll\b"),     "I'll"),
    (re.compile(r"\bi'd\b"),      "I'd"),
    (re.compile(r"\bwanna\b"),    "want to"),
    (re.compile(r"\bgonna\b"),    "going to"),
    (re.compile(r"\bgotta\b"),    "have to"),
    (re.compile(r"\bkinda\b"),    "kind of"),
    (re.compile(r"\bsorta\b"),    "sort of"),
]

# Sentence-initial capitalization after . ! ?
SENTENCE_START = re.compile(r"([.!?]\s+)([a-z])")

DELETE_TRIGGERS = [
    "delete that", "scratch that", "ignore that", "strike that",
    "never mind that",
]

# Explicit rewrite/polish commands — Tier 2 LLM fires ONLY when user says one.
# Otherwise transcription is inserted verbatim (after Tier 1 regex cleanup).
REWRITE_TRIGGERS = [
    "polish that",
    "rewrite that",
    "clean that up",
    "make that professional",
    "fix that",
    "make that formal",
    "make that casual",
]

# Global voice control commands — these take over the whole engine,
# never insert anything into the active field.
CONTROL_PAUSE_TRIGGERS = [
    "pause flow",
    "flow pause",
    "hey flow pause",
]
CONTROL_RESUME_TRIGGERS = [
    "resume flow",
    "flow resume",
    "hey flow resume",
]


def detect_control_command(text: str) -> Optional[str]:
    """
    Returns 'pause' / 'resume' / None.
    Match is done on a lowered, punctuation-stripped version.
    """
    low = re.sub(r"[^\w\s]", "", text.lower()).strip()
    for trig in CONTROL_PAUSE_TRIGGERS:
        if trig in low:
            return "pause"
    for trig in CONTROL_RESUME_TRIGGERS:
        if trig in low:
            return "resume"
    return None


def clean_text(raw: str,
               vocab: Optional[dict] = None,
               speller: Optional["Speller"] = None,
               category: str = "formal") -> str:
    """
    Context-aware cleanup pipeline. Behaviour adapts per app category:
      formal   — full punctuation, sentence caps, grammar fixes (email / docs)
      chat     — no trailing period, contractions kept, lowercase OK (Slack / iMessage)
      notes    — filler removal only, voice preserved (Notes / Obsidian)
      code     — no trailing period, no caps, no grammar expansions (VS Code)
      terminal — raw passthrough, nothing touched
    """
    if not raw:
        return raw

    if category == "terminal":
        return raw.strip()

    text = raw.strip()

    # 1. Vocabulary replacements (all categories)
    if vocab:
        for wrong, right in vocab.get("replacements", {}).items():
            text = re.sub(rf"\b{re.escape(wrong)}\b", right, text,
                          flags=re.IGNORECASE)

    # 2. Voice commands (all except terminal — already returned)
    for phrase, replacement in VOICE_COMMANDS.items():
        text = re.sub(rf"\b{re.escape(phrase)}\b", replacement, text,
                      flags=re.IGNORECASE)

    # 3. Filler removal (all categories)
    text = SAFE_FILLERS.sub("", text)

    if category == "notes":
        # Notes: fillers gone, everything else preserved as-is
        text = re.sub(r"  +", " ", text).strip()
        return text

    # 4. Collapse repeated words
    text = REPEATED_WORD.sub(r"\1", text)

    # 5. Grammar fixes — skip for code (wanna→want to would mangle identifiers)
    if category not in ("code",):
        for pattern, repl in GRAMMAR_FIXES:
            text = pattern.sub(repl, text)

    # 6. Spelling correction — skip for code (identifiers would get mangled)
    if speller is not None and category not in ("code", "terminal"):
        text = speller.correct(text)

    # 7. Capitalization — skip for code and chat
    text = text.strip()
    if category not in ("code", "chat"):
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        text = SENTENCE_START.sub(
            lambda m: m.group(1) + m.group(2).upper(), text,
        )

    # 8. Whitespace + punctuation spacing (all)
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:])([A-Za-z])", r"\1 \2", text)

    # 9. Trailing period — only for formal category
    if (category == "formal"
            and text and text[-1] not in ".!?,;:\"'"
            and len(text.split()) >= 3):
        text += "."

    return text


def detect_delete_trigger(text: str) -> bool:
    low = text.lower()
    return any(t in low for t in DELETE_TRIGGERS)

def strip_delete_trigger(text: str) -> str:
    """Remove 'scratch that' etc and return what comes after."""
    low = text.lower()
    for trig in DELETE_TRIGGERS:
        idx = low.find(trig)
        if idx >= 0:
            after = text[idx + len(trig):].lstrip(" ,.—-")
            return after
    return text


def detect_rewrite_trigger(text: str) -> Optional[str]:
    """
    Returns the trigger phrase if user explicitly asked for AI polish.
    e.g. "polish that" → returns "polish that"
    """
    low = text.lower()
    for trig in REWRITE_TRIGGERS:
        if trig in low:
            return trig
    return None


def strip_rewrite_trigger(text: str) -> str:
    """Remove 'polish that' etc — return what the user actually said."""
    low = text.lower()
    for trig in REWRITE_TRIGGERS:
        idx = low.find(trig)
        if idx >= 0:
            before = text[:idx].rstrip(" ,.")
            after  = text[idx + len(trig):].lstrip(" ,.")
            # Usually the content is BEFORE the command ("write the email polish that")
            # but sometimes after ("polish that write the email")
            return (before + " " + after).strip() or text
    return text


# ─────────────────────────────────────────────────────────────────────────────
# TIER 2 — SymSpell spelling correction
#
# Dictionary-based. Uses edit-distance precomputed lookup. ~1M words/sec.
# Preserves known vocabulary words (user's custom entries won't be "corrected").
# Total RAM cost: ~15MB.
# ─────────────────────────────────────────────────────────────────────────────

class Speller:
    """Wraps SymSpell for fast English spell correction."""

    def __init__(self):
        self._sym = None
        self._ready = False
        self._user_words = set()    # words to never "correct"

    def load(self):
        if self._ready:
            return
        try:
            from symspellpy import SymSpell, Verbosity
            import pkg_resources

            sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            dict_path = pkg_resources.resource_filename(
                "symspellpy", "frequency_dictionary_en_82_765.txt"
            )
            sym.load_dictionary(dict_path, term_index=0, count_index=1)
            self._sym = sym
            self._ready = True
        except Exception as e:
            print(f"[speller] load failed: {e}")
            self._sym = None

    def add_user_words(self, words):
        """Words in user vocabulary should not be 'corrected' away."""
        for w in words:
            if w:
                self._user_words.add(w.lower())
                if self._sym is not None:
                    # Register with high frequency so it wins the ranking
                    try:
                        self._sym.create_dictionary_entry(w, 10_000_000)
                    except Exception:
                        pass

    def correct(self, text: str) -> str:
        if not self._ready:
            self.load()
        if self._sym is None:
            return text

        from symspellpy import Verbosity
        # Correct word-by-word, preserving punctuation and casing
        def fix_word(match):
            word = match.group(0)
            lower = word.lower()
            # Skip very short / numeric / punctuation words
            if len(lower) <= 2 or not lower.isalpha():
                return word
            # Skip user vocab words
            if lower in self._user_words:
                return word
            # Lookup
            try:
                suggestions = self._sym.lookup(
                    lower, Verbosity.TOP, max_edit_distance=2,
                    include_unknown=True,
                )
                if not suggestions:
                    return word
                best = suggestions[0].term
                # Only apply if different and not a massive change
                if best != lower and len(best) <= len(lower) + 2:
                    # Preserve original capitalization
                    if word[0].isupper():
                        best = best[0].upper() + best[1:]
                    return best
            except Exception:
                pass
            return word

        return re.sub(r"\b[a-zA-Z]+\b", fix_word, text)


# ─────────────────────────────────────────────────────────────────────────────
# TIER 3 CLEANUP — LFM2.5-350M via llama-cpp-python (OPT-IN ONLY)
#
# LFM2-350M (Liquid AI, Nov 2025) — hybrid conv+attention, CPU-optimised.
# Beats Gemma-3-270M on IFEval (65.1 vs 51.2) which is what matters for
# cleanup fidelity (avoids hallucinated rewrites). Runs alongside Whisper
# in <600MB RAM, ~30-60 tok/s on 4-core i5.
# Gated: only fires on ~15-25% of utterances where rules aren't enough.
# ─────────────────────────────────────────────────────────────────────────────

LFM_MODEL_URL = (
    "https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/"
    "LFM2.5-350M-Q4_K_M.gguf"
)
LFM_MODEL_FILE = MODELS_DIR / "LFM2.5-350M-Q4_K_M.gguf"


class LLMCleanup:
    """
    Lazy-loaded Gemma-3-270M for transcript cleanup.
    Gated: only fires when raw text is long enough AND has confidence markers.
    """
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self._llm = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.model_path.exists()

    def load(self):
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=512,
                n_threads=2,          # leave room for Whisper
                n_batch=64,
                verbose=False,
            )
        except Exception as e:
            print(f"[llm] load failed: {e}")
            self._llm = None

    def cleanup(self, raw: str, style_hint: str = "") -> str:
        """Clean up a transcript. Returns raw if LLM unavailable or fails."""
        if not self.available:
            return raw
        with self._lock:
            if self._llm is None:
                self.load()
            if self._llm is None:
                return raw

            style = style_hint or "natural written style"
            # Use create_chat_completion so llama.cpp applies the model's
            # native chat template from the GGUF metadata. Trying to hand-roll
            # the template produces garbage (LFM2 != ChatML).
            try:
                out = self._llm.create_chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a voice transcript cleaner. "
                                "Your ONLY job: remove filler words (um, uh, "
                                "you know, I mean, like, sort of), fix "
                                "punctuation, fix capitalization. "
                                "NEVER paraphrase. NEVER add new words. "
                                "NEVER change meaning. "
                                f"Target style: {style}. "
                                "Output ONLY the cleaned sentence. "
                                "No preamble, no explanation."
                            ),
                        },
                        {"role": "user", "content": raw},
                    ],
                    max_tokens=128,
                    temperature=0.1,
                    top_p=0.9,
                )
                cleaned = out["choices"][0]["message"]["content"].strip()
                # Strip common LLM preamble
                for prefix in [
                    "Here is the cleaned sentence:",
                    "Here's the cleaned sentence:",
                    "Cleaned:", "Output:",
                ]:
                    if cleaned.startswith(prefix):
                        cleaned = cleaned[len(prefix):].strip()
                # Strip quotes if LLM wrapped the output
                if cleaned.startswith('"') and cleaned.endswith('"'):
                    cleaned = cleaned[1:-1]
                return cleaned or raw
            except Exception as e:
                print(f"[llm] cleanup failed: {e}")
                return raw


# NOTE: We deliberately do NOT auto-trigger the LLM based on Whisper confidence.
# Auto-rewriting emails would be a massive trust violation.
#
# Policy: LLM fires ONLY when:
#   1. User says an explicit rewrite trigger ("polish that", "rewrite that"), OR
#   2. User opens the "Polish last" menu item in the tray
#
# Everything else = verbatim transcription + Tier 1 regex cleanup only.


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM — active app detection (cross-platform)
# ─────────────────────────────────────────────────────────────────────────────

def get_active_app() -> str:
    try:
        if IS_MAC:
            from AppKit import NSWorkspace
            app = NSWorkspace.sharedWorkspace().frontmostApplication()
            return app.localizedName() if app else ""
        elif IS_WIN:
            import ctypes
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            length = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            # Strip document name — rough app guess
            title = buf.value
            for sep in [" — ", " - ", " – "]:
                if sep in title:
                    title = title.split(sep)[-1]
            return title
        elif IS_LINUX:
            import subprocess
            try:
                out = subprocess.check_output(
                    ["xdotool", "getactivewindow", "getwindowname"],
                    stderr=subprocess.DEVNULL, timeout=1,
                ).decode().strip()
                return out
            except Exception:
                return ""
    except Exception:
        pass
    return ""


def get_style_for_app(app_name: str) -> str:
    cat = get_app_category(app_name)
    return PER_APP_STYLES.get(cat, "")


# ─────────────────────────────────────────────────────────────────────────────
# CURSOR INSERTION (cross-platform)
# ─────────────────────────────────────────────────────────────────────────────

def insert_text(text: str, method: str = "clipboard_paste"):
    if not text:
        return

    if IS_MAC:
        # On macOS pynput.keyboard.Controller goes through ctypes → CGEvent →
        # TSMGetInputSourceProperty which asserts main-queue and crashes from
        # any background thread. Use subprocess-only approach.
        try:
            import subprocess
            subprocess.run(["pbcopy"], input=text.encode("utf-8"),
                           timeout=3, check=True)
            result = subprocess.run(
                ["osascript", "-e",
                 'tell application "System Events" to keystroke "v" '
                 'using {command down}'],
                timeout=3, capture_output=True,
            )
            if result.returncode != 0:
                err = result.stderr.decode().strip()
                if "not allowed" in err.lower() or "accessibility" in err.lower():
                    # Accessibility permission not granted — raise so caller can surface it
                    raise PermissionError(
                        "Accessibility access required. "
                        "Go to System Settings → Privacy & Security → Accessibility → enable Flow."
                    )
        except PermissionError:
            raise
        except Exception as e:
            print(f"[insert] mac paste failed: {e}")
        return

    # ── Windows / Linux ──────────────────────────────────────────────────────
    if method == "clipboard_paste":
        try:
            import pyperclip
            from pynput.keyboard import Controller, Key

            prev = ""
            try:
                prev = pyperclip.paste()
            except Exception:
                pass

            pyperclip.copy(text)
            time.sleep(0.03)

            kb = Controller()
            with kb.pressed(Key.ctrl):
                kb.press("v")
                kb.release("v")

            def restore():
                time.sleep(0.5)
                try:
                    if prev:
                        pyperclip.copy(prev)
                except Exception:
                    pass
            threading.Thread(target=restore, daemon=True).start()
        except Exception as e:
            print(f"[insert] clipboard paste failed: {e}")
    else:
        try:
            from pynput.keyboard import Controller
            Controller().type(text)
        except Exception as e:
            print(f"[insert] typewrite failed: {e}")


def play_beep(up: bool = True):
    """Short platform-native beep on record start/stop."""
    try:
        if IS_MAC:
            os.system("afplay /System/Library/Sounds/Tink.aiff &" if up
                      else "afplay /System/Library/Sounds/Pop.aiff &")
        elif IS_WIN:
            import winsound
            winsound.Beep(1200 if up else 900, 60)
        else:
            sys.stdout.write("\a"); sys.stdout.flush()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL HOTKEY
# ─────────────────────────────────────────────────────────────────────────────

class HotkeyManager:
    def __init__(self, hotkey_str: str,
                 on_press: Callable, on_release: Callable):
        self.hotkey_str = hotkey_str
        self.on_press   = on_press
        self.on_release = on_release
        self._listener  = None
        self._active    = False
        self._paused    = False

    def pause(self):   self._paused = True
    def resume(self):  self._paused = False

    def start(self):
        try:
            from pynput import keyboard
            combo = keyboard.HotKey.parse(self.hotkey_str)
            current = set()

            def press(key):
                if self._paused: return
                try:
                    canonical = self._listener.canonical(key)
                except Exception:
                    canonical = key
                if canonical in combo:
                    current.add(canonical)
                    if all(k in current for k in combo) and not self._active:
                        self._active = True
                        self.on_press()

            def release(key):
                try:
                    canonical = self._listener.canonical(key)
                except Exception:
                    canonical = key
                if canonical in current:
                    current.discard(canonical)
                    if self._active:
                        self._active = False
                        self.on_release()

            self._listener = keyboard.Listener(on_press=press, on_release=release)
            self._listener.daemon = True
            self._listener.start()
        except Exception as e:
            print(f"[hotkey] start failed: {e}")
            if IS_MAC:
                print("[hotkey] Grant Accessibility permission in System Settings")

    def stop(self):
        if self._listener:
            try: self._listener.stop()
            except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
# DICTATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class DictationEngine:
    """
    Toggle-recorded streaming engine.
    - Click hotkey or mic button to START recording
    - Audio is buffered continuously
    - Every CHUNK_SECS (or on VAD silence boundary), flush buffer to Whisper
    - Cleaned text is inserted at cursor immediately (dictation mode)
      OR appended to in-app feed (meeting mode)
    - Click again to STOP
    """
    RATE         = 16000
    CHUNK        = 1024
    # TUMBLING WINDOW: grow buffer, transcribe the full thing every ROUND_S.
    # Whisper's encoder always processes a 30s mel window — so a 2s chunk
    # costs the same as a 25s chunk. Longer buffer = astronomically better RTFx.
    ROUND_S      = 2.5      # transcribe every 2.5s (MLX encoder takes ~2.5s)
    FIRST_ROUND  = 0.8      # first transcription fires after just 0.8s of speech
    MAX_BUF_S    = 25.0     # cap buffer (stay within single 30s encoder window)
    SILENCE_S    = 2.0      # 2s real silence before commit — never cuts mid-speech
    VAD_RMS      = 0.020    # silence threshold — raised from 0.013 to reduce false mid-sentence cuts
    MIN_FLUSH_S  = 0.5      # transcribe even short audio (MLX handles it fine)

    MODE_DICTATION = "dictation"
    MODE_MEETING   = "meeting"
    MODE_CAPTURE   = "capture"   # save thought, don't insert anywhere

    def __init__(self, cfg: dict,
                 llm: LLMCleanup,
                 speller: Speller,
                 on_status: Callable,
                 on_text:   Callable,
                 on_control: Optional[Callable] = None):
        self.cfg = cfg
        self.llm = llm
        self.speller = speller
        self.on_status = on_status
        self.on_text   = on_text
        self.on_control = on_control

        self.mode = self.MODE_DICTATION   # toggle to MODE_MEETING for meetings

        # Streaming state
        self._model     = None
        self._audio_q   = queue.Queue()    # raw chunks from mic callback
        self._stream    = None
        self._rec       = False
        self._lock      = threading.Lock()

        # Stream worker controls
        self._stream_thread: Optional[threading.Thread] = None

        # Meeting mode buffer for full audio + speaker tracking
        self._meeting_buf: List[np.ndarray] = []
        self._last_inserted = ""
        self._last_speaker  = 0
        self._silence_gap_s = 0.0
        self._on_meeting_text: Optional[Callable] = None

        self._buf   = []
        self._pa = None
        self._last_raw      = ""
        # Thread pool for parallel chunk transcription — transcribe while
        # still recording the next chunk (eliminates wait-for-inference gap)
        from concurrent.futures import ThreadPoolExecutor
        self._transcribe_pool = ThreadPoolExecutor(
            max_workers=max(1, self.cfg.get("num_workers", 1))
        )

        # Silero VAD (optional — speech pre-filter before Whisper, eliminates silence hallucinations)
        self._silero_vad    = None
        self._silero_get_ts = None
        self._torch         = None

        # Two-pass confidence router (optional — tiny.en drafts, full model only on uncertain chunks)
        self._draft_model = None

        # Parakeet TDT 0.6B v3 subprocess (Python 3.10 + NeMo venv)
        self._parakeet_proc = None
        self._parakeet_lock = threading.Lock()

        # Auto-vocabulary learning: track "scratch that" corrections.
        # {wrong_phrase → {correction → count}}. At count=2, auto-add to vocab.
        self._correction_counter: dict = {}

    # Known Whisper hallucination phrases (output on silence/noise)
    HALLUCINATIONS = {
        "thank you", "thanks for watching", "thanks for listening",
        "subscribe", "like and subscribe", "see you next time",
        "bye", "goodbye", "the end", "you", "thank you so much",
        "thanks", "i'll see you in the next video",
        "please subscribe", "thank you for watching",
        "click the mic", "your dictations", "meeting transcripts",
        "below to record", "to start dictating",
        "once upon a time", "in the door", "in the dog",
        "clip the mic", "the mic in", "to record",
        "i'm going to", "let's get started", "welcome back",
    }

    # Additional hallucination detection: if output contains mostly
    # repeated words or very low avg_logprob, it's garbage
    @staticmethod
    def _is_hallucination(text, avg_logprob=-999):
        if not text or len(text.strip()) < 2:
            return True
        low = text.lower().strip('.,!?  ')
        # Check exact match
        if low in DictationEngine.HALLUCINATIONS:
            return True
        # Check if any hallucination phrase is CONTAINED
        for h in DictationEngine.HALLUCINATIONS:
            if h in low:
                return True
        words = text.split()
        # Repeated single word/phrase (e.g. "hello hello hello")
        if len(words) >= 3 and len(set(w.lower() for w in words)) == 1:
            return True
        # Common hallucination patterns: very short + generic
        if len(words) <= 2 and low in {
            "hello", "hi", "okay", "so", "yeah", "yes", "no",
            "um", "uh", "ah", "oh", "hmm", "huh", "hey",
            "once upon a time", "the end",
        }:
            return True
        # Very low confidence = garbage (avg_logprob -0.8 ≈ 45% prob/token — typical hallucination floor)
        if avg_logprob < -0.8:
            return True
        return False

    # MLX model name mapping
    MLX_MODELS = {
        "distil-large-v3": "mlx-community/distil-whisper-large-v3",
        "large-v3":        "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo":  "mlx-community/whisper-large-v3-turbo",
    }

    def load(self):
        # ── Silero VAD (optional, all platforms) ─────────────────────────────
        # Loaded first — small model (~1MB), negligible RAM, eliminates silence hallucinations.
        # Falls back silently to RMS VAD if torch/silero-vad not installed.
        try:
            import torch
            from silero_vad import load_silero_vad, get_speech_timestamps
            self.on_status("Loading VAD…")
            self._silero_vad    = load_silero_vad()
            self._silero_get_ts = get_speech_timestamps
            self._torch         = torch
        except Exception:
            self._silero_vad = None  # RMS fallback

        # ── Parakeet TDT 0.6B v3 (fastest option — 19-37x RTFx on MPS) ──────
        if self.cfg.get("model") == "parakeet-tdt-0.6b-v3":
            self._load_parakeet()
            return

        # ── Whisper model ─────────────────────────────────────────────────────
        # Use MLX on macOS (Metal GPU) — 3x faster encoder than CPU
        # Fall back to faster-whisper on Windows/Linux
        if IS_MAC:
            try:
                import mlx_whisper
                self._use_mlx = True
                self._mlx_model = self.MLX_MODELS.get(
                    self.cfg["model"], "mlx-community/distil-whisper-large-v3"
                )
                self.on_status("Getting ready…")
                mlx_whisper.transcribe(
                    np.zeros(16000, dtype=np.float32),
                    path_or_hf_repo=self._mlx_model,
                    verbose=False,
                )
                self.on_status("Ready")
                return
            except Exception as e:
                sys.stderr.write(f"[load] MLX failed, falling back to CPU: {e}\n")
                self._use_mlx = False

        self._use_mlx = False
        from faster_whisper import WhisperModel
        self.on_status("Getting ready…")
        self._model = WhisperModel(
            self.cfg["model"],
            device="cpu",
            compute_type="int8",
            cpu_threads=self.cfg["threads_per_worker"],
            num_workers=self.cfg["num_workers"],
        )

        # ── Two-pass confidence router (CPU only) ─────────────────────────────
        try:
            self._draft_model = WhisperModel(
                "tiny.en", device="cpu", compute_type="int8",
                cpu_threads=min(2, self.cfg["threads_per_worker"]),
            )
        except Exception:
            self._draft_model = None

        # ── Warmup — pre-compile CTranslate2 kernels ─────────────────────────
        # Without this, the first real transcription stalls 3–8s while CTranslate2
        # JIT-compiles its BLAS kernels. With this, first tap is instant.
        try:
            _dummy = np.zeros(self.RATE, dtype=np.float32)
            list(self._model.transcribe(
                _dummy, beam_size=1, language="en", vad_filter=True,
            ))
            if self._draft_model:
                list(self._draft_model.transcribe(_dummy, beam_size=1))
            del _dummy
        except Exception:
            pass
        self.on_status("Ready")

        self.on_status("Ready")

    # ── Parakeet subprocess management ──────────────────────────────────────

    # Workers sit in engine/workers/ alongside engine/flow.py
    PARAKEET_WORKER = str(Path(__file__).parent / "workers" / "parakeet_worker.py")

    @staticmethod
    def _find_nemo_python() -> str:
        """Find the NeMo-capable Python interpreter on any OS."""
        import sys, shutil
        # Standard install location across all platforms: ~/.flow/nemo_env
        venv = Path.home() / ".flow" / "nemo_env"
        candidates = [
            venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python3"),
            venv / ("Scripts/python"     if sys.platform == "win32" else "bin/python"),
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        # Fallback: system python3 (works if NeMo installed globally)
        return shutil.which("python3") or shutil.which("python") or sys.executable

    NEMO_PYTHON: str = ""   # set in __init_subclass__ / resolved lazily

    def _load_parakeet(self):
        """Spawn parakeet_worker.py using the best available Python interpreter."""
        import subprocess, shutil
        python = self._find_nemo_python()
        if not Path(python).exists() and python not in ("python3", "python"):
            self.on_status("Parakeet: Python interpreter not found — run setup script")
            self._use_mlx = False
            self._use_parakeet = False
            return
        self.on_status("Loading Parakeet (first load ~15s)…")
        env = os.environ.copy()
        lang = self.cfg.get("language") or ""
        if lang:
            env["PARAKEET_LANG"] = lang   # e.g. "en" → lock to English
        proc = subprocess.Popen(
            [python, self.PARAKEET_WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        # Wait for {"ready": true} line (up to 60s — model download + warmup)
        import select
        deadline = time.time() + 60
        while time.time() < deadline:
            if proc.poll() is not None:
                err = proc.stderr.read(2000).decode(errors="replace")
                self.on_status(f"Parakeet failed: {err[:80]}")
                self._use_parakeet = False
                return
            line = proc.stdout.readline()
            if not line:
                continue
            try:
                msg = json.loads(line.decode().strip())
                if msg.get("ready"):
                    break
            except Exception:
                continue
        else:
            proc.kill()
            self.on_status("Parakeet: timed out loading")
            self._use_parakeet = False
            return
        self._parakeet_proc = proc
        self._use_parakeet = True

        # Tight round intervals — Parakeet processes in ~80ms so 0.5s rounds
        # give word-by-word feel vs the 2.5s default tuned for slow MLX Whisper
        self.ROUND_S    = 0.5
        self.FIRST_ROUND = 0.3

        # Pre-warm MPS with real speech-shaped audio so first user recording
        # doesn't pay the MPS kernel compilation cost (fixes low first RTFx)
        try:
            sr = 16000
            warm_audio = (np.random.randn(sr * 3) * 0.15).astype(np.float32)
            self._transcribe_parakeet(warm_audio)
        except Exception:
            pass

        self.on_status("Ready")

    def _transcribe_parakeet(self, audio: np.ndarray):
        """Send audio to the parakeet worker subprocess, return (text, rtfx)."""
        import struct
        with self._parakeet_lock:
            proc = self._parakeet_proc
            if proc is None or proc.poll() is not None:
                return "", 0
            try:
                n = len(audio)
                proc.stdin.write(struct.pack("<I", n))
                proc.stdin.write(audio.astype(np.float32).tobytes())
                proc.stdin.flush()
                line = proc.stdout.readline()
                if not line:
                    return "", 0
                msg = json.loads(line.decode().strip())
                return msg.get("text", ""), msg.get("rtfx", 0)
            except Exception as e:
                sys.stderr.write(f"[parakeet] error: {e}\n")
                return "", 0

    def _kill_parakeet(self):
        proc = self._parakeet_proc
        if proc:
            try:
                proc.stdin.close()
                proc.kill()
            except Exception:
                pass
            self._parakeet_proc = None

    # ── PUBLIC: toggle entry points ─────────────────────────────────────────

    def is_recording(self) -> bool:
        return self._rec

    def toggle(self, vocab: dict,
               vocab_callback: Optional[Callable] = None,
               on_meeting_text: Optional[Callable] = None):
        """
        Single click-to-toggle entry. Starts streaming if stopped, stops if running.
        """
        if self._rec:
            self._stop_streaming()
        else:
            self._start_streaming(vocab, vocab_callback, on_meeting_text)

    def set_mode(self, mode: str):
        """Switch between dictation and meeting modes (must be stopped first)."""
        if self._rec:
            self._stop_streaming()
        self.mode = mode

    # ── INTERNAL: streaming start/stop ──────────────────────────────────────

    def _start_streaming(self, vocab, vocab_callback, on_meeting_text):
        try:
            import sounddevice as sd
        except ImportError:
            self.on_status("sounddevice missing"); return

        # Reset all per-session state
        with self._lock:
            self._rec = True
            self._audio_q = queue.Queue()
            self._meeting_buf = []
            self._silence_gap_s = 0.0
            self._cur_vocab = vocab
            self._cur_vocab_cb = vocab_callback
            self._on_meeting_text = on_meeting_text

        def cb(indata, frames, time_info, status):
            if self._rec:
                self._audio_q.put_nowait(indata[:, 0].copy())

        # On macOS, sounddevice/PortAudio crashes when opened from ANY thread
        # except the true main thread (TSMGetInputSourceProperty assertion).
        # pywebview owns the main thread. AppHelper.callAfter doesn't help.
        #
        # Solution: run audio capture in a SUBPROCESS with its own main thread.
        # The subprocess writes raw float32 audio to a temp file or pipe.
        # We read from it in our streaming worker.
        if IS_MAC:
            self._start_subprocess_recorder()
        else:
            try:
                mic_device = self.cfg.get("mic_device")
                self._stream = sd.InputStream(
                    samplerate=self.RATE, channels=1, dtype="float32",
                    blocksize=self.CHUNK, callback=cb,
                    device=mic_device,  # None = system default
                )
                self._stream.start()
            except Exception as e:
                self.on_status(f"Mic error: {e}")
                self._rec = False

        if self.cfg.get("sound_feedback"):
            play_beep(up=True)
        self.on_status("● Recording")

        # Spawn the streaming worker that flushes buffered audio every
        # CHUNK_SECS or on VAD silence boundary
        self._stream_thread = threading.Thread(
            target=self._stream_worker, daemon=True
        )
        self._stream_thread.start()

    def _start_subprocess_recorder(self):
        """
        macOS: run audio capture in a separate Python subprocess.
        The subprocess has its own main thread, so PortAudio/CoreAudio
        can safely call TSMGetInputSourceProperty without crashing.
        Audio data is piped back via stdout as raw float32 bytes.
        """
        import subprocess
        mic_device = self.cfg.get("mic_device")
        device_arg = f"device={mic_device!r}" if mic_device is not None else ""
        script = f"""
import sounddevice as sd
import sys
RATE = {self.RATE}
CHUNK = {self.CHUNK}
stream = sd.InputStream(samplerate=RATE, channels=1, dtype='float32', blocksize=CHUNK{', ' + device_arg if device_arg else ''})
stream.start()
try:
    while True:
        data, overflowed = stream.read(CHUNK)
        sys.stdout.buffer.write(data[:, 0].tobytes())
        sys.stdout.buffer.flush()
except (KeyboardInterrupt, BrokenPipeError):
    pass
finally:
    stream.stop()
    stream.close()
"""
        try:
            self._recorder_proc = subprocess.Popen(
                [sys.executable, "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            # Reader thread: reads from subprocess stdout → audio queue
            def _reader():
                chunk_bytes = self.CHUNK * 4  # float32 = 4 bytes
                while self._rec and self._recorder_proc.poll() is None:
                    data = self._recorder_proc.stdout.read(chunk_bytes)
                    if data and len(data) == chunk_bytes:
                        arr = np.frombuffer(data, dtype=np.float32)
                        self._audio_q.put_nowait(arr)

            threading.Thread(target=_reader, daemon=True, name="mic-reader").start()
        except Exception as e:
            self.on_status(f"Mic subprocess error: {e}")
            self._rec = False

    def _stop_streaming(self):
        with self._lock:
            if not self._rec:
                return
            self._rec = False

        # Drain the queue immediately so the worker doesn't keep transcribing
        # stale audio after the user clicks stop
        while not self._audio_q.empty():
            try: self._audio_q.get_nowait()
            except Exception: break

        # Kill subprocess recorder if on macOS
        if IS_MAC and hasattr(self, '_recorder_proc') and self._recorder_proc:
            try:
                self._recorder_proc.kill()
                self._recorder_proc.wait(timeout=2)
            except Exception:
                pass
            self._recorder_proc = None
        else:
            try:
                if self._stream is not None:
                    self._stream.stop(); self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self.cfg.get("sound_feedback"):
            play_beep(up=False)
        self.on_status("Stopped")

    # ── INTERNAL: streaming worker (the new core) ───────────────────────────

    def _stream_worker(self):
        """
        TUMBLING WINDOW + LocalAgreement-2 streaming.

        Whisper encoder ALWAYS processes 30s mel — so bigger buffer = better RTFx:
          2s buf  → encoder(30s) ≈ 2s → RTFx 1x   (waste)
          10s buf → encoder(30s) ≈ 2s → RTFx 5x   (good)
          25s buf → encoder(30s) ≈ 2s → RTFx 12x  (great)

        LocalAgreement-2 (Machacek et al. 2023):
          Words are emitted only when they appear at the same position across
          TWO consecutive transcription passes. This prevents flicker — Whisper
          often revises early words as more audio arrives. Only stable words
          (confirmed by the second pass) appear in the live preview.
          Result: words appear word-by-word as you speak, not after 2s silence.

        Architecture:
          - THIS thread: reads mic audio, grows buffer (never blocks)
          - BACKGROUND thread: transcribes snapshot, emits newly confirmed words
          - On silence: final transcription → commit all remaining → reset
        """
        buf = np.array([], dtype=np.float32)
        silence_chunks = 0
        silence_chunks_target = int(self.SILENCE_S * self.RATE / self.CHUNK)
        last_round = time.perf_counter()
        max_buf_samples = int(self.MAX_BUF_S * self.RATE)
        transcribing = False
        detected_lang = None
        first_round_done = False   # first round fires early (0.8s)

        # LocalAgreement-2 state — reset per utterance
        la_prev_words:      list  = []   # words from previous round
        la_confirmed_count: int   = 0    # how many words already emitted as confirmed
        la_confirmed_text:  str   = ""   # full confirmed text so far
        la_last_parakeet:   str   = ""   # last text emitted via Parakeet path
        utterance_script:   str   = ""   # "cyrillic" | "latin" | "" — locked per utterance
        session_script:     str   = ""   # dominant script seen in this session — biases new utterances
        sentence_ended:     bool  = False  # True when last Parakeet text ended with sentence punctuation

        def _script(text: str) -> str:
            """Detect dominant script: 'cyrillic', 'latin', or ''."""
            cyr = sum(1 for c in text if 'Ѐ' <= c <= 'ӿ')
            lat = sum(1 for c in text if c.isascii() and c.isalpha())
            if cyr + lat == 0:
                return ""
            return "cyrillic" if cyr >= lat else "latin"

        def _la_reset():
            nonlocal la_prev_words, la_confirmed_count, la_confirmed_text, la_last_parakeet
            nonlocal utterance_script, sentence_ended
            la_prev_words      = []
            la_confirmed_count = 0
            la_confirmed_text  = ""
            la_last_parakeet   = ""
            utterance_script   = ""      # reset per utterance
            sentence_ended     = False
            # session_script intentionally NOT reset — biases next utterance

        def _bg_transcribe(snapshot, is_final):
            """
            Transcribe snapshot → emit newly LocalAgreement-confirmed words.
            On final: emit everything remaining unconditionally.
            """
            nonlocal transcribing, detected_lang, first_round_done
            nonlocal la_prev_words, la_confirmed_count, la_confirmed_text, la_last_parakeet
            nonlocal utterance_script, session_script, sentence_ended
            try:
                text, lang = self._transcribe_buffer_with_lang(snapshot, use_stream_model=False)
                if not text:
                    return

                first_round_done = True
                if lang and lang != detected_lang:
                    detected_lang = lang
                    self.on_status(f"lang:{lang}")

                app = get_active_app() if self.cfg["active_app_context"] else ""
                cat = get_app_category(app)

                if is_final:
                    vocab = self._cur_vocab or {}
                    speller = self.speller if self.cfg.get("spell_check", True) else None
                    cleaned = (clean_text(text, vocab, speller=speller, category=cat)
                               if self.cfg["regex_cleanup"] else text)
                    self.on_text(cleaned, app, 0, 1)  # 1 = final
                    sys.stderr.write(f"[stream] FINAL: {cleaned[:80]}\n"); sys.stderr.flush()
                    self._commit_text(cleaned, "")
                elif getattr(self, '_use_parakeet', False):
                    # ── Language lock: detect script on first round, discard flips ──
                    s = _script(text)
                    if not utterance_script and s:
                        # Bias toward session script if ambiguous (single short word)
                        if session_script and len(text.split()) <= 2 and s != session_script:
                            utterance_script = session_script
                        else:
                            utterance_script = s
                    if s:
                        session_script = s   # update session-level bias
                    if utterance_script and s and s != utterance_script:
                        # Script flipped mid-utterance — discard this result
                        sys.stderr.write(f"[stream] script flip {s}≠{utterance_script}, discarded\n")
                        sys.stderr.flush()
                        return
                    # ── Emit only when text changed ───────────────────────────────
                    if text != la_last_parakeet:
                        la_last_parakeet = text
                        sentence_ended = text.rstrip().endswith(('.', '?', '!', '।', '。', '…'))
                        self.on_text(text, app, 0, 0)
                        sys.stderr.write(f"[stream] parakeet: {text[:60]}\n"); sys.stderr.flush()
                else:
                    # ── LocalAgreement-2 ──────────────────────────────────
                    # Find the longest prefix of new_words that matches prev_words
                    # at the same positions. Words that agree across passes = stable.
                    new_words = text.split()
                    stable_end = 0
                    for i in range(min(len(la_prev_words), len(new_words))):
                        if la_prev_words[i].lower() == new_words[i].lower():
                            stable_end = i + 1
                        else:
                            break  # stop at first disagreement

                    # Emit words confirmed for the first time this pass
                    if stable_end > la_confirmed_count:
                        newly = new_words[la_confirmed_count:stable_end]
                        emit  = " ".join(newly)
                        la_confirmed_text  += (" " if la_confirmed_text else "") + emit
                        la_confirmed_count  = stable_end
                        self.on_text(la_confirmed_text, app, 0, 0)
                        sys.stderr.write(f"[stream] +{len(newly)} confirmed: {emit}\n")
                        sys.stderr.flush()

                    la_prev_words = new_words
            finally:
                transcribing = False

        chunks_received = 0
        while self._rec or not self._audio_q.empty():
            try:
                chunk = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                # No audio, but check if we should kick off a round
                now = time.perf_counter()
                buf_secs = len(buf) / self.RATE
                if (not transcribing
                        and buf_secs >= self.MIN_FLUSH_S
                        and now - last_round >= (self.FIRST_ROUND if not first_round_done else self.ROUND_S)):
                    snapshot = buf.copy()
                    transcribing = True
                    last_round = now
                    sys.stderr.write(f"[stream] round: {buf_secs:.1f}s buf, bg transcribe\n"); sys.stderr.flush()
                    threading.Thread(target=_bg_transcribe,
                                     args=(snapshot, False), daemon=True).start()
                continue

            # VAD
            rms = float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0

            chunks_received += 1
            if chunks_received % 50 == 1:
                sys.stderr.write(f"[stream] chunks:{chunks_received} buf:{len(buf)/self.RATE:.1f}s rms:{rms:.4f}\n"); sys.stderr.flush()
            if rms < self.VAD_RMS:
                silence_chunks += 1
            else:
                silence_chunks = 0

            buf = np.concatenate([buf, chunk])

            if self.mode == self.MODE_MEETING:
                self._meeting_buf.append(chunk)

            buf_secs = len(buf) / self.RATE
            now = time.perf_counter()

            # ── SILENCE → finalize & commit ──────────────────────────
            # After a sentence-ending punctuation, accept half the normal silence
            effective_silence_target = (silence_chunks_target // 2
                                        if sentence_ended else silence_chunks_target)
            hit_silence = (silence_chunks >= effective_silence_target
                           and buf_secs >= self.MIN_FLUSH_S)

            if hit_silence:
                # Wait for any in-progress transcription to finish
                while transcribing:
                    time.sleep(0.05)
                # Final transcription on full buffer
                snapshot = buf.copy()
                transcribing = True
                last_round = now
                # Run final synchronously — we need to reset buffer after
                _bg_transcribe(snapshot, is_final=True)
                buf = np.array([], dtype=np.float32)
                silence_chunks = 0
                first_round_done = False    # next utterance fires fast again
                _la_reset()                 # reset LocalAgreement state for next utterance
                continue

            # ── Cap buffer → force commit ────────────────────────────
            if len(buf) >= max_buf_samples:
                while transcribing:
                    time.sleep(0.05)
                snapshot = buf.copy()
                _bg_transcribe(snapshot, is_final=True)
                buf = np.array([], dtype=np.float32)
                silence_chunks = 0
                first_round_done = False
                last_round = now
                _la_reset()
                continue

            # ── ROUND: periodic background transcription ─────────────
            if (not transcribing
                    and buf_secs >= self.MIN_FLUSH_S
                    and now - last_round >= self.ROUND_S):
                snapshot = buf.copy()
                transcribing = True
                last_round = now
                threading.Thread(target=_bg_transcribe,
                                 args=(snapshot, False), daemon=True).start()

        # Flush remaining on stop
        while transcribing:
            time.sleep(0.05)
        if buf.size and len(buf) / self.RATE >= self.MIN_FLUSH_S:
            text = self._transcribe_buffer(buf)
            if text:
                self._commit_text(text, "")

    def _transcribe_buffer_with_lang(self, audio: np.ndarray, use_stream_model: bool = False):
        """Returns (text, language_code) tuple. Uses MLX (Metal) on Mac, faster-whisper on others."""
        duration = len(audio) / self.RATE
        if duration < 0.3:
            return "", None

        # ── Silero VAD pre-filter ─────────────────────────────────────────────
        # Check for actual speech before calling Whisper. Eliminates hallucinations
        # on silent/noise chunks entirely. Cost: ~1ms. Benefit: no Whisper call on silence.
        # Skip for Parakeet — it handles silence internally and doesn't hallucinate.
        if self._silero_vad is not None and not getattr(self, '_use_parakeet', False):
            try:
                t = self._torch.from_numpy(audio)
                ts = self._silero_get_ts(t, self._silero_vad, sampling_rate=self.RATE,
                                         threshold=0.4, min_speech_duration_ms=200)
                if not ts:
                    return "", None
            except Exception:
                pass  # silero failed — fall through to RMS-gated Whisper

        try:
            t0 = time.perf_counter()
            lang_cfg = self.cfg.get("language") or None

            if getattr(self, '_use_parakeet', False):
                # Parakeet TDT 0.6B v3 — MPS, 19-37x RTFx
                text, rtfx = self._transcribe_parakeet(audio)
                elapsed = time.perf_counter() - t0
                if rtfx == 0:
                    rtfx = round(duration / elapsed, 1) if elapsed > 0 else 0
                self.on_status(f"● {duration:.1f}s  RTFx {rtfx:.0f}x  [parakeet]")
                return text, "en"

            if getattr(self, '_use_mlx', False):
                # MLX path — Metal GPU, ~2.5s encoder (3x faster than CPU)
                import mlx_whisper
                kwargs = dict(
                    path_or_hf_repo=self._mlx_model,
                    verbose=False,
                    condition_on_previous_text=False,
                    temperature=0.0,
                    no_speech_threshold=0.6,
                )
                if lang_cfg:
                    kwargs["language"] = lang_cfg
                result = mlx_whisper.transcribe(audio, **kwargs)
                elapsed = time.perf_counter() - t0
                rtfx = duration / elapsed if elapsed > 0 else 0
                raw = result.get("text", "").strip()
                raw = re.sub(r'\s*\n+\s*', ' ', raw)
                raw = re.sub(r'^\s*[-•]\s*', '', raw)
                raw = re.sub(r'\s{2,}', ' ', raw).strip()
                detected = result.get("language", None)
                avg_lp = -0.5  # MLX doesn't expose logprob easily
                if self._is_hallucination(raw, avg_lp):
                    return "", None
                self.on_status(f"● {duration:.1f}s  RTFx {rtfx:.1f}x")
                return raw, detected
            else:
                # faster-whisper path (CPU) — Windows/Linux fallback
                # ── Two-pass confidence router ────────────────────────────────
                # tiny.en runs first (~10ms). If avg_logprob >= -0.5 (confident),
                # skip the full model entirely. Escalate only on uncertain chunks.
                lang = lang_cfg or "en"
                if self._draft_model is not None:
                    try:
                        draft_segs, _ = self._draft_model.transcribe(
                            audio, beam_size=1, language="en",
                            no_speech_threshold=0.6,
                            condition_on_previous_text=False,
                        )
                        draft_segs = list(draft_segs)
                        if draft_segs:
                            avg_lp_draft = sum(s.avg_logprob for s in draft_segs) / len(draft_segs)
                            if avg_lp_draft >= -0.5:
                                # Draft confident — use result directly (fast path)
                                raw = " ".join(s.text.strip() for s in draft_segs).strip()
                                raw = re.sub(r'\s*\n+\s*', ' ', raw)
                                raw = re.sub(r'\s{2,}', ' ', raw).strip()
                                if not self._is_hallucination(raw, avg_lp_draft):
                                    elapsed = time.perf_counter() - t0
                                    rtfx = duration / elapsed if elapsed > 0 else 0
                                    self.on_status(f"● {duration:.1f}s  RTFx {rtfx:.1f}x  [fast]")
                                    return raw, "en"
                    except Exception:
                        pass  # draft failed — fall through to full model

                # Full model (slow path — uncertain chunks or draft unavailable)
                segs, info = self._model.transcribe(
                    audio, beam_size=1, language=lang_cfg, task="transcribe",
                    vad_filter=True, no_speech_threshold=0.6,
                    condition_on_previous_text=False,
                )
                segs = list(segs)
                elapsed = time.perf_counter() - t0
                rtfx = duration / elapsed if elapsed > 0 else 0
                raw_parts = []
                for s in segs:
                    t = s.text.strip()
                    t = re.sub(r'\s*\n+\s*', ' ', t)
                    t = re.sub(r'^\s*[-•]\s*', '', t)
                    t = re.sub(r'\s{2,}', ' ', t).strip()
                    if t:
                        raw_parts.append(t)
                raw = " ".join(raw_parts).strip()
                avg_lp = (sum(s.avg_logprob for s in segs) / len(segs)) if segs else -999
                if self._is_hallucination(raw, avg_lp):
                    return "", None
                detected = getattr(info, 'language', None)
                self.on_status(f"● {duration:.1f}s  RTFx {rtfx:.1f}x")
                return raw, detected
        except Exception as e:
            self.on_status(f"Error: {e}")
            return "", None

    def _transcribe_buffer(self, audio: np.ndarray, use_stream_model: bool = False) -> str:
        """Wrapper — routes to _transcribe_buffer_with_lang, returns text only."""
        text, _ = self._transcribe_buffer_with_lang(audio, use_stream_model)
        return text

    def _commit_text(self, final_text: str, committed_text: str):
        """Finalize text: clean, dispatch, insert at cursor."""
        if not final_text:
            return
        vocab = self._cur_vocab or {}
        vocab_cb = self._cur_vocab_cb

        # Voice control
        control = detect_control_command(final_text)
        if control and self.on_control:
            self.on_control(control)
            self.on_status(f"Voice: {control}")
            return

        # "scratch that" correction
        if (self.mode == self.MODE_DICTATION
                and detect_delete_trigger(final_text)
                and self._last_inserted):
            correction = strip_delete_trigger(final_text)
            if correction and vocab_cb:
                vocab_cb(self._last_inserted, correction)
            self._track_correction(self._last_inserted, correction)
            self._backspace_last()
            final_text = correction

        app = get_active_app() if self.cfg["active_app_context"] else ""
        cat = get_app_category(app)
        style = PER_APP_STYLES.get(cat, "")
        speller = self.speller if self.cfg.get("spell_check", True) else None
        cleaned = (clean_text(final_text, vocab, speller=speller, category=cat)
                   if self.cfg["regex_cleanup"] else final_text)

        # Tier 3 LLM polish
        if (detect_rewrite_trigger(final_text)
                and self.cfg["llm_cleanup"]
                and self.llm.available):
            to_polish = strip_rewrite_trigger(final_text)
            to_polish = clean_text(to_polish, vocab, speller=speller, category=cat)
            self.on_status("Polishing…")
            polished = self.llm.cleanup(to_polish, style_hint=style)
            cleaned = clean_text(polished, vocab, speller=speller, category=cat)

        self._last_raw = final_text
        self._last_inserted = cleaned

        if self.mode == self.MODE_CAPTURE:
            append_thought(cleaned)
            self.on_text(cleaned, "capture", 0, 0)
            self.on_status("Thought saved")
            return
        elif self.mode == self.MODE_MEETING:
            speaker = self._infer_speaker()
            if self._on_meeting_text:
                self._on_meeting_text(cleaned, speaker, 0)
        else:
            self.on_text(cleaned, app, 0, 0)
            try:
                insert_text(cleaned, method=self.cfg["insert_method"])
            except PermissionError as e:
                self.on_status(f"⚠ {e}")
                return

        append_history(cleaned, app)
        self.on_status("✓ committed")

    def _track_correction(self, wrong: str, right: str):
        """Auto-learn vocabulary after 2 identical 'scratch that' corrections."""
        if not wrong or not right:
            return
        key = wrong.lower().strip()
        if key not in self._correction_counter:
            self._correction_counter[key] = {}
        self._correction_counter[key][right] = (
            self._correction_counter[key].get(right, 0) + 1
        )
        if self._correction_counter[key][right] >= 2:
            vocab = load_vocab()
            vocab["replacements"][key] = right
            save_vocab(vocab)
            self._cur_vocab = vocab
            del self._correction_counter[key]
            self.on_status(f"Learned: {key!r} → {right!r}")

    # _process_chunk removed — replaced by _transcribe_buffer + _commit_text

    def _infer_speaker(self) -> int:
        """
        V1 silence-gap diarization: every chunk that comes after a long
        silence gap is considered a new speaker. Far from perfect but
        ~80% accurate for two-person meetings, free, real-time.
        """
        return self._last_speaker

    def _backspace_last(self):
        """Delete the previously inserted text from the focused field."""
        if not self._last_inserted:
            return
        n = len(self._last_inserted)
        try:
            if IS_MAC:
                # Use osascript — avoids importing pynput Controller here
                # which can stall if accessibility not granted
                import subprocess
                script = (
                    f"tell application \"System Events\" "
                    f"to repeat {n} times\n"
                    f"  key code 51\n"
                    f"end repeat"
                )
                subprocess.run(
                    ["osascript", "-e", script],
                    timeout=3, capture_output=True,
                )
            else:
                from pynput.keyboard import Controller, Key
                kb = Controller()
                for _ in range(n):
                    kb.press(Key.backspace)
                    kb.release(Key.backspace)
                    time.sleep(0.001)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM AUDIO CAPTURE (Meeting mode's killer feature)
#
# Captures what your speakers are PLAYING — i.e. the other side of a Zoom call,
# a podcast, a YouTube video — without joining as a bot. This is the
# differentiator that competes with Granola.
#
# macOS:   ScreenCaptureKit (13+) via Apple's AVAudioEngine tap on display
# Windows: WASAPI loopback (built into Windows Vista+)
# Linux:   PulseAudio monitor source
#
# This module provides a single capture() function that yields float32 chunks
# from system output. The Meeting engine combines this with mic input.
# ─────────────────────────────────────────────────────────────────────────────

class SystemAudioCapture:
    """
    Cross-platform system audio capture. Yields float32 mono chunks at 16kHz.
    Falls back gracefully if the platform doesn't expose system audio.
    """
    RATE = 16000

    def __init__(self, on_chunk: Callable, on_status: Callable):
        self.on_chunk  = on_chunk
        self.on_status = on_status
        self._stream   = None
        self._running  = False

    def available(self) -> bool:
        """Returns True if system audio capture is supported on this OS."""
        if IS_MAC:
            try:
                import sounddevice as sd
                # Look for BlackHole or similar virtual devices
                devs = sd.query_devices()
                for d in devs:
                    name = d.get("name", "").lower()
                    if "blackhole" in name or "loopback" in name or "soundflower" in name:
                        return True
                # macOS 14.4+ has built-in system audio via ScreenCaptureKit
                # via the new "Audio Capture" API. Check for it.
                try:
                    import platform
                    ver = tuple(int(x) for x in platform.mac_ver()[0].split(".")[:2])
                    return ver >= (14, 4)
                except Exception:
                    return False
            except ImportError:
                return False
        elif IS_WIN:
            try:
                import sounddevice as sd
                # WASAPI loopback is exposed as a special device
                hostapis = sd.query_hostapis()
                return any("WASAPI" in h.get("name", "") for h in hostapis)
            except Exception:
                return False
        elif IS_LINUX:
            try:
                import subprocess
                # Check for PulseAudio
                r = subprocess.run(["pactl", "info"], capture_output=True, timeout=2)
                return r.returncode == 0
            except Exception:
                return False
        return False

    def start(self):
        if self._running:
            return
        self._running = True

        if IS_WIN:
            self._start_wasapi()
        elif IS_MAC:
            self._start_mac()
        elif IS_LINUX:
            self._start_pulse()

    def _start_wasapi(self):
        """Windows: WASAPI loopback — built into Windows."""
        try:
            import sounddevice as sd
            wasapi_idx = None
            for i, h in enumerate(sd.query_hostapis()):
                if "WASAPI" in h.get("name", ""):
                    wasapi_idx = i
                    break
            if wasapi_idx is None:
                self.on_status("WASAPI not found")
                return

            # Find default output device, use loopback variant
            default_out = sd.default.device[1]
            extra_settings = sd.WasapiSettings(loopback=True)

            def cb(indata, frames, t, status):
                if self._running:
                    # Downmix to mono if stereo, resample if needed
                    if indata.ndim > 1 and indata.shape[1] > 1:
                        mono = indata.mean(axis=1)
                    else:
                        mono = indata[:, 0] if indata.ndim > 1 else indata
                    self.on_chunk(mono.astype(np.float32))

            self._stream = sd.InputStream(
                samplerate=self.RATE, channels=1, dtype="float32",
                callback=cb, device=default_out,
                extra_settings=extra_settings,
            )
            self._stream.start()
            self.on_status("System audio: WASAPI loopback")
        except Exception as e:
            self.on_status(f"WASAPI failed: {e}")

    def _start_mac(self):
        """macOS: prefer BlackHole if installed, else warn."""
        try:
            import sounddevice as sd
            devs = sd.query_devices()
            blackhole_idx = None
            for i, d in enumerate(devs):
                name = d.get("name", "").lower()
                if "blackhole" in name and d.get("max_input_channels", 0) > 0:
                    blackhole_idx = i
                    break

            if blackhole_idx is None:
                self.on_status(
                    "Install BlackHole for system audio: brew install blackhole-2ch"
                )
                return

            def cb(indata, frames, t, status):
                if self._running:
                    if indata.ndim > 1 and indata.shape[1] > 1:
                        mono = indata.mean(axis=1)
                    else:
                        mono = indata[:, 0] if indata.ndim > 1 else indata
                    self.on_chunk(mono.astype(np.float32))

            self._stream = sd.InputStream(
                samplerate=self.RATE, channels=1, dtype="float32",
                callback=cb, device=blackhole_idx,
            )
            self._stream.start()
            self.on_status("System audio: BlackHole")
        except Exception as e:
            self.on_status(f"Mac audio capture failed: {e}")

    def _start_pulse(self):
        """Linux: PulseAudio monitor source."""
        try:
            import sounddevice as sd
            devs = sd.query_devices()
            monitor_idx = None
            for i, d in enumerate(devs):
                name = d.get("name", "").lower()
                if "monitor" in name and d.get("max_input_channels", 0) > 0:
                    monitor_idx = i
                    break

            if monitor_idx is None:
                self.on_status("No PulseAudio monitor source found")
                return

            def cb(indata, frames, t, status):
                if self._running:
                    if indata.ndim > 1 and indata.shape[1] > 1:
                        mono = indata.mean(axis=1)
                    else:
                        mono = indata[:, 0] if indata.ndim > 1 else indata
                    self.on_chunk(mono.astype(np.float32))

            self._stream = sd.InputStream(
                samplerate=self.RATE, channels=1, dtype="float32",
                callback=cb, device=monitor_idx,
            )
            self._stream.start()
            self.on_status("System audio: PulseAudio monitor")
        except Exception as e:
            self.on_status(f"PulseAudio capture failed: {e}")

    def stop(self):
        self._running = False
        if self._stream:
            try:
                self._stream.stop(); self._stream.close()
            except Exception:
                pass
            self._stream = None


# ─────────────────────────────────────────────────────────────────────────────
# MIC INDICATOR — floating pill, cross-platform
# ─────────────────────────────────────────────────────────────────────────────

class MicIndicator:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.win: Optional[tk.Toplevel] = None
        self.dot_canvas = None
        self.dot = None
        self.label = None

    def show(self, status="Recording"):
        if self.win is not None:
            if self.label: self.label.config(text=status)
            return
        win = tk.Toplevel(self.root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        try: win.attributes("-alpha", 0.96)
        except Exception: pass
        win.configure(bg=Theme.LEVEL3)

        w, h = 190, 46
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        win.geometry(f"{w}x{h}+{(sw-w)//2}+{sh-140}")

        frame = tk.Frame(
            win, bg=Theme.LEVEL3,
            highlightbackground=Theme.BORDER_PRIMARY, highlightthickness=1,
        )
        frame.pack(fill="both", expand=True)

        self.dot_canvas = tk.Canvas(frame, width=14, height=14,
                                     bg=Theme.LEVEL3, highlightthickness=0)
        self.dot_canvas.pack(side="left", padx=(16, 10), pady=15)
        self.dot = self.dot_canvas.create_oval(3, 3, 11, 11,
                                                fill=Theme.VIOLET, outline="")

        self.label = tk.Label(
            frame, text=status, font=Theme.FONT_SMALL,
            fg=Theme.TEXT_PRIMARY, bg=Theme.LEVEL3,
        )
        self.label.pack(side="left", padx=(0, 14))

        self.win = win
        self._pulse()

    def _pulse(self):
        if self.win is None: return
        try:
            cur = self.dot_canvas.itemcget(self.dot, "fill")
            self.dot_canvas.itemconfig(
                self.dot,
                fill=Theme.VIOLET_HOVER if cur == Theme.VIOLET else Theme.VIOLET,
            )
            self.win.after(400, self._pulse)
        except Exception:
            pass

    def hide(self):
        if self.win is not None:
            try: self.win.destroy()
            except Exception: pass
            self.win = None


# ─────────────────────────────────────────────────────────────────────────────
# ONBOARDING FLOW
# ─────────────────────────────────────────────────────────────────────────────

class Onboarding:
    """
    Modal overlay that walks new users through:
      1. Welcome
      2. Permissions
      3. Benchmark (thread sweep)
      4. Practice dictation
      5. Done
    """
    def __init__(self, root: tk.Tk, cfg: dict, on_done: Callable):
        self.root = root
        self.cfg  = cfg
        self.on_done = on_done
        self.step = 0

        self.win = tk.Toplevel(root)
        self.win.title("Welcome to Flow")
        self.win.geometry("620x500+200+150")
        self.win.configure(bg=Theme.MARKETING_BLACK)
        self.win.transient(root)
        self.win.grab_set()
        self.win.lift()
        self.win.attributes("-topmost", True)
        self.win.after(100, lambda: self.win.attributes("-topmost", False))
        self.win.focus_force()

        self.container = tk.Frame(self.win, bg=Theme.MARKETING_BLACK)
        self.container.pack(fill="both", expand=True, padx=40, pady=40)

        self.steps = [
            self._welcome, self._permissions,
            self._benchmark, self._practice, self._done,
        ]
        self._render()

    def _render(self):
        for w in self.container.winfo_children():
            w.destroy()
        self.steps[self.step]()

    def _next(self):
        self.step += 1
        if self.step >= len(self.steps):
            self.win.destroy()
            self.on_done()
        else:
            self._render()

    def _h1(self, text):
        tk.Label(
            self.container, text=text, font=Theme.FONT_H1,
            fg=Theme.TEXT_PRIMARY, bg=Theme.MARKETING_BLACK,
        ).pack(anchor="w", pady=(0, 8))

    def _p(self, text):
        tk.Label(
            self.container, text=text, font=Theme.FONT_BODY,
            fg=Theme.TEXT_SECONDARY, bg=Theme.MARKETING_BLACK,
            wraplength=520, justify="left",
        ).pack(anchor="w", pady=(0, 18))

    def _btn(self, text, cmd, primary=True):
        return tk.Button(
            self.container, text=text,
            font=Theme.FONT_BODY,
            bg=Theme.INDIGO if primary else Theme.LEVEL3,
            fg="#ffffff" if primary else Theme.TEXT_PRIMARY,
            activebackground=Theme.VIOLET_HOVER,
            activeforeground="#ffffff",
            relief="flat", bd=0,
            padx=24, pady=10, cursor="hand2", command=cmd,
        )

    # Step 1: Welcome
    def _welcome(self):
        self._h1("Flow")
        self._p("Voice dictation that works offline, on any app. "
                "Press a hotkey, speak, release — your words appear wherever your cursor is.")
        tk.Label(
            self.container,
            text="• Works on planes. No internet needed.\n"
                 "• Pay once. No subscriptions.\n"
                 "• Learns your vocabulary over time.\n"
                 "• Whisper distil-large-v3 — fastest accurate model.",
            font=Theme.FONT_BODY, justify="left",
            fg=Theme.TEXT_TERTIARY, bg=Theme.MARKETING_BLACK,
        ).pack(anchor="w", pady=(0, 24))
        self._btn("Get Started  →", self._next).pack(anchor="w")

    # Step 2: Permissions
    def _permissions(self):
        self._h1("Grant Permissions")
        if IS_MAC:
            self._p("Flow needs two permissions from macOS:\n\n"
                    "  1. Microphone — to hear your voice\n"
                    "  2. Accessibility — to type into other apps\n\n"
                    "Click the button to open System Settings.")
            self._btn("Open System Settings",
                      lambda: os.system("open 'x-apple.systempreferences:"
                                         "com.apple.preference.security?Privacy_Accessibility'"),
                      primary=False).pack(anchor="w", pady=(0,12))
        elif IS_WIN:
            self._p("Flow needs microphone access. Windows will prompt "
                    "you the first time it captures audio.")
        else:
            self._p("On Linux, make sure your user has audio access and "
                    "xdotool is installed for active-window detection:\n\n"
                    "  sudo apt install xdotool")
        self._btn("I've granted access  →", self._next).pack(anchor="w", pady=(20,0))

    # Step 3: Benchmark
    def _benchmark(self):
        self._h1("Tuning for your machine")
        self._p("Running a quick benchmark to find the optimal thread count.")
        status = tk.Label(
            self.container, text="Starting…", font=Theme.FONT_SMALL,
            fg=Theme.VIOLET, bg=Theme.MARKETING_BLACK,
        )
        status.pack(anchor="w", pady=(0, 12))

        def run():
            try:
                from faster_whisper import WhisperModel
                import tempfile, wave, multiprocessing
                # Generate 3s tone
                sr = 16000
                t = np.linspace(0, 3, 3*sr, dtype=np.float32)
                tone = (np.sin(2*np.pi*440*t) * 32767).astype(np.int16)
                tmp = Path(tempfile.gettempdir()) / "flow_bench.wav"
                with wave.open(str(tmp), "w") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
                    wf.writeframes(tone.tobytes())

                cores = multiprocessing.cpu_count()
                candidates = [(2, 3), (1, 4), (2, 2), (1, 2)]
                best = None; best_rtfx = 0
                for workers, threads in candidates:
                    if workers * threads > cores: continue
                    status.config(text=f"Testing {workers}w × {threads}t…")
                    self.root.update_idletasks()
                    m = WhisperModel("tiny.en", device="cpu",
                                     compute_type="int8",
                                     cpu_threads=threads,
                                     num_workers=workers)
                    list(m.transcribe(str(tmp), beam_size=1)[0])  # warm
                    t0 = time.perf_counter()
                    list(m.transcribe(str(tmp), beam_size=1)[0])
                    elapsed = time.perf_counter() - t0
                    rtfx = 3.0 / elapsed
                    if rtfx > best_rtfx:
                        best_rtfx, best = rtfx, (workers, threads)
                    del m

                if best:
                    w, t_ = best
                    self.cfg["num_workers"] = w
                    self.cfg["threads_per_worker"] = t_
                    self.cfg["benchmark_rtfx"] = round(best_rtfx, 2)
                    save_config(self.cfg)

                status.config(
                    text=f"✓ Optimal: {best[0]}w × {best[1]}t → RTFx {best_rtfx:.1f}x"
                         if best else "Benchmark failed",
                    fg=Theme.SUCCESS,
                )
                self._btn("Continue  →", self._next).pack(anchor="w", pady=(16,0))
                try: tmp.unlink()
                except Exception: pass
            except Exception as e:
                status.config(text=f"Error: {e}", fg=Theme.ERROR)
                self._btn("Skip  →", self._next, primary=False).pack(anchor="w", pady=(16,0))

        threading.Thread(target=run, daemon=True).start()

    # Step 4: Practice
    def _practice(self):
        self._h1("Try it")
        self._p(f"Hold {self.cfg['hotkey']} anywhere, say:\n\n"
                f'   "Hello Flow, this is my first dictation."\n\n'
                "Release the keys and watch it appear below.")
        self.practice_result = tk.Label(
            self.container, text="…", font=Theme.FONT_BODY,
            fg=Theme.VIOLET, bg=Theme.PANEL_DARK,
            padx=16, pady=12, wraplength=480, justify="left",
        )
        self.practice_result.pack(anchor="w", fill="x", pady=(0,16))
        self._btn("Done  →", self._next).pack(anchor="w")

    def update_practice(self, text: str):
        if hasattr(self, "practice_result") and self.practice_result:
            try: self.practice_result.config(text=text)
            except Exception: pass

    # Step 5: Done
    def _done(self):
        self._h1("You're set")
        self._p(f"Hold {self.cfg['hotkey']} anywhere to dictate. "
                "Flow will keep running in your tray.\n\n"
                "Open the main window any time from the tray icon "
                "to review history, add vocabulary, or change settings.")
        self._btn("Finish", self._next).pack(anchor="w")


# ─────────────────────────────────────────────────────────────────────────────
# TRAY ICON (cross-platform via pystray)
# ─────────────────────────────────────────────────────────────────────────────

def make_tray_icon_image(color="#5e6ad2", size=64):
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Simple filled circle for "F"
    d.ellipse((4, 4, size-4, size-4), fill=color)
    # White "F"
    try:
        from PIL import ImageFont
        try: font = ImageFont.truetype("Helvetica.ttf", 36)
        except Exception: font = ImageFont.load_default()
        d.text((20, 12), "F", fill="white", font=font)
    except Exception:
        pass
    return img


class TrayApp:
    """pystray-backed menu bar icon — runs in its own thread."""
    def __init__(self, flow_app):
        self.flow = flow_app
        self.icon = None
        self._running = False

    def start(self):
        try:
            import pystray
            from pystray import MenuItem, Menu

            def do_show(icon, item):   self.flow.show_window()
            def do_pause(icon, item):  self.flow.pause_for(30)
            def do_resume(icon, item): self.flow.resume()
            def do_quit(icon, item):
                try: icon.stop()
                except Exception: pass
                self.flow.quit()

            def menu_builder():
                paused = self.flow.is_paused()
                return Menu(
                    MenuItem(
                        f"● Recording" if self.flow.is_recording else "○ Ready",
                        None, enabled=False,
                    ),
                    Menu.SEPARATOR,
                    MenuItem(f"Model: {self.flow.cfg['model']}", None, enabled=False),
                    MenuItem(
                        f"Compute: {self.flow.cfg['num_workers']}w × "
                        f"{self.flow.cfg['threads_per_worker']}t",
                        None, enabled=False,
                    ),
                    Menu.SEPARATOR,
                    MenuItem("Resume" if paused else "Pause 30 min",
                             do_resume if paused else do_pause),
                    Menu.SEPARATOR,
                    MenuItem("Open Flow…", do_show, default=True),
                    Menu.SEPARATOR,
                    MenuItem("Quit", do_quit),
                )

            img = make_tray_icon_image()
            self.icon = pystray.Icon("Flow", img, "Flow", menu=menu_builder())

            def refresh_loop():
                while self._running:
                    try:
                        self.icon.menu = menu_builder()
                        self.icon.update_menu()
                    except Exception:
                        pass
                    time.sleep(2)

            self._running = True
            threading.Thread(target=refresh_loop, daemon=True).start()
            # pystray.run() blocks — must be on a non-UI thread on non-Mac,
            # and on the main thread on Mac. We run it on the main thread via
            # run_detached where possible.
            if hasattr(self.icon, "run_detached"):
                self.icon.run_detached()
            else:
                threading.Thread(target=self.icon.run, daemon=True).start()
        except Exception as e:
            print(f"[tray] failed to start: {e}")

    def stop(self):
        self._running = False
        try:
            if self.icon: self.icon.stop()
        except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

class FlowApp:
    def __init__(self):
        self.cfg   = load_config()
        self.vocab = load_vocab()
        self.suggestions = load_suggestions()

        self.root = tk.Tk()
        self.root.title("Flow")
        self.root.geometry("780x600+100+100")
        self.root.configure(bg=Theme.MARKETING_BLACK)
        self.root.minsize(640, 480)
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(200, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()

        self.mic_indicator = MicIndicator(self.root)
        self.llm = LLMCleanup(LFM_MODEL_FILE)
        self.speller = Speller()
        # Register user vocab so speller never "fixes" custom words
        self.speller.add_user_words(
            list(self.vocab.get("replacements", {}).values())
        )
        self.engine: Optional[DictationEngine] = None
        self.hotkey: Optional[HotkeyManager]   = None
        self.tray:   Optional[TrayApp]         = None
        self.onboarding: Optional[Onboarding]  = None

        self.is_recording = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close_window)

        if not self.cfg.get("onboarded"):
            self.onboarding = Onboarding(self.root, self.cfg, self._finish_onboarding)
            # Don't start engine until onboarding picks thread config
        else:
            threading.Thread(target=self._boot, daemon=True).start()

    # ── BOOT ─────────────────────────────────────────────────────────────────

    def _boot(self):
        self.engine = DictationEngine(
            cfg=self.cfg,
            llm=self.llm,
            speller=self.speller,
            on_status=self._set_status,
            on_text=self._on_transcription,
            on_control=self._on_voice_control,
        )
        # Pre-load speller so first dictation is instant
        self.speller.load()
        try:
            self.engine.load()
            self._start_hotkey()
            self._start_tray()
            self._check_pause_loop()
        except Exception as e:
            self._set_status(f"Boot error: {e}")

    def _finish_onboarding(self):
        self.cfg["onboarded"] = True
        save_config(self.cfg)
        threading.Thread(target=self._boot, daemon=True).start()

    def _start_hotkey(self):
        self.hotkey = HotkeyManager(
            self.cfg["hotkey"],
            on_press=self._on_hotkey_down,
            on_release=self._on_hotkey_up,
        )
        self.hotkey.start()
        self._set_status(f"Ready  •  {self.cfg['hotkey']} to dictate")

    def _start_tray(self):
        self.tray = TrayApp(self)
        self.tray.start()

    def _check_pause_loop(self):
        """Auto-resume when pause timer expires."""
        if self.cfg.get("paused_until", 0) > 0:
            if time.time() >= self.cfg["paused_until"]:
                self.cfg["paused_until"] = 0
                save_config(self.cfg)
                if self.hotkey: self.hotkey.resume()
                self._set_status("Resumed")
        self.root.after(5000, self._check_pause_loop)

    # ── TRAY ACTIONS ─────────────────────────────────────────────────────────

    def show_window(self):
        self.root.after(0, lambda: (self.root.deiconify(), self.root.lift()))

    def pause_for(self, minutes: int):
        self.cfg["paused_until"] = time.time() + minutes * 60
        save_config(self.cfg)
        if self.hotkey: self.hotkey.pause()
        self._set_status(f"Paused for {minutes} min")

    def resume(self):
        self.cfg["paused_until"] = 0
        save_config(self.cfg)
        if self.hotkey: self.hotkey.resume()
        self._set_status("Resumed")

    def is_paused(self) -> bool:
        return self.cfg.get("paused_until", 0) > time.time()

    def quit(self):
        self._on_close()

    # ── UI BUILD ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        bg = Theme.MARKETING_BLACK

        hdr = tk.Frame(self.root, bg=bg)
        hdr.pack(fill="x", padx=Theme.S_LG, pady=(Theme.S_LG, Theme.S_SM))

        tk.Label(hdr, text="Flow", font=Theme.FONT_H1,
                 fg=Theme.TEXT_PRIMARY, bg=bg).pack(side="left")

        self.status_lbl = tk.Label(
            hdr, text="Booting…", font=Theme.FONT_SMALL,
            fg=Theme.TEXT_TERTIARY, bg=bg,
        )
        self.status_lbl.pack(side="right")

        # Tabs
        tab_bar = tk.Frame(self.root, bg=bg)
        tab_bar.pack(fill="x", padx=Theme.S_LG, pady=(0, Theme.S_SM))
        self.current_tab = tk.StringVar(value="transcript")
        for name, key in [("Transcript", "transcript"),
                          ("Vocabulary", "vocab"),
                          ("Suggestions", "sugg"),
                          ("Settings",   "settings")]:
            tk.Button(
                tab_bar, text=name, font=Theme.FONT_SMALL,
                fg=Theme.TEXT_SECONDARY, bg=bg,
                activebackground=Theme.LEVEL3, activeforeground=Theme.TEXT_PRIMARY,
                relief="flat", bd=0, padx=12, pady=6, cursor="hand2",
                command=lambda k=key: self._switch_tab(k),
            ).pack(side="left", padx=(0, 4))

        tk.Frame(self.root, bg=Theme.BORDER_PRIMARY, height=1).pack(
            fill="x", padx=Theme.S_LG, pady=(0, Theme.S_SM))

        self.content = tk.Frame(self.root, bg=bg)
        self.content.pack(fill="both", expand=True,
                          padx=Theme.S_LG, pady=(0, Theme.S_LG))

        self.tab_frames = {
            "transcript": self._build_transcript_tab(),
            "vocab":      self._build_vocab_tab(),
            "sugg":       self._build_suggestions_tab(),
            "settings":   self._build_settings_tab(),
        }
        self._switch_tab("transcript")

        footer = tk.Frame(self.root, bg=bg)
        footer.pack(fill="x", padx=Theme.S_LG, pady=(0, Theme.S_MD))
        tk.Label(
            footer,
            text=f"Hold  {self.cfg['hotkey']}  anywhere to dictate",
            font=Theme.FONT_MICRO, fg=Theme.TEXT_TERTIARY, bg=bg,
        ).pack(side="left")

    def _build_transcript_tab(self) -> tk.Frame:
        f = tk.Frame(self.content, bg=Theme.MARKETING_BLACK)
        self.feed = scrolledtext.ScrolledText(
            f, font=Theme.FONT_BODY,
            bg=Theme.PANEL_DARK, fg=Theme.TEXT_PRIMARY,
            relief="flat", wrap="word", padx=16, pady=14,
            insertbackground=Theme.TEXT_PRIMARY,
            highlightthickness=1, highlightbackground=Theme.BORDER_PRIMARY,
        )
        self.feed.pack(fill="both", expand=True)
        self.feed.tag_config("ts",    foreground=Theme.TEXT_TERTIARY, font=Theme.FONT_MICRO)
        self.feed.tag_config("app",   foreground=Theme.VIOLET,        font=Theme.FONT_MICRO)
        self.feed.tag_config("meta",  foreground=Theme.TEXT_QUATERNARY, font=Theme.FONT_MICRO)
        self.feed.tag_config("text",  foreground=Theme.TEXT_PRIMARY)
        return f

    def _build_vocab_tab(self) -> tk.Frame:
        f = tk.Frame(self.content, bg=Theme.MARKETING_BLACK)
        tk.Label(f, text="Vocabulary", font=Theme.FONT_H3,
                 fg=Theme.TEXT_PRIMARY, bg=Theme.MARKETING_BLACK).pack(anchor="w")
        tk.Label(f,
            text="Heard-as → should-be replacements. Applied to every transcription.",
            font=Theme.FONT_SMALL, wraplength=660, justify="left",
            fg=Theme.TEXT_TERTIARY, bg=Theme.MARKETING_BLACK,
        ).pack(anchor="w", pady=(0, Theme.S_MD))

        row = tk.Frame(f, bg=Theme.MARKETING_BLACK)
        row.pack(fill="x", pady=(0, Theme.S_SM))
        self.wrong_var = tk.StringVar()
        self.right_var = tk.StringVar()
        for var, ph in [(self.wrong_var, "Heard as…"),
                        (self.right_var, "Should be…")]:
            e = tk.Entry(
                row, textvariable=var, font=Theme.FONT_BODY, width=22,
                bg=Theme.PANEL_DARK, fg=Theme.TEXT_PRIMARY,
                insertbackground=Theme.TEXT_PRIMARY, relief="flat",
                highlightthickness=1, highlightbackground=Theme.BORDER_PRIMARY,
                highlightcolor=Theme.VIOLET,
            )
            e.pack(side="left", padx=(0, Theme.S_SM), ipady=6)

        tk.Button(
            row, text="Add", font=Theme.FONT_SMALL,
            bg=Theme.INDIGO, fg="#ffffff",
            activebackground=Theme.VIOLET_HOVER, activeforeground="#ffffff",
            relief="flat", bd=0, padx=16, pady=6, cursor="hand2",
            command=self._add_vocab,
        ).pack(side="left")

        self.vocab_list = scrolledtext.ScrolledText(
            f, font=Theme.FONT_MONO,
            bg=Theme.PANEL_DARK, fg=Theme.TEXT_PRIMARY,
            relief="flat", padx=14, pady=12, height=14,
            highlightthickness=1, highlightbackground=Theme.BORDER_PRIMARY,
        )
        self.vocab_list.pack(fill="both", expand=True, pady=(Theme.S_SM, 0))
        self._refresh_vocab_display()
        return f

    def _build_suggestions_tab(self) -> tk.Frame:
        f = tk.Frame(self.content, bg=Theme.MARKETING_BLACK)
        tk.Label(f, text="Learning Suggestions", font=Theme.FONT_H3,
                 fg=Theme.TEXT_PRIMARY, bg=Theme.MARKETING_BLACK).pack(anchor="w")
        tk.Label(f,
            text="When you say 'scratch that' and re-dictate, Flow notices "
                 "the correction and suggests a vocabulary entry here.",
            font=Theme.FONT_SMALL, wraplength=660, justify="left",
            fg=Theme.TEXT_TERTIARY, bg=Theme.MARKETING_BLACK,
        ).pack(anchor="w", pady=(0, Theme.S_MD))

        self.sugg_frame = tk.Frame(f, bg=Theme.MARKETING_BLACK)
        self.sugg_frame.pack(fill="both", expand=True)
        self._refresh_suggestions_display()
        return f

    def _build_settings_tab(self) -> tk.Frame:
        f = tk.Frame(self.content, bg=Theme.MARKETING_BLACK)
        tk.Label(f, text="Settings", font=Theme.FONT_H3,
                 fg=Theme.TEXT_PRIMARY, bg=Theme.MARKETING_BLACK,
                 ).pack(anchor="w", pady=(0, Theme.S_MD))

        def row(label, builder):
            r = tk.Frame(f, bg=Theme.MARKETING_BLACK)
            r.pack(fill="x", pady=Theme.S_SM)
            tk.Label(r, text=label, font=Theme.FONT_BODY,
                     fg=Theme.TEXT_SECONDARY, bg=Theme.MARKETING_BLACK,
                     width=22, anchor="w").pack(side="left")
            builder(r)

        def mk_model(r):
            cb = ttk.Combobox(
                r, values=["distil-large-v3", "distil-medium.en",
                           "medium.en", "small.en", "base.en"],
                state="readonly", width=22, font=Theme.FONT_SMALL,
            )
            cb.set(self.cfg["model"])
            cb.bind("<<ComboboxSelected>>",
                    lambda e: self._update_cfg("model", cb.get()))
            cb.pack(side="left")
        row("Whisper model", mk_model)

        def mk_hotkey(r):
            e = tk.Entry(r, font=Theme.FONT_MONO, width=22,
                         bg=Theme.PANEL_DARK, fg=Theme.TEXT_PRIMARY,
                         insertbackground=Theme.TEXT_PRIMARY, relief="flat",
                         highlightthickness=1, highlightbackground=Theme.BORDER_PRIMARY)
            e.insert(0, self.cfg["hotkey"])
            e.bind("<FocusOut>", lambda ev: self._update_cfg("hotkey", e.get()))
            e.pack(side="left", ipady=4)
            tk.Label(r, text="  (restart to apply)",
                     font=Theme.FONT_MICRO, fg=Theme.TEXT_QUATERNARY,
                     bg=Theme.MARKETING_BLACK).pack(side="left")
        row("Hotkey", mk_hotkey)

        def mk_bool(r, key, text):
            v = tk.BooleanVar(value=self.cfg.get(key, False))
            tk.Checkbutton(
                r, text=text, variable=v,
                font=Theme.FONT_SMALL,
                fg=Theme.TEXT_SECONDARY, bg=Theme.MARKETING_BLACK,
                selectcolor=Theme.PANEL_DARK,
                activebackground=Theme.MARKETING_BLACK,
                activeforeground=Theme.TEXT_PRIMARY,
                command=lambda: self._update_cfg(key, v.get()),
            ).pack(side="left")

        row("Tier 1 regex", lambda r: mk_bool(r, "regex_cleanup", "Filler removal, grammar rules, punctuation"))
        row("Tier 2 spelling", lambda r: mk_bool(r, "spell_check", "SymSpell dictionary correction"))
        row("Tier 3 AI polish",
            lambda r: mk_bool(r, "llm_cleanup",
                              f"LFM2.5-350M {'(opt-in — say ''polish that'')' if self.llm.available else '(not installed)'}"))
        row("Sound feedback", lambda r: mk_bool(r, "sound_feedback", "Beep on start/stop"))
        row("App context",    lambda r: mk_bool(r, "active_app_context", "Tag with active app"))

        def mk_compute(r):
            tk.Label(r, text="workers:", font=Theme.FONT_MICRO,
                     fg=Theme.TEXT_TERTIARY, bg=Theme.MARKETING_BLACK).pack(side="left")
            w = ttk.Spinbox(r, from_=1, to=4, width=4, font=Theme.FONT_SMALL)
            w.set(self.cfg["num_workers"])
            w.bind("<FocusOut>", lambda e: self._update_cfg("num_workers", int(w.get())))
            w.pack(side="left", padx=4)
            tk.Label(r, text="  threads:", font=Theme.FONT_MICRO,
                     fg=Theme.TEXT_TERTIARY, bg=Theme.MARKETING_BLACK).pack(side="left")
            t = ttk.Spinbox(r, from_=1, to=8, width=4, font=Theme.FONT_SMALL)
            t.set(self.cfg["threads_per_worker"])
            t.bind("<FocusOut>", lambda e: self._update_cfg("threads_per_worker", int(t.get())))
            t.pack(side="left", padx=4)
        row("Compute", mk_compute)

        return f

    # ── TAB SWITCHING ────────────────────────────────────────────────────────

    def _switch_tab(self, key):
        self.current_tab.set(key)
        for k, frame in self.tab_frames.items():
            frame.pack_forget()
        self.tab_frames[key].pack(fill="both", expand=True)

    # ── STATE HELPERS ────────────────────────────────────────────────────────

    def _update_cfg(self, key, value):
        self.cfg[key] = value
        save_config(self.cfg)

    def _add_vocab(self):
        w = self.wrong_var.get().strip()
        r = self.right_var.get().strip()
        if w and r:
            self.vocab["replacements"][w] = r
            save_vocab(self.vocab)
            self.wrong_var.set(""); self.right_var.set("")
            self._refresh_vocab_display()

    def _refresh_vocab_display(self):
        self.vocab_list.config(state="normal")
        self.vocab_list.delete("1.0", "end")
        if self.vocab["replacements"]:
            for w, r in sorted(self.vocab["replacements"].items()):
                self.vocab_list.insert("end", f"  {w}  →  {r}\n")
        else:
            self.vocab_list.insert("end", "\n  (no replacements yet — add above or use 'scratch that')\n")
        self.vocab_list.config(state="disabled")

    def _refresh_suggestions_display(self):
        for w in self.sugg_frame.winfo_children():
            w.destroy()
        if not self.suggestions:
            tk.Label(
                self.sugg_frame,
                text="(no suggestions yet)",
                font=Theme.FONT_SMALL,
                fg=Theme.TEXT_TERTIARY, bg=Theme.MARKETING_BLACK,
            ).pack(anchor="w", pady=20)
            return

        for idx, sugg in enumerate(list(self.suggestions)):
            card = tk.Frame(
                self.sugg_frame, bg=Theme.PANEL_DARK,
                highlightbackground=Theme.BORDER_PRIMARY, highlightthickness=1,
            )
            card.pack(fill="x", pady=(0, Theme.S_SM))
            tk.Label(
                card, text=f'"{sugg["wrong"]}"  →  "{sugg["right"]}"',
                font=Theme.FONT_BODY, fg=Theme.TEXT_PRIMARY, bg=Theme.PANEL_DARK,
                padx=14, pady=8, anchor="w",
            ).pack(side="left", fill="x", expand=True)

            def accept(i=idx, s=sugg):
                self.vocab["replacements"][s["wrong"]] = s["right"]
                save_vocab(self.vocab)
                self.suggestions.pop(i)
                save_suggestions(self.suggestions)
                self._refresh_vocab_display()
                self._refresh_suggestions_display()

            def reject(i=idx):
                self.suggestions.pop(i)
                save_suggestions(self.suggestions)
                self._refresh_suggestions_display()

            tk.Button(
                card, text="Accept", font=Theme.FONT_MICRO,
                bg=Theme.INDIGO, fg="#ffffff",
                relief="flat", bd=0, padx=12, pady=4, cursor="hand2",
                command=accept,
            ).pack(side="right", padx=(0, 8))
            tk.Button(
                card, text="Reject", font=Theme.FONT_MICRO,
                bg=Theme.LEVEL3, fg=Theme.TEXT_SECONDARY,
                relief="flat", bd=0, padx=12, pady=4, cursor="hand2",
                command=reject,
            ).pack(side="right", padx=(0, 8))

    # ── CALLBACKS ────────────────────────────────────────────────────────────

    def _set_status(self, msg: str):
        try:
            self.root.after(0, lambda: self.status_lbl.config(text=msg))
        except Exception:
            pass

    def _learn_correction(self, wrong: str, right: str):
        """Called by engine when 'scratch that' auto-correction happens."""
        # Store as a suggestion (user must accept to commit)
        self.suggestions.append({
            "wrong": wrong, "right": right,
            "ts": datetime.now().isoformat(),
        })
        save_suggestions(self.suggestions)
        self.root.after(0, self._refresh_suggestions_display)

    def _on_transcription(self, text: str, app: str, rtfx: float, logprob: float):
        ts = datetime.now().strftime("%H:%M:%S")
        def _do():
            self.feed.insert("end", f"\n{ts}  ", "ts")
            if app:
                self.feed.insert("end", f"@{app}  ", "app")
            self.feed.insert("end", f"RTFx {rtfx:.1f}x\n", "meta")
            self.feed.insert("end", f"{text}\n", "text")
            self.feed.see("end")
            if self.onboarding:
                self.onboarding.update_practice(text)
        self.root.after(0, _do)

    def _on_voice_control(self, action: str):
        """Called when engine detects 'pause flow' / 'resume flow'."""
        if action == "pause":
            self.pause_for(30)
        elif action == "resume":
            self.resume()

    def _on_hotkey_down(self):
        if self.is_paused(): return
        self.is_recording = True
        self.root.after(0, lambda: self.mic_indicator.show("● Listening"))
        if self.engine: self.engine.start_recording()

    def _on_hotkey_up(self):
        self.is_recording = False
        self.root.after(0, self.mic_indicator.hide)
        if self.engine:
            threading.Thread(
                target=lambda: self.engine.stop_recording_and_transcribe(
                    self.vocab, vocab_callback=self._learn_correction,
                ),
                daemon=True,
            ).start()

    # ── LIFECYCLE ────────────────────────────────────────────────────────────

    def _on_close_window(self):
        """Close window → hide to tray, keep app running."""
        self.root.withdraw()

    def _on_close(self):
        try:
            if self.hotkey: self.hotkey.stop()
            if self.tray:   self.tray.stop()
        except Exception: pass
        try:
            if hasattr(self, 'engine') and hasattr(self.engine, '_kill_parakeet'):
                self.engine._kill_parakeet()
        except Exception: pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────────────────────────────────────

def check_deps():
    missing = []
    for pkg, imp in [
        ("faster-whisper", "faster_whisper"),
        ("sounddevice",    "sounddevice"),
        ("pynput",         "pynput"),
        ("pyperclip",      "pyperclip"),
        ("pystray",        "pystray"),
        ("Pillow",         "PIL"),
        ("numpy",          "numpy"),
        ("symspellpy",     "symspellpy"),
    ]:
        try: __import__(imp)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"pip install {' '.join(missing)}")
        sys.exit(1)


if __name__ == "__main__":
    check_deps()
    FlowApp().run()
