// Flow — frontend
// Talks to Python backend via HTTP + WebSocket (no pywebview dependency)

let currentStep = 0;
const TOTAL_STEPS = 5;

// ─── API bridge ───────────────────────────────────────────────────────────────
// Works in both modes:
//   1. Tauri/browser: HTTP POST to /api/<method>
//   2. Legacy pywebview: window.pywebview.api.<method>  (fallback)

const BASE_URL = window.__FLOW_BASE__ || "http://127.0.0.1:7878";

async function api(method, ...args) {
  // pywebview fallback (dev mode)
  if (window.pywebview && window.pywebview.api && window.pywebview.api[method]) {
    try {
      return await window.pywebview.api[method](...args);
    } catch (e) {
      console.error("[api/pywebview]", method, e);
      return null;
    }
  }
  // HTTP mode
  try {
    const body = args.length ? { args } : {};
    const r = await fetch(`${BASE_URL}/api/${method}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) { console.error("[api]", method, r.status); return null; }
    const j = await r.json();
    if (j.error) { console.error("[api]", method, j.error); return null; }
    return j.result;
  } catch (e) {
    console.error("[api]", method, e);
    return null;
  }
}

// ─── WebSocket — real-time state + word streaming ─────────────────────────────

let _ws = null;
let _wsReady = false;
let _wsReconnectMs = 500;

function connectWS() {
  const wsUrl = BASE_URL.replace("http", "ws") + "/ws";
  try {
    _ws = new WebSocket(wsUrl);
  } catch (e) {
    setTimeout(connectWS, _wsReconnectMs);
    return;
  }

  _ws.onopen = () => {
    _wsReady = true;
    _wsReconnectMs = 500;
    setInterval(() => _ws && _ws.readyState === 1 && _ws.send("ping"), 20000);
  };

  _ws.onmessage = (ev) => {
    try {
      const msg = JSON.parse(ev.data);
      if (msg.type === "state") applyState(msg.data);
      if (msg.type === "words") pushWords(msg.words);
    } catch (e) {}
  };

  _ws.onerror = () => {};
  _ws.onclose = () => {
    _wsReady = false;
    _wsReconnectMs = Math.min(_wsReconnectMs * 2, 8000);
    setTimeout(connectWS, _wsReconnectMs);
  };
}

// ─── State application (replaces pollState) ──────────────────────────────────

let lastTranscriptCount = 0;
let _lowConfTipShown = false;

function applyState(state) {
  if (!state) return;

  // Status dot
  const dot = document.getElementById("status-dot");
  const txt = document.getElementById("status-text");
  if (dot) dot.classList.toggle("recording", !!state.recording);
  if (txt) {
    if (state.recording) {
      txt.textContent = state.status || "Listening…";
    } else if (state.status && state.status !== "Ready"
               && !state.status.startsWith("Starting")
               && !state.status.startsWith("Welcome")) {
      txt.textContent = state.status;
    } else {
      const hk = (window._flowHotkey || "Alt+Space")
        .replace(/<|>/g, "").replace(/\+/g, " + ");
      txt.innerHTML = `<span style="opacity:0.5;font-size:11px">${hk} to speak</span>`;
    }
  }

  // Mic buttons
  document.querySelectorAll(".mic-btn").forEach(btn =>
    btn.classList.toggle("recording", !!state.recording)
  );

  // RTFx
  if (state.last_rtfx) {
    const meta = document.getElementById("status-meta");
    if (meta && !state.recording)
      meta.textContent = `RTFx ${state.last_rtfx.toFixed(1)}x`;
  }

  // Mode pills
  if (state.mode) {
    ["dictation", "meeting", "capture"].forEach(m => {
      const btn = document.getElementById(`mode-${m}`);
      if (btn) btn.classList.toggle("active", m === state.mode);
    });
  }

  // Language notifier
  const langNotif = document.getElementById("lang-notifier");
  if (state.detected_lang && state.recording) {
    if (!langNotif) {
      const n = document.createElement("div");
      n.id = "lang-notifier";
      n.style.cssText =
        "position:fixed;bottom:120px;left:50%;transform:translateX(-50%);" +
        "padding:8px 16px;background:rgba(94,106,210,0.95);color:#fff;" +
        "border-radius:20px;font-size:13px;z-index:9999;cursor:pointer;" +
        "backdrop-filter:blur(10px);box-shadow:0 4px 12px rgba(0,0,0,0.3);" +
        "display:flex;gap:10px;align-items:center;";
      n.innerHTML =
        `<span>Detected: <b>${state.detected_lang.toUpperCase()}</b></span>` +
        `<button onclick="setLang('${state.detected_lang}')" ` +
        `style="background:#fff;color:#5e6ad2;border:none;border-radius:12px;` +
        `padding:3px 10px;font-size:11px;cursor:pointer;font-weight:600;">Use</button>` +
        `<button onclick="dismissLang()" ` +
        `style="background:transparent;color:rgba(255,255,255,0.7);border:none;` +
        `cursor:pointer;font-size:14px;">✕</button>`;
      document.body.appendChild(n);
    }
  } else if (langNotif) {
    langNotif.remove();
  }

  // Speed toast / low-conf tip (consume from state)
  if (state.speed_toast) {
    showSpeedToast(state.speed_toast);
    // Mark consumed server-side
    api("pop_toast").catch(() => {});
  }
  if (state.low_conf_tip && !_lowConfTipShown) {
    _lowConfTipShown = true;
    showLowConfTip();
    api("pop_toast").catch(() => {});
  }

  // New transcriptions
  if (state.transcriptions && state.transcriptions.length > lastTranscriptCount) {
    const feed = document.getElementById("feed");
    if (feed) {
      const emptyEl = document.getElementById("feed-empty-state");
      if (emptyEl) emptyEl.remove();
      const streamEl = feed.querySelector(".streaming");
      if (streamEl) streamEl.remove();
      for (let i = lastTranscriptCount; i < state.transcriptions.length; i++) {
        const t = state.transcriptions[i];
        const entry = document.createElement("div");
        entry.className = "entry";
        const appBadge = t.app && t.app !== "capture"
          ? `<span class="entry-app" style="display:inline-flex;align-items:center;gap:3px">
               <span style="opacity:0.5;font-size:9px">↑</span>${escape(t.app)}
             </span>`
          : t.app === "capture"
          ? `<span class="entry-app" style="color:var(--sage)">💭 thought</span>`
          : "";
        entry.innerHTML = `
          <div class="entry-meta"><span>${escape(t.time)}</span>${appBadge}</div>
          <div class="entry-text">${escape(t.text)}</div>
        `;
        feed.appendChild(entry);
      }
      feed.scrollTop = feed.scrollHeight;
      lastTranscriptCount = state.transcriptions.length;
    }
  }
}

// ─── Word streaming ───────────────────────────────────────────────────────────

function pushWords(words) {
  if (!words || !words.length) return;
  const feed = document.getElementById("feed");
  if (!feed) return;
  let streamEl = feed.querySelector(".streaming");
  if (!streamEl) {
    streamEl = document.createElement("div");
    streamEl.className = "entry streaming";
    streamEl.innerHTML = '<div class="entry-text"></div>';
    feed.appendChild(streamEl);
  }
  const textEl = streamEl.querySelector(".entry-text");
  const oldCursor = textEl.querySelector(".cursor-blink");
  if (oldCursor) oldCursor.remove();
  for (const w of words) {
    const span = document.createElement("span");
    span.className = "word-fade";
    span.textContent = (textEl.childNodes.length > 0 ? " " : "") + w;
    textEl.appendChild(span);
  }
  const cursor = document.createElement("span");
  cursor.className = "cursor-blink";
  cursor.textContent = "▋";
  textEl.appendChild(cursor);
  feed.scrollTop = feed.scrollHeight;
}

// ─── Toasts ───────────────────────────────────────────────────────────────────

function showSpeedToast(msg) {
  let t = document.getElementById("speed-toast");
  if (!t) {
    t = document.createElement("div");
    t.id = "speed-toast";
    t.style.cssText =
      "position:fixed;bottom:90px;left:50%;transform:translateX(-50%) translateY(8px);" +
      "padding:7px 14px;background:rgba(122,184,145,0.18);color:rgba(255,255,255,0.9);" +
      "border:1px solid rgba(122,184,145,0.35);border-radius:20px;font-size:11px;" +
      "z-index:9999;backdrop-filter:blur(12px);opacity:0;" +
      "transition:opacity 0.25s,transform 0.25s;pointer-events:none;" +
      "white-space:nowrap;font-weight:500;";
    document.body.appendChild(t);
  }
  t.textContent = "⚡ " + msg;
  t.style.opacity = "1";
  t.style.transform = "translateX(-50%) translateY(0)";
  clearTimeout(t._timer);
  t._timer = setTimeout(() => {
    t.style.opacity = "0";
    t.style.transform = "translateX(-50%) translateY(4px)";
  }, 2800);
}

function showLowConfTip() {
  const feed = document.getElementById("feed");
  if (!feed) return;
  const tip = document.createElement("div");
  tip.className = "entry";
  tip.style.cssText = "background:rgba(212,154,74,0.10);border:1px solid rgba(212,154,74,0.25);";
  tip.innerHTML = `
    <div class="entry-text" style="font-size:11px;color:var(--ink-3)">
      💡 Not quite right? Say <span class="kbd" style="font-size:10px">scratch that</span>
      to undo and re-speak.
    </div>
  `;
  feed.appendChild(tip);
  feed.scrollTop = feed.scrollHeight;
  setTimeout(() => tip.remove(), 8000);
}

function showToast(msg) {
  let t = document.getElementById("toast");
  if (!t) {
    t = document.createElement("div");
    t.id = "toast";
    t.style.cssText =
      "position:fixed;bottom:24px;left:50%;transform:translateX(-50%);" +
      "padding:10px 16px;background:rgba(20,22,28,0.9);color:#f7f8f8;" +
      "border-radius:8px;font-size:12px;z-index:9999;backdrop-filter:blur(20px);" +
      "box-shadow:0 8px 24px rgba(0,0,0,0.5),0 0 0 1px rgba(255,255,255,0.1) inset;";
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.style.opacity = "1";
  clearTimeout(t._timer);
  t._timer = setTimeout(() => {
    t.style.opacity = "0";
    t.style.transition = "opacity 0.3s";
  }, 3000);
}

// ─── Onboarding ───────────────────────────────────────────────────────────────

function showStep(n) {
  document.querySelectorAll(".onb-step").forEach(el =>
    el.classList.toggle("active", parseInt(el.dataset.step) === n)
  );
  document.querySelectorAll(".onb-dot").forEach(d => {
    const i = parseInt(d.dataset.step);
    d.classList.toggle("active", i === n);
    d.classList.toggle("done", i < n);
  });
  currentStep = n;
  if (window.lucide) lucide.createIcons();
  if (n === 1) { document.getElementById("perm-continue")?.setAttribute("disabled","true"); checkPerms(); }
  if (n === 2) runBenchmark();
  if (n === 3) {
    document.getElementById("practice-result").textContent = "Waiting for you to speak…";
    document.getElementById("practice-result").classList.remove("has-text");
  }
}

function next() { if (currentStep < TOTAL_STEPS - 1) showStep(currentStep + 1); }
function prev() { if (currentStep > 0) showStep(currentStep - 1); }

async function grantMic() {
  document.getElementById("perm-mic-btn").textContent = "Asking…";
  await api("trigger_mic_prompt");
  setTimeout(checkPerms, 600);
}

async function grantAcc() {
  document.getElementById("perm-acc-btn").textContent = "Opening…";
  await api("trigger_accessibility_prompt");
  setTimeout(checkPerms, 1000);
}

async function checkPerms() {
  const p = await api("check_permissions");
  if (!p) return;
  const micItem = document.getElementById("perm-mic");
  const micBtn  = document.getElementById("perm-mic-btn");
  const accItem = document.getElementById("perm-acc");
  const accBtn  = document.getElementById("perm-acc-btn");
  if (p.mic) {
    micItem?.classList.add("granted");
    if (micBtn) micBtn.outerHTML = '<span class="perm-status-ok">✓</span>';
  } else {
    if (micBtn && micBtn.textContent !== "Grant") micBtn.textContent = "Grant";
  }
  if (p.accessibility) {
    accItem?.classList.add("granted");
    if (accBtn) accBtn.outerHTML = '<span class="perm-status-ok">✓</span>';
  } else {
    if (accBtn && accBtn.textContent !== "Grant") accBtn.textContent = "Grant";
  }
  const cont = document.getElementById("perm-continue");
  if (p.mic && p.accessibility) cont?.removeAttribute("disabled");
}

async function runBenchmark() {
  const txt = document.getElementById("bench-text");
  const continueBtn = document.getElementById("bench-continue");
  const cores = navigator.hardwareConcurrency || "a few";
  const messages = [
    "Waking up the speech engine…",
    `Found ${cores} CPU cores. Nice machine.`,
    "Teaching it to listen really, really well…",
    "Calibrating for your voice…",
    "Making sure it's faster than your typing speed…",
    "Almost there…",
  ];
  const benchPromise = api("run_benchmark");
  let i = 0;
  const interval = setInterval(() => {
    if (i < messages.length) txt.textContent = messages[i++];
  }, 600);
  const result = await benchPromise;
  await new Promise(r => setTimeout(r, Math.max(0, 2500 - i * 600)));
  clearInterval(interval);
  if (result && result.best) {
    const qLabel = result.quality
      ? result.quality.charAt(0).toUpperCase() + result.quality.slice(1) : "";
    txt.textContent = `✓ ${result.cores} cores · ${qLabel} quality · optimised for your machine`;
    txt.style.color = "var(--ok)";
    if (result.quality) _syncQualityPills(result.model || QUALITY_MAP[result.quality]?.model);
  } else {
    txt.textContent = "✓ Configured with defaults";
    txt.style.color = "var(--ok)";
  }
  txt.parentElement?.querySelector(".spinner")?.style?.setProperty("display","none");
  continueBtn.disabled = false;
}

async function finishOnboarding() {
  await api("finish_onboarding");
  document.getElementById("onboarding").classList.add("hidden");
  document.getElementById("main").classList.remove("hidden");
  initMainApp();
}

// ─── Main app ─────────────────────────────────────────────────────────────────

async function initMainApp() {
  const cfg = await api("get_config");
  if (cfg) {
    const set = (id, val) => { const e = document.getElementById(id); if (e) e.value = val; };
    const setChk = (id, val) => { const e = document.getElementById(id); if (e) e.checked = !!val; };
    set("set-model", cfg.model || "distil-large-v3");
    _syncQualityPills(cfg.model || "distil-large-v3");
    if (cfg.hotkey) window._flowHotkey = cfg.hotkey;
    const loginChk = document.getElementById("set-login");
    if (loginChk) loginChk.checked = !!cfg.launch_at_login;
    set("set-workers", cfg.num_workers || 2);
    set("set-threads", cfg.threads_per_worker || 3);
    set("set-hotkey", cfg.hotkey || "<alt>+<space>");
    setChk("set-regex", cfg.regex_cleanup !== false);
    setChk("set-spell", cfg.spell_check !== false);
    setChk("set-llm", cfg.llm_cleanup);
    setChk("set-sound", cfg.sound_feedback !== false);
    setChk("set-context", cfg.active_app_context !== false);
    const sm = document.getElementById("status-meta");
    if (sm) sm.textContent = cfg.model || "—";
    loadMicPicker(cfg.mic_device ?? null);
  }

  ["set-model","set-workers","set-threads","set-hotkey",
   "set-regex","set-spell","set-llm","set-sound","set-context"].forEach(id => {
    document.getElementById(id)?.addEventListener("change", saveSettings);
  });

  refreshVocab();
  connectWS();

  // Fallback poll if WS unavailable (pywebview mode or WS failure)
  setInterval(async () => {
    if (_wsReady) return;
    const state = await api("get_state");
    if (state) applyState(state);
  }, 800);

  // Stats ticker
  setInterval(async () => {
    const s = await api("get_stats");
    if (!s || !s.words) return;
    const meta = document.getElementById("status-meta");
    const rec = document.getElementById("status-dot")?.classList.contains("recording");
    if (meta && !rec)
      meta.textContent = s.words > 0 ? `${s.words} words · ${s.time_str} saved today` : "";
  }, 5000);

  if (window.lucide) lucide.createIcons();
}

function switchView(view) {
  document.querySelectorAll(".dock-view-btn").forEach(b =>
    b.classList.toggle("active", b.dataset.view === view)
  );
  document.querySelectorAll(".view").forEach(v =>
    v.classList.toggle("active", v.dataset.view === view)
  );
  if (view === "vocab") refreshVocab();
  if (view === "history") loadHistoryView();
  if (view === "thoughts") loadThoughtsView();
  if (view === "settings") { loadMicPicker(); loadModelList(); }
  if (window.lucide) lucide.createIcons();
}

async function toggleRecording() {
  await api("toggle_recording");
}

// ─── Mic picker ───────────────────────────────────────────────────────────────

async function loadMicPicker(currentDevice) {
  const sel = document.getElementById("set-mic");
  if (!sel) return;
  const mics = await api("list_mics");
  if (!mics) return;
  while (sel.options.length > 1) sel.remove(1);
  mics.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m.index;
    opt.textContent = m.name;
    if (m.index === currentDevice) opt.selected = true;
    sel.appendChild(opt);
  });
  if (currentDevice == null) sel.value = "default";
}

async function setMic(val) {
  await api("set_mic", val === "default" ? "default" : parseInt(val));
  showToast("Microphone updated — takes effect on next recording");
}

async function setLoginToggle(enabled) {
  const ok = await api("set_launch_at_login", enabled);
  if (!ok) {
    document.getElementById("set-login").checked = !enabled;
    showToast("Could not update login setting");
  } else {
    showToast(enabled ? "Flow will start at login" : "Login start disabled");
  }
}

// ─── Model download ───────────────────────────────────────────────────────────

const QUALITY_MAP = {
  fast:     { model: "tiny.en",          workers: 1, threads: 2 },
  balanced: { model: "distil-medium.en", workers: 2, threads: 3 },
  best:     { model: "distil-large-v3",  workers: 2, threads: 4 },
};

const MODEL_SIZES = { "tiny.en": 75, "distil-medium.en": 394, "distil-large-v3": 756 };

async function loadModelList() {
  const el = document.getElementById("model-download-list");
  if (!el) return;
  el.innerHTML = "";
  for (const [modelId, sizeMb] of Object.entries(MODEL_SIZES)) {
    const status = await api("check_model_cached", modelId);
    const cached = status && status.cached;
    const row = document.createElement("div");
    row.className = "setting-row";
    row.id = `model-row-${modelId.replace(/\./g, "-")}`;
    row.innerHTML = `
      <div class="setting-label">
        <div class="setting-name">${modelId}</div>
        <div class="setting-desc">${cached
          ? `✓ Downloaded · ${status.size_mb} MB`
          : `${sizeMb} MB · not yet downloaded`}</div>
      </div>
      ${cached
        ? `<span style="font-size:11px;color:var(--ok)">Ready</span>`
        : `<button class="btn-primary" style="padding:5px 14px;font-size:11px;margin:0"
             id="dl-btn-${modelId.replace(/\./g,"-")}"
             onclick="startDownload('${modelId}')">Download</button>`}
    `;
    el.appendChild(row);
  }
}

let _dlPollTimer = null;
async function startDownload(modelId) {
  const btn = document.getElementById(`dl-btn-${modelId.replace(/\./g, "-")}`);
  if (btn) { btn.disabled = true; btn.textContent = "Starting…"; }
  await api("download_model", modelId);
  _dlPollTimer = setInterval(() => pollDownload(modelId), 800);
}

async function pollDownload(modelId) {
  const p = await api("get_download_progress");
  if (!p) return;
  const btn = document.getElementById(`dl-btn-${modelId.replace(/\./g, "-")}`);
  if (p.error) {
    clearInterval(_dlPollTimer);
    if (btn) { btn.disabled = false; btn.textContent = "Retry"; }
    showToast("Download failed: " + p.error);
    return;
  }
  if (btn) btn.textContent = p.pct < 100 ? `${p.pct}%` : "Installing…";
  if (p.done && p.pct >= 100) {
    clearInterval(_dlPollTimer);
    showToast(`✓ ${modelId} ready`);
    loadModelList();
  }
}

// ─── Quality picker ───────────────────────────────────────────────────────────

async function setQuality(level) {
  const q = QUALITY_MAP[level];
  if (!q) return;
  document.querySelectorAll(".quality-pill").forEach(p =>
    p.classList.toggle("active", p.id === `q-${level}`)
  );
  document.getElementById("set-model")?.setAttribute("value", q.model);
  const sm = document.getElementById("status-meta");
  if (sm) sm.textContent = q.model;
  await api("update_config", { model: q.model, num_workers: q.workers, threads_per_worker: q.threads });
  showToast(`Quality set to ${level.charAt(0).toUpperCase() + level.slice(1)}`);
}

function _syncQualityPills(model) {
  const level = Object.entries(QUALITY_MAP).find(([, v]) => v.model === model)?.[0];
  if (level) {
    document.querySelectorAll(".quality-pill").forEach(p =>
      p.classList.toggle("active", p.id === `q-${level}`)
    );
  }
}

// ─── File / URL transcription ─────────────────────────────────────────────────

async function handleDrop(e) {
  e.preventDefault();
  e.currentTarget.style.background = "rgba(255,255,255,0.02)";
  await browseAndTranscribe();
}

async function handleFileSelect(e) {
  await browseAndTranscribe();
}

async function browseAndTranscribe() {
  // Tauri: use dialog API if available
  if (window.__TAURI__) {
    try {
      const { open } = window.__TAURI__.dialog;
      const path = await open({
        multiple: false,
        filters: [{ name: "Audio/Video", extensions: ["mp3","wav","m4a","flac","ogg","mp4","mov","webm","opus"] }],
      });
      if (!path) return;
      showFileResult("⏳ Transcribing…", "loading");
      const r = await api("transcribe_file", path);
      renderFileResult(r);
      return;
    } catch (e) {}
  }
  // Fallback: server pick dialog
  const path = await api("pick_file_via_dialog");
  if (!path) return;
  showFileResult("⏳ Transcribing…", "loading");
  const r = await api("transcribe_file", path);
  renderFileResult(r);
}

function renderFileResult(r) {
  if (r && r.ok) {
    showFileResult(
      `<div class="entry-meta" style="margin-bottom:8px">${r.duration.toFixed(1)}s audio · ${r.elapsed.toFixed(1)}s · RTFx ${r.rtfx.toFixed(1)}x · ${r.language}</div>` +
      `<div class="entry-text">${escape(r.text)}</div>`, "ok"
    );
  } else {
    showFileResult("Error: " + (r ? r.error : "unknown"), "err");
  }
}

async function handleUrl() {
  const url = document.getElementById("url-input").value.trim();
  if (!url) return;
  showFileResult("⏳ Downloading and transcribing…", "loading");
  const r = await api("transcribe_url", url);
  renderFileResult(r);
}

function showFileResult(html, kind) {
  const el = document.getElementById("file-result");
  el.classList.remove("hidden");
  el.innerHTML = html;
}

// ─── History / Thoughts ───────────────────────────────────────────────────────

let _searchTimer = null;
async function handleSearch(e) {
  clearTimeout(_searchTimer);
  _searchTimer = setTimeout(async () => {
    const q = e.target.value.trim();
    const results = q ? await api("search_history", q) : await api("get_history");
    renderHistory(results || []);
  }, 200);
}

async function loadHistoryView() {
  renderHistory(await api("get_history", 100) || []);
}

function renderHistory(items) {
  const list = document.getElementById("history-list");
  if (!items || !items.length) {
    list.innerHTML = '<div class="feed-empty"><div class="feed-empty-text">No transcripts found</div></div>';
    return;
  }
  list.innerHTML = "";
  items.forEach(item => {
    const card = document.createElement("div");
    card.className = "entry";
    const time = item.ts ? new Date(item.ts).toLocaleString() : "";
    card.innerHTML = `
      <div class="entry-meta">
        <span>${escape(time)}</span>
        ${item.app ? `<span class="entry-app">${escape(item.app)}</span>` : ""}
      </div>
      <div class="entry-text">${escape(item.text || "")}</div>
    `;
    list.appendChild(card);
  });
}

async function loadThoughtsView() {
  const items = await api("get_thoughts", 100);
  const list = document.getElementById("thoughts-list");
  if (!items || !items.length) {
    list.innerHTML =
      '<div class="feed-empty"><div style="font-size:24px;margin-bottom:8px;opacity:0.4">💭</div>' +
      '<div class="feed-empty-text">No captured thoughts yet</div>' +
      '<div style="font-size:11px;color:var(--ink-4);margin-top:4px">Double-tap your shortcut to enter capture mode</div></div>';
    return;
  }
  list.innerHTML = "";
  items.forEach(item => {
    const card = document.createElement("div");
    card.className = "entry";
    const time = item.ts ? new Date(item.ts).toLocaleString() : "";
    card.innerHTML = `
      <div class="entry-meta"><span>${escape(time)}</span></div>
      <div class="entry-text">${escape(item.text || "")}</div>
    `;
    list.appendChild(card);
  });
}

// ─── Mode / settings ─────────────────────────────────────────────────────────

async function setMode(mode) {
  await api("set_mode", mode);
  ["dictation","meeting","capture"].forEach(m => {
    document.getElementById(`mode-${m}`)?.classList.toggle("active", m === mode);
  });
  if (mode === "capture") switchView("thoughts");
}

async function saveSettings() {
  const cfg = {
    model:              document.getElementById("set-model")?.value,
    num_workers:        parseInt(document.getElementById("set-workers")?.value),
    threads_per_worker: parseInt(document.getElementById("set-threads")?.value),
    hotkey:             document.getElementById("set-hotkey")?.value,
    regex_cleanup:      document.getElementById("set-regex")?.checked,
    spell_check:        document.getElementById("set-spell")?.checked,
    llm_cleanup:        document.getElementById("set-llm")?.checked,
    sound_feedback:     document.getElementById("set-sound")?.checked,
    active_app_context: document.getElementById("set-context")?.checked,
  };
  await api("update_config", cfg);
}

async function addVocab() {
  const wrong = document.getElementById("vocab-wrong").value.trim();
  const right = document.getElementById("vocab-right").value.trim();
  if (!wrong || !right) return;
  await api("add_vocab", wrong, right);
  document.getElementById("vocab-wrong").value = "";
  document.getElementById("vocab-right").value = "";
  refreshVocab();
}

async function refreshVocab() {
  const list = await api("get_vocab");
  const el = document.getElementById("vocab-list");
  if (!list || !Object.keys(list).length) {
    el.innerHTML = '<div class="empty-state sm"><div class="empty-text">No replacements yet — add above or use "scratch that"</div></div>';
    return;
  }
  el.innerHTML = "";
  for (const [w, r] of Object.entries(list).sort()) {
    const row = document.createElement("div");
    row.className = "vocab-row";
    row.innerHTML = `<span>${escape(w)}</span><span class="arrow">→</span><span>${escape(r)}</span>`;
    el.appendChild(row);
  }
}

async function refreshSuggestions() {
  const list = await api("get_suggestions");
  const el = document.getElementById("suggestions-list");
  const badge = document.getElementById("sugg-badge");
  if (!el) return;
  if (!list || !list.length) {
    el.innerHTML = '<div class="feed-empty"><div class="feed-empty-text">No suggestions yet</div></div>';
    if (badge) { badge.classList.remove("has-count"); badge.textContent = ""; }
    return;
  }
  if (badge) { badge.classList.add("has-count"); badge.textContent = list.length; }
  el.innerHTML = "";
  list.forEach((s, i) => {
    const card = document.createElement("div");
    card.className = "entry";
    card.innerHTML = `
      <div class="entry-meta">
        <span>"${escape(s.wrong)}"</span>
        <span class="entry-app">→</span>
        <span>"${escape(s.right)}"</span>
      </div>
      <div style="display:flex;gap:8px;margin-top:8px">
        <button class="btn-primary" style="padding:5px 12px;font-size:11px;margin-top:0"
          onclick="acceptSuggestion(${i})">Accept</button>
        <button class="btn-ghost" style="padding:5px 12px;font-size:11px"
          onclick="rejectSuggestion(${i})">Reject</button>
      </div>
    `;
    el.appendChild(card);
  });
}

async function acceptSuggestion(i) {
  await api("accept_suggestion", i); refreshVocab(); refreshSuggestions();
}
async function rejectSuggestion(i) {
  await api("reject_suggestion", i); refreshSuggestions();
}

async function setLang(code) {
  await api("set_language", code);
  document.getElementById("lang-notifier")?.remove();
  showToast(`Language set to ${code.toUpperCase()}`);
}
async function dismissLang() {
  await api("dismiss_lang");
  document.getElementById("lang-notifier")?.remove();
}

async function detachToMini() {
  await api("detach_to_mini");
}

// ─── Utility ──────────────────────────────────────────────────────────────────

function escape(s) {
  return String(s)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

// ─── Boot ─────────────────────────────────────────────────────────────────────

async function boot() {
  // Try HTTP first, then pywebview fallback
  let cfg = null;
  try {
    const r = await fetch(`${BASE_URL}/api/get_config`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
    });
    if (r.ok) { const j = await r.json(); cfg = j.result; }
  } catch (e) {
    // pywebview mode
    if (window.pywebview) {
      await new Promise(res => {
        const t = setInterval(() => { if (window.pywebview?.api) { clearInterval(t); res(); }}, 50);
        setTimeout(() => { clearInterval(t); res(); }, 8000);
      });
      cfg = await window.pywebview.api.get_config?.();
    }
  }

  if (cfg && cfg.onboarded) {
    document.getElementById("onboarding").classList.add("hidden");
    document.getElementById("main").classList.remove("hidden");
    initMainApp();
  } else {
    showStep(0);
  }
  if (window.lucide) lucide.createIcons();
}

window.addEventListener("pywebviewready", boot);
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => setTimeout(boot, 100));
} else {
  setTimeout(boot, 300);
}
