use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, TrayIconBuilder, TrayIconEvent},
    AppHandle, Manager, RunEvent, WebviewUrl, WebviewWindowBuilder,
};
use tauri_plugin_shell::{process::CommandChild, ShellExt};

// ─── Python server process ────────────────────────────────────────────────────

struct Server {
    child: CommandChild,
    port: u16,
    base_url: String,
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

fn read_port_file() -> Option<u16> {
    let path = dirs_home().join(".flow").join("port");
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

fn spawn_server(app: &AppHandle) -> Result<Server, String> {
    // Remove stale port file so we wait for a fresh write from this run
    let _ = std::fs::remove_file(dirs_home().join(".flow").join("port"));

    // Spawn the bundled flow_server sidecar (no Python required on target)
    let sidecar = app
        .shell()
        .sidecar("flow_server")
        .map_err(|e| format!("flow_server sidecar not found: {e}\n(run build.sh first to create it)"))?;

    let (_rx, child) = sidecar
        .spawn()
        .map_err(|e| format!("Failed to start server: {e}"))?;

    // Poll until the server writes its port and responds to /health
    let deadline = Instant::now() + Duration::from_secs(30);
    let port = loop {
        if Instant::now() > deadline {
            let _ = child.kill();
            return Err("Server did not start within 30 seconds".into());
        }
        if let Some(p) = read_port_file() {
            let url = format!("http://127.0.0.1:{p}/health");
            if ureq::get(&url)
                .timeout(Duration::from_secs(2))
                .call()
                .map(|r| r.status() == 200)
                .unwrap_or(false)
            {
                break p;
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    };

    Ok(Server {
        child,
        port,
        base_url: format!("http://127.0.0.1:{port}"),
    })
}

// ─── Tauri commands (callable from JS if needed) ──────────────────────────────

#[tauri::command]
fn show_main(app: AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.show();
        let _ = w.set_focus();
    }
}

#[tauri::command]
fn hide_main(app: AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.hide();
    }
}

// ─── App entry ────────────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let server_state: Arc<Mutex<Option<Server>>> = Arc::new(Mutex::new(None));
    let server_state_clone = server_state.clone();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(move |app| {
            let handle = app.handle().clone();

            match spawn_server(&handle) {
                Ok(srv) => {
                    let base_url = srv.base_url.clone();
                    *server_state_clone.lock().unwrap() = Some(srv);
                    build_mini_window(&handle, &base_url)?;
                    build_main_window(&handle, &base_url)?;
                    build_tray(&handle)?;
                }
                Err(e) => {
                    eprintln!("Flow startup error: {e}");
                    // Give the error a moment to be visible in logs, then exit
                    std::thread::spawn(|| {
                        std::thread::sleep(Duration::from_millis(500));
                        std::process::exit(1);
                    });
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![show_main, hide_main])
        .build(tauri::generate_context!())
        .expect("error building app")
        .run(move |_app, event| {
            if let RunEvent::Exit = event {
                if let Ok(mut guard) = server_state.lock() {
                    if let Some(srv) = guard.take() {
                        let _ = srv.child.kill();
                    }
                }
            }
        });
}

// ─── Window builders ─────────────────────────────────────────────────────────

fn build_mini_window(app: &AppHandle, base_url: &str) -> Result<(), tauri::Error> {
    let url = format!("{base_url}/mini_bar.html");
    WebviewWindowBuilder::new(app, "mini", WebviewUrl::External(url.parse().unwrap()))
        .title("Flow")
        .inner_size(220.0, 52.0)
        .min_inner_size(220.0, 52.0)
        .max_inner_size(220.0, 52.0)
        .decorations(false)
        .always_on_top(true)
        .resizable(false)
        .skip_taskbar(true)
        .focused(false)
        .visible(true)
        .build()?;
    Ok(())
}

fn build_main_window(app: &AppHandle, base_url: &str) -> Result<(), tauri::Error> {
    WebviewWindowBuilder::new(app, "main", WebviewUrl::External(base_url.parse().unwrap()))
        .title("Flow Settings")
        .inner_size(420.0, 680.0)
        .min_inner_size(380.0, 560.0)
        .decorations(true)
        .resizable(true)
        .skip_taskbar(false)
        .focused(true)
        .visible(true)
        .center()
        .build()?;
    Ok(())
}

// ─── System tray ─────────────────────────────────────────────────────────────

fn build_tray(app: &AppHandle) -> Result<(), tauri::Error> {
    let open_i = MenuItem::with_id(app, "open", "Open Flow",  true, None::<&str>)?;
    let quit_i  = MenuItem::with_id(app, "quit", "Quit Flow", true, None::<&str>)?;
    let menu    = Menu::with_items(app, &[&open_i, &quit_i])?;

    let handle = app.clone();
    TrayIconBuilder::with_id("tray")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .tooltip("Flow — offline dictation")
        .on_menu_event(move |_tray, event| match event.id.as_ref() {
            "open" => {
                if let Some(w) = handle.get_webview_window("main") {
                    let _ = w.show();
                    let _ = w.set_focus();
                }
            }
            "quit" => handle.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click { button: MouseButton::Left, .. } = event {
                if let Some(w) = tray.app_handle().get_webview_window("mini") {
                    let _ = w.set_focus();
                }
            }
        })
        .build(app)?;
    Ok(())
}
