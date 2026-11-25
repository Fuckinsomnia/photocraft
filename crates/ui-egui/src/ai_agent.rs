//! The AI Agent dock group: lists AI agents installed on this PC, opened by AI Agent ›
//! AI Agent Panel (`window.toggle.aiAgent`) into the right dock below Layers. Detection runs
//! on a background thread (`agent_detect`); connecting swaps the body for `ai_chat::view`.

use crate::PhotocraftApp;
use crate::agent_detect::{AgentKind, DetectedAgent};
use serde::{Deserialize, Serialize};

/// (button, OpenAI-compatible base URL, current models; first = default) — one click fills
/// the custom form. Model ids checked against provider docs 2026-11; "Load models" refetches.
const CLOUD_PRESETS: [(&str, &str, &[&str]); 5] = [
    (
        "Gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        &["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-4-hydrogen"],
    ),
    ("Claude", "https://api.anthropic.com/v1", &["claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5", "claude-fable-5-2"]),
    ("OpenAI", "https://api.openai.com/v1", &["gpt-5.4-mini", "gpt-5.5", "gpt-6.2-astra"]),
    (
        "OpenRouter",
        "https://openrouter.ai/api/v1",
        &["openrouter/free", "nvidia/nemotron-3-ultra-550b-a55b:free", "baidu/ernie-5.5-flash-preview:free", "nvidia/nemotron-3.5-lightning:free"],
    ),
    ("Groq", "https://api.groq.com/openai/v1", &["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]),
];

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AiAgentUi {
    pub agents: Vec<DetectedAgent>,
    pub scanned: bool,
    /// The connection settings page (agent list + custom endpoint); the landing view is one Connect button.
    pub settings: bool,
    /// Custom OpenAI-compatible endpoint form (Hermes proxy, LM Studio, any `/v1` server).
    pub custom_endpoint: String,
    pub custom_model: String,
    /// Picker options for `custom_model`: the clicked preset's list or a fetched `/models` list.
    pub custom_models: Vec<String>,
    #[serde(skip)]
    pub custom_key: String,
    pub error: Option<String>,
}

/// Shell commands for the panel (AI Agent › Refresh Agent List).
pub fn command(app: &mut PhotocraftApp, ctx: &egui::Context, id: &str, params: &serde_json::Value) -> Option<Result<serde_json::Value, String>> {
    if id == "aiAgent.refresh" {
        start_scan(app, ctx);
        return Some(Ok(serde_json::json!({"scanning": app.ai_agent_scan.is_some()})));
    }
    if id == "aiAgent.settings" {
        app.ui.ai_agent.settings = params["open"].as_bool().unwrap_or(true);
        return Some(Ok(serde_json::json!({"settings": app.ui.ai_agent.settings})));
    }
    None
}

fn start_scan(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if app.ai_agent_scan.is_some() {
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let (tx, rx) = std::sync::mpsc::channel();
        let ctx = ctx.clone();
        let spawned = std::thread::Builder::new().name("ai-agent-detect".into()).spawn(move || {
            let _ = tx.send(crate::agent_detect::detect());
            ctx.request_repaint();
        });
        match spawned {
            Ok(_) => app.ai_agent_scan = Some(rx),
            Err(_) => app.ui.ai_agent.scanned = true,
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = ctx;
        app.ui.ai_agent.scanned = true;
    }
}

fn poll(app: &mut PhotocraftApp) {
    let Some(rx) = app.ai_agent_scan.as_ref() else { return };
    match rx.try_recv() {
        Ok(agents) => {
            app.ui.ai_agent.agents = agents;
            app.ui.ai_agent.scanned = true;
            app.ai_agent_scan = None;
        }
        Err(std::sync::mpsc::TryRecvError::Empty) => {}
        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
            app.ai_agent_scan = None;
            app.ui.ai_agent.scanned = true;
        }
    }
}

/// Fetch the endpoint's `/models` list on a background thread so the picker shows what the
/// server actually offers (preset lists are a snapshot and can go stale).
fn load_models(app: &mut PhotocraftApp, ctx: &egui::Context) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if app.ai_models_rx.is_some() {
            return;
        }
        let endpoint = app.ui.ai_agent.custom_endpoint.clone();
        let key = app.ui.ai_agent.custom_key.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let ctx = ctx.clone();
        let spawned = std::thread::Builder::new().name("ai-agent-models".into()).spawn(move || {
            let result = crate::ai_chat::models_body(&endpoint, &key).and_then(|b| crate::ai_chat::parse_models_compat(&b));
            let _ = tx.send(result);
            ctx.request_repaint();
        });
        match spawned {
            Ok(_) => app.ai_models_rx = Some(rx),
            Err(_) => app.ui.ai_agent.error = Some(crate::i18n::t("Could not start the model list request.").to_string()),
        }
    }
    #[cfg(target_arch = "wasm32")]
    let _ = (app, ctx);
}

fn poll_models(app: &mut PhotocraftApp) {
    let Some(rx) = app.ai_models_rx.as_ref() else { return };
    match rx.try_recv() {
        Ok(Ok(models)) => {
            if !models.is_empty() {
                let wanted = crate::ai_chat::model_key(&app.ui.ai_agent.custom_model);
                app.ui.ai_agent.custom_model = models.iter().find(|m| crate::ai_chat::model_key(m) == wanted).cloned().unwrap_or_else(|| models[0].clone());
                app.ui.ai_agent.custom_models = models;
            }
            app.ui.ai_agent.error = None;
            app.ai_models_rx = None;
        }
        Ok(Err(e)) => {
            app.ui.ai_agent.error = Some(e);
            app.ai_models_rx = None;
        }
        Err(std::sync::mpsc::TryRecvError::Empty) => {}
        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
            app.ai_models_rx = None;
        }
    }
}

/// The dock group's body (`panels::dock_body`): the agent list and custom-endpoint form, or
/// the connected agent's chat. The dock frames it and gives it its height.
pub fn panel(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    poll(app);
    poll_models(app);
    if !app.ui.ai_agent.scanned && app.ai_agent_scan.is_none() {
        start_scan(app, ui.ctx());
    }
    let t = crate::theme::Tokens::get(ui.ctx());
    crate::ai_chat::poll(app, ui.ctx());
    if app.ui.ai_chat.agent.is_some() {
        crate::ai_chat::view(app, ui, &t);
        return;
    }
    if !app.ui.ai_agent.settings {
        ui.add_space(6.0);
        ui.label(
            egui::RichText::new(crate::i18n::t("Chat with an AI agent installed on this PC and let it edit or generate images without leaving PhotoCraft."))
                .color(t.text_dim),
        );
        ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
            ui.add_space(12.0);
            if cfg!(target_arch = "wasm32") {
                ui.weak(crate::i18n::t("Agent detection and chat require the desktop app."));
            } else if ui.add(egui::Button::new(egui::RichText::new(crate::i18n::t("Connect")).size(18.0))).clicked() {
                app.ui.ai_agent.settings = true;
            }
        });
        return;
    }
    egui::ScrollArea::vertical().id_salt("ai-agent-list").max_height(ui.available_height()).auto_shrink([false, false]).show(ui, |ui| {
        if ui.small_button(crate::i18n::t("Back")).clicked() {
            app.ui.ai_agent.settings = false;
        }
        ui.add_space(6.0);
        if cfg!(target_arch = "wasm32") {
            ui.weak(crate::i18n::t("Agent detection and chat require the desktop app."));
        } else if app.ai_agent_scan.is_some() {
            ui.weak(crate::i18n::t("Scanning this PC for AI agents…"));
        } else if app.ui.ai_agent.agents.is_empty() {
            ui.weak(crate::i18n::t("No AI agents found. Install a local server such as Ollama or a coding-agent CLI, then refresh."));
        } else {
            let agents = app.ui.ai_agent.agents.clone();
            for a in &agents {
                row(app, ui, &t, a);
            }
        }
        if cfg!(not(target_arch = "wasm32")) {
            if let Some(e) = app.ui.ai_agent.error.clone() {
                ui.colored_label(t.warning, e);
            }
            ui.separator();
            ui.label(egui::RichText::new(crate::i18n::t("Cloud AI (API key)")).strong().color(t.text));
            ui.horizontal_wrapped(|ui| {
                for (name, url, models) in CLOUD_PRESETS {
                    if ui.small_button(crate::i18n::t(name)).clicked() {
                        app.ui.ai_agent.custom_endpoint = url.to_string();
                        app.ui.ai_agent.custom_models = models.iter().map(|m| m.to_string()).collect();
                        app.ui.ai_agent.custom_model = models[0].to_string();
                        app.ui.ai_agent.error = None;
                    }
                }
            });
            ui.weak(crate::i18n::t("Presets fill the form below; paste the provider's API key. Local servers usually need none."));
            ui.add_space(6.0);
            ui.label(egui::RichText::new(crate::i18n::t("Custom OpenAI-compatible endpoint")).strong().color(t.text));
            ui.add(egui::TextEdit::singleline(&mut app.ui.ai_agent.custom_endpoint).hint_text("http://127.0.0.1:8645/v1"));
            ui.horizontal(|ui| {
                ui.label(crate::i18n::t("Model"));
                let mut models = app.ui.ai_agent.custom_models.clone();
                if !app.ui.ai_agent.custom_model.is_empty() && !models.contains(&app.ui.ai_agent.custom_model) {
                    models.push(app.ui.ai_agent.custom_model.clone());
                }
                let selected =
                    if app.ui.ai_agent.custom_model.is_empty() { crate::i18n::t("Pick a model").to_string() } else { app.ui.ai_agent.custom_model.clone() };
                egui::ComboBox::from_id_salt("ai-agent-model").selected_text(selected).show_ui(ui, |ui| {
                    for m in &models {
                        ui.selectable_value(&mut app.ui.ai_agent.custom_model, m.clone(), m.as_str());
                    }
                });
                if ui.add_enabled(app.ai_models_rx.is_none(), egui::Button::new(crate::i18n::t("Load models")).small()).clicked() {
                    load_models(app, ui.ctx());
                }
            });
            ui.add(egui::TextEdit::singleline(&mut app.ui.ai_agent.custom_key).hint_text(crate::i18n::t("API key (optional)")).password(true));
            if ui.small_button(crate::i18n::t("Connect endpoint")).clicked() {
                let params = serde_json::json!({
                    "id": "compat",
                    "endpoint": app.ui.ai_agent.custom_endpoint,
                    "model": app.ui.ai_agent.custom_model,
                    "key": app.ui.ai_agent.custom_key,
                });
                let result = crate::ai_chat::command(app, ui.ctx(), "aiAgent.connect", &params);
                app.ui.ai_agent.error = result.and_then(|r| r.err());
            }
        }
        ui.add_space(8.0);
        if ui.button(crate::i18n::t("Refresh")).clicked() {
            start_scan(app, ui.ctx());
        }
    });
}

fn row(app: &mut PhotocraftApp, ui: &mut egui::Ui, t: &crate::theme::Tokens, a: &DetectedAgent) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(&a.name).strong().color(t.text));
        ui.weak(match a.kind {
            AgentKind::Cli => crate::i18n::t("CLI agent"),
            AgentKind::Server => crate::i18n::t("Local server"),
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let (label, color) = if a.running { (crate::i18n::t("Running"), t.accent) } else { (crate::i18n::t("Installed"), t.text_dim) };
            ui.label(egui::RichText::new(label).small().color(color));
        });
    });
    ui.label(egui::RichText::new(&a.location).small().color(t.text_faint));
    let connectable =
        (a.id == "ollama" && a.running && a.kind == AgentKind::Server) || (a.id == "hermes-proxy" && a.running) || (a.kind == AgentKind::Cli && a.running);
    if connectable && ui.small_button(crate::i18n::t("Connect")).clicked() {
        let _ = crate::ai_chat::command(app, ui.ctx(), "aiAgent.connect", &serde_json::json!({"id": a.id}));
    }
    ui.add_space(6.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panel_lists_detected_agents_with_status_and_refresh() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.panels.ai_agent = true;
        app.ui.ai_agent.scanned = true;
        app.ui.ai_agent.agents = vec![
            DetectedAgent { id: "ollama".into(), name: "Ollama".into(), kind: AgentKind::Server, location: "127.0.0.1:11434".into(), running: true },
            DetectedAgent { id: "claude".into(), name: "Claude Code".into(), kind: AgentKind::Cli, location: "C:/tools/claude.cmd".into(), running: false },
        ];
        let mut h = Harness::builder().with_size(egui::vec2(900.0, 700.0)).build_ui_state(|ui, app| panel(app, ui), app);
        h.run_steps(3);
        h.get_by_label("Connect").click();
        h.run_steps(2);
        assert!(h.get_by_label("Ollama").rect().width() > 0.0);
        assert!(h.get_by_label("Claude Code").rect().width() > 0.0);
        assert!(h.get_by_label("Running").rect().width() > 0.0);
        assert!(h.get_by_label("Installed").rect().width() > 0.0);
        assert!(h.get_by_label("127.0.0.1:11434").rect().width() > 0.0);
        assert!(h.get_by_label("Refresh").rect().width() > 0.0);
    }

    #[test]
    fn refresh_command_starts_a_background_scan() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        let r = crate::menus::invoke(&mut app, &ctx, "aiAgent.refresh", serde_json::json!({})).unwrap();
        assert_eq!(r["scanning"], serde_json::json!(cfg!(not(target_arch = "wasm32"))));
        if cfg!(not(target_arch = "wasm32")) {
            assert!(app.ai_agent_scan.is_some());
        }
    }

    #[test]
    fn custom_form_reports_a_bad_endpoint_without_connecting() {
        if cfg!(target_arch = "wasm32") {
            return;
        }
        use egui_kittest::{Harness, kittest::Queryable};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.panels.ai_agent = true;
        app.ui.ai_agent.scanned = true;
        app.ui.ai_agent.custom_endpoint = "api.example.com/v1".into();
        let mut h = Harness::builder().with_size(egui::vec2(900.0, 700.0)).build_ui_state(|ui, app| panel(app, ui), app);
        h.run_steps(3);
        h.get_by_label("Connect").click();
        h.run_steps(2);
        h.get_by_label("Connect endpoint").click();
        h.run_steps(2);
        assert!(h.state().ui.ai_chat.agent.is_none());
        assert!(h.state().ui.ai_agent.error.as_deref().unwrap_or_default().contains("https://"));
    }

    #[test]
    fn cloud_presets_fill_the_custom_endpoint_form() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.panels.ai_agent = true;
        app.ui.ai_agent.scanned = true;
        let mut h = Harness::builder().with_size(egui::vec2(900.0, 700.0)).build_ui_state(|ui, app| panel(app, ui), app);
        h.run_steps(3);
        h.get_by_label("Connect").click();
        h.run_steps(2);
        h.get_by_label("Gemini").click();
        h.run_steps(2);
        assert_eq!(h.state().ui.ai_agent.custom_endpoint, "https://generativelanguage.googleapis.com/v1beta/openai");
        assert_eq!(h.state().ui.ai_agent.custom_model, "gemini-3.8-flash");
        h.get_by_value("gemini-3.8-flash").click();
        h.run_steps(2);
        h.get_by_label("gemini-3.5-flash-lite").click();
        h.run_steps(2);
        assert_eq!(h.state().ui.ai_agent.custom_model, "gemini-3.5-flash-lite");
        h.get_by_label("Claude").click();
        h.run_steps(2);
        assert_eq!(h.state().ui.ai_agent.custom_endpoint, "https://api.anthropic.com/v1");
        assert_eq!(h.state().ui.ai_agent.custom_model, "claude-sonnet-5-5");
    }

    #[test]
    fn load_models_fills_the_picker_from_the_endpoint() {
        if cfg!(target_arch = "wasm32") {
            return;
        }
        use egui_kittest::{Harness, kittest::Queryable};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            if let Ok((mut sock, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let _ = sock.read(&mut buf);
                let body = r#"{"data":[{"id":"model-a"},{"id":"model-b"}]}"#;
                let resp = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                let _ = sock.write_all(resp.as_bytes());
            }
        });
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.panels.ai_agent = true;
        app.ui.ai_agent.scanned = true;
        app.ui.ai_agent.custom_endpoint = format!("http://127.0.0.1:{port}/v1");
        let mut h = Harness::builder().with_size(egui::vec2(900.0, 700.0)).build_ui_state(|ui, app| panel(app, ui), app);
        h.run_steps(3);
        h.get_by_label("Connect").click();
        h.run_steps(2);
        h.get_by_label("Load models").click();
        for _ in 0..50 {
            h.run_steps(2);
            if h.state().ai_models_rx.is_none() {
                break;
            }
        }
        assert_eq!(h.state().ui.ai_agent.custom_models, vec!["model-a".to_string(), "model-b".to_string()], "{:?}", h.state().ui.ai_agent.error);
        assert_eq!(h.state().ui.ai_agent.custom_model, "model-a");
        h.get_by_value("model-a").click();
        h.run_steps(2);
        h.get_by_label("model-b").click();
        h.run_steps(2);
        assert_eq!(h.state().ui.ai_agent.custom_model, "model-b");
    }

    #[test]
    fn fetched_models_keep_the_preset_pick_across_the_models_prefix() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.ai_agent.custom_model = "gemini-3.8-flash".into();
        let (tx, rx) = std::sync::mpsc::channel();
        tx.send(Ok(vec!["models/gemini-2.5-flash".to_string(), "models/gemini-3.8-flash".to_string()])).unwrap();
        app.ai_models_rx = Some(rx);
        poll_models(&mut app);
        assert_eq!(app.ui.ai_agent.custom_model, "models/gemini-3.8-flash");
        assert!(app.ai_models_rx.is_none());
    }

    #[test]
    fn running_cli_row_offers_connect_and_uses_the_cli_backend() {
        if cfg!(target_arch = "wasm32") {
            return;
        }
        use egui_kittest::{Harness, kittest::Queryable};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.panels.ai_agent = true;
        app.ui.ai_agent.scanned = true;
        app.ui.ai_agent.agents = vec![DetectedAgent {
            id: "hermes".into(),
            name: "Hermes Agent".into(),
            kind: AgentKind::Cli,
            location: "C:/tools/hermes.cmd".into(),
            running: true,
        }];
        let mut h = Harness::builder().with_size(egui::vec2(900.0, 700.0)).build_ui_state(|ui, app| panel(app, ui), app);
        h.run_steps(3);
        h.get_by_label("Connect").click();
        h.run_steps(2);
        h.get_by_label("Connect").click();
        h.run_steps(3);
        assert_eq!(h.state().ui.ai_chat.agent.as_deref(), Some("hermes"));
        assert_eq!(h.state().ui.ai_chat.backend, crate::ai_chat::Backend::Cli);
    }

    #[test]
    fn menu_toggle_reveals_the_dock_group_below_layers() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        crate::menus::invoke(&mut app, &ctx, "window.toggle.aiAgent", serde_json::json!({})).unwrap();
        assert!(app.ui.panels.ai_agent);
        assert!(!app.ui.dock.is_collapsed(crate::dock::Group::AiAgent));
        app.ui.ai_agent.scanned = true;
        let mut h = Harness::builder().with_size(egui::vec2(1100.0, 800.0)).build_ui_state(
            |ui, app| {
                if !ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    return;
                }
                crate::panels::right_dock(app, ui);
                egui::CentralPanel::default().show(ui, |_| {});
            },
            app,
        );
        let theme = h.state().ui.theme;
        PhotocraftApp::setup_context(&h.ctx, theme);
        h.run_steps(4);
        assert!(h.get_by_label("AI Agent").rect().width() > 0.0);
        assert!(h.get_by_label("Connect").rect().width() > 0.0);
        let rects = crate::dock::last_rects(&h.ctx);
        let layers = rects.iter().find(|(g, _)| *g == crate::dock::Group::Layers).map(|(_, r)| *r).unwrap();
        let ai = rects.iter().find(|(g, _)| *g == crate::dock::Group::AiAgent).map(|(_, r)| *r).unwrap();
        assert!(ai.top() >= layers.bottom(), "AI Agent sits below Layers: {layers:?} {ai:?}");
    }

    #[test]
    fn landing_shows_only_connect_until_it_opens_the_settings() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.ai_agent.scanned = true;
        app.ui.ai_agent.agents =
            vec![DetectedAgent { id: "ollama".into(), name: "Ollama".into(), kind: AgentKind::Server, location: "127.0.0.1:11434".into(), running: true }];
        let mut h = Harness::builder().with_size(egui::vec2(900.0, 700.0)).build_ui_state(|ui, app| panel(app, ui), app);
        h.run_steps(3);
        assert!(h.get_by_label("Connect").rect().width() > 0.0);
        assert!(h.query_by_label("Ollama").is_none());
        assert!(h.query_by_label("Refresh").is_none());
        assert!(h.query_by_label("Connect endpoint").is_none());
        h.get_by_label("Connect").click();
        h.run_steps(2);
        assert!(h.get_by_label("Ollama").rect().width() > 0.0);
        assert!(h.get_by_label("Refresh").rect().width() > 0.0);
        h.get_by_label("Back").click();
        h.run_steps(2);
        assert!(h.query_by_label("Ollama").is_none());
        assert!(h.get_by_label("Connect").rect().width() > 0.0);
    }
}
