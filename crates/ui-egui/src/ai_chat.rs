//! Embedded chat with a detected local agent or a cloud provider. The agent list lives in
//! `ai_agent`; connecting swaps the panel body for `view` here. Requests run on a
//! background thread and stream back as `ChatEvent`s over mpsc, like `ai_agent`'s scan.
//!
//! Ollama speaks plain HTTP on loopback; cloud providers (Gemini, Claude, OpenAI, …) speak
//! https. The client is a minimal HTTP/1.1 request with `Connection: close` over TCP or TLS
//! (native-tls: schannel on Windows), decoding chunked bodies so a streaming reply is just
//! NDJSON/SSE lines to EOF, local or cloud alike.

use crate::PhotocraftApp;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AiChatUi {
    /// Detected agent id the chat is connected to; `None` shows the agent list.
    pub agent: Option<String>,
    pub endpoint: String,
    pub models: Vec<String>,
    pub model: String,
    pub messages: Vec<ChatMsg>,
    pub input: String,
    pub busy: bool,
    pub error: Option<String>,
    /// Command rounds already executed for the current user turn (cap in `finish_turn`).
    pub rounds: u8,
    /// When the current agent round started, for the elapsed-time hint (not persisted).
    #[serde(skip)]
    pub round_started: Option<std::time::Instant>,
    /// Protocol the connected agent speaks.
    pub backend: Backend,
    /// Bearer key for OpenAI-compatible endpoints; never part of prompts, transcripts, or serialized state.
    #[serde(skip)]
    pub key: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatMsg {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    #[default]
    Ollama,
    /// OpenAI-compatible `/v1` server with an SSE stream: Hermes proxy, Ollama `/v1`, LM Studio, …
    Compat,
    /// One-shot CLI process: the transcript goes in as the prompt, stdout streams back.
    Cli,
}

pub enum ChatEvent {
    Models(Vec<String>),
    Delta(String),
    Done,
    Failed(String),
}

/// Shell commands: `aiAgent.connect`, `aiAgent.disconnect`, `aiAgent.send`.
pub fn command(app: &mut PhotocraftApp, ctx: &egui::Context, id: &str, params: &serde_json::Value) -> Option<Result<serde_json::Value, String>> {
    match id {
        "aiAgent.connect" => Some(connect(app, ctx, params)),
        "aiAgent.disconnect" => {
            disconnect(app);
            Some(Ok(serde_json::json!({"connected": false})))
        }
        "aiAgent.send" => Some(send(app, ctx, params["text"].as_str())),
        "aiAgent.generate" => Some(generate(app, params["svg"].as_str().unwrap_or(""))),
        _ => None,
    }
}

pub fn disconnect(app: &mut PhotocraftApp) {
    app.ai_chat_rx = None;
    app.ui.ai_chat = AiChatUi::default();
    app.ui.ai_agent.settings = false;
}

fn connect(app: &mut PhotocraftApp, ctx: &egui::Context, params: &serde_json::Value) -> Result<serde_json::Value, String> {
    if cfg!(target_arch = "wasm32") {
        let _ = ctx;
        return Err(crate::i18n::t("Chat requires the desktop app.").to_string());
    }
    let id = params["id"].as_str().unwrap_or("ollama");
    let row = app.ui.ai_agent.agents.iter().find(|a| a.id == id).cloned();
    let (backend, endpoint, fallback) = match id {
        "compat" => {
            let endpoint = params["endpoint"].as_str().unwrap_or("").trim().to_string();
            parse_http_url(&endpoint)?;
            (Backend::Compat, endpoint, params["model"].as_str().unwrap_or("").to_string())
        }
        "hermes-proxy" => {
            let row = row.ok_or_else(|| crate::i18n::t("Agent not detected or not running. Refresh the list."))?;
            if !row.running {
                return Err(crate::i18n::t("Start the proxy first in a terminal: hermes proxy start").to_string());
            }
            (Backend::Compat, format!("http://{}/v1", row.location), String::new())
        }
        "ollama" => {
            let row = row
                .filter(|a| a.running && a.kind == crate::agent_detect::AgentKind::Server)
                .ok_or_else(|| crate::i18n::t("Agent not detected or not running. Refresh the list."))?;
            (Backend::Ollama, row.location, String::new())
        }
        _ => {
            let row = row
                .filter(|a| a.running && a.kind == crate::agent_detect::AgentKind::Cli)
                .ok_or_else(|| crate::i18n::t("Agent not detected or not running. Refresh the list."))?;
            (Backend::Cli, row.location, row.name)
        }
    };
    app.ui.ai_agent.settings = false;
    let chat = &mut app.ui.ai_chat;
    chat.agent = Some(id.to_string());
    chat.backend = backend;
    chat.endpoint = endpoint.clone();
    chat.key = params["key"].as_str().unwrap_or("").to_string();
    chat.error = None;
    chat.busy = true;
    let key = chat.key.clone();
    #[cfg(target_arch = "wasm32")]
    let _ = (&endpoint, &key, &fallback, backend);
    #[cfg(not(target_arch = "wasm32"))]
    start(app, ctx, move |tx, ctx| {
        let listed = match backend {
            Backend::Ollama => tags_body(&endpoint).and_then(|b| parse_tags(&b)),
            Backend::Compat => models_body(&endpoint, &key)
                .and_then(|b| parse_models_compat(&b))
                .or_else(|e| if fallback.is_empty() { Err(e) } else { Ok(vec![fallback.clone()]) }),
            Backend::Cli => Ok(vec![fallback.clone()]),
        };
        let _ = tx.send(match listed {
            Ok(models) => ChatEvent::Models(models),
            Err(e) => ChatEvent::Failed(e),
        });
        ctx.request_repaint();
    });
    Ok(serde_json::json!({"connecting": true}))
}

fn send(app: &mut PhotocraftApp, ctx: &egui::Context, text: Option<&str>) -> Result<serde_json::Value, String> {
    if cfg!(target_arch = "wasm32") {
        let _ = (ctx, text);
        return Err(crate::i18n::t("Chat requires the desktop app.").to_string());
    }
    if app.ui.ai_chat.agent.is_none() {
        return Err(crate::i18n::t("Not connected to an agent.").to_string());
    }
    if app.ui.ai_chat.busy {
        return Err(crate::i18n::t("Still waiting for the agent.").to_string());
    }
    if app.ui.ai_chat.model.is_empty() {
        return Err(crate::i18n::t("No model available yet.").to_string());
    }
    let text = match text {
        Some(t) => t.trim().to_string(),
        None => app.ui.ai_chat.input.trim().to_string(),
    };
    if text.is_empty() {
        return Ok(serde_json::json!({"sent": false}));
    }
    let chat = &mut app.ui.ai_chat;
    chat.messages.push(ChatMsg { role: "user".into(), content: text });
    chat.input.clear();
    chat.error = None;
    chat.busy = true;
    chat.rounds = 0;
    #[cfg(not(target_arch = "wasm32"))]
    start_chat(app, ctx);
    Ok(serde_json::json!({"sent": true}))
}

#[cfg(not(target_arch = "wasm32"))]
fn start_chat(app: &mut PhotocraftApp, ctx: &egui::Context) {
    app.ui.ai_chat.round_started = Some(std::time::Instant::now());
    let backend = app.ui.ai_chat.backend;
    let endpoint = app.ui.ai_chat.endpoint.clone();
    let model = app.ui.ai_chat.model.clone();
    let key = app.ui.ai_chat.key.clone();
    let agent = app.ui.ai_chat.agent.clone().unwrap_or_default();
    let mut messages = vec![ChatMsg { role: "system".into(), content: system_prompt(app) }];
    messages.extend(app.ui.ai_chat.messages.iter().cloned());
    start(app, ctx, move |tx, ctx| {
        if backend == Backend::Cli {
            stream_cli(&endpoint, &agent, &messages, &tx, &ctx);
            return;
        }
        use std::io::BufRead;
        // Greedy decoding: a 3B model follows the command protocol far more reliably without sampling.
        let body = match backend {
            Backend::Ollama => serde_json::json!({"model": model, "messages": messages, "stream": true, "options": {"temperature": 0}}),
            _ => serde_json::json!({"model": model, "messages": messages, "stream": true, "temperature": 0}),
        }
        .to_string();
        let reader = match backend {
            Backend::Ollama => chat_reader(&endpoint, &body),
            _ => compat_reader(&endpoint, &key, &body),
        };
        let reader = match reader {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(ChatEvent::Failed(e));
                ctx.request_repaint();
                return;
            }
        };
        let parse = if backend == Backend::Ollama { parse_stream_line } else { parse_sse_line };
        let mut done = false;
        // A round that dribbles on forever is cancelled like a hung CLI agent (240 s watchdog).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(240);
        for line in reader.lines() {
            if std::time::Instant::now() > deadline {
                let _ = tx.send(ChatEvent::Failed("The agent round timed out after 240 s; try a shorter prompt.".into()));
                ctx.request_repaint();
                return;
            }
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx.send(ChatEvent::Failed(stream_error(e)));
                    break;
                }
            };
            match parse(&line) {
                StreamLine::Delta(d) => {
                    let _ = tx.send(ChatEvent::Delta(d));
                }
                StreamLine::Done => {
                    done = true;
                    let _ = tx.send(ChatEvent::Done);
                    break;
                }
                StreamLine::Ignore => {}
            }
            ctx.request_repaint();
        }
        if !done {
            let _ = tx.send(ChatEvent::Done);
        }
        ctx.request_repaint();
    });
}

/// A silent socket is the read timeout (120 s in `http_request`), not a broken cable: say so.
#[cfg(not(target_arch = "wasm32"))]
fn stream_error(e: std::io::Error) -> String {
    if matches!(e.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock) {
        "The agent sent nothing for 120 s; the round was cancelled. Try a shorter prompt.".into()
    } else {
        e.to_string()
    }
}

/// One-shot flags per coding-agent CLI; the transcript (system prompt included) is the prompt.
fn cli_args(id: &str, prompt: &str) -> Vec<String> {
    let flags: &[&str] = match id {
        "codex" => &["exec", "--skip-git-repo-check"],
        "hermes" => &["-z"],
        _ => &["-p"],
    };
    flags.iter().map(|s| s.to_string()).chain(std::iter::once(prompt.to_string())).collect()
}

/// The whole transcript as one prompt: one-shot CLIs keep no session of ours between turns.
pub fn cli_prompt(messages: &[ChatMsg]) -> String {
    messages.iter().map(|m| format!("{}: {}", m.role, m.content)).collect::<Vec<_>>().join("\n\n")
}

/// Streams a CLI agent's stdout line by line, with a watchdog so a hung agent cannot freeze
/// the panel forever; stderr becomes the error text when the process fails without output.
#[cfg(not(target_arch = "wasm32"))]
fn stream_cli(program: &str, id: &str, messages: &[ChatMsg], tx: &std::sync::mpsc::Sender<ChatEvent>, ctx: &egui::Context) {
    use std::io::{BufRead, Read};
    let mut child = match std::process::Command::new(program)
        .args(cli_args(id, &cli_prompt(messages)))
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.send(ChatEvent::Failed(format!("Cannot run {program}: {e}")));
            ctx.request_repaint();
            return;
        }
    };
    let Some(stdout) = child.stdout.take() else {
        let _ = child.kill();
        let _ = tx.send(ChatEvent::Failed("CLI agent gave no output pipe.".into()));
        ctx.request_repaint();
        return;
    };
    let (lx, lr) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in std::io::BufReader::new(stdout).lines() {
            match line {
                Ok(l) => {
                    if lx.send(l).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
    let mut got = false;
    loop {
        match lr.recv_timeout(std::time::Duration::from_secs(240)) {
            Ok(line) => {
                got = true;
                let _ = tx.send(ChatEvent::Delta(format!("{line}\n")));
                ctx.request_repaint();
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                let _ = child.kill();
                let _ = tx.send(ChatEvent::Failed("CLI agent timed out after 240 s.".into()));
                ctx.request_repaint();
                return;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    let status = child.wait();
    let mut err = String::new();
    if let Some(mut se) = child.stderr.take() {
        let _ = se.read_to_string(&mut err);
    }
    let event = match status {
        Ok(s) if s.success() || got => ChatEvent::Done,
        Ok(s) => ChatEvent::Failed(format!("CLI agent exited with {s}: {}", err.trim())),
        Err(e) => ChatEvent::Failed(e.to_string()),
    };
    let _ = tx.send(event);
    ctx.request_repaint();
}

/// The only command ids the agent may run; everything else is refused before dispatch.
const ALLOWED: &[&str] = &[
    "file.new",
    "layer.new.layer",
    "layer.duplicate",
    "layer.delete",
    "layer.setProps",
    "layer.arrange.bringForward",
    "layer.arrange.sendBackward",
    "edit.fill",
    "edit.undo",
    "edit.redo",
    "select.all",
    "select.deselect",
    "select.rect",
    "image.imageRotation.flipCanvasHorizontal",
    "image.imageRotation.flipCanvasVertical",
    "image.adjustments.desaturate",
    "view.fitOnScreen",
];

/// Extracts the agent's command: a fenced cmd block, or a reply that is only the
/// bare JSON object (small models often skip the fence).
pub fn parse_cmd(text: &str) -> Option<(String, serde_json::Value)> {
    let block = match text.find("```cmd") {
        Some(i) => {
            let start = i + 6;
            let end = text[start..].find("```")? + start;
            &text[start..end]
        }
        None => text,
    };
    let v: serde_json::Value = serde_json::from_str(block.trim()).ok()?;
    let id = v["id"].as_str()?.to_string();
    Some((id, v.get("params").cloned().unwrap_or_else(|| serde_json::json!({}))))
}

/// Extracts a fenced ```svg picture block, if any: the agent's generated image.
pub fn parse_svg(text: &str) -> Option<String> {
    let start = text.find("```svg")? + 6;
    let end = text[start..].find("```")? + start;
    let svg = text[start..end].trim();
    svg.starts_with('<').then(|| svg.to_string())
}

fn system_prompt(app: &PhotocraftApp) -> String {
    let context = match app.session.active() {
        Some(active) => {
            let doc = &active.doc;
            let layers: Vec<String> = doc.layers.iter().map(|l| format!("id={} name={:?} opacity={:.2}", l.id.0, l.name, l.opacity)).collect();
            format!("Open document {:?} ({:?}), layers bottom-to-top: [{}].", doc.name, doc.size, layers.join(", "))
        }
        None => "No document is open; run file.new to create one before pixel edits.".to_string(),
    };
    format!(
        "You are an AI assistant embedded in PhotoCraft, a Photoshop-like editor. Always reply in the same language as the user's latest message. You may edit the open document by emitting ONE command per reply in a fenced block:\n```cmd\n{{\"id\": \"<id>\", \"params\": {{...}}}}\n```\nA reply that is only the bare JSON object is also accepted. Its result arrives as the next user message; then emit another command or give a final answer without a block (at most four commands per turn). After a result fulfils the request, stop with a short final answer and no block. Allowed ids and params (? = optional): file.new {{width?, height?, name?, background: white|black|transparent|#rrggbb? (a hex colour goes in background, never in backgroundColor)}}; layer.new.layer {{name?, opacity 0..1?}}; layer.duplicate {{layer?}}; layer.delete {{layer?}}; layer.setProps {{layer?, name?, visible?, opacity 0..1?, blend?}}; layer.arrange.bringForward {{layer?}}; layer.arrange.sendBackward {{layer?}}; edit.fill {{contents: foreground|background|white|black|color, color? \"#rrggbb\", opacity 0..100?}}; edit.undo {{}}; edit.redo {{}}; select.all {{}}; select.deselect {{}}; select.rect {{x, y, width, height}}; image.imageRotation.flipCanvasHorizontal {{}}; image.imageRotation.flipCanvasVertical {{}}; image.adjustments.desaturate {{}}; view.fitOnScreen {{}}. Any other id is refused. Example — user: \"open new document\"; assistant reply:\n```cmd\n{{\"id\": \"file.new\", \"params\": {{\"width\": 800, \"height\": 600}}}}\n```\nIf a command result says refused, emit a corrected block with an allowed id. To draw or generate ANY picture (sun, landscape, logo, illustration), you MUST reply with ONE fenced ```svg block holding a complete <svg> document (xmlns, width and height in px); its shapes are added as new layers to the open document, centred on its canvas (a new document only when none is open). file.new only makes a blank document, never a picture. Example — user: \"generate a picture of a yellow sun on blue sky\"; assistant reply:\n```svg\n<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"200\" height=\"200\"><rect width=\"200\" height=\"200\" fill=\"#87ceeb\"/><circle cx=\"100\" cy=\"100\" r=\"60\" fill=\"#ffd700\"/></svg>\n```\nTo recolour an existing layer, use select.rect (if needed) then edit.fill with contents:'color' and color:'#rrggbb'; svg generation always creates NEW layers, so never use it just to recolour.\n{context} Final-answer rule: match the user's latest language exactly; never switch languages. Be brief."
    )
}

/// Runs the command block of the last assistant message (if any), feeds the result back
/// as a user message, and reports whether another agent round should follow.
pub fn finish_turn(app: &mut PhotocraftApp, ctx: &egui::Context) -> bool {
    let Some(last) = app.ui.ai_chat.messages.last() else {
        return false;
    };
    if last.role != "assistant" {
        return false;
    }
    let (label, result) = if let Some((id, params)) = parse_cmd(&last.content) {
        let result = if !ALLOWED.contains(&id.as_str()) {
            format!(
                "refused: {id} is not in the allowed command list; allowed ids: {}. Emit a corrected command block with one of these ids.",
                ALLOWED.join(", ")
            )
        } else {
            let run = if photocraft_engine::commands::find(&id).is_some() { app.run(&id, params) } else { crate::menus::invoke(app, ctx, &id, params) };
            match run {
                Ok(v) => format!("ok: {v}. If this fulfils the request, reply with a short final answer and no block."),
                Err(e) => format!("error: {e}"),
            }
        };
        (format!("command {id}"), result)
    } else if let Some(svg) = parse_svg(&last.content) {
        let result = match generate(app, &svg) {
            Ok(v) => format!("ok: {v}. If this fulfils the request, reply with a short final answer and no block."),
            Err(e) => format!("error: {e}"),
        };
        ("generate svg".to_string(), result)
    } else {
        return false;
    };
    app.ui.ai_chat.messages.push(ChatMsg { role: "user".into(), content: format!("[{label}] {result}") });
    app.ui.ai_chat.rounds = app.ui.ai_chat.rounds.saturating_add(1);
    app.ui.ai_chat.rounds < 4
}

#[cfg(not(target_arch = "wasm32"))]
fn start(app: &mut PhotocraftApp, ctx: &egui::Context, job: impl FnOnce(std::sync::mpsc::Sender<ChatEvent>, egui::Context) + Send + 'static) {
    let (tx, rx) = std::sync::mpsc::channel();
    let ctx = ctx.clone();
    let spawned = std::thread::Builder::new().name("ai-chat".into()).spawn(move || job(tx, ctx));
    match spawned {
        Ok(_) => app.ai_chat_rx = Some(rx),
        Err(_) => {
            app.ui.ai_chat.busy = false;
            app.ui.ai_chat.error = Some(crate::i18n::t("Could not start the chat thread.").to_string());
        }
    }
}

/// Adds the agent's generated SVG to the open document as new layers: the shapes are imported
/// into a scratch document, copied across with `layer.copyToDocument` (one undoable edit) and the
/// scratch is closed again. With no document open, the SVG becomes a new document instead.
fn generate(app: &mut PhotocraftApp, svg: &str) -> Result<serde_json::Value, String> {
    if svg.trim().is_empty() {
        return Err("No SVG text.".into());
    }
    if svg.len() > 64 * 1024 {
        return Err("SVG is larger than 64 KiB; generate a simpler picture.".into());
    }
    let Some(dest) = app.session.active_index() else {
        let name = app.open_name("AI generation.svg");
        let warnings = app.open_automation_bytes(&name, svg.as_bytes())?;
        let layers = app.session.active().map(|a| a.doc.layers.len()).unwrap_or(0);
        return Ok(serde_json::json!({"document": name, "layers": layers, "warnings": warnings}));
    };
    let import = app.services.import.as_ref().ok_or("no importer configured")?;
    let depth = app.session.prefs().file_handling.rasterize_svg_groups_deeper_than as usize;
    let (doc, warnings) = import("AI generation.svg", svg.as_bytes(), depth)?;
    let ids: Vec<u64> = doc.layers.iter().map(|l| l.id.0).collect();
    if ids.is_empty() {
        return Err("The SVG produced no layers.".into());
    }
    app.session.add_document(doc, None);
    let src = app.session.active_index().ok_or("no document")?;
    let copied = app.run("layer.copyToDocument", serde_json::json!({"document": dest, "source": src, "layers": ids, "center": true}));
    let closed = app.run("file.close", serde_json::json!({"document": src}));
    let value = copied?;
    closed?;
    let name = app.session.active().map(|a| a.doc.name.clone()).unwrap_or_default();
    Ok(serde_json::json!({"document": name, "layers": value["layers"], "warnings": warnings}))
}

pub fn poll(app: &mut PhotocraftApp, ctx: &egui::Context) {
    while let Some(rx) = app.ai_chat_rx.as_ref() {
        match rx.try_recv() {
            Ok(event) => apply(app, ctx, event),
            Err(std::sync::mpsc::TryRecvError::Empty) => break,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                app.ai_chat_rx = None;
                app.ui.ai_chat.busy = false;
                break;
            }
        }
    }
}

fn apply(app: &mut PhotocraftApp, ctx: &egui::Context, event: ChatEvent) {
    let chat = &mut app.ui.ai_chat;
    match event {
        ChatEvent::Models(models) => {
            chat.busy = false;
            let wanted = model_key(&chat.model);
            chat.model = models.iter().find(|m| model_key(m) == wanted).cloned().or_else(|| models.first().cloned()).unwrap_or_default();
            if models.is_empty() {
                chat.error = Some(crate::i18n::t("No models installed. In a terminal run: ollama pull qwen2.5-coder:3b").to_string());
            }
            chat.models = models;
        }
        ChatEvent::Delta(d) => match chat.messages.last_mut() {
            Some(m) if m.role == "assistant" => m.content.push_str(&d),
            _ => chat.messages.push(ChatMsg { role: "assistant".into(), content: d }),
        },
        ChatEvent::Done => {
            chat.busy = false;
            let again = finish_turn(app, ctx);
            #[cfg(not(target_arch = "wasm32"))]
            if again {
                start_chat(app, ctx);
            }
            #[cfg(target_arch = "wasm32")]
            let _ = (again, ctx);
        }
        ChatEvent::Failed(e) => {
            chat.busy = false;
            chat.error = Some(e);
        }
    }
}

pub fn parse_tags(body: &str) -> Result<Vec<String>, String> {
    let v: serde_json::Value = serde_json::from_str(body).map_err(|e| format!("Bad model list: {e}"))?;
    let models = v["models"].as_array().ok_or_else(|| "Bad model list: no models array".to_string())?;
    Ok(models.iter().filter_map(|m| m["name"].as_str().map(str::to_string)).collect())
}

pub enum StreamLine {
    Delta(String),
    Done,
    Ignore,
}

pub fn parse_stream_line(line: &str) -> StreamLine {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
        return StreamLine::Ignore;
    };
    if v["done"].as_bool() == Some(true) {
        return StreamLine::Done;
    }
    match v["message"]["content"].as_str() {
        Some(s) if !s.is_empty() => StreamLine::Delta(s.to_string()),
        _ => StreamLine::Ignore,
    }
}

/// OpenAI-compatible model list: `{"data":[{"id":…}]}`.
pub fn parse_models_compat(body: &str) -> Result<Vec<String>, String> {
    let v: serde_json::Value = serde_json::from_str(body).map_err(|e| format!("Bad model list: {e}"))?;
    let models = v["data"].as_array().ok_or_else(|| "Bad model list: no data array".to_string())?;
    Ok(models.iter().filter_map(|m| m["id"].as_str().map(str::to_string)).collect())
}

/// Gemini lists ids as `models/gemini-…` while its OpenAI-compatible chat also accepts the
/// bare name; compare selections without the prefix so a picked model survives the fetch.
pub(crate) fn model_key(id: &str) -> &str {
    id.strip_prefix("models/").unwrap_or(id)
}

/// One SSE line of an OpenAI-compatible stream: `data: {…delta…}`, `data: [DONE]`, or noise
/// (comments, keep-alives, role-only first chunk).
pub fn parse_sse_line(line: &str) -> StreamLine {
    let line = line.trim();
    if line == "data: [DONE]" {
        return StreamLine::Done;
    }
    let Some(json) = line.strip_prefix("data: ") else {
        return StreamLine::Ignore;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else {
        return StreamLine::Ignore;
    };
    match v["choices"][0]["delta"]["content"].as_str() {
        Some(s) if !s.is_empty() => StreamLine::Delta(s.to_string()),
        _ => StreamLine::Ignore,
    }
}

#[cfg(not(target_arch = "wasm32"))]
trait Wire: std::io::Read + std::io::Write + Send {}
#[cfg(not(target_arch = "wasm32"))]
impl Wire for std::net::TcpStream {}
#[cfg(not(target_arch = "wasm32"))]
impl Wire for native_tls::TlsStream<std::net::TcpStream> {}

/// HTTP/1.1 `Transfer-Encoding: chunked` decoder: Go servers and cloud APIs chunk their
/// streams, so the body is reassembled into plain bytes before NDJSON/SSE line parsing.
#[cfg(not(target_arch = "wasm32"))]
struct Chunked {
    inner: Box<dyn std::io::Read + Send>,
    left: usize,
    after_data: bool,
    done: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl Chunked {
    fn new(inner: Box<dyn std::io::Read + Send>) -> Self {
        Self { inner, left: 0, after_data: false, done: false }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn read_size_line(inner: &mut dyn std::io::Read) -> std::io::Result<String> {
    let mut line = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        inner.read_exact(&mut byte)?;
        if byte[0] == b'\n' {
            break;
        }
        line.push(byte[0]);
        if line.len() > 1024 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "oversized chunk header"));
        }
    }
    Ok(String::from_utf8_lossy(&line).trim_end_matches('\r').to_string())
}

#[cfg(not(target_arch = "wasm32"))]
impl std::io::Read for Chunked {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.done || buf.is_empty() {
            return Ok(0);
        }
        loop {
            if self.left > 0 {
                let n = buf.len().min(self.left);
                let read = self.inner.read(&mut buf[..n])?;
                if read == 0 {
                    return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "chunked body ended mid-chunk"));
                }
                self.left -= read;
                self.after_data = self.left == 0;
                return Ok(read);
            }
            if self.after_data {
                let mut crlf = [0u8; 2];
                self.inner.read_exact(&mut crlf)?;
                self.after_data = false;
            }
            let line = read_size_line(self.inner.as_mut())?;
            let hex = line.split(';').next().unwrap_or_default().trim().to_string();
            let size = usize::from_str_radix(&hex, 16).map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("bad chunk size: {line}")))?;
            if size == 0 {
                while !read_size_line(self.inner.as_mut())?.is_empty() {}
                self.done = true;
                return Ok(0);
            }
            self.left = size;
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn http_request(host: &str, port: u16, tls: bool, method: &str, path: &str, body: &str, key: &str) -> Result<Box<dyn std::io::Read + Send>, String> {
    use std::io::{Read, Write};
    use std::net::ToSocketAddrs;
    let addr = (host, port)
        .to_socket_addrs()
        .map_err(|_| format!("Bad agent address: {host}:{port}"))?
        .next()
        .ok_or_else(|| format!("Bad agent address: {host}:{port}"))?;
    let tcp = std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_secs(5)).map_err(|e| format!("Cannot reach {host}:{port}: {e}"))?;
    tcp.set_read_timeout(Some(std::time::Duration::from_secs(120))).map_err(|e| e.to_string())?;
    let mut stream: Box<dyn Wire> = if tls {
        let connector = native_tls::TlsConnector::new().map_err(|e| format!("TLS init failed: {e}"))?;
        Box::new(connector.connect(host, tcp).map_err(|e| format!("TLS handshake with {host} failed: {e}"))?)
    } else {
        Box::new(tcp)
    };
    let auth = if key.is_empty() { String::new() } else { format!("Authorization: Bearer {key}\r\n") };
    let host_header = if (tls && port != 443) || (!tls && port != 80) { format!("{host}:{port}") } else { host.to_string() };
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {host_header}\r\nConnection: close\r\n{auth}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).map_err(|e| e.to_string())?;
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        stream.read_exact(&mut byte).map_err(|e| e.to_string())?;
        head.push(byte[0]);
        if head.ends_with(b"\r\n\r\n") {
            break;
        }
        if head.len() > 16 * 1024 {
            return Err("Agent sent an oversized header.".into());
        }
    }
    let head = String::from_utf8_lossy(&head);
    let status = head.lines().next().unwrap_or_default();
    if !status.contains(" 200 ") {
        // Surface the provider's reason (overload, quota, unknown model), not just the status.
        let mut reader: Box<dyn std::io::Read + Send> =
            if head.to_ascii_lowercase().contains("transfer-encoding: chunked") { Box::new(Chunked::new(stream)) } else { stream };
        let mut body = String::new();
        let _ = reader.by_ref().take(2048).read_to_string(&mut body);
        let detail = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
            .unwrap_or_else(|| body.trim().chars().take(200).collect());
        return Err(format!("Agent replied: {} — {detail}", status.trim()));
    }
    if head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        return Ok(Box::new(Chunked::new(stream)));
    }
    Ok(stream)
}

#[cfg(not(target_arch = "wasm32"))]
fn split_host_port(endpoint: &str) -> Result<(String, u16), String> {
    let (host, port) = endpoint.rsplit_once(':').ok_or_else(|| format!("Bad agent address: {endpoint}"))?;
    let port = port.parse().map_err(|_| format!("Bad agent address: {endpoint}"))?;
    Ok((host.to_string(), port))
}

#[cfg(not(target_arch = "wasm32"))]
fn tags_body(endpoint: &str) -> Result<String, String> {
    use std::io::Read;
    let (host, port) = split_host_port(endpoint)?;
    let mut stream = http_request(&host, port, false, "GET", "/api/tags", "", "")?;
    let mut body = String::new();
    stream.read_to_string(&mut body).map_err(|e| e.to_string())?;
    Ok(body)
}

#[cfg(not(target_arch = "wasm32"))]
fn chat_reader(endpoint: &str, body: &str) -> Result<std::io::BufReader<Box<dyn std::io::Read + Send>>, String> {
    let (host, port) = split_host_port(endpoint)?;
    let stream = http_request(&host, port, false, "POST", "/api/chat", body, "")?;
    Ok(std::io::BufReader::new(stream))
}

/// Split an `http(s)://host[:port]/prefix` base URL into host, port, path prefix (default
/// `/v1`) and whether TLS is needed; https reaches cloud providers (Gemini, Claude, …).
pub fn parse_http_url(base: &str) -> Result<(String, u16, String, bool), String> {
    let trimmed = base.trim();
    let (rest, tls, default_port) = if let Some(r) = trimmed.strip_prefix("https://") {
        (r, true, 443u16)
    } else if let Some(r) = trimmed.strip_prefix("http://") {
        (r, false, 80u16)
    } else {
        return Err(crate::i18n::t("Endpoints must start with http:// or https://.").to_string());
    };
    let (host_port, prefix) = match rest.find('/') {
        Some(i) => (&rest[..i], rest[i..].trim_end_matches('/').to_string()),
        None => (rest, String::new()),
    };
    let (host, port) = match host_port.rsplit_once(':') {
        Some((h, p)) => (h, p.parse::<u16>().map_err(|_| format!("Bad endpoint port: {p}"))?),
        None => (host_port, default_port),
    };
    if host.is_empty() {
        return Err(format!("Bad endpoint address: {host_port}"));
    }
    Ok((host.to_string(), port, if prefix.is_empty() { "/v1".into() } else { prefix }, tls))
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn models_body(base: &str, key: &str) -> Result<String, String> {
    use std::io::Read;
    let (host, port, prefix, tls) = parse_http_url(base)?;
    let mut stream = http_request(&host, port, tls, "GET", &format!("{prefix}/models"), "", key)?;
    let mut body = String::new();
    stream.read_to_string(&mut body).map_err(|e| e.to_string())?;
    Ok(body)
}

#[cfg(not(target_arch = "wasm32"))]
fn compat_reader(base: &str, key: &str, body: &str) -> Result<std::io::BufReader<Box<dyn std::io::Read + Send>>, String> {
    let (host, port, prefix, tls) = parse_http_url(base)?;
    let stream = http_request(&host, port, tls, "POST", &format!("{prefix}/chat/completions"), body, key)?;
    Ok(std::io::BufReader::new(stream))
}

/// The chat body shown inside the AI Agent dock panel while connected.
pub fn view(app: &mut PhotocraftApp, ui: &mut egui::Ui, t: &crate::theme::Tokens) {
    let name = app
        .ui
        .ai_agent
        .agents
        .iter()
        .find(|a| Some(&a.id) == app.ui.ai_chat.agent.as_ref())
        .map(|a| a.name.clone())
        .unwrap_or_else(|| app.ui.ai_chat.agent.clone().unwrap_or_default());
    let models = app.ui.ai_chat.models.clone();
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(name).strong().color(t.text));
        let mut model = app.ui.ai_chat.model.clone();
        let selected = if model.is_empty() { crate::i18n::t("No model") } else { model.as_str() };
        egui::ComboBox::from_id_salt("ai-chat-model").selected_text(selected).show_ui(ui, |ui| {
            for m in &models {
                if ui.selectable_label(m == &model, m.as_str()).clicked() {
                    model = m.clone();
                }
            }
        });
        app.ui.ai_chat.model = model;
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button(crate::i18n::t("Disconnect")).clicked() {
                disconnect(app);
            }
        });
    });
    if let Some(e) = app.ui.ai_chat.error.clone() {
        ui.colored_label(t.warning, e);
    }
    ui.add_space(4.0);
    let room = (ui.available_height() - 40.0).max(60.0);
    egui::ScrollArea::vertical().stick_to_bottom(true).max_height(room).show(ui, |ui| {
        for m in &app.ui.ai_chat.messages {
            ui.strong(if m.role == "user" { crate::i18n::t("You") } else { crate::i18n::t("Agent") });
            ui.label(m.content.as_str());
            ui.add_space(6.0);
        }
        if app.ui.ai_chat.busy {
            let waiting = crate::i18n::t("Waiting for the agent…");
            // An empty transcript means the connect round (model list); anything else is a turn.
            if !app.ui.ai_chat.messages.is_empty() {
                let secs = app.ui.ai_chat.round_started.map(|t| t.elapsed().as_secs()).unwrap_or(0);
                ui.weak(format!("{waiting} round {}/4, {secs}s", app.ui.ai_chat.rounds + 1));
                ui.ctx().request_repaint_after(std::time::Duration::from_secs(1));
            } else {
                ui.weak(waiting);
            }
        }
    });
    ui.add_space(4.0);
    let busy = app.ui.ai_chat.busy;
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        let send = ui.add_enabled(!busy, egui::Button::new(crate::i18n::t("Send")));
        let entry = ui.add_enabled(!busy, egui::TextEdit::singleline(&mut app.ui.ai_chat.input).hint_text(crate::i18n::t("Message…")));
        let enter = entry.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        if send.clicked() || enter {
            let _ = command(app, ui.ctx(), "aiAgent.send", &serde_json::json!({}));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tags_reads_model_names() {
        let body = r#"{"models":[{"name":"qwen2.5-coder:3b","size":1},{"name":"llama3.2:latest"}]}"#;
        assert_eq!(parse_tags(body).unwrap(), vec!["qwen2.5-coder:3b".to_string(), "llama3.2:latest".to_string()]);
        assert!(parse_tags("not json").is_err());
        assert!(parse_tags("{}").is_err());
    }

    #[test]
    fn models_event_keeps_the_picked_model_across_the_models_prefix() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.ai_chat.model = "gemini-3.8-flash".into();
        apply(&mut app, &egui::Context::default(), ChatEvent::Models(vec!["models/gemini-2.5-flash".into(), "models/gemini-3.8-flash".into()]));
        assert_eq!(app.ui.ai_chat.model, "models/gemini-3.8-flash");
        assert_eq!(app.ui.ai_chat.models.len(), 2);
    }

    #[test]
    fn non_200_replies_include_the_server_reason() {
        if cfg!(target_arch = "wasm32") {
            return;
        }
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            if let Ok((mut sock, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let _ = sock.read(&mut buf);
                let body = r#"{"error":{"message":"The model is overloaded. Please try again later."}}"#;
                let resp = format!("HTTP/1.1 503 Service Unavailable\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                let _ = sock.write_all(resp.as_bytes());
            }
        });
        let err = models_body(&format!("http://127.0.0.1:{port}/v1"), "").unwrap_err();
        assert!(err.contains("503"), "{err}");
        assert!(err.contains("overloaded"), "{err}");
    }

    #[test]
    fn stream_lines_split_deltas_from_done_and_noise() {
        match parse_stream_line(r#"{"message":{"role":"assistant","content":"hel"},"done":false}"#) {
            StreamLine::Delta(d) => assert_eq!(d, "hel"),
            _ => panic!("expected a delta"),
        }
        assert!(matches!(parse_stream_line(r#"{"message":{"content":""},"done":true}"#), StreamLine::Done));
        assert!(matches!(parse_stream_line(r#"{"message":{"content":""},"done":false}"#), StreamLine::Ignore));
        assert!(matches!(parse_stream_line("keep-alive noise"), StreamLine::Ignore));
    }

    #[test]
    fn chat_view_shows_messages_model_and_controls() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.panels.ai_agent = true;
        app.ui.ai_agent.scanned = true;
        app.ui.ai_chat.agent = Some("ollama".into());
        app.ui.ai_chat.models = vec!["qwen2.5-coder:3b".into()];
        app.ui.ai_chat.model = "qwen2.5-coder:3b".into();
        app.ui.ai_chat.messages =
            vec![ChatMsg { role: "user".into(), content: "hi there".into() }, ChatMsg { role: "assistant".into(), content: "hello!".into() }];
        let mut h = Harness::builder().with_size(egui::vec2(900.0, 700.0)).build_ui_state(|ui, app| crate::ai_agent::panel(app, ui), app);
        h.run_steps(3);
        assert!(h.get_by_label("hi there").rect().width() > 0.0);
        assert!(h.get_by_label("hello!").rect().width() > 0.0);
        assert!(h.get_by_label("You").rect().width() > 0.0);
        assert!(h.get_by_label("Agent").rect().width() > 0.0);
        assert!(h.get_by_label("Send").rect().width() > 0.0);
        assert!(h.get_by_label("Disconnect").rect().width() > 0.0);
        // Empty input: Send is a no-op, no request thread starts.
        h.get_by_label("Send").click();
        h.run_steps(2);
        assert_eq!(h.state().ui.ai_chat.messages.len(), 2);
        assert!(h.state().ai_chat_rx.is_none());
    }

    #[test]
    fn connect_and_send_validate_before_any_network() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        let err = crate::menus::invoke(&mut app, &ctx, "aiAgent.connect", serde_json::json!({})).unwrap_err();
        if cfg!(target_arch = "wasm32") {
            assert!(err.contains("desktop"), "{err}");
        } else {
            assert!(err.contains("not detected"), "{err}");
        }
        let err = crate::menus::invoke(&mut app, &ctx, "aiAgent.send", serde_json::json!({"text": "hi"})).unwrap_err();
        assert!(err.contains("Not connected"), "{err}");
        assert!(app.ai_chat_rx.is_none());
    }

    #[test]
    fn disconnect_returns_to_the_connect_landing() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.ui.ai_chat.agent = Some("ollama".into());
        app.ui.ai_agent.settings = true;
        app.ui.ai_chat.messages = vec![ChatMsg { role: "user".into(), content: "x".into() }];
        crate::menus::invoke(&mut app, &ctx, "aiAgent.disconnect", serde_json::json!({})).unwrap();
        assert!(app.ui.ai_chat.agent.is_none());
        assert!(app.ui.ai_chat.messages.is_empty());
        assert!(!app.ui.ai_agent.settings);
    }

    #[test]
    fn parse_cmd_reads_fenced_and_bare_json_commands() {
        let text = "Sure!\n```cmd\n{\"id\": \"layer.new.layer\", \"params\": {\"name\": \"AI\"}}\n```\nDone.";
        let (id, params) = parse_cmd(text).unwrap();
        assert_eq!(id, "layer.new.layer");
        assert_eq!(params["name"], serde_json::json!("AI"));
        let (id, _) = parse_cmd("{\"id\": \"file.new\", \"params\": {}}").unwrap();
        assert_eq!(id, "file.new");
        assert!(parse_cmd("```json\n{\"id\": \"x\"}\n```").is_none());
        assert!(parse_cmd("no block here").is_none());
        assert!(parse_cmd("use {\"id\": \"x\"} please").is_none());
    }

    #[test]
    fn finish_turn_refuses_a_bare_json_unknown_id() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.ui.ai_chat.messages = vec![ChatMsg { role: "assistant".into(), content: "{\"id\": \"newDoc\", \"params\": {}}".into() }];
        assert!(finish_turn(&mut app, &ctx));
        let last = app.ui.ai_chat.messages.last().unwrap();
        assert!(last.content.contains("refused: newDoc"), "{}", last.content);
        assert!(last.content.contains("file.new"), "{}", last.content);
        assert_eq!(app.ui.ai_chat.rounds, 1);
    }

    #[test]
    fn finish_turn_creates_a_document_with_file_new() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        assert!(app.session.active().is_none());
        app.ui.ai_chat.messages = vec![ChatMsg { role: "assistant".into(), content: "{\"id\": \"file.new\", \"params\": {\"name\": \"AI\"}}".into() }];
        assert!(finish_turn(&mut app, &ctx));
        assert!(app.session.active().is_some());
        let last = app.ui.ai_chat.messages.last().unwrap();
        assert!(last.content.starts_with("[command file.new] ok"), "{}", last.content);
    }

    #[test]
    fn finish_turn_refuses_commands_outside_the_allowlist() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.ui.ai_chat.messages = vec![ChatMsg { role: "assistant".into(), content: "```cmd {\"id\": \"file.saveAs\"} ```".into() }];
        assert!(finish_turn(&mut app, &ctx));
        let last = app.ui.ai_chat.messages.last().unwrap();
        assert_eq!(last.role, "user");
        assert!(last.content.contains("refused"), "{}", last.content);
        assert_eq!(app.ui.ai_chat.rounds, 1);
    }

    #[test]
    fn finish_turn_runs_an_allowed_command_and_feeds_the_result_back() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.ui.ai_chat.messages = vec![ChatMsg { role: "assistant".into(), content: "```cmd {\"id\": \"edit.undo\"} ```".into() }];
        assert!(finish_turn(&mut app, &ctx));
        let last = app.ui.ai_chat.messages.last().unwrap();
        assert!(last.content.starts_with("[command edit.undo]"), "{}", last.content);
        assert_eq!(app.ui.ai_chat.rounds, 1);
    }

    #[test]
    fn finish_turn_ends_without_a_block_and_at_the_round_cap() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.ui.ai_chat.messages = vec![ChatMsg { role: "assistant".into(), content: "all done".into() }];
        assert!(!finish_turn(&mut app, &ctx));
        assert_eq!(app.ui.ai_chat.messages.len(), 1);
        app.ui.ai_chat.rounds = 3;
        app.ui.ai_chat.messages = vec![ChatMsg { role: "assistant".into(), content: "```cmd {\"id\": \"edit.undo\"} ```".into() }];
        assert!(!finish_turn(&mut app, &ctx), "the fourth round must not start another");
        assert_eq!(app.ui.ai_chat.rounds, 4);
    }

    #[test]
    fn chat_view_shows_the_command_block_transparently() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.panels.ai_agent = true;
        app.ui.ai_agent.scanned = true;
        app.ui.ai_chat.agent = Some("ollama".into());
        let block = "```cmd {\"id\": \"edit.undo\"} ```";
        app.ui.ai_chat.messages = vec![ChatMsg { role: "assistant".into(), content: block.into() }];
        let mut h = Harness::builder().with_size(egui::vec2(900.0, 700.0)).build_ui_state(|ui, app| crate::ai_agent::panel(app, ui), app);
        h.run_steps(3);
        assert!(h.get_by_label(block).rect().width() > 0.0);
    }

    #[test]
    fn parse_svg_reads_the_fenced_picture_only() {
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"8\" height=\"8\"><rect width=\"8\" height=\"8\" fill=\"red\"/></svg>";
        let text = format!("Here you go:\n```svg\n{svg}\n```\nEnjoy.");
        assert_eq!(parse_svg(&text).unwrap(), svg);
        assert!(parse_svg("no picture").is_none());
        assert!(parse_svg("```cmd {\"id\": \"file.new\"} ```").is_none());
    }

    #[test]
    fn finish_turn_opens_a_generated_svg_as_a_document() {
        let mut app = PhotocraftApp::new(
            photocraft_engine::Session::new(),
            crate::Services {
                import: Some(Box::new(|name: &str, bytes: &[u8], depth: usize| {
                    photocraft_io::import_with_svg_group_depth(name, bytes, depth).map(|r| (r.document, r.warnings)).map_err(|e| e.to_string())
                })),
                ..Default::default()
            },
        );
        let ctx = egui::Context::default();
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"8\" height=\"8\"><rect width=\"8\" height=\"8\" fill=\"red\"/></svg>";
        app.ui.ai_chat.messages = vec![ChatMsg { role: "assistant".into(), content: format!("```svg\n{svg}\n```") }];
        assert!(finish_turn(&mut app, &ctx));
        assert!(app.session.active().is_some());
        let last = app.ui.ai_chat.messages.last().unwrap();
        assert!(last.content.starts_with("[generate svg] ok"), "{}", last.content);
    }

    #[test]
    fn finish_turn_adds_a_generated_svg_to_the_open_document() {
        let mut app = PhotocraftApp::new(
            photocraft_engine::Session::new(),
            crate::Services {
                import: Some(Box::new(|name: &str, bytes: &[u8], depth: usize| {
                    photocraft_io::import_with_svg_group_depth(name, bytes, depth).map(|r| (r.document, r.warnings)).map_err(|e| e.to_string())
                })),
                ..Default::default()
            },
        );
        let ctx = egui::Context::default();
        app.run("file.new", serde_json::json!({"width": 800, "height": 600, "name": "Canvas"})).unwrap();
        let before = app.session.active().unwrap().doc.layers.len();
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"8\" height=\"8\"><rect width=\"8\" height=\"8\" fill=\"red\"/></svg>";
        app.ui.ai_chat.messages = vec![ChatMsg { role: "assistant".into(), content: format!("```svg\n{svg}\n```") }];
        assert!(finish_turn(&mut app, &ctx));
        assert_eq!(app.session.documents().len(), 1, "the scratch document is closed again");
        let active = app.session.active().unwrap();
        assert_eq!(active.doc.name, "Canvas");
        assert_eq!(active.doc.layers.len(), before + 1, "the shape landed in the open document");
        let last = app.ui.ai_chat.messages.last().unwrap();
        assert!(last.content.starts_with("[generate svg] ok"), "{}", last.content);
        assert!(last.content.contains("Canvas"), "{}", last.content);
        app.run("edit.undo", serde_json::json!({})).unwrap();
        assert_eq!(app.session.active().unwrap().doc.layers.len(), before, "the added layers are undoable");
    }

    #[test]
    fn busy_chat_shows_the_round_and_elapsed_hint() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.panels.ai_agent = true;
        app.ui.ai_agent.scanned = true;
        app.ui.ai_chat.agent = Some("ollama".into());
        app.ui.ai_chat.busy = true;
        app.ui.ai_chat.rounds = 1;
        app.ui.ai_chat.messages = vec![
            ChatMsg { role: "user".into(), content: "[command edit.undo] ok: {}".into() },
            ChatMsg { role: "assistant".into(), content: "```svg\n<svg xmlns=".into() },
        ];
        let mut h = Harness::builder().with_size(egui::vec2(900.0, 700.0)).build_ui_state(|ui, app| crate::ai_agent::panel(app, ui), app);
        h.run_steps(3);
        assert!(h.get_by_label("Waiting for the agent… round 2/4, 0s").rect().width() > 0.0);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn stream_error_names_the_silent_socket() {
        let quiet = std::io::Error::new(std::io::ErrorKind::TimedOut, "os error 10060");
        assert!(stream_error(quiet).contains("120 s"));
        let broken = std::io::Error::new(std::io::ErrorKind::ConnectionReset, "reset");
        assert_eq!(stream_error(broken), "reset");
    }

    #[test]
    fn finish_turn_reports_a_failed_svg_import() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.ui.ai_chat.messages = vec![ChatMsg { role: "assistant".into(), content: "```svg\n<svg/>\n```".into() }];
        assert!(finish_turn(&mut app, &ctx));
        let last = app.ui.ai_chat.messages.last().unwrap();
        assert!(last.content.starts_with("[generate svg] error"), "{}", last.content);
        assert!(app.session.active().is_none());
    }

    #[test]
    fn sse_lines_split_deltas_from_done_and_noise() {
        match parse_sse_line(r#"data: {"choices":[{"delta":{"content":"hel"}}]}"#) {
            StreamLine::Delta(d) => assert_eq!(d, "hel"),
            _ => panic!("expected a delta"),
        }
        assert!(matches!(parse_sse_line("data: [DONE]"), StreamLine::Done));
        assert!(matches!(parse_sse_line(r#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#), StreamLine::Ignore));
        assert!(matches!(parse_sse_line(": keep-alive"), StreamLine::Ignore));
        assert!(matches!(parse_sse_line(""), StreamLine::Ignore));
    }

    #[test]
    fn http_urls_parse_to_host_port_prefix_and_tls() {
        let (host, port, prefix, tls) = parse_http_url("http://127.0.0.1:8645/v1").unwrap();
        assert_eq!((host.as_str(), port, prefix.as_str(), tls), ("127.0.0.1", 8645, "/v1", false));
        let (host, port, prefix, tls) = parse_http_url("https://api.example.com/v1").unwrap();
        assert_eq!((host.as_str(), port, prefix.as_str(), tls), ("api.example.com", 443, "/v1", true));
        let (host, port, _, tls) = parse_http_url("http://localhost:1234/").unwrap();
        assert_eq!((host.as_str(), port, tls), ("localhost", 1234, false));
        let (_, _, prefix, _) = parse_http_url("http://127.0.0.1:8645/v1/").unwrap();
        assert_eq!(prefix, "/v1");
        let (_, port, prefix, tls) = parse_http_url("https://api.example.com").unwrap();
        assert_eq!((port, prefix.as_str(), tls), (443, "/v1", true));
        assert!(parse_http_url("127.0.0.1:8645").is_err());
        assert!(parse_http_url("ftp://api.example.com/v1").is_err());
        assert!(parse_http_url("https://api.example.com:notaport/v1").is_err());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn chunked_decoder_reassembles_chunks_extensions_and_small_reads() {
        let mut c = Chunked::new(Box::new(std::io::Cursor::new(b"4;ext=1\r\nWiki\r\n5\r\npedia\r\n0\r\n\r\n".to_vec())));
        let mut out = String::new();
        std::io::Read::read_to_string(&mut c, &mut out).unwrap();
        assert_eq!(out, "Wikipedia");
        let mut empty = Chunked::new(Box::new(std::io::Cursor::new(b"0\r\n\r\n".to_vec())));
        let mut s = String::new();
        std::io::Read::read_to_string(&mut empty, &mut s).unwrap();
        assert_eq!(s, "");
        let mut small = Chunked::new(Box::new(std::io::Cursor::new(b"3\r\nabc\r\n3\r\ndef\r\n0\r\n\r\n".to_vec())));
        let mut buf = [0u8; 2];
        let mut acc = Vec::new();
        loop {
            let n = std::io::Read::read(&mut small, &mut buf).unwrap();
            if n == 0 {
                break;
            }
            acc.extend_from_slice(&buf[..n]);
        }
        assert_eq!(acc, b"abcdef");
    }

    #[test]
    fn compat_model_list_reads_data_ids() {
        let body = r#"{"data":[{"id":"qwen2.5-coder:3b"},{"id":"llama3.2"}]}"#;
        assert_eq!(parse_models_compat(body).unwrap(), vec!["qwen2.5-coder:3b".to_string(), "llama3.2".to_string()]);
        assert!(parse_models_compat("not json").is_err());
        assert!(parse_models_compat("{}").is_err());
    }

    #[test]
    fn connect_compat_validates_the_endpoint_before_any_thread() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        let err = crate::menus::invoke(&mut app, &ctx, "aiAgent.connect", serde_json::json!({"id": "compat", "endpoint": "api.example.com/v1"})).unwrap_err();
        assert!(err.contains("https://"), "{err}");
        assert!(app.ai_chat_rx.is_none());
        assert!(app.ui.ai_chat.agent.is_none());
    }

    #[test]
    fn connect_cli_row_uses_the_cli_backend_and_the_agents_own_config() {
        use crate::agent_detect::{AgentKind, DetectedAgent};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.ui.ai_agent.agents = vec![DetectedAgent {
            id: "hermes".into(),
            name: "Hermes Agent".into(),
            kind: AgentKind::Cli,
            location: "C:/tools/hermes.cmd".into(),
            running: true,
        }];
        crate::menus::invoke(&mut app, &ctx, "aiAgent.connect", serde_json::json!({"id": "hermes", "key": "sk-secret"})).unwrap();
        assert_eq!(app.ui.ai_chat.agent.as_deref(), Some("hermes"));
        assert_eq!(app.ui.ai_chat.backend, Backend::Cli);
        assert_eq!(app.ui.ai_chat.endpoint, "C:/tools/hermes.cmd");
        // The key is kept for compat endpoints only; it never enters the transcript or prompt.
        assert!(app.ui.ai_chat.messages.is_empty());
        assert!(!system_prompt(&app).contains("sk-secret"));
        crate::menus::invoke(&mut app, &ctx, "aiAgent.disconnect", serde_json::json!({})).unwrap();
    }

    #[test]
    fn api_keys_are_not_serialized_in_ui_state() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.ai_agent.custom_key = "dummy-custom-key".into();
        app.ui.ai_chat.key = "dummy-chat-key".into();
        let state = serde_json::to_value(&app.ui).unwrap();
        assert!(state["ai_agent"].get("custom_key").is_none());
        assert!(state["ai_chat"].get("key").is_none());
        let restored: crate::UiState = serde_json::from_value(state).unwrap();
        assert!(restored.ai_agent.custom_key.is_empty());
        assert!(restored.ai_chat.key.is_empty());
    }

    #[test]
    fn system_prompt_requires_language_matching_and_recolour_with_fill() {
        let app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let prompt = system_prompt(&app);
        assert!(prompt.contains("Always reply in the same language as the user's latest message"));
        assert!(prompt.contains("never use it just to recolour"));
        assert!(prompt.contains("edit.fill with contents:'color'"));
    }

    #[test]
    fn cli_prompt_flattens_the_transcript_and_args_carry_it() {
        let msgs = vec![ChatMsg { role: "system".into(), content: "S".into() }, ChatMsg { role: "user".into(), content: "U".into() }];
        assert_eq!(cli_prompt(&msgs), "system: S\n\nuser: U");
        assert_eq!(cli_args("hermes", "P"), vec!["-z".to_string(), "P".to_string()]);
        assert_eq!(cli_args("codex", "P"), vec!["exec".to_string(), "--skip-git-repo-check".to_string(), "P".to_string()]);
        assert_eq!(cli_args("claude", "P"), vec!["-p".to_string(), "P".to_string()]);
    }
}
