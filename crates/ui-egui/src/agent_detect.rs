//! Detection of AI agents installed on this PC: coding-agent CLIs found on PATH and
//! local model servers (Ollama, LM Studio) probed on their loopback ports. Runs on a
//! background thread from the AI Agent panel; the web build shows a note instead.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AgentKind {
    Cli,
    Server,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetectedAgent {
    pub id: String,
    pub name: String,
    pub kind: AgentKind,
    /// Executable path for CLIs, loopback endpoint for running servers.
    pub location: String,
    /// A server answered its port probe; a CLI counts as running once found.
    pub running: bool,
}

const CLI_AGENTS: [(&str, &str); 7] = [
    ("claude", "Claude Code"),
    ("gemini", "Gemini CLI"),
    ("codex", "Codex CLI"),
    ("hermes", "Hermes Agent"),
    ("qoder", "Qoder CLI"),
    ("codebuddy", "CodeBuddy CLI (WorkBuddy)"),
    ("agy", "Antigravity CLI"),
];

const CLI_INSTALL_FALLBACKS: [(&str, &str); 3] = [
    ("qoder", ".qoder/bin/qodercli/qodercli.exe"),
    ("codebuddy", "AppData/Local/Programs/WorkBuddyAI/resources/app.asar.unpacked/cli/dist/codebuddy-lite-wb.mjs"),
    ("agy", ".gemini/bin/agy.exe"),
];

/// (id, name, port, PATH binary, install fallback relative to the user profile).
const SERVER_AGENTS: [(&str, &str, u16, &str, &str); 2] = [
    ("ollama", "Ollama", 11434, "ollama", "AppData/Local/Programs/Ollama/ollama.exe"),
    ("lmstudio", "LM Studio", 1234, "lms", "AppData/Local/Programs/LM Studio/LM Studio.exe"),
];

fn candidates(base: &str) -> [String; 4] {
    [format!("{base}.exe"), format!("{base}.cmd"), format!("{base}.bat"), base.to_owned()]
}

pub fn find_on_path(path_var: &str, base: &str) -> Option<PathBuf> {
    std::env::split_paths(path_var).find_map(|dir| candidates(base).into_iter().map(|n| dir.join(n)).find(|p| p.is_file()))
}

pub fn server_up(endpoint: &str) -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::net::{TcpStream, ToSocketAddrs};
        let Ok(addrs) = endpoint.to_socket_addrs() else { return false };
        addrs.into_iter().any(|a| TcpStream::connect_timeout(&a, std::time::Duration::from_millis(300)).is_ok())
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = endpoint;
        false
    }
}

fn cli_install_fallback(id: &str, profile: &Path, path_var: &str) -> Option<PathBuf> {
    let (_, rel) = CLI_INSTALL_FALLBACKS.iter().find(|(agent, _)| *agent == id)?;
    if id == "codebuddy" && find_on_path(path_var, "node").is_none() {
        return None;
    }
    let p = profile.join(rel);
    p.is_file().then_some(p)
}

pub fn detect_with(path_var: &str, profile: &Path, probe: impl Fn(&str) -> bool) -> Vec<DetectedAgent> {
    let mut out = Vec::new();
    for (id, name) in CLI_AGENTS {
        if let Some(p) = find_on_path(path_var, id).or_else(|| cli_install_fallback(id, profile, path_var)) {
            out.push(DetectedAgent { id: id.into(), name: name.into(), kind: AgentKind::Cli, location: p.display().to_string(), running: true });
        }
    }
    for (id, name, port, bin, rel) in SERVER_AGENTS {
        let endpoint = format!("127.0.0.1:{port}");
        let running = probe(&endpoint);
        let installed = find_on_path(path_var, bin).or_else(|| {
            let p = profile.join(rel);
            p.is_file().then_some(p)
        });
        if running || installed.is_some() {
            let location = if running { endpoint } else { installed.map(|p| p.display().to_string()).unwrap_or(endpoint) };
            out.push(DetectedAgent { id: id.into(), name: name.into(), kind: AgentKind::Server, location, running });
        }
    }
    // Hermes can front its OAuth login with a local OpenAI-compatible proxy (default port 8645).
    if out.iter().any(|a| a.id == "hermes") {
        let endpoint = "127.0.0.1:8645";
        out.push(DetectedAgent {
            id: "hermes-proxy".into(),
            name: "Hermes Proxy".into(),
            kind: AgentKind::Server,
            location: endpoint.into(),
            running: probe(endpoint),
        });
    }
    out
}

pub fn detect() -> Vec<DetectedAgent> {
    let path = std::env::var("PATH").unwrap_or_default();
    let profile = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).map(PathBuf::from).unwrap_or_default();
    detect_with(&path, &profile, server_up)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pc-agent-detect-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn path_scan_finds_shims_and_bare_binaries_and_lists_cli_agents() {
        let dir = scratch_dir("path");
        std::fs::write(dir.join("claude.cmd"), b"@echo off").unwrap();
        std::fs::write(dir.join("gemini"), b"#!/bin/sh").unwrap();
        std::fs::write(dir.join("qoder.cmd"), b"@echo off").unwrap();
        std::fs::write(dir.join("codebuddy.exe"), b"").unwrap();
        std::fs::write(dir.join("agy"), b"#!/bin/sh").unwrap();
        let path = std::env::join_paths([dir.as_path()]).unwrap().into_string().unwrap();
        assert!(find_on_path(&path, "claude").is_some());
        assert!(find_on_path(&path, "gemini").is_some());
        assert!(find_on_path(&path, "codex").is_none());
        let agents = detect_with(&path, &dir, |_| false);
        let ids: Vec<&str> = agents.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids, ["claude", "gemini", "qoder", "codebuddy", "agy"]);
        assert_eq!(agents.iter().find(|a| a.id == "codebuddy").unwrap().name, "CodeBuddy CLI (WorkBuddy)");
        assert!(agents.iter().all(|a| a.kind == AgentKind::Cli && a.running));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn profile_fallbacks_require_real_cli_entries_and_node_for_workbuddy() {
        let dir = scratch_dir("fallbacks");
        for rel in [
            ".qoder/bin/qodercli/qodercli.exe",
            "AppData/Local/Programs/WorkBuddyAI/resources/app.asar.unpacked/cli/dist/codebuddy-lite-wb.mjs",
            ".gemini/bin/agy.exe",
        ] {
            let p = dir.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, b"").unwrap();
        }
        let bin = dir.join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(bin.join("node.exe"), b"").unwrap();
        let path = std::env::join_paths([bin.as_path()]).unwrap().into_string().unwrap();
        let agents = detect_with(&path, &dir, |_| false);
        let ids: Vec<&str> = agents.iter().map(|a| a.id.as_str()).collect();
        assert_eq!(ids, ["qoder", "codebuddy", "agy"]);
        assert!(agents.iter().find(|a| a.id == "codebuddy").unwrap().location.ends_with("codebuddy-lite-wb.mjs"));
        std::fs::remove_file(bin.join("node.exe")).unwrap();
        let agents = detect_with(&path, &dir, |_| false);
        assert!(agents.iter().all(|a| a.id != "codebuddy"));
        assert!(agents.iter().any(|a| a.id == "agy"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn server_probe_distinguishes_listening_from_closed_port() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        assert!(server_up(&addr));
        drop(listener);
        assert!(!server_up(&addr));
    }

    #[test]
    fn running_server_lists_endpoint_and_stopped_install_lists_path() {
        let dir = scratch_dir("server");
        std::fs::write(dir.join("ollama.exe"), b"").unwrap();
        let path = std::env::join_paths([dir.as_path()]).unwrap().into_string().unwrap();
        let up = detect_with(&path, &dir, |e| e == "127.0.0.1:11434");
        let ollama = up.iter().find(|a| a.id == "ollama").unwrap();
        assert!(ollama.running);
        assert_eq!(ollama.kind, AgentKind::Server);
        assert_eq!(ollama.location, "127.0.0.1:11434");
        let down = detect_with(&path, &dir, |_| false);
        let ollama = down.iter().find(|a| a.id == "ollama").unwrap();
        assert!(!ollama.running);
        assert!(ollama.location.ends_with("ollama.exe"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn hermes_install_lists_the_cli_and_its_login_proxy() {
        let dir = scratch_dir("hermes");
        std::fs::write(dir.join("hermes.cmd"), b"@echo off").unwrap();
        let path = std::env::join_paths([dir.as_path()]).unwrap().into_string().unwrap();
        let up = detect_with(&path, &dir, |e| e == "127.0.0.1:8645");
        let cli = up.iter().find(|a| a.id == "hermes").unwrap();
        assert!(cli.kind == AgentKind::Cli && cli.running);
        let proxy = up.iter().find(|a| a.id == "hermes-proxy").unwrap();
        assert!(proxy.kind == AgentKind::Server && proxy.running);
        assert_eq!(proxy.location, "127.0.0.1:8645");
        let down = detect_with(&path, &dir, |_| false);
        assert!(!down.iter().find(|a| a.id == "hermes-proxy").unwrap().running);
        let none = detect_with("", &dir, |_| false);
        assert!(none.iter().all(|a| a.id != "hermes-proxy"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
