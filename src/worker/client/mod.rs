//! tokimo-perception-worker client library.
//!
//! Provides [`AiWorkerClient`] — a drop-in replacement for
//! `tokimo_perception::AiService` that speaks the ai-worker RPC protocol over a
//! pluggable transport, plus a [`Supervisor`] that manages the local
//! worker process lifecycle (spawn, idle-exit, auto-respawn).
//!
//! The goal of the split is **physical** memory reclamation: when the worker
//! is idle long enough the process exits; the next RPC re-spawns it.

#![allow(clippy::module_name_repetitions)]

#[allow(clippy::module_inception)]
mod client;
mod settings;
mod supervisor;

pub use super::protocol::transport::{AnyTransport, Transport, UdsTransport};
pub use super::protocol::types as wire;
pub use super::protocol::{RpcError, RpcResult};

pub use client::{AiWorkerClient, StreamingSttSession};
pub use settings::{AiWorkerMode, AiWorkerSettings};
pub use supervisor::{Supervisor, SupervisorConfig};

use std::path::Path;

/// Resolve the ONNX Runtime shared library path and set `ORT_DYLIB_PATH` if
/// not already set. Search order:
///   1. `bin/onnxruntime/current/lib/` (deps.toml-managed, preferred)
///   2. `<data_local_path>/vendors/onnxruntime/lib/` (legacy compat)
///   3. perception python sidecar venv (dev fallback)
pub fn resolve_ort_dylib_path(data_local_path: &Path) {
    if std::env::var("ORT_DYLIB_PATH").is_ok() {
        return;
    }

    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent() // packages/
        .and_then(|p| p.parent()) // repo root
        .map(std::path::Path::to_path_buf);

    // 1. New deps.toml-managed location
    let mut resolved = false;
    if let Some(root) = repo_root.as_ref() {
        let new_path = root.join("bin/onnxruntime/current/lib/libonnxruntime.so");
        if new_path.exists() {
            unsafe { std::env::set_var("ORT_DYLIB_PATH", &new_path) };
            resolved = true;
        }
    }
    // 2. Legacy install location
    if !resolved {
        let canonical = data_local_path.join("vendors/onnxruntime/lib/libonnxruntime.so");
        if canonical.exists() {
            unsafe { std::env::set_var("ORT_DYLIB_PATH", &canonical) };
            resolved = true;
        }
    }
    // 3. perception python sidecar venv — onnxruntime installed as a Python dependency
    if !resolved
        && let Some(venv) = repo_root.map(|r| r.join("packages/tokimo-perception/python/.venv"))
        && let Ok(entries) = std::fs::read_dir(venv.join("lib"))
    {
        'outer: for py_ver in entries.flatten() {
            let capi = py_ver.path().join("site-packages/onnxruntime/capi");
            if let Ok(capi_entries) = std::fs::read_dir(&capi) {
                for entry in capi_entries.flatten() {
                    let name = entry.file_name();
                    let name = name.to_string_lossy();
                    if name.starts_with("libonnxruntime.so") {
                        unsafe {
                            std::env::set_var("ORT_DYLIB_PATH", entry.path());
                        };
                        break 'outer;
                    }
                }
            }
        }
    }
}
