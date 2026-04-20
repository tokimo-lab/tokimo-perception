//! Settings stored in the `system_config` DB table and fed into the client.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AiWorkerMode {
    /// Spawn a local worker process and connect over UDS.
    #[default]
    Auto,
    /// Connect to an already-running remote worker over HTTP (or unix path).
    Remote,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiWorkerSettings {
    pub mode: AiWorkerMode,
    /// When `mode = Remote`, the base URL (e.g. `http://ai-worker:5679`).
    /// When `mode = Auto`, optional UDS override (otherwise
    /// `{data_local}/ai-worker.sock`).
    pub remote_url: Option<String>,
    /// When true, never idle-exit the worker (keep it warm).
    pub keepalive_always: bool,
    /// Idle seconds before graceful worker shutdown. `None` = 900 (15 min).
    /// `Some(0)` = disabled (same effect as `keepalive_always`).
    pub idle_timeout_secs: Option<u32>,
    /// Path to the `tokimo-perception-worker` binary (when `mode = Auto`).
    pub worker_binary: Option<String>,
    /// Directory containing AI models.
    pub models_dir: Option<String>,
}

impl Default for AiWorkerSettings {
    fn default() -> Self {
        Self {
            mode: AiWorkerMode::Auto,
            remote_url: None,
            keepalive_always: false,
            idle_timeout_secs: None,
            worker_binary: None,
            models_dir: None,
        }
    }
}

impl AiWorkerSettings {
    pub fn effective_idle_secs(&self) -> u32 {
        if self.keepalive_always {
            return 0;
        }
        self.idle_timeout_secs.unwrap_or(900)
    }
}
