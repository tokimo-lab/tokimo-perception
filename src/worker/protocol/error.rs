use serde::{Deserialize, Serialize};

/// Structured error carried over the RPC transport.
///
/// Bridges to `tokimo-server`'s `AppError` on the client side.
#[derive(Debug, Clone, thiserror::Error, Serialize, Deserialize)]
pub enum RpcError {
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("model not ready: {0}")]
    ModelNotReady(String),
    #[error("disabled: {0}")]
    Disabled(String),
    #[error("internal: {0}")]
    Internal(String),
    #[error("transport: {0}")]
    Transport(String),
    #[error("worker shutting down")]
    ShuttingDown,
    #[error("timed out")]
    Timeout,
}

pub type RpcResult<T> = Result<T, RpcError>;

impl From<std::io::Error> for RpcError {
    fn from(e: std::io::Error) -> Self {
        Self::Transport(e.to_string())
    }
}

impl From<rmp_serde::encode::Error> for RpcError {
    fn from(e: rmp_serde::encode::Error) -> Self {
        Self::Transport(format!("encode: {e}"))
    }
}

impl From<rmp_serde::decode::Error> for RpcError {
    fn from(e: rmp_serde::decode::Error) -> Self {
        Self::Transport(format!("decode: {e}"))
    }
}

impl From<reqwest::Error> for RpcError {
    fn from(e: reqwest::Error) -> Self {
        if e.is_timeout() {
            Self::Timeout
        } else {
            Self::Transport(e.to_string())
        }
    }
}
