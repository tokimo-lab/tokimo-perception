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

mod client;
mod settings;
mod supervisor;

pub use super::protocol::transport::{AnyTransport, Transport, UdsTransport};
pub use super::protocol::types as wire;
pub use super::protocol::{RpcError, RpcResult};

pub use client::{AiWorkerClient, StreamingSttSession};
pub use settings::{AiWorkerMode, AiWorkerSettings};
pub use supervisor::{Supervisor, SupervisorConfig};
