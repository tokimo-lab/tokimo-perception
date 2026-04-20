//! Length-prefixed MessagePack frame codec.
//!
//! Wire format:
//! ```text
//! [ u32 BE length ][ rmp bytes ]
//! ```
//!
//! Used for:
//! - UDS request/response (one request frame + one response frame; response
//!   is `Result<T, RpcError>` encoded as rmp).
//! - UDS streaming channels (multiple frames both directions).
//!
//! HTTP transport piggy-backs on standard HTTP bodies (rmp in / rmp out for
//! unary calls; `application/x-ndjson` style newline-framed rmp for streams).

use serde::{Serialize, de::DeserializeOwned};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use super::error::{RpcError, RpcResult};

/// Maximum single-frame payload (32 MiB). Protects against malformed senders.
pub const MAX_FRAME_BYTES: u32 = 32 * 1024 * 1024;

pub async fn write_frame<W, T>(w: &mut W, value: &T) -> RpcResult<()>
where
    W: AsyncWrite + Unpin,
    T: Serialize,
{
    let bytes = rmp_serde::to_vec_named(value)?;
    if bytes.len() > MAX_FRAME_BYTES as usize {
        return Err(RpcError::BadRequest(format!(
            "frame too large: {} bytes > {MAX_FRAME_BYTES}",
            bytes.len()
        )));
    }
    let len = bytes.len() as u32;
    w.write_all(&len.to_be_bytes()).await?;
    w.write_all(&bytes).await?;
    w.flush().await?;
    Ok(())
}

pub async fn read_frame<R, T>(r: &mut R) -> RpcResult<T>
where
    R: AsyncRead + Unpin,
    T: DeserializeOwned,
{
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf);
    if len > MAX_FRAME_BYTES {
        return Err(RpcError::BadRequest(format!(
            "frame too large: {len} bytes > {MAX_FRAME_BYTES}"
        )));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf).await?;
    let value = rmp_serde::from_slice::<T>(&buf)?;
    Ok(value)
}

/// Try to read a frame; returns `Ok(None)` on clean EOF.
pub async fn read_frame_opt<R, T>(r: &mut R) -> RpcResult<Option<T>>
where
    R: AsyncRead + Unpin,
    T: DeserializeOwned,
{
    let mut len_buf = [0u8; 4];
    match r.read_exact(&mut len_buf).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let len = u32::from_be_bytes(len_buf);
    if len > MAX_FRAME_BYTES {
        return Err(RpcError::BadRequest(format!(
            "frame too large: {len} bytes > {MAX_FRAME_BYTES}"
        )));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf).await?;
    let value = rmp_serde::from_slice::<T>(&buf)?;
    Ok(Some(value))
}
