//! Core RPC dispatch: given a route + (decoded) request bytes, invoke the
//! matching AiService method and return response bytes.
//!
//! Shared by both the UDS listener and the HTTP axum router.

use std::sync::Arc;

use rust_models::worker::protocol::error::{RpcError, RpcResult};
use rust_models::worker::protocol::routes;
use rust_models::worker::protocol::types as wire;
use rust_models::AiService;
use serde::Serialize;
use tokio::sync::mpsc;

use crate::convert;

fn map_err(e: String) -> RpcError {
    let lower = e.to_lowercase();
    if lower.contains("disabled") {
        RpcError::Disabled(e)
    } else if lower.contains("not found") || lower.contains("missing") {
        RpcError::NotFound(e)
    } else if lower.contains("not ready") || lower.contains("not present") {
        RpcError::ModelNotReady(e)
    } else {
        RpcError::Internal(e)
    }
}

fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> RpcResult<T> {
    rmp_serde::from_slice(bytes).map_err(Into::into)
}

fn encode<T: Serialize>(v: &T) -> RpcResult<Vec<u8>> {
    rmp_serde::to_vec_named(v).map_err(Into::into)
}

/// Run a unary RPC. Returns rmp-encoded `Result<Res, RpcError>`.
pub async fn dispatch_unary(ai: &Arc<AiService>, route: &str, req_bytes: &[u8]) -> Vec<u8> {
    let result = unary_inner(ai, route, req_bytes).await;
    // Always encode a Result<Value, RpcError>; even errors go through as Ok-encoded outer frame.
    match &result {
        Ok(b) => b.clone(),
        Err(e) => encode::<RpcResult<()>>(&Err(e.clone())).unwrap_or_default(),
    }
}

async fn unary_inner(ai: &Arc<AiService>, route: &str, req_bytes: &[u8]) -> RpcResult<Vec<u8>> {
    match route {
        routes::PING => encode::<RpcResult<wire::Pong>>(&Ok(wire::Pong {
            version: env!("CARGO_PKG_VERSION").to_string(),
            accel_provider: convert::accel_to_wire(rust_models::active_ep()),
        })),
        routes::INFO => {
            let cfg = ai.config();
            let st = ai.status();
            let info = wire::WorkerInfo {
                accel_provider: convert::accel_to_wire(st.accel_provider),
                models_dir: cfg.models_dir.clone(),
                ocr_sidecar_url: cfg.ocr_sidecar_url.clone(),
                ocr_det_max_side: cfg.ocr_det_max_side,
                ocr_enabled: ai.is_ocr_enabled(),
                clip_enabled: ai.is_clip_enabled(),
                face_enabled: ai.is_face_enabled(),
                stt_enabled: ai.is_stt_enabled(),
                models_ready: ai.models_ready(),
                ocr_models_ready: ai.ocr_models_ready(),
                ocr_server_models_ready: ai.ocr_server_models_ready(),
                ocr_mobile_models_ready: ai.ocr_mobile_models_ready(),
                clip_models_ready: ai.clip_models_ready(),
                face_models_ready: ai.face_models_ready(),
                stt_model_ready: ai.stt_model_ready(),
                streaming_stt_model_ready: ai.streaming_stt_model_ready(),
                ocr_loaded: st.ocr_loaded,
                clip_loaded: st.clip_loaded,
                face_loaded: st.face_loaded,
                stt_loaded: st.stt_loaded,
            };
            encode::<RpcResult<wire::WorkerInfo>>(&Ok(info))
        }
        routes::STATUS => encode::<RpcResult<wire::AiStatus>>(&Ok(convert::status_to_wire(ai.status()))),
        routes::LIST_OCR_MODELS => {
            let list: Vec<wire::OcrModelInfo> = ai
                .ocr_available_models()
                .into_iter()
                .map(convert::ocr_model_info_to_wire)
                .collect();
            encode::<RpcResult<Vec<wire::OcrModelInfo>>>(&Ok(list))
        }
        routes::STT_MODELS_STATUS => {
            let list: Vec<wire::SttModelStatus> = ai
                .stt_models_status()
                .into_iter()
                .map(convert::stt_status_to_wire)
                .collect();
            encode::<RpcResult<Vec<wire::SttModelStatus>>>(&Ok(list))
        }
        routes::OCR => {
            let req: wire::OcrRequest = decode(req_bytes)?;
            let out = ai
                .ocr(&req.image, req.model.as_deref())
                .await
                .map_err(map_err)?;
            let items: Vec<_> = out.into_iter().map(convert::ocr_item_to_wire).collect();
            encode::<RpcResult<Vec<wire::OcrItem>>>(&Ok(items))
        }
        routes::OCR_HYBRID => {
            let req: wire::OcrHybridRequest = decode(req_bytes)?;
            let det = req
                .det_model
                .as_deref()
                .ok_or_else(|| RpcError::BadRequest("det_model required".into()))?;
            let vlm = req
                .vlm_model
                .as_deref()
                .ok_or_else(|| RpcError::BadRequest("vlm_model required".into()))?;
            let (items, _debug) = ai.ocr_hybrid(&req.image, det, vlm).await.map_err(map_err)?;
            let items: Vec<_> = items.into_iter().map(convert::ocr_item_to_wire).collect();
            encode::<RpcResult<Vec<wire::OcrItem>>>(&Ok(items))
        }
        routes::CLIP_IMAGE => {
            let req: wire::ClipImageRequest = decode(req_bytes)?;
            let v = ai.clip_image(&req.image).await.map_err(map_err)?;
            encode::<RpcResult<wire::VecResponse>>(&Ok(wire::VecResponse { data: v }))
        }
        routes::CLIP_TEXT => {
            let req: wire::ClipTextRequest = decode(req_bytes)?;
            let v = ai.clip_text(&req.text).await.map_err(map_err)?;
            encode::<RpcResult<wire::VecResponse>>(&Ok(wire::VecResponse { data: v }))
        }
        routes::CLIP_CLASSIFY => {
            let req: wire::ClipClassifyRequest = decode(req_bytes)?;
            let tags = ai.clip_classify(&req.vector).await.map_err(map_err)?;
            let tags: Vec<_> = tags.into_iter().map(convert::tag_to_wire).collect();
            encode::<RpcResult<Vec<wire::TagResult>>>(&Ok(tags))
        }
        routes::FACE_DETECT => {
            let req: wire::FaceRequest = decode(req_bytes)?;
            let out = ai.detect_faces(&req.image).await.map_err(map_err)?;
            let faces: Vec<_> = out.into_iter().map(convert::face_to_wire).collect();
            encode::<RpcResult<Vec<wire::FaceDetection>>>(&Ok(faces))
        }
        routes::STT_TRANSCRIBE => {
            let req: wire::SttTranscribeRequest = decode(req_bytes)?;
            let text = ai.transcribe_audio(&req.wav).await.map_err(map_err)?;
            encode::<RpcResult<wire::SttTranscribeResponse>>(&Ok(wire::SttTranscribeResponse { text }))
        }
        routes::STT_TRANSCRIBE_PCM => {
            let req: wire::SttTranscribePcmRequest = decode(req_bytes)?;
            let text = ai
                .transcribe_pcm(req.samples, req.sample_rate)
                .await
                .map_err(map_err)?;
            encode::<RpcResult<wire::SttTranscribeResponse>>(&Ok(wire::SttTranscribeResponse { text }))
        }
        routes::SHUTDOWN => {
            // Caller handles the actual exit; we just ack.
            encode::<RpcResult<wire::ShutdownResponse>>(&Ok(wire::ShutdownResponse { ok: true }))
        }
        other => Err(RpcError::NotFound(format!("route not found: {other}"))),
    }
}

/// Streaming response: write progress frames to `tx`.
/// Used for model download endpoints.
pub fn dispatch_server_stream(
    ai: Arc<AiService>,
    route: &str,
    req_bytes: &[u8],
    tx: mpsc::Sender<RpcResult<wire::ProgressFrame>>,
) {
    let route = route.to_string();
    let req_bytes = req_bytes.to_vec();
    tokio::spawn(async move {
        let res = server_stream_inner(&ai, &route, &req_bytes, tx.clone()).await;
        if let Err(e) = res {
            let _ = tx.send(Err(e)).await;
        }
    });
}

async fn server_stream_inner(
    ai: &Arc<AiService>,
    route: &str,
    req_bytes: &[u8],
    tx: mpsc::Sender<RpcResult<wire::ProgressFrame>>,
) -> RpcResult<()> {
    match route {
        routes::ENSURE_CATEGORY => {
            let req: wire::EnsureCategoryRequest = decode(req_bytes)?;
            let tx_clone = tx.clone();
            // ProgressFn is a boxed async closure — see rust-models::models::ProgressFn
            let progress: rust_models::models::ProgressFn = Box::new(move |file, status, pct, dl, total| {
                let frame = wire::ProgressFrame::Progress {
                    file_name: file.to_string(),
                    status: status.to_string(),
                    percent: u32::from(pct),
                    downloaded_bytes: dl,
                    total_bytes: total,
                };
                // Non-blocking: drop progress updates if channel is full/closed.
                let _ = tx_clone.try_send(Ok(frame));
            });
            match req.category {
                Some(c) => {
                    let cat = convert::category_from_wire(c);
                    ai.ensure_category_with_progress(cat, progress).await.map_err(map_err)?;
                }
                None => {
                    ai.ensure_models_with_progress(progress).await.map_err(map_err)?;
                }
            }
            let _ = tx.send(Ok(wire::ProgressFrame::Done)).await;
            Ok(())
        }
        routes::DOWNLOAD_STT => {
            let req: wire::DownloadSttRequest = decode(req_bytes)?;
            let tx_clone = tx.clone();
            let progress: rust_models::models::ProgressFn = Box::new(move |file, status, pct, dl, total| {
                let frame = wire::ProgressFrame::Progress {
                    file_name: file.to_string(),
                    status: status.to_string(),
                    percent: u32::from(pct),
                    downloaded_bytes: dl,
                    total_bytes: total,
                };
                let _ = tx_clone.try_send(Ok(frame));
            });
            ai.download_stt_model(&req.model_id, progress).await.map_err(map_err)?;
            let _ = tx.send(Ok(wire::ProgressFrame::Done)).await;
            Ok(())
        }
        other => Err(RpcError::NotFound(format!("stream route not found: {other}"))),
    }
}
