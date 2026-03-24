mod clip;
mod config;
mod face;
mod models;
mod ocr;
mod tokenizer;

use std::sync::Arc;

use axum::{
    Router,
    extract::{DefaultBodyLimit, Multipart, State},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;
use tower_http::cors::CorsLayer;

use config::Config;

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

pub struct AppState {
    pub config: Config,
    pub clip: OnceCell<clip::ClipService>,
    pub ocr: OnceCell<ocr::OcrService>,
    pub face: OnceCell<face::FaceService>,
}

impl AppState {
    fn new(config: Config) -> Self {
        Self {
            config,
            clip: OnceCell::new(),
            ocr: OnceCell::new(),
            face: OnceCell::new(),
        }
    }

    async fn clip(&self) -> Result<&clip::ClipService, String> {
        self.clip
            .get_or_try_init(|| async {
                clip::ClipService::new(&self.config.models_dir)
            })
            .await
    }

    async fn ocr(&self) -> Result<&ocr::OcrService, String> {
        self.ocr
            .get_or_try_init(|| async {
                ocr::OcrService::new(&self.config.models_dir)
            })
            .await
    }

    async fn face(&self) -> Result<&face::FaceService, String> {
        self.face
            .get_or_try_init(|| async {
                face::FaceService::new(&self.config.models_dir)
            })
            .await
    }
}

type SharedState = Arc<AppState>;

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

fn verify_api_key(headers: &HeaderMap, expected: &str) -> Result<(), (StatusCode, &'static str)> {
    let key = headers
        .get("api-key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if key != expected {
        return Err((StatusCode::UNAUTHORIZED, "Invalid API key"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// JSON response helpers
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct CheckResponse {
    result: &'static str,
    title: &'static str,
    services: Services,
    #[serde(skip_serializing_if = "Option::is_none")]
    detector_backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    recognition_model: Option<String>,
}

#[derive(Serialize)]
struct Services {
    ocr: bool,
    clip: bool,
    face: bool,
}

// OCR response — mtphotos_ai format: flat object with texts, scores (f64), boxes (quad arrays)
#[derive(Serialize)]
struct OcrResponse {
    texts: Vec<String>,
    scores: Vec<f64>,
    boxes: Vec<Vec<Vec<f64>>>, // boxes[i] = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
}

// CLIP response — mtphotos_ai format: { vec: [f32; 512] }
#[derive(Serialize)]
struct ClipVecResponse {
    vec: Vec<f32>,
}

#[derive(Deserialize)]
struct ClipTxtRequest {
    text: String,
}

// Face response — mtphotos_face_api format: flat Vec<FaceDetection>
#[derive(Serialize)]
struct FaceDetection {
    embedding: Vec<f64>,
    facial_area: FaceArea,
    face_confidence: f64,
}

#[derive(Serialize)]
struct FaceArea {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

// ---------------------------------------------------------------------------
// Read image bytes from multipart
// ---------------------------------------------------------------------------

async fn read_multipart_image(mut multipart: Multipart) -> Result<Vec<u8>, String> {
    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" {
            return field
                .bytes()
                .await
                .map(|b| b.to_vec())
                .map_err(|e| format!("Failed to read file: {e}"));
        }
    }
    Err("No 'file' field in multipart".into())
}

fn decode_image_bytes(bytes: &[u8]) -> Result<image::DynamicImage, String> {
    image::load_from_memory(bytes).map_err(|e| format!("Invalid image: {e}"))
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn handler_index(State(state): State<SharedState>) -> impl IntoResponse {
    let cfg = &state.config;
    let html = format!(
        r#"<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Tokimo AI</title>
<style>body{{font-family:system-ui;max-width:600px;margin:40px auto;padding:0 20px}}</style></head>
<body><h1>🧠 Tokimo AI</h1>
<p><b>Status:</b> Running</p>
<p><b>OCR:</b> {}</p>
<p><b>CLIP:</b> {}</p>
<p><b>Face:</b> {}</p>
</body></html>"#,
        if cfg.enable_ocr { "enabled" } else { "disabled" },
        if cfg.enable_clip { "enabled" } else { "disabled" },
        if cfg.enable_face { "enabled" } else { "disabled" },
    );
    axum::response::Html(html)
}

async fn handler_check(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, (StatusCode, &'static str)> {
    verify_api_key(&headers, &state.config.api_auth_key)?;
    let cfg = &state.config;
    Ok(axum::Json(CheckResponse {
        result: "pass",
        title: "Tokimo AI Service",
        services: Services {
            ocr: cfg.enable_ocr,
            clip: cfg.enable_clip,
            face: cfg.enable_face,
        },
        detector_backend: cfg.enable_face.then(|| cfg.detector_backend.clone()),
        recognition_model: cfg.enable_face.then(|| cfg.recognition_model.clone()),
    }))
}

async fn handler_ocr(
    State(state): State<SharedState>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Result<impl IntoResponse, (StatusCode, &'static str)> {
    verify_api_key(&headers, &state.config.api_auth_key)?;
    if !state.config.enable_ocr {
        return Err((StatusCode::SERVICE_UNAVAILABLE, "OCR disabled"));
    }

    let empty_result = || OcrResponse {
        texts: vec![],
        scores: vec![],
        boxes: vec![],
    };

    let bytes = match read_multipart_image(multipart).await {
        Ok(b) => b,
        Err(msg) => {
            tracing::warn!("OCR multipart error: {msg}");
            return Ok(axum::Json(empty_result()));
        }
    };

    let img = match decode_image_bytes(&bytes) {
        Ok(i) => i,
        Err(msg) => {
            tracing::warn!("OCR image decode error: {msg}");
            return Ok(axum::Json(empty_result()));
        }
    };

    let svc = state.ocr().await.map_err(|e| {
        tracing::error!("OCR init failed: {e}");
        (StatusCode::INTERNAL_SERVER_ERROR, "OCR init failed")
    })?;

    let result = svc.recognize(&img);

    match result {
        Ok(items) => {
            let mut texts = Vec::new();
            let mut scores = Vec::new();
            let mut boxes = Vec::new();
            for item in items {
                texts.push(item.text);
                scores.push(item.score as f64);
                // Convert (x, y, w, h) to quad: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                let x1 = item.x as f64;
                let y1 = item.y as f64;
                let x2 = x1 + item.w as f64;
                let y2 = y1 + item.h as f64;
                boxes.push(vec![
                    vec![x1, y1],
                    vec![x2, y1],
                    vec![x2, y2],
                    vec![x1, y2],
                ]);
            }
            Ok(axum::Json(OcrResponse {
                texts,
                scores,
                boxes,
            }))
        }
        Err(msg) => {
            tracing::warn!("OCR failed: {msg}");
            Ok(axum::Json(empty_result()))
        }
    }
}

async fn handler_clip_img(
    State(state): State<SharedState>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Result<impl IntoResponse, (StatusCode, &'static str)> {
    verify_api_key(&headers, &state.config.api_auth_key)?;
    if !state.config.enable_clip {
        return Err((StatusCode::SERVICE_UNAVAILABLE, "CLIP disabled"));
    }

    let bytes = match read_multipart_image(multipart).await {
        Ok(b) => b,
        Err(msg) => {
            tracing::warn!("CLIP img multipart error: {msg}");
            return Ok(axum::Json(ClipVecResponse { vec: vec![] }));
        }
    };

    let img = match decode_image_bytes(&bytes) {
        Ok(i) => i,
        Err(msg) => {
            tracing::warn!("CLIP img decode error: {msg}");
            return Ok(axum::Json(ClipVecResponse { vec: vec![] }));
        }
    };

    let svc = state.clip().await.map_err(|e| {
        tracing::error!("CLIP init failed: {e}");
        (StatusCode::INTERNAL_SERVER_ERROR, "CLIP init failed")
    })?;

    match svc.embed_image(&img) {
        Ok(vec) => Ok(axum::Json(ClipVecResponse { vec })),
        Err(msg) => {
            tracing::warn!("CLIP img failed: {msg}");
            Ok(axum::Json(ClipVecResponse { vec: vec![] }))
        }
    }
}

async fn handler_clip_txt(
    State(state): State<SharedState>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<ClipTxtRequest>,
) -> Result<impl IntoResponse, (StatusCode, &'static str)> {
    verify_api_key(&headers, &state.config.api_auth_key)?;
    if !state.config.enable_clip {
        return Err((StatusCode::SERVICE_UNAVAILABLE, "CLIP disabled"));
    }

    let svc = state.clip().await.map_err(|e| {
        tracing::error!("CLIP init failed: {e}");
        (StatusCode::INTERNAL_SERVER_ERROR, "CLIP init failed")
    })?;

    match svc.embed_text(&req.text) {
        Ok(vec) => Ok(axum::Json(ClipVecResponse { vec })),
        Err(msg) => {
            tracing::warn!("CLIP txt failed: {msg}");
            Ok(axum::Json(ClipVecResponse { vec: vec![] }))
        }
    }
}

async fn handler_represent(
    State(state): State<SharedState>,
    headers: HeaderMap,
    multipart: Multipart,
) -> Result<impl IntoResponse, (StatusCode, &'static str)> {
    verify_api_key(&headers, &state.config.api_auth_key)?;
    if !state.config.enable_face {
        return Err((StatusCode::SERVICE_UNAVAILABLE, "Face disabled"));
    }

    let bytes = match read_multipart_image(multipart).await {
        Ok(b) => b,
        Err(msg) => {
            tracing::warn!("Face multipart error: {msg}");
            let empty: Vec<FaceDetection> = vec![];
            return Ok(axum::Json(empty));
        }
    };

    let img = match decode_image_bytes(&bytes) {
        Ok(i) => i,
        Err(msg) => {
            tracing::warn!("Face image decode error: {msg}");
            let empty: Vec<FaceDetection> = vec![];
            return Ok(axum::Json(empty));
        }
    };

    let svc = state.face().await.map_err(|e| {
        tracing::error!("Face init failed: {e}");
        (StatusCode::INTERNAL_SERVER_ERROR, "Face init failed")
    })?;

    match svc.detect_faces(&img) {
        Ok(faces) => {
            let results: Vec<FaceDetection> = faces
                .into_iter()
                .map(|f| FaceDetection {
                    embedding: f.embedding.into_iter().map(|v| v as f64).collect(),
                    facial_area: FaceArea {
                        x: f.x as f64,
                        y: f.y as f64,
                        w: f.w as f64,
                        h: f.h as f64,
                    },
                    face_confidence: f.confidence as f64,
                })
                .collect();
            Ok(axum::Json(results))
        }
        Err(msg) => {
            tracing::warn!("Face detection failed: {msg}");
            let empty: Vec<FaceDetection> = vec![];
            Ok(axum::Json(empty))
        }
    }
}

async fn handler_restart(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, (StatusCode, &'static str)> {
    verify_api_key(&headers, &state.config.api_auth_key)?;
    Ok(axum::Json(serde_json::json!({ "result": "pass" })))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "tokimo_ai=info".parse().unwrap()),
        )
        .init();

    let config = Config::from_env();
    let port = config.http_port;

    tracing::info!("Tokimo AI starting...");
    tracing::info!("  OCR:  {}", if config.enable_ocr { "on" } else { "off" });
    tracing::info!("  CLIP: {}", if config.enable_clip { "on" } else { "off" });
    tracing::info!("  Face: {}", if config.enable_face { "on" } else { "off" });

    // Ensure models exist
    if let Err(e) = models::ensure_models(&config).await {
        tracing::error!("Model download failed: {e}");
        std::process::exit(1);
    }

    let state = Arc::new(AppState::new(config));

    let app = Router::new()
        .route("/", get(handler_index))
        .route("/check", post(handler_check))
        .route("/ocr", post(handler_ocr))
        .route("/clip/img", post(handler_clip_img))
        .route("/clip/txt", post(handler_clip_txt))
        .route("/represent", post(handler_represent))
        .route("/restart", post(handler_restart))
        .route("/restart_v2", post(handler_restart))
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024)) // 50MB
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    tracing::info!("Listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
