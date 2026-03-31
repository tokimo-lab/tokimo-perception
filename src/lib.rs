//! Unified AI service: OCR + CLIP + Face recognition + STT via ONNX Runtime / sherpa-onnx.
//!
//! This crate is used as a library by `tokimo-server` — no HTTP server.
//! Models are lazy-loaded on first use and automatically evicted after 3 minutes
//! of inactivity to free memory.

pub mod clip;
pub mod config;
pub mod face;
pub mod models;

/// Snapshot of AI service status for the system-info API.
#[derive(Debug, Clone)]
pub struct AiStatus {
    /// Whether CUDA was requested in config.
    pub cuda_enabled: bool,
    pub ocr_loaded: bool,
    pub clip_loaded: bool,
    pub face_loaded: bool,
    pub stt_loaded: bool,
}
pub mod ocr;
pub mod ocr_attention;
pub mod ocr_backend;
pub mod ocr_detector;
pub mod ocr_manager;
pub mod ocr_parseq;
pub mod ocr_trocr;
pub mod stt;
mod tokenizer;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use config::AiConfig;
use tokio::sync::{OnceCell, RwLock};

/// How long an idle model stays in memory before eviction.
const MODEL_IDLE_TIMEOUT: Duration = Duration::from_secs(180); // 3 minutes

/// Build an ONNX Runtime session from a model file.
/// When CUDA is enabled (via `AI_ENABLE_CUDA`), registers the CUDA execution provider;
/// if the GPU is inaccessible or drivers are misconfigured the session creation will fail
/// with a clear error — there is no silent CPU fallback.
pub fn build_session(path: impl AsRef<std::path::Path>) -> ort::Result<ort::session::Session> {
    use ort::session::Session;

    let path = path.as_ref();
    if ENABLE_CUDA.load(Ordering::Relaxed) {
        Session::builder()?
            .with_execution_providers([ort::ep::CUDA::default().build()])?
            .commit_from_file(path)
    } else {
        Session::builder()?.commit_from_file(path)
    }
}

/// Whether CUDA was enabled after auto-detection (set once at init).
static ENABLE_CUDA: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Detect whether the CUDA environment is ready for ONNX Runtime.
/// Checks: (1) ORT CUDA provider .so exists, (2) required CUDA runtime libs are installed.
fn detect_cuda_environment() -> bool {
    // 1. Check ORT CUDA provider shared library
    let ort_path = match std::env::var("ORT_DYLIB_PATH") {
        Ok(p) => p,
        Err(_) => {
            tracing::debug!("CUDA detection: ORT_DYLIB_PATH not set");
            return false;
        }
    };
    let ort_dir = std::path::Path::new(&ort_path)
        .parent()
        .unwrap_or(std::path::Path::new("."));
    if !ort_dir.join("libonnxruntime_providers_cuda.so").exists() {
        tracing::debug!("CUDA detection: libonnxruntime_providers_cuda.so not found");
        return false;
    }

    // 2. Check required CUDA runtime libraries (ORT 1.24.x)
    let required = [
        "libcudart.so.12",
        "libcublas.so.12",
        "libcublasLt.so.12",
        "libcufft.so.11",
        "libcurand.so.10",
        "libcudnn.so.9",
    ];
    let lib_dirs = [
        "/usr/lib/x86_64-linux-gnu",
        "/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/local/cuda-12/targets/x86_64-linux/lib",
    ];
    let ldconfig = std::process::Command::new("ldconfig")
        .arg("-p")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();

    for lib in required {
        let found = ldconfig.contains(lib)
            || lib_dirs
                .iter()
                .any(|dir| std::path::Path::new(dir).join(lib).exists());
        if !found {
            tracing::debug!("CUDA detection: {lib} not found");
            return false;
        }
    }

    true
}

/// Returns whether CUDA is enabled (auto-detected at init).
pub fn is_cuda_enabled() -> bool {
    ENABLE_CUDA.load(Ordering::Relaxed)
}

/// Epoch-based timestamp (seconds since an arbitrary point).
fn epoch_secs() -> u64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Shared AI service state. Holds lazy-loaded ONNX model sessions.
///
/// Wrap in `Arc` and store in your app state. All methods are `&self`.
/// Models are evicted from memory after [`MODEL_IDLE_TIMEOUT`] of inactivity.
pub struct AiService {
    config: AiConfig,
    ocr_manager: OnceCell<ocr_manager::OcrManager>,
    // Heavy models — evictable (wrapped in Arc for shared ownership during eviction)
    clip: RwLock<Option<Arc<clip::ClipService>>>,
    clip_last_use: AtomicU64,
    face: RwLock<Option<Arc<face::FaceService>>>,
    face_last_use: AtomicU64,
    stt: RwLock<Option<stt::SttService>>,
    stt_last_use: AtomicU64,
    streaming_stt: RwLock<Option<stt::StreamingSttService>>,
    streaming_stt_last_use: AtomicU64,
}

impl AiService {
    pub fn new(config: AiConfig) -> Arc<Self> {
        // Auto-detect CUDA: user wants it AND environment supports it.
        let use_cuda = config.enable_cuda && detect_cuda_environment();
        ENABLE_CUDA.store(use_cuda, Ordering::Relaxed);

        // Initialize ONNX Runtime environment (CPU only at global level).
        // CUDA EP is applied per-session in build_session().
        let applied = ort::init().commit();
        if applied {
            if use_cuda {
                tracing::info!("ONNX Runtime initialized (CUDA enabled per-session)");
            } else if config.enable_cuda {
                tracing::info!(
                    "ONNX Runtime initialized (CUDA requested but environment not ready, using CPU)"
                );
            } else {
                tracing::info!("ONNX Runtime initialized (CPU only)");
            }
        }

        Arc::new(Self {
            config,
            ocr_manager: OnceCell::new(),
            clip: RwLock::new(None),
            clip_last_use: AtomicU64::new(0),
            face: RwLock::new(None),
            face_last_use: AtomicU64::new(0),
            stt: RwLock::new(None),
            stt_last_use: AtomicU64::new(0),
            streaming_stt: RwLock::new(None),
            streaming_stt_last_use: AtomicU64::new(0),
        })
    }

    /// Start background eviction loop. Call once after creating the service.
    pub fn start_idle_eviction(self: &Arc<Self>) {
        let svc = Arc::clone(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                svc.evict_idle().await;
            }
        });
    }

    /// Evict models that haven't been used for [`MODEL_IDLE_TIMEOUT`].
    async fn evict_idle(&self) {
        let now = epoch_secs();
        let threshold = MODEL_IDLE_TIMEOUT.as_secs();
        let mut evicted_any = false;

        // CLIP
        let clip_last = self.clip_last_use.load(Ordering::Relaxed);
        if clip_last > 0 && now.saturating_sub(clip_last) >= threshold {
            let mut guard = self.clip.write().await;
            if guard.is_some() {
                *guard = None;
                tracing::info!("Evicted idle CLIP model from memory");
                evicted_any = true;
            }
        }

        // Face
        let face_last = self.face_last_use.load(Ordering::Relaxed);
        if face_last > 0 && now.saturating_sub(face_last) >= threshold {
            let mut guard = self.face.write().await;
            if guard.is_some() {
                *guard = None;
                tracing::info!("Evicted idle Face model from memory");
                evicted_any = true;
            }
        }

        // OCR
        if let Some(mgr) = self.ocr_manager.get() {
            let ocr_last = mgr.last_use_epoch();
            if ocr_last > 0 && now.saturating_sub(ocr_last) >= threshold {
                mgr.evict_all().await;
                evicted_any = true;
            }
        }

        // STT (SenseVoice)
        let stt_last = self.stt_last_use.load(Ordering::Relaxed);
        if stt_last > 0 && now.saturating_sub(stt_last) >= threshold {
            let mut guard = self.stt.write().await;
            if guard.is_some() {
                *guard = None;
                tracing::info!("Evicted idle SenseVoice STT model from memory");
                evicted_any = true;
            }
        }

        // Streaming STT (Zipformer)
        let stream_last = self.streaming_stt_last_use.load(Ordering::Relaxed);
        if stream_last > 0 && now.saturating_sub(stream_last) >= threshold {
            let mut guard = self.streaming_stt.write().await;
            if guard.is_some() {
                *guard = None;
                tracing::info!("Evicted idle streaming STT model from memory");
                evicted_any = true;
            }
        }

        // After evicting models, ask glibc to return freed memory to the OS.
        // Without this, RSS stays high because glibc keeps freed pages in its arena.
        if evicted_any {
            #[cfg(target_os = "linux")]
            {
                unsafe { libc::malloc_trim(0) };
                tracing::info!("Called malloc_trim to release memory to OS");
            }
        }
    }

    pub fn config(&self) -> &AiConfig {
        &self.config
    }

    pub fn models_dir(&self) -> &str {
        &self.config.models_dir
    }

    /// Report current AI service status for the system-info API.
    pub fn status(&self) -> AiStatus {
        AiStatus {
            cuda_enabled: self.config.enable_cuda,
            ocr_loaded: self.ocr_manager.get().is_some_and(|m| m.has_loaded_backends()),
            clip_loaded: self.clip.try_read().is_ok_and(|g| g.is_some()),
            face_loaded: self.face.try_read().is_ok_and(|g| g.is_some()),
            stt_loaded: self.stt.try_read().is_ok_and(|g| g.is_some()),
        }
    }

    // ── Feature toggles ──────────────────────────────────────────────────

    pub fn is_ocr_enabled(&self) -> bool {
        self.config.enable_ocr
    }

    pub fn is_clip_enabled(&self) -> bool {
        self.config.enable_clip
    }

    pub fn is_face_enabled(&self) -> bool {
        self.config.enable_face
    }

    pub fn is_stt_enabled(&self) -> bool {
        self.config.enable_stt
    }

    // ── Model download ───────────────────────────────────────────────────

    /// Download any missing models. Call during startup.
    pub async fn ensure_models(&self) -> Result<(), String> {
        models::ensure_models(&self.config).await
    }

    /// Check whether all enabled model files exist on disk.
    pub fn models_ready(&self) -> bool {
        models::all_models_present(&self.config)
    }

    /// Check whether CLIP model files exist on disk.
    pub fn clip_models_ready(&self) -> bool {
        models::clip_models_present(&self.config)
    }

    /// Check whether OCR model files exist on disk.
    pub fn ocr_models_ready(&self) -> bool {
        models::ocr_models_present(&self.config)
    }

    /// Check whether face model files exist on disk.
    pub fn face_models_ready(&self) -> bool {
        models::face_models_present(&self.config)
    }

    // ── OCR ──────────────────────────────────────────────────────────────

    /// Run OCR on raw image bytes. Returns detected text regions.
    ///
    /// `model_name` selects the backend (e.g. `"pp-ocrv5-server"`).
    /// Pass `None` to use the default server model.
    pub async fn ocr(
        &self,
        image_bytes: &[u8],
        model_name: Option<&str>,
    ) -> Result<Vec<ocr::OcrItem>, String> {
        if !self.config.enable_ocr {
            return Err("OCR is disabled".into());
        }
        let manager = self
            .ocr_manager
            .get_or_try_init(|| async {
                Ok::<_, String>(ocr_manager::OcrManager::with_max_side(
                    self.config.models_dir.clone(),
                    self.config.ocr_sidecar_url.clone(),
                    self.config.ocr_det_max_side,
                ))
            })
            .await?;
        let model = model_name.unwrap_or(ocr_manager::DEFAULT_MODEL);
        manager.ocr(model, image_bytes).await
    }

    /// Hybrid OCR: `det_model` provides bounding boxes, `vlm_model` provides
    /// accurate text. Results are merged by the sidecar.
    /// Returns (merged_items, optional_debug_info).
    pub async fn ocr_hybrid(
        &self,
        image_bytes: &[u8],
        det_model: &str,
        vlm_model: &str,
    ) -> Result<(Vec<ocr::OcrItem>, Option<serde_json::Value>), String> {
        if !self.config.enable_ocr {
            return Err("OCR is disabled".into());
        }
        let manager = self
            .ocr_manager
            .get_or_try_init(|| async {
                Ok::<_, String>(ocr_manager::OcrManager::with_max_side(
                    self.config.models_dir.clone(),
                    self.config.ocr_sidecar_url.clone(),
                    self.config.ocr_det_max_side,
                ))
            })
            .await?;
        manager.ocr_hybrid(det_model, vlm_model, image_bytes).await
    }

    /// List available OCR models and their status.
    pub async fn ocr_available_models(&self) -> Vec<ocr_manager::OcrModelInfo> {
        match self.ocr_manager.get() {
            Some(mgr) => mgr.available_models(),
            None => vec![],
        }
    }

    // ── CLIP ─────────────────────────────────────────────────────────────

    /// Embed an image → 512-dim CLIP vector.
    pub async fn clip_image(&self, image_bytes: &[u8]) -> Result<Vec<f32>, String> {
        if !self.config.enable_clip {
            return Err("CLIP is disabled".into());
        }
        let img =
            image::load_from_memory(image_bytes).map_err(|e| format!("Invalid image: {e}"))?;
        let svc = self.get_or_init_clip().await?;
        svc.embed_image(&img)
    }

    /// Embed text → 512-dim CLIP vector.
    pub async fn clip_text(&self, text: &str) -> Result<Vec<f32>, String> {
        if !self.config.enable_clip {
            return Err("CLIP is disabled".into());
        }
        let svc = self.get_or_init_clip().await?;
        svc.embed_text(text)
    }

    async fn get_or_init_clip(&self) -> Result<Arc<clip::ClipService>, String> {
        self.clip_last_use.store(epoch_secs(), Ordering::Relaxed);
        {
            let guard = self.clip.read().await;
            if let Some(svc) = guard.as_ref() {
                return Ok(Arc::clone(svc));
            }
        }
        let mut guard = self.clip.write().await;
        if let Some(svc) = guard.as_ref() {
            return Ok(Arc::clone(svc));
        }
        let svc = Arc::new(clip::ClipService::new(&self.config.models_dir)?);
        *guard = Some(Arc::clone(&svc));
        Ok(svc)
    }

    // ── Face ─────────────────────────────────────────────────────────────

    /// Detect faces and extract 512-dim embeddings.
    pub async fn detect_faces(
        &self,
        image_bytes: &[u8],
    ) -> Result<Vec<face::FaceDetection>, String> {
        if !self.config.enable_face {
            return Err("Face recognition is disabled".into());
        }
        let img =
            image::load_from_memory(image_bytes).map_err(|e| format!("Invalid image: {e}"))?;
        let svc = self.get_or_init_face().await?;
        svc.detect_faces(&img)
    }

    async fn get_or_init_face(&self) -> Result<Arc<face::FaceService>, String> {
        self.face_last_use.store(epoch_secs(), Ordering::Relaxed);
        {
            let guard = self.face.read().await;
            if let Some(svc) = guard.as_ref() {
                return Ok(Arc::clone(svc));
            }
        }
        let mut guard = self.face.write().await;
        if let Some(svc) = guard.as_ref() {
            return Ok(Arc::clone(svc));
        }
        let svc = Arc::new(face::FaceService::new(&self.config.models_dir)?);
        *guard = Some(Arc::clone(&svc));
        Ok(svc)
    }

    // ── STT ──────────────────────────────────────────────────────────────

    /// Transcribe WAV audio bytes to text using SenseVoice (sherpa-onnx).
    pub async fn transcribe_audio(&self, wav_bytes: &[u8]) -> Result<String, String> {
        if !self.config.enable_stt {
            return Err("STT is disabled".into());
        }
        let svc = self.get_or_init_stt().await?;
        let bytes = wav_bytes.to_vec();
        tokio::task::spawn_blocking(move || svc.transcribe(&bytes))
            .await
            .map_err(|e| format!("Transcription task panicked: {e}"))?
    }

    /// Transcribe raw f32 PCM samples using SenseVoice (for streaming refinement).
    pub async fn transcribe_pcm(
        &self,
        samples: Vec<f32>,
        sample_rate: i32,
    ) -> Result<String, String> {
        if !self.config.enable_stt {
            return Err("STT is disabled".into());
        }
        let svc = self.get_or_init_stt().await?;
        tokio::task::spawn_blocking(move || svc.transcribe_pcm(&samples, sample_rate))
            .await
            .map_err(|e| format!("Transcription task panicked: {e}"))?
    }

    async fn get_or_init_stt(&self) -> Result<stt::SttService, String> {
        self.stt_last_use.store(epoch_secs(), Ordering::Relaxed);
        {
            let guard = self.stt.read().await;
            if let Some(svc) = guard.as_ref() {
                return Ok(svc.clone());
            }
        }
        let mut guard = self.stt.write().await;
        if let Some(svc) = guard.as_ref() {
            return Ok(svc.clone());
        }
        let svc = stt::SttService::new(&self.config.models_dir, stt::DEFAULT_MODEL)?;
        *guard = Some(svc.clone());
        Ok(svc)
    }

    /// Get STT model status information.
    pub fn stt_models_status(&self) -> Vec<stt::SttModelStatus> {
        stt::models_status(&self.config.models_dir)
    }

    /// Check if the default STT model is ready.
    pub fn stt_model_ready(&self) -> bool {
        stt::model_exists(&self.config.models_dir, stt::DEFAULT_MODEL)
    }

    /// Check if the streaming STT model is ready.
    pub fn streaming_stt_model_ready(&self) -> bool {
        stt::model_exists(&self.config.models_dir, stt::STREAMING_MODEL)
    }

    /// Get or init the streaming STT service.
    pub async fn streaming_stt(&self) -> Result<stt::StreamingSttService, String> {
        if !self.config.enable_stt {
            return Err("STT is disabled".into());
        }
        self.get_or_init_streaming_stt().await
    }

    async fn get_or_init_streaming_stt(&self) -> Result<stt::StreamingSttService, String> {
        self.streaming_stt_last_use
            .store(epoch_secs(), Ordering::Relaxed);
        {
            let guard = self.streaming_stt.read().await;
            if let Some(svc) = guard.as_ref() {
                return Ok(svc.clone());
            }
        }
        let mut guard = self.streaming_stt.write().await;
        if let Some(svc) = guard.as_ref() {
            return Ok(svc.clone());
        }
        let svc = stt::StreamingSttService::new(&self.config.models_dir)?;
        *guard = Some(svc.clone());
        Ok(svc)
    }

    /// Download a specific STT model with progress reporting.
    ///
    /// `on_progress` is called with `(status, progress_0_100)`.
    pub async fn download_stt_model<F>(&self, model_id: &str, on_progress: F) -> Result<(), String>
    where
        F: Fn(&str, u8) + Send + 'static,
    {
        let model = stt::ALL_MODELS
            .iter()
            .find(|m| m.id() == model_id)
            .copied()
            .ok_or_else(|| format!("Unknown STT model: {model_id}"))?;

        let stt_dir = format!("{}/stt", self.config.models_dir);
        tokio::fs::create_dir_all(&stt_dir)
            .await
            .map_err(|e| format!("Failed to create stt dir: {e}"))?;

        let url = model.download_url();
        let archive_path = format!("{stt_dir}/_download_{}.tar.bz2", model.id());

        tracing::info!(
            "Downloading STT model: {} → {stt_dir}",
            model.display_name()
        );
        on_progress("downloading", 0);

        // Streaming download with progress
        let resp = reqwest::get(url)
            .await
            .map_err(|e| format!("HTTP request failed: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}: {url}", resp.status()));
        }

        let total_size = resp.content_length().unwrap_or(0);
        let mut downloaded: u64 = 0;
        let mut last_pct: u8 = 0;

        let mut file = tokio::fs::File::create(&archive_path)
            .await
            .map_err(|e| format!("Create file failed: {e}"))?;

        use tokio::io::AsyncWriteExt;
        let mut stream = resp.bytes_stream();
        use futures_util::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("Download stream error: {e}"))?;
            file.write_all(&chunk)
                .await
                .map_err(|e| format!("Write error: {e}"))?;
            downloaded += chunk.len() as u64;

            if total_size > 0 {
                let pct = ((downloaded as f64 / total_size as f64) * 100.0).min(100.0) as u8;
                if pct != last_pct {
                    last_pct = pct;
                    on_progress("downloading", pct);
                }
            }
        }
        file.flush()
            .await
            .map_err(|e| format!("Flush error: {e}"))?;
        drop(file);

        let size_mb = downloaded as f64 / (1024.0 * 1024.0);
        tracing::info!("Download complete: {size_mb:.1} MB");
        on_progress("extracting", 0);

        // Extract using tar (tar.bz2 format) on blocking thread
        let archive_clone = archive_path.clone();
        let stt_dir_clone = stt_dir.clone();
        let extract_result = tokio::task::spawn_blocking(move || {
            std::process::Command::new("tar")
                .args(["xjf", &archive_clone, "-C", &stt_dir_clone])
                .status()
        })
        .await
        .map_err(|e| format!("Extract task panicked: {e}"))?
        .map_err(|e| format!("Failed to run tar: {e}"))?;
        if !extract_result.success() {
            let _ = tokio::fs::remove_file(&archive_path).await;
            return Err("tar extraction failed".into());
        }

        // Clean up archive
        let _ = tokio::fs::remove_file(&archive_path).await;

        on_progress("completed", 100);
        tracing::info!(
            "STT model {} downloaded and extracted",
            model.display_name()
        );
        Ok(())
    }
}
