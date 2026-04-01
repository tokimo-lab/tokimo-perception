/// Download model files if missing.
use crate::config::AiConfig;
use std::path::Path;

struct ModelFile {
    rel_path: &'static str,
    url: &'static str,
    enabled: bool,
}

/// Progress callback: (file_name, status, percent 0-100, downloaded_bytes, total_bytes)
pub type ProgressFn = Box<dyn Fn(&str, &str, u8, u64, u64) + Send + Sync>;

pub async fn ensure_models(config: &AiConfig) -> Result<(), String> {
    ensure_models_with_progress(config, None).await
}

pub async fn ensure_models_with_progress(
    config: &AiConfig,
    on_progress: Option<ProgressFn>,
) -> Result<(), String> {
    let dir = &config.models_dir;

    let files = [
        // CLIP models (from MT-Photos release, publicly accessible)
        ModelFile {
            rel_path: "clip/vit-b-16.img.fp32.onnx",
            url: "https://github.com/MT-Photos/mt-photos-ai/releases/download/v1.1.0/vit-b-16.img.fp32.onnx",
            enabled: config.enable_clip,
        },
        ModelFile {
            rel_path: "clip/vit-b-16.txt.fp32.onnx",
            url: "https://github.com/MT-Photos/mt-photos-ai/releases/download/v1.1.0/vit-b-16.txt.fp32.onnx",
            enabled: config.enable_clip,
        },
        // OCR models — PP-OCRv5 server (ONNX converted from PaddleOCR)
        ModelFile {
            rel_path: "ocr/PP-OCRv5_server_det.onnx",
            url: "https://github.com/tokimo-lab/tokimo-ai-models/releases/download/v0.3.0/PP-OCRv5_server_det.onnx",
            enabled: config.enable_ocr,
        },
        ModelFile {
            rel_path: "ocr/PP-OCRv5_cls.onnx",
            url: "https://github.com/tokimo-lab/tokimo-ai-models/releases/download/v0.3.0/PP-OCRv5_cls.onnx",
            enabled: config.enable_ocr,
        },
        ModelFile {
            rel_path: "ocr/PP-OCRv5_server_rec.onnx",
            url: "https://github.com/tokimo-lab/tokimo-ai-models/releases/download/v0.3.0/PP-OCRv5_server_rec.onnx",
            enabled: config.enable_ocr,
        },
        // OCR models — PP-OCRv5 mobile (lightweight variant)
        ModelFile {
            rel_path: "ocr/PP-OCRv5_mobile_det.onnx",
            url: "https://github.com/tokimo-lab/tokimo-ai-models/releases/download/v0.3.0/PP-OCRv5_mobile_det.onnx",
            enabled: config.enable_ocr,
        },
        ModelFile {
            rel_path: "ocr/PP-OCRv5_mobile_rec.onnx",
            url: "https://github.com/tokimo-lab/tokimo-ai-models/releases/download/v0.3.0/PP-OCRv5_mobile_rec.onnx",
            enabled: config.enable_ocr,
        },
        // OCR models — Attention recognition (generated locally via export script)
        // This model is NOT auto-downloaded — run scripts/export-attention-models.py
        // to generate it from PaddleOCR SARNet weights.
        // Face models (InsightFace buffalo_l pack, publicly accessible)
        ModelFile {
            rel_path: "face/det_10g.onnx",
            url: "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip#det_10g.onnx",
            enabled: config.enable_face,
        },
        ModelFile {
            rel_path: "face/w600k_r50.onnx",
            url: "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip#w600k_r50.onnx",
            enabled: config.enable_face,
        },
    ];

    // Deduplicate zip downloads: face models come from a single buffalo_l.zip
    let mut zip_downloaded: std::collections::HashSet<String> = std::collections::HashSet::new();

    for f in &files {
        if !f.enabled {
            continue;
        }
        let full_path = format!("{}/{}", dir, f.rel_path);
        if Path::new(&full_path).exists() {
            tracing::info!("Model OK: {}", f.rel_path);
            if let Some(cb) = on_progress.as_ref() {
                cb(f.rel_path, "ready", 100, 0, 0);
            }
            continue;
        }

        // Handle zip archives: URL contains #filename to extract
        if let Some((zip_url, entry_name)) = f.url.split_once('#') {
            if !zip_downloaded.contains(zip_url) {
                tracing::info!("Downloading archive: {}", zip_url);
                let parent = Path::new(&full_path).parent().ok_or("Invalid path")?;
                download_and_extract_zip(zip_url, parent.to_str().unwrap_or("."), f.rel_path, &on_progress).await?;
                zip_downloaded.insert(zip_url.to_string());
            }
            // The zip may have a subdirectory, move the file if needed
            let extracted = format!("{}/{}", dir, f.rel_path);
            if !Path::new(&extracted).exists() {
                // Try finding it in a subdirectory (buffalo_l/det_10g.onnx → face/det_10g.onnx)
                let parent_dir = Path::new(&full_path).parent().unwrap();
                let mut found = false;
                if let Ok(mut entries) = tokio::fs::read_dir(parent_dir).await {
                    while let Ok(Some(entry)) = entries.next_entry().await {
                        let path = entry.path();
                        if path.is_dir() {
                            let candidate = path.join(entry_name);
                            if candidate.exists() {
                                tokio::fs::rename(&candidate, &full_path)
                                    .await
                                    .map_err(|e| format!("Move failed: {e}"))?;
                                found = true;
                                tracing::info!("  Extracted: {}", f.rel_path);
                                break;
                            }
                        }
                    }
                }
                if !found && !Path::new(&full_path).exists() {
                    return Err(format!("Model not found after extraction: {}", f.rel_path));
                }
            }
            if let Some(cb) = on_progress.as_ref() {
                cb(f.rel_path, "ready", 100, 0, 0);
            }
        } else {
            tracing::info!("Downloading: {} → {}", f.url, full_path);
            download_file(f.url, &full_path, f.rel_path, &on_progress).await?;
        }
    }

    Ok(())
}

/// Check whether all enabled model files exist on disk (no download).
pub fn all_models_present(config: &AiConfig) -> bool {
    let dir = &config.models_dir;
    let checks: Vec<(&str, bool)> = vec![
        ("clip/vit-b-16.img.fp32.onnx", config.enable_clip),
        ("clip/vit-b-16.txt.fp32.onnx", config.enable_clip),
        // Server variant
        ("ocr/PP-OCRv5_server_det.onnx", config.enable_ocr),
        ("ocr/PP-OCRv5_cls.onnx", config.enable_ocr),
        ("ocr/PP-OCRv5_server_rec.onnx", config.enable_ocr),
        // Mobile variant
        ("ocr/PP-OCRv5_mobile_det.onnx", config.enable_ocr),
        ("ocr/PP-OCRv5_mobile_rec.onnx", config.enable_ocr),
        ("face/det_10g.onnx", config.enable_face),
        ("face/w600k_r50.onnx", config.enable_face),
    ];
    for (rel, enabled) in checks {
        if enabled && !Path::new(&format!("{dir}/{rel}")).exists() {
            return false;
        }
    }
    true
}

/// Check whether CLIP model files exist on disk.
pub fn clip_models_present(config: &AiConfig) -> bool {
    let dir = &config.models_dir;
    Path::new(&format!("{dir}/clip/vit-b-16.img.fp32.onnx")).exists()
        && Path::new(&format!("{dir}/clip/vit-b-16.txt.fp32.onnx")).exists()
}

/// Check whether OCR model files exist on disk (at least server variant).
pub fn ocr_models_present(config: &AiConfig) -> bool {
    let dir = &config.models_dir;
    Path::new(&format!("{dir}/ocr/PP-OCRv5_server_det.onnx")).exists()
        && Path::new(&format!("{dir}/ocr/PP-OCRv5_cls.onnx")).exists()
        && Path::new(&format!("{dir}/ocr/PP-OCRv5_server_rec.onnx")).exists()
}

/// Check whether face detection/recognition model files exist on disk.
pub fn face_models_present(config: &AiConfig) -> bool {
    let dir = &config.models_dir;
    Path::new(&format!("{dir}/face/det_10g.onnx")).exists()
        && Path::new(&format!("{dir}/face/w600k_r50.onnx")).exists()
}

/// Download a single model file to the given destination path.
pub async fn download_model_file(url: &str, dest: &str) -> Result<(), String> {
    download_file(url, dest, dest, &None).await
}

async fn download_file(url: &str, dest: &str, label: &str, on_progress: &Option<ProgressFn>) -> Result<(), String> {
    use futures_util::StreamExt;

    let parent = Path::new(dest).parent().ok_or("Invalid path")?;
    tokio::fs::create_dir_all(parent)
        .await
        .map_err(|e| format!("mkdir failed: {e}"))?;

    let resp = reqwest::get(url)
        .await
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("HTTP {}: {}", resp.status(), url));
    }

    let total_size = resp.content_length().unwrap_or(0);
    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut last_pct: u8 = 0;
    let mut buf = Vec::with_capacity(total_size as usize);

    if let Some(cb) = on_progress.as_ref() {
        cb(label, "downloading", 0, 0, total_size);
    }

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Download stream error: {e}"))?;
        downloaded += chunk.len() as u64;
        buf.extend_from_slice(&chunk);

        if let Some(cb) = on_progress.as_ref() {
            let pct = if total_size > 0 {
                ((downloaded as f64 / total_size as f64) * 100.0).min(100.0) as u8
            } else {
                0
            };
            if pct != last_pct {
                last_pct = pct;
                cb(label, "downloading", pct, downloaded, total_size);
            }
        }
    }

    tokio::fs::write(dest, &buf)
        .await
        .map_err(|e| format!("Write failed: {e}"))?;

    let size_mb = buf.len() as f64 / (1024.0 * 1024.0);
    tracing::info!("  Done: {:.1} MB", size_mb);

    if let Some(cb) = on_progress.as_ref() {
        cb(label, "ready", 100, downloaded, total_size);
    }

    Ok(())
}

async fn download_and_extract_zip(url: &str, dest_dir: &str, label: &str, on_progress: &Option<ProgressFn>) -> Result<(), String> {
    use futures_util::StreamExt;

    tokio::fs::create_dir_all(dest_dir)
        .await
        .map_err(|e| format!("mkdir failed: {e}"))?;

    let resp = reqwest::get(url)
        .await
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("HTTP {}: {}", resp.status(), url));
    }

    let total_size = resp.content_length().unwrap_or(0);
    let mut stream = resp.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut last_pct: u8 = 0;
    let mut buf = Vec::with_capacity(total_size as usize);

    if let Some(cb) = on_progress.as_ref() {
        cb(label, "downloading", 0, 0, total_size);
    }

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Download stream error: {e}"))?;
        downloaded += chunk.len() as u64;
        buf.extend_from_slice(&chunk);

        if let Some(cb) = on_progress.as_ref() {
            let pct = if total_size > 0 {
                ((downloaded as f64 / total_size as f64) * 100.0).min(100.0) as u8
            } else {
                0
            };
            if pct != last_pct {
                last_pct = pct;
                cb(label, "downloading", pct, downloaded, total_size);
            }
        }
    }

    let size_mb = buf.len() as f64 / (1024.0 * 1024.0);
    tracing::info!("  Downloaded: {:.1} MB, extracting...", size_mb);

    if let Some(cb) = on_progress.as_ref() {
        cb(label, "extracting", 0, downloaded, total_size);
    }

    let cursor = std::io::Cursor::new(buf);
    let mut archive = zip::ZipArchive::new(cursor).map_err(|e| format!("ZIP open failed: {e}"))?;

    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| format!("ZIP entry error: {e}"))?;
        let name = file.name().to_string();
        if !name.ends_with(".onnx") {
            continue;
        }
        let out_path = format!("{}/{}", dest_dir, name);
        if let Some(parent) = Path::new(&out_path).parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("mkdir failed: {e}"))?;
        }
        let mut out_file =
            std::fs::File::create(&out_path).map_err(|e| format!("File create failed: {e}"))?;
        std::io::copy(&mut file, &mut out_file).map_err(|e| format!("Extract failed: {e}"))?;
        tracing::info!("  Extracted: {}", name);
    }

    Ok(())
}
