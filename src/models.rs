/// Download model files if missing.
use crate::config::AiConfig;
use rust_client_api::model_downloader::ModelDownloader;
use std::path::Path;

struct ModelFile {
    rel_path: &'static str,
    url: &'static str,
    category: ModelCategory,
}

/// Which category of models to download.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelCategory {
    OcrServer,
    OcrMobile,
    Clip,
    Face,
}

/// Progress callback: (`file_name`, status, percent 0-100, `downloaded_bytes`, `total_bytes`)
pub type ProgressFn = rust_client_api::model_downloader::ProgressFn;

const MODEL_FILES: &[ModelFile] = &[
    // CLIP models (from MT-Photos release, publicly accessible)
    ModelFile {
        rel_path: "clip/vit-b-16.img.fp32.onnx",
        url: "https://github.com/MT-Photos/mt-photos-ai/releases/download/v1.1.0/vit-b-16.img.fp32.onnx",
        category: ModelCategory::Clip,
    },
    ModelFile {
        rel_path: "clip/vit-b-16.txt.fp32.onnx",
        url: "https://github.com/MT-Photos/mt-photos-ai/releases/download/v1.1.0/vit-b-16.txt.fp32.onnx",
        category: ModelCategory::Clip,
    },
    // OCR models — PP-OCRv5 server (ONNX, from HuggingFace bukuroo/PPOCRv5-ONNX)
    ModelFile {
        rel_path: "ocr/PP-OCRv5_server_det.onnx",
        url: "https://huggingface.co/bukuroo/PPOCRv5-ONNX/resolve/main/ppocrv5-server-det.onnx",
        category: ModelCategory::OcrServer,
    },
    ModelFile {
        rel_path: "ocr/PP-OCRv5_cls.onnx",
        url: "https://huggingface.co/bukuroo/PPOCRv5-ONNX/resolve/main/ppocrv5-cls.onnx",
        category: ModelCategory::OcrServer,
    },
    ModelFile {
        rel_path: "ocr/PP-OCRv5_server_rec.onnx",
        url: "https://huggingface.co/bukuroo/PPOCRv5-ONNX/resolve/main/ppocrv5-server-rec.onnx",
        category: ModelCategory::OcrServer,
    },
    // OCR models — PP-OCRv5 mobile (lightweight variant)
    ModelFile {
        rel_path: "ocr/PP-OCRv5_mobile_det.onnx",
        url: "https://huggingface.co/bukuroo/PPOCRv5-ONNX/resolve/main/ppocrv5-mobile-det.onnx",
        category: ModelCategory::OcrMobile,
    },
    ModelFile {
        rel_path: "ocr/PP-OCRv5_mobile_rec.onnx",
        url: "https://huggingface.co/bukuroo/PPOCRv5-ONNX/resolve/main/ppocrv5-mobile-rec.onnx",
        category: ModelCategory::OcrMobile,
    },
    // Mobile also needs the shared classifier
    ModelFile {
        rel_path: "ocr/PP-OCRv5_cls.onnx",
        url: "https://huggingface.co/bukuroo/PPOCRv5-ONNX/resolve/main/ppocrv5-cls.onnx",
        category: ModelCategory::OcrMobile,
    },
    // Face models (InsightFace buffalo_l pack, publicly accessible)
    ModelFile {
        rel_path: "face/det_10g.onnx",
        url: "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip#det_10g.onnx",
        category: ModelCategory::Face,
    },
    ModelFile {
        rel_path: "face/w600k_r50.onnx",
        url: "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip#w600k_r50.onnx",
        category: ModelCategory::Face,
    },
];

pub async fn ensure_models(config: &AiConfig) -> Result<(), String> {
    ensure_models_with_progress(config, None).await
}

pub async fn ensure_models_with_progress(config: &AiConfig, on_progress: Option<ProgressFn>) -> Result<(), String> {
    let enabled: Vec<ModelCategory> = [
        (config.enable_clip, ModelCategory::Clip),
        (config.enable_ocr, ModelCategory::OcrServer),
        (config.enable_ocr, ModelCategory::OcrMobile),
        (config.enable_face, ModelCategory::Face),
    ]
    .into_iter()
    .filter(|(e, _)| *e)
    .map(|(_, c)| c)
    .collect();

    for cat in enabled {
        download_category(config, cat, &on_progress).await?;
    }
    Ok(())
}

/// Download models for a single category.
pub async fn ensure_category_with_progress(
    config: &AiConfig,
    category: ModelCategory,
    on_progress: Option<ProgressFn>,
) -> Result<(), String> {
    download_category(config, category, &on_progress).await
}

async fn download_category(
    config: &AiConfig,
    category: ModelCategory,
    on_progress: &Option<ProgressFn>,
) -> Result<(), String> {
    let dir = &config.models_dir;
    let downloader = ModelDownloader::new(reqwest::Client::new());
    let mut zip_downloaded: std::collections::HashSet<String> = std::collections::HashSet::new();

    for f in MODEL_FILES {
        if f.category != category {
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
                downloader
                    .download_and_extract_zip(zip_url, parent.to_str().unwrap_or("."), f.rel_path, on_progress)
                    .await
                    .map_err(|e| e.to_string())?;
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
            downloader
                .download_file(f.url, &full_path, f.rel_path, on_progress)
                .await
                .map_err(|e| e.to_string())?;
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

/// Check whether OCR Server model files exist on disk.
pub fn ocr_server_models_present(config: &AiConfig) -> bool {
    let dir = &config.models_dir;
    Path::new(&format!("{dir}/ocr/PP-OCRv5_server_det.onnx")).exists()
        && Path::new(&format!("{dir}/ocr/PP-OCRv5_cls.onnx")).exists()
        && Path::new(&format!("{dir}/ocr/PP-OCRv5_server_rec.onnx")).exists()
}

/// Check whether OCR Mobile model files exist on disk.
pub fn ocr_mobile_models_present(config: &AiConfig) -> bool {
    let dir = &config.models_dir;
    Path::new(&format!("{dir}/ocr/PP-OCRv5_mobile_det.onnx")).exists()
        && Path::new(&format!("{dir}/ocr/PP-OCRv5_cls.onnx")).exists()
        && Path::new(&format!("{dir}/ocr/PP-OCRv5_mobile_rec.onnx")).exists()
}

/// Check whether OCR model files exist on disk (at least server variant).
pub fn ocr_models_present(config: &AiConfig) -> bool {
    ocr_server_models_present(config)
}

/// Check whether face detection/recognition model files exist on disk.
pub fn face_models_present(config: &AiConfig) -> bool {
    let dir = &config.models_dir;
    Path::new(&format!("{dir}/face/det_10g.onnx")).exists() && Path::new(&format!("{dir}/face/w600k_r50.onnx")).exists()
}

/// Download a single model file to the given destination path.
pub async fn download_model_file(url: &str, dest: &str) -> Result<(), String> {
    let downloader = ModelDownloader::new(reqwest::Client::new());
    downloader
        .download_file(url, dest, dest, &None)
        .await
        .map_err(|e| e.to_string())
}
