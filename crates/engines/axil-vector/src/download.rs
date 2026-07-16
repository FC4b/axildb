use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::models::EmbeddingModel;

/// HuggingFace base URL for model files.
const HF_BASE: &str = "https://huggingface.co";

/// Model file manifest: (repo, file_path, expected_file_name).
fn model_files(model: &EmbeddingModel) -> Vec<(&'static str, &'static str, &'static str)> {
    match model {
        EmbeddingModel::BgeSmall => vec![
            ("BAAI/bge-small-en-v1.5", "onnx/model.onnx", "model.onnx"),
            ("BAAI/bge-small-en-v1.5", "tokenizer.json", "tokenizer.json"),
        ],
        EmbeddingModel::BgeSmallInt8 => vec![
            (
                "BAAI/bge-small-en-v1.5",
                "onnx/model_quantized.onnx",
                "model.onnx",
            ),
            ("BAAI/bge-small-en-v1.5", "tokenizer.json", "tokenizer.json"),
        ],
        EmbeddingModel::BgeBase => vec![
            ("BAAI/bge-base-en-v1.5", "onnx/model.onnx", "model.onnx"),
            ("BAAI/bge-base-en-v1.5", "tokenizer.json", "tokenizer.json"),
        ],
        EmbeddingModel::Nomic => vec![
            (
                "nomic-ai/nomic-embed-text-v1.5",
                "onnx/model.onnx",
                "model.onnx",
            ),
            (
                "nomic-ai/nomic-embed-text-v1.5",
                "tokenizer.json",
                "tokenizer.json",
            ),
        ],
        EmbeddingModel::BgeM3 => vec![
            // Xenova hosts a Transformers.js-compatible ONNX export that
            // works directly with the onnxruntime session used here. The
            // upstream BAAI/bge-m3 repo ships only PyTorch weights.
            ("Xenova/bge-m3", "onnx/model.onnx", "model.onnx"),
            ("Xenova/bge-m3", "tokenizer.json", "tokenizer.json"),
        ],
        EmbeddingModel::GteModernbertBase => vec![
            // Alibaba-NLP ships the ONNX export directly under onnx/model.onnx
            // (no Xenova mirror needed). 8k context, mean-pooled.
            (
                "Alibaba-NLP/gte-modernbert-base",
                "onnx/model.onnx",
                "model.onnx",
            ),
            (
                "Alibaba-NLP/gte-modernbert-base",
                "tokenizer.json",
                "tokenizer.json",
            ),
        ],
        EmbeddingModel::Custom { .. } => vec![],
    }
}

/// Download model files from HuggingFace to the local model directory.
///
/// Downloads to `~/.axil/models/<model_name>/`.
/// Skips files that already exist.
pub fn download_model(model: &EmbeddingModel) -> Result<PathBuf, String> {
    let dir = model
        .model_dir()
        .ok_or_else(|| "cannot resolve model directory".to_string())?;

    fs::create_dir_all(&dir)
        .map_err(|e| format!("failed to create model directory {}: {e}", dir.display()))?;

    let files = model_files(model);
    if files.is_empty() {
        return Err("custom models cannot be auto-downloaded".into());
    }

    for (repo, file_path, local_name) in &files {
        let dest = dir.join(local_name);
        if dest.exists() {
            // File exists — ensure it has a checksum sidecar.
            if !sidecar_path(&dest).exists() {
                let hash = sha256_file(&dest)?;
                let _ = fs::write(sidecar_path(&dest), format!("{hash}  {local_name}\n"));
                eprintln!("  [checksum] {} (computed for existing file)", local_name);
            } else {
                eprintln!("  [skip] {} (already exists)", local_name);
            }
            continue;
        }

        let url = format!("{}/{}/resolve/main/{}", HF_BASE, repo, file_path);

        eprintln!("  [download] {} → {}", url, dest.display());
        download_file(&url, &dest)?;
    }

    Ok(dir)
}

/// Check if a model's files are present locally.
pub fn is_model_available(model: &EmbeddingModel) -> bool {
    let Some(dir) = model.model_dir() else {
        return false;
    };
    dir.join("model.onnx").exists() && dir.join("tokenizer.json").exists()
}

/// Download a single file from a URL to a local path, retrying on failure.
///
/// HuggingFace intermittently 504s / rate-limits (especially shared CI
/// runner IPs), so transient failures get a short exponential backoff
/// before giving up — a one-shot request turns a blip into a hard error
/// for whatever command triggered the auto-download.
fn download_file(url: &str, dest: &Path) -> Result<(), String> {
    const ATTEMPTS: u32 = 3;
    let mut last_err = String::new();
    for attempt in 1..=ATTEMPTS {
        match download_file_once(url, dest) {
            Ok(()) => return Ok(()),
            Err(e) => {
                last_err = e;
                if attempt < ATTEMPTS {
                    let wait = 2u64.pow(attempt);
                    eprintln!(
                        "  [retry] attempt {attempt}/{ATTEMPTS} failed: {last_err} — retrying in {wait}s"
                    );
                    std::thread::sleep(std::time::Duration::from_secs(wait));
                }
            }
        }
    }
    Err(format!("{last_err} (after {ATTEMPTS} attempts)"))
}

/// Single download attempt.
///
/// Downloads to a `.tmp` file first, computes SHA256 checksum, then atomically
/// renames on success. Stores the checksum in a `.sha256` sidecar file.
fn download_file_once(url: &str, dest: &Path) -> Result<(), String> {
    let tmp = dest.with_extension("tmp");

    // Use ureq HTTP client instead of external curl binary.
    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("HTTP download failed for {url}: {e}"))?;

    let mut file = fs::File::create(&tmp)
        .map_err(|e| format!("failed to create temp file {}: {e}", tmp.display()))?;
    std::io::copy(&mut response.into_body().as_reader(), &mut file).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        format!("failed to write download to {}: {e}", tmp.display())
    })?;

    // Verify the file was written and is non-empty.
    let meta = fs::metadata(&tmp).map_err(|e| format!("failed to verify download: {e}"))?;
    if meta.len() == 0 {
        let _ = fs::remove_file(&tmp);
        return Err(format!("downloaded file is empty: {}", dest.display()));
    }

    let hash = sha256_file(&tmp)?;
    eprintln!("  [sha256] {hash}");

    // Atomic rename: tmp → dest.
    fs::rename(&tmp, dest).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        format!(
            "failed to rename {} → {}: {e}",
            tmp.display(),
            dest.display()
        )
    })?;

    // Store checksum as sidecar file for future verification.
    let checksum_path = sidecar_path(dest);
    let _ = fs::write(
        &checksum_path,
        format!(
            "{hash}  {}\n",
            dest.file_name().unwrap_or_default().to_string_lossy()
        ),
    );

    let size_mb = meta.len() as f64 / (1024.0 * 1024.0);
    eprintln!("  [done] {} ({:.1} MB)", dest.display(), size_mb);

    Ok(())
}

/// Compute the SHA256 hex digest of a file.
pub fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file =
        fs::File::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("read error: {e}"))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Verify a downloaded model file against its stored SHA256 checksum.
///
/// Returns `Ok(true)` if checksum matches, `Ok(false)` if mismatch,
/// `Err` if no checksum file exists or file is unreadable.
pub fn verify_checksum(file_path: &Path) -> Result<bool, String> {
    let checksum_path = sidecar_path(file_path);
    if !checksum_path.exists() {
        return Err(format!("no checksum file at {}", checksum_path.display()));
    }
    let stored =
        fs::read_to_string(&checksum_path).map_err(|e| format!("failed to read checksum: {e}"))?;
    let expected_hash = stored
        .split_whitespace()
        .next()
        .ok_or_else(|| "checksum file is empty".to_string())?;
    let actual_hash = sha256_file(file_path)?;
    Ok(actual_hash == expected_hash)
}

/// Verify all files for a model with full SHA256 re-hash. Returns list of (file_name, ok).
pub fn verify_model(model: &EmbeddingModel) -> Result<Vec<(String, bool)>, String> {
    let dir = model
        .model_dir()
        .ok_or_else(|| "cannot resolve model directory".to_string())?;
    let files = model_files(model);
    if files.is_empty() {
        return Err("custom models have no manifest for verification".into());
    }
    let mut results = Vec::new();
    for (_, _, local_name) in &files {
        let path = dir.join(local_name);
        let ok = verify_checksum(&path).unwrap_or(false);
        results.push((local_name.to_string(), ok));
    }
    Ok(results)
}

/// Check that checksum sidecar files exist for a model (no re-hashing).
pub fn has_checksums(model: &EmbeddingModel) -> Result<Vec<(String, bool)>, String> {
    let dir = model
        .model_dir()
        .ok_or_else(|| "cannot resolve model directory".to_string())?;
    let files = model_files(model);
    if files.is_empty() {
        return Err("custom models have no manifest for verification".into());
    }
    let mut results = Vec::new();
    for (_, _, local_name) in &files {
        let path = dir.join(local_name);
        let has = sidecar_path(&path).exists();
        results.push((local_name.to_string(), has));
    }
    Ok(results)
}

/// Sidecar checksum path: `model.onnx` → `model.onnx.sha256`
fn sidecar_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".sha256");
    PathBuf::from(s)
}

/// Remove a downloaded model's files from disk.
///
/// Only works for built-in models (BgeSmall, BgeBase, Nomic).
/// Rejects Custom models to prevent deleting unrelated directories.
pub fn remove_model(model: &EmbeddingModel) -> Result<(), String> {
    if matches!(model, EmbeddingModel::Custom { .. }) {
        return Err("cannot auto-remove custom models — delete the files manually".into());
    }

    let dir = model
        .model_dir()
        .ok_or_else(|| "cannot resolve model directory".to_string())?;

    if !dir.exists() {
        return Err(format!("model not found at {}", dir.display()));
    }

    fs::remove_dir_all(&dir).map_err(|e| format!("failed to remove {}: {e}", dir.display()))?;

    Ok(())
}

/// List all downloaded models with their sizes.
pub fn list_models() -> Vec<(EmbeddingModel, PathBuf, u64)> {
    let models = [
        EmbeddingModel::BgeSmall,
        EmbeddingModel::BgeSmallInt8,
        EmbeddingModel::BgeBase,
        EmbeddingModel::Nomic,
        EmbeddingModel::BgeM3,
        EmbeddingModel::GteModernbertBase,
    ];

    let mut result = Vec::new();
    for model in models {
        if is_model_available(&model) {
            if let Some(dir) = model.model_dir() {
                let size = dir_size(&dir);
                result.push((model, dir, size));
            }
        }
    }
    result
}

/// Total size of a directory in bytes.
fn dir_size(path: &Path) -> u64 {
    fs::read_dir(path)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0)
}
