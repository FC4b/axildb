//! Reranker model downloader. Mirrors [`axil_vector::download`] shape so
//! tooling (`axil rerank download <model>`, doctor, install) can treat
//! both kinds of models with the same UI primitives.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::models::RerankModel;

const HF_BASE: &str = "https://huggingface.co";

/// Download model.onnx + tokenizer.json from HuggingFace to
/// `~/.axil/rerank-models/<name>/`. Returns the model dir on success.
pub fn download(model: &RerankModel) -> Result<PathBuf, String> {
    let dir = model
        .model_dir()
        .ok_or_else(|| "cannot resolve rerank model directory".to_string())?;
    fs::create_dir_all(&dir).map_err(|e| format!("failed to create {}: {e}", dir.display()))?;

    let files = model.repo_files();
    if files.is_empty() {
        return Err("custom rerankers must be placed manually — no auto-download".into());
    }

    for (repo, file_path, local_name) in &files {
        let dest = dir.join(local_name);
        if dest.exists() {
            // Existing file: ensure sidecar so doctor can verify later.
            if !sidecar(&dest).exists() {
                let hash = sha256_file(&dest)?;
                let _ = fs::write(sidecar(&dest), format!("{hash}  {local_name}\n"));
            }
            continue;
        }
        let url = format!("{HF_BASE}/{repo}/resolve/main/{file_path}");
        eprintln!("  [rerank-download] {url} → {}", dest.display());
        fetch(&url, &dest)?;
    }

    Ok(dir)
}

/// True iff `model.onnx` + `tokenizer.json` both present in the model dir.
pub fn is_available(model: &RerankModel) -> bool {
    let Some(dir) = model.model_dir() else {
        return false;
    };
    dir.join("model.onnx").exists() && dir.join("tokenizer.json").exists()
}

fn fetch(url: &str, dest: &Path) -> Result<(), String> {
    let tmp = dest.with_extension("tmp");
    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("HTTP download failed for {url}: {e}"))?;
    let mut file = fs::File::create(&tmp).map_err(|e| format!("create temp: {e}"))?;
    std::io::copy(&mut response.into_body().as_reader(), &mut file).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        format!("write download: {e}")
    })?;
    let meta = fs::metadata(&tmp).map_err(|e| format!("stat temp: {e}"))?;
    if meta.len() == 0 {
        let _ = fs::remove_file(&tmp);
        return Err(format!("downloaded file empty: {}", dest.display()));
    }
    let hash = sha256_file(&tmp)?;
    fs::rename(&tmp, dest).map_err(|e| format!("rename: {e}"))?;
    let _ = fs::write(
        sidecar(dest),
        format!(
            "{hash}  {}\n",
            dest.file_name().unwrap_or_default().to_string_lossy()
        ),
    );
    Ok(())
}

fn sha256_file(p: &Path) -> Result<String, String> {
    let mut f = fs::File::open(p).map_err(|e| format!("open {}: {e}", p.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = f.read(&mut buf).map_err(|e| format!("read: {e}"))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sidecar(p: &Path) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(".sha256");
    PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_available_false_when_missing() {
        // Custom-path model pointing at a definitely-not-present file.
        let m = RerankModel::Custom(PathBuf::from("/tmp/__nope__/model.onnx"));
        assert!(!is_available(&m));
    }

    #[test]
    fn sidecar_appends_sha256_suffix() {
        let p = PathBuf::from("/tmp/foo/model.onnx");
        assert_eq!(sidecar(&p), PathBuf::from("/tmp/foo/model.onnx.sha256"));
    }
}
