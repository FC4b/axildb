//! Embedding model files: where each registered model's files come from, how
//! they are checked, and where they live.
//!
//! Every registered model is pinned: each file is fetched from one Hugging
//! Face commit and installed only when its size and sha256 match the values
//! compiled in here, so an upstream change to a model repo cannot change what
//! a new install embeds with. Custom models are never downloaded; their only
//! integrity check is a `.sha256` sidecar recorded next to the file
//! ([`verify_checksum`]).

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use crate::models::EmbeddingModel;

/// HuggingFace base URL for model files.
const HF_BASE: &str = "https://huggingface.co";

/// One file of a registered model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelFile {
    /// Path inside the Hugging Face repo.
    pub remote: &'static str,
    /// File name in the local model directory.
    pub local: &'static str,
    /// Expected sha256 of the file, lowercase hex.
    pub sha256: &'static str,
    /// Expected size in bytes.
    pub size: u64,
}

/// A registered model's files, pinned to one Hugging Face revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PinnedModel {
    /// Hugging Face repo id.
    pub repo: &'static str,
    /// Git revision (commit sha) every file is fetched from.
    pub revision: &'static str,
    /// The files the model needs, all in one directory.
    pub files: &'static [ModelFile],
}

impl PinnedModel {
    /// Pinned download URL of one file.
    pub fn url(&self, file: &ModelFile) -> String {
        format!(
            "{HF_BASE}/{}/resolve/{}/{}",
            self.repo, self.revision, file.remote
        )
    }

    /// Bytes downloaded for a fresh install.
    pub fn total_size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }
}

/// The WordPiece tokenizer the BGE English models and nomic-embed-text share:
/// byte-identical in every repo that uses it below.
const BERT_UNCASED_TOKENIZER: ModelFile = ModelFile {
    remote: "tokenizer.json",
    local: "tokenizer.json",
    sha256: "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    size: 711_396,
};

const BGE_SMALL: PinnedModel = PinnedModel {
    repo: "BAAI/bge-small-en-v1.5",
    revision: "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    files: &[
        ModelFile {
            remote: "onnx/model.onnx",
            local: "model.onnx",
            sha256: "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
            size: 133_093_490,
        },
        BERT_UNCASED_TOKENIZER,
    ],
};

/// BAAI's repo ships no quantized export. Xenova's export of the same
/// checkpoint does, and its `onnx/model.onnx` is byte-identical to BAAI's.
const BGE_SMALL_INT8: PinnedModel = PinnedModel {
    repo: "Xenova/bge-small-en-v1.5",
    revision: "ea104dacec62c0de699686887e3f920caeb4f3e3",
    files: &[
        ModelFile {
            remote: "onnx/model_quantized.onnx",
            local: "model.onnx",
            sha256: "6c9c6101a956d62dfb5e7190c538226c0c5bb9cb27b651234b6df063ee7dbfe4",
            size: 34_014_426,
        },
        BERT_UNCASED_TOKENIZER,
    ],
};

const BGE_BASE: PinnedModel = PinnedModel {
    repo: "BAAI/bge-base-en-v1.5",
    revision: "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    files: &[
        ModelFile {
            remote: "onnx/model.onnx",
            local: "model.onnx",
            sha256: "9bc579acdba21c253c62a9bf866891355a63ffa3442b52c8a37d75b2ccb91848",
            size: 435_811_539,
        },
        BERT_UNCASED_TOKENIZER,
    ],
};

const NOMIC: PinnedModel = PinnedModel {
    repo: "nomic-ai/nomic-embed-text-v1.5",
    revision: "e9b6763023c676ca8431644204f50c2b100d9aab",
    files: &[
        ModelFile {
            remote: "onnx/model.onnx",
            local: "model.onnx",
            sha256: "147d5aa88c2101237358e17796cf3a227cead1ec304ec34b465bb08e9d952965",
            size: 547_310_275,
        },
        BERT_UNCASED_TOKENIZER,
    ],
};

/// Xenova hosts a Transformers.js-compatible ONNX export that works directly
/// with the onnxruntime session used here; the upstream BAAI/bge-m3 repo
/// ships only PyTorch weights. The graph keeps its weights in an external
/// `model.onnx_data` file, which onnxruntime looks for next to `model.onnx`.
const BGE_M3: PinnedModel = PinnedModel {
    repo: "Xenova/bge-m3",
    revision: "4de13258303883538bd53b696b452bf8099f0858",
    files: &[
        ModelFile {
            remote: "onnx/model.onnx",
            local: "model.onnx",
            sha256: "5d89a0010dd39aa2cfa8b22bb49f06904c5bbf5877135f877da419480f40cde3",
            size: 607_298,
        },
        ModelFile {
            remote: "onnx/model.onnx_data",
            local: "model.onnx_data",
            sha256: "1eebfb28493f67bba03ce0ef64bfdc7fc5a3bd9d7493f818bb1d78cd798416b4",
            size: 2_266_820_608,
        },
        ModelFile {
            remote: "tokenizer.json",
            local: "tokenizer.json",
            sha256: "6710678b12670bc442b99edc952c4d996ae309a7020c1fa0096dd245c2faf790",
            size: 17_082_821,
        },
    ],
};

/// Alibaba-NLP ships the ONNX export directly under `onnx/model.onnx`.
const GTE_MODERNBERT_BASE: PinnedModel = PinnedModel {
    repo: "Alibaba-NLP/gte-modernbert-base",
    revision: "e7f32e3c00f91d699e8c43b53106206bcc72bb22",
    files: &[
        ModelFile {
            remote: "onnx/model.onnx",
            local: "model.onnx",
            sha256: "947f31df7effaeec4edb57c50e4ed7e0f2034d9336063f92615b92e3e0d24d78",
            size: 596_392_315,
        },
        ModelFile {
            remote: "tokenizer.json",
            local: "tokenizer.json",
            sha256: "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
            size: 3_583_228,
        },
    ],
};

/// IBM's repo ships only PyTorch and safetensors weights. The onnx-community
/// export (Transformers.js, fp32) keeps its weights in an external
/// `model.onnx_data`, as bge-m3's does. Its `tokenizer.json` is IBM's
/// tokenizer re-serialized (merges written as pairs): the same vocabulary,
/// merges, normalizer, pre-tokenizer and post-processor.
const GRANITE_SMALL_R2: PinnedModel = PinnedModel {
    repo: "onnx-community/granite-embedding-small-english-r2-ONNX",
    revision: "1dc7835ba0cb9c76a3618d0bf0c427c97671b3c8",
    files: &[
        ModelFile {
            remote: "onnx/model.onnx",
            local: "model.onnx",
            sha256: "cddb145cd1147ec24a3908b2ca2602b98b20a3d198365cff270b7cb26c98179e",
            size: 182_504,
        },
        ModelFile {
            remote: "onnx/model.onnx_data",
            local: "model.onnx_data",
            sha256: "86a3a705d4598615894d89540ea71a3d9bbdb17a315e79edcd5dfc737222834b",
            size: 194_844_160,
        },
        ModelFile {
            remote: "tokenizer.json",
            local: "tokenizer.json",
            sha256: "feeb83348dcb033bc6b9d2e1f7906ca9eb2d122845000c9416d894d7c2927149",
            size: 2_128_614,
        },
    ],
};

/// The pinned files of a registered model; `None` for a custom model.
pub fn pinned_model(model: &EmbeddingModel) -> Option<&'static PinnedModel> {
    match model {
        EmbeddingModel::BgeSmall => Some(&BGE_SMALL),
        EmbeddingModel::BgeSmallInt8 => Some(&BGE_SMALL_INT8),
        EmbeddingModel::BgeBase => Some(&BGE_BASE),
        EmbeddingModel::Nomic => Some(&NOMIC),
        EmbeddingModel::BgeM3 => Some(&BGE_M3),
        EmbeddingModel::GteModernbertBase => Some(&GTE_MODERNBERT_BASE),
        EmbeddingModel::GraniteSmallR2 => Some(&GRANITE_SMALL_R2),
        EmbeddingModel::Custom { .. } => None,
    }
}

/// Make a registered model's directory, `~/.axil/models/<model_name>/`, hold
/// its pinned files.
///
/// A file already there is kept when its size and sha256 match the pin (it is
/// hashed, not fetched again), so an existing install of the pinned bytes
/// costs one read. A missing or mismatching file is fetched from the pinned
/// revision and installed only once it verifies. Safe to run from several
/// processes at once: each download writes its own temporary file, renamed
/// into place only after it verifies.
pub fn download_model(model: &EmbeddingModel) -> Result<PathBuf, String> {
    let pinned =
        pinned_model(model).ok_or_else(|| "custom models cannot be auto-downloaded".to_string())?;
    let dir = model
        .model_dir()
        .ok_or_else(|| "cannot resolve model directory".to_string())?;
    download_into(pinned, &dir, fetch)?;
    Ok(dir)
}

/// [`download_model`] into `dir`, with `fetch(url, path)` writing each file.
fn download_into<F>(pinned: &PinnedModel, dir: &Path, fetch: F) -> Result<(), String>
where
    F: Fn(&str, &Path) -> Result<(), String>,
{
    fs::create_dir_all(dir)
        .map_err(|e| format!("failed to create model directory {}: {e}", dir.display()))?;
    for file in pinned.files {
        install(dir, file, &pinned.url(file), &fetch)?;
    }
    Ok(())
}

/// Check if a model's files are present locally.
///
/// For a registered model, every pinned file must be there at its pinned
/// size. This is the cheap check `VectorEngine::with_model` runs before it
/// loads a model, downloading when it fails: a corrupt file of the right
/// size passes it, and [`verify_model`] or [`download_model`] hash. For a
/// custom model, `model.onnx` and `tokenizer.json` must exist in its
/// directory.
pub fn is_model_available(model: &EmbeddingModel) -> bool {
    let Some(dir) = model.model_dir() else {
        return false;
    };
    match pinned_model(model) {
        Some(pinned) => pinned.files.iter().all(|f| has_pinned_size(&dir, f)),
        None => dir.join("model.onnx").exists() && dir.join("tokenizer.json").exists(),
    }
}

/// Per-file version of [`is_model_available`] for a registered model:
/// whether each pinned file is present at its pinned size, without hashing.
pub fn check_sizes(model: &EmbeddingModel) -> Result<Vec<(String, bool)>, String> {
    let pinned = pinned_model(model)
        .ok_or_else(|| "custom models have no pinned files to check".to_string())?;
    let dir = model
        .model_dir()
        .ok_or_else(|| "cannot resolve model directory".to_string())?;
    Ok(pinned
        .files
        .iter()
        .map(|f| (f.local.to_string(), has_pinned_size(&dir, f)))
        .collect())
}

/// Whether `dir` holds `file` at its pinned size (symlinks followed).
fn has_pinned_size(dir: &Path, file: &ModelFile) -> bool {
    fs::metadata(dir.join(file.local))
        .map(|m| m.len() == file.size)
        .unwrap_or(false)
}

/// Attempts per file; a transfer error is retried, a verification failure is not.
const ATTEMPTS: u32 = 3;

/// Install `file` into `dir` from `url`, with `fetch(url, path)` writing its
/// bytes to the path it is given.
///
/// Each attempt writes to its own temporary file in `dir` (process id, time
/// and a counter in the name), so concurrent downloaders never share one.
/// That file's size and sha256 are checked on disk after it is fully
/// written, and only then is it renamed over the final name, which is atomic
/// within one directory. A final file that already verifies is kept: it is
/// either from an earlier run or from a concurrent downloader that finished
/// first, and both hold the pinned bytes.
///
/// HuggingFace intermittently 504s and rate-limits (especially shared CI
/// runner IPs), so a transfer error gets a short exponential backoff before
/// giving up.
fn install<F>(dir: &Path, file: &ModelFile, url: &str, fetch: F) -> Result<(), String>
where
    F: Fn(&str, &Path) -> Result<(), String>,
{
    let dest = dir.join(file.local);
    match check(&dest, file) {
        Ok(()) => {
            axil_core::diagln!("  [ok] {} (matches the pinned sha256)", file.local);
            return Ok(());
        }
        Err(e) if dest.exists() => axil_core::diagln!("  [replace] {e}"),
        Err(_) => {}
    }
    axil_core::diagln!("  [download] {url} → {}", dest.display());
    let mut last = String::new();
    for attempt in 1..=ATTEMPTS {
        let tmp = temp_path(dir, file.local);
        if let Err(e) = fetch(url, &tmp) {
            let _ = fs::remove_file(&tmp);
            last = e;
            if attempt < ATTEMPTS {
                let wait = 2u64.pow(attempt);
                axil_core::diagln!(
                    "  [retry] attempt {attempt}/{ATTEMPTS} failed: {last} — retrying in {wait}s"
                );
                std::thread::sleep(Duration::from_secs(wait));
            }
            continue;
        }
        if let Err(e) = check(&tmp, file) {
            let _ = fs::remove_file(&tmp);
            return Err(e);
        }
        if verifies(&dest, file) {
            let _ = fs::remove_file(&tmp);
            return Ok(());
        }
        return match fs::rename(&tmp, &dest) {
            Ok(()) => {
                let size_mb = file.size as f64 / (1024.0 * 1024.0);
                axil_core::diagln!(
                    "  [done] {} ({size_mb:.1} MB, sha256 verified)",
                    dest.display()
                );
                Ok(())
            }
            Err(e) => {
                let _ = fs::remove_file(&tmp);
                if verifies(&dest, file) {
                    Ok(())
                } else {
                    Err(format!("failed to install {}: {e}", dest.display()))
                }
            }
        };
    }
    Err(format!(
        "HTTP download failed for {url}: {last} (after {ATTEMPTS} attempts)"
    ))
}

/// A temporary path in `dir` that no other download, in this process or
/// another, will use.
fn temp_path(dir: &Path, local: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    dir.join(format!(
        ".{local}.{}.{nanos}.{}.part",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

/// Whether `path` holds exactly the pinned bytes of `file`.
fn verifies(path: &Path, file: &ModelFile) -> bool {
    check(path, file).is_ok()
}

/// Check `path` against `file`'s pinned size, then its sha256.
fn check(path: &Path, file: &ModelFile) -> Result<(), String> {
    let size = fs::metadata(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .len();
    if size != file.size {
        return Err(format!(
            "{}: {size} bytes, pinned {} for {}",
            path.display(),
            file.size,
            file.remote
        ));
    }
    let actual = sha256_file(path)?;
    if actual != file.sha256 {
        return Err(format!(
            "{}: sha256 {actual} does not match the pinned {} for {}",
            path.display(),
            file.sha256,
            file.remote
        ));
    }
    Ok(())
}

/// Stream `url` into `path`.
fn fetch(url: &str, path: &Path) -> Result<(), String> {
    let response = ureq::get(url).call().map_err(|e| e.to_string())?;
    let mut reader = response.into_body().into_reader();
    let mut out =
        fs::File::create(path).map_err(|e| format!("failed to create {}: {e}", path.display()))?;
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = reader.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])
            .map_err(|e| format!("failed to write {}: {e}", path.display()))?;
    }
    out.sync_all().map_err(|e| e.to_string())
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

/// Verify a file against the SHA256 checksum recorded in its `.sha256`
/// sidecar. This is the integrity check for custom models, which have no
/// pin; registered models are checked against their pins by
/// [`verify_model`].
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

/// Verify all files for a model with a full SHA256 re-hash. Returns list of
/// (file_name, ok).
///
/// A registered model's files are checked against their pinned size and
/// sha256. A custom model's files (the ONNX file and the `tokenizer.json`
/// next to it) are checked against their recorded sidecars; a file without
/// one counts as not ok.
pub fn verify_model(model: &EmbeddingModel) -> Result<Vec<(String, bool)>, String> {
    let dir = model
        .model_dir()
        .ok_or_else(|| "cannot resolve model directory".to_string())?;
    if let Some(pinned) = pinned_model(model) {
        return Ok(pinned
            .files
            .iter()
            .map(|f| (f.local.to_string(), verifies(&dir.join(f.local), f)))
            .collect());
    }
    Ok(custom_files(model, &dir)
        .into_iter()
        .map(|path| {
            let ok = verify_checksum(&path).unwrap_or(false);
            (file_name(&path), ok)
        })
        .collect())
}

/// Check that each of a model's files has a checksum to verify against,
/// without re-hashing. Always true for a registered model's files, whose
/// checksums are pinned in this crate; for a custom model, whether each file
/// has a recorded `.sha256` sidecar.
pub fn has_checksums(model: &EmbeddingModel) -> Result<Vec<(String, bool)>, String> {
    let dir = model
        .model_dir()
        .ok_or_else(|| "cannot resolve model directory".to_string())?;
    if let Some(pinned) = pinned_model(model) {
        return Ok(pinned
            .files
            .iter()
            .map(|f| (f.local.to_string(), true))
            .collect());
    }
    Ok(custom_files(model, &dir)
        .into_iter()
        .map(|path| {
            let has = sidecar_path(&path).exists();
            (file_name(&path), has)
        })
        .collect())
}

/// A custom model's ONNX file and the `tokenizer.json` next to it.
fn custom_files(model: &EmbeddingModel, dir: &Path) -> Vec<PathBuf> {
    let onnx = match model {
        EmbeddingModel::Custom { path, .. } => path.clone(),
        _ => dir.join("model.onnx"),
    };
    vec![onnx, dir.join("tokenizer.json")]
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
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
        EmbeddingModel::GraniteSmallR2,
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

#[cfg(test)]
mod tests {
    use super::*;

    const REGISTERED: [EmbeddingModel; 7] = [
        EmbeddingModel::BgeSmall,
        EmbeddingModel::BgeSmallInt8,
        EmbeddingModel::BgeBase,
        EmbeddingModel::Nomic,
        EmbeddingModel::BgeM3,
        EmbeddingModel::GteModernbertBase,
        EmbeddingModel::GraniteSmallR2,
    ];

    /// A pinned file for `bytes`, as the tests' download payload.
    fn pinned(local: &'static str, bytes: &[u8]) -> ModelFile {
        ModelFile {
            remote: local,
            local,
            sha256: Box::leak(format!("{:x}", Sha256::digest(bytes)).into_boxed_str()),
            size: bytes.len() as u64,
        }
    }

    /// A fetch that writes `bytes` in small chunks, yielding between them, so
    /// concurrent writers interleave.
    fn slow_writer(bytes: &[u8]) -> impl Fn(&str, &Path) -> Result<(), String> + '_ {
        move |_, path| {
            let mut out = fs::File::create(path).map_err(|e| e.to_string())?;
            for chunk in bytes.chunks(4096) {
                out.write_all(chunk).map_err(|e| e.to_string())?;
                std::thread::sleep(Duration::from_micros(200));
            }
            Ok(())
        }
    }

    fn never_fetch(_: &str, _: &Path) -> Result<(), String> {
        panic!("a verified file must not be fetched")
    }

    fn leftovers(dir: &Path) -> Vec<String> {
        fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".part"))
            .collect()
    }

    #[test]
    fn every_registered_model_is_pinned_to_a_commit() {
        for model in &REGISTERED {
            let pinned = pinned_model(model).unwrap_or_else(|| panic!("{model} has no pin"));
            assert_eq!(pinned.revision.len(), 40, "{model}: {}", pinned.revision);
            assert!(pinned.revision.chars().all(|c| c.is_ascii_hexdigit()));
            let locals: Vec<&str> = pinned.files.iter().map(|f| f.local).collect();
            assert!(locals.contains(&"model.onnx"), "{model}: {locals:?}");
            assert!(locals.contains(&"tokenizer.json"), "{model}: {locals:?}");
            for file in pinned.files {
                assert_eq!(file.sha256.len(), 64, "{model}: {}", file.remote);
                assert!(file
                    .sha256
                    .chars()
                    .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)));
                assert!(file.size > 0, "{model}: {}", file.remote);
                let url = pinned.url(file);
                assert!(
                    url.contains(&format!("/resolve/{}/", pinned.revision)),
                    "{url}"
                );
                assert!(!url.contains("/resolve/main/"), "{url}");
            }
        }
        let custom = EmbeddingModel::Custom {
            path: PathBuf::from("/models/mine.onnx"),
            dimensions: 8,
            pooling: crate::models::PoolingStrategy::Cls,
            max_seq_len: 16,
        };
        assert!(pinned_model(&custom).is_none());
    }

    #[test]
    fn an_installed_file_that_matches_its_pin_is_kept_without_a_fetch() {
        let file = pinned("model.onnx", b"pinned bytes");
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("model.onnx");
        fs::write(&dest, b"pinned bytes").unwrap();
        install(dir.path(), &file, "https://example.test/m", never_fetch).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"pinned bytes");
    }

    #[test]
    fn a_mismatching_installed_file_is_replaced_by_the_pinned_one() {
        let file = pinned("model.onnx", b"pinned bytes");
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("model.onnx");
        // Same size, different bytes: only the hash catches it.
        fs::write(&dest, b"forged bytes").unwrap();
        install(dir.path(), &file, "u", slow_writer(b"pinned bytes")).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"pinned bytes");
        assert!(leftovers(dir.path()).is_empty());
    }

    #[test]
    fn a_download_that_fails_verification_is_never_installed() {
        let file = pinned("model.onnx", b"pinned bytes");
        let dir = tempfile::tempdir().unwrap();
        let err = install(dir.path(), &file, "u", slow_writer(b"forged bytes")).unwrap_err();
        assert!(err.contains("does not match the pinned"), "{err}");
        let err = install(dir.path(), &file, "u", slow_writer(b"short")).unwrap_err();
        assert!(err.contains("bytes, pinned"), "{err}");
        assert!(!dir.path().join("model.onnx").exists());
        assert!(leftovers(dir.path()).is_empty());
    }

    #[test]
    fn concurrent_downloads_install_one_verified_file() {
        let payload: Vec<u8> = (0..256 * 1024u32).map(|i| (i * 31 % 251) as u8).collect();
        let file = pinned("model.onnx", &payload);
        let dir = tempfile::tempdir().unwrap();
        let start = std::sync::Barrier::new(4);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        start.wait();
                        install(dir.path(), &file, "u", slow_writer(&payload))
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap().unwrap();
            }
        });
        assert_eq!(fs::read(dir.path().join("model.onnx")).unwrap(), payload);
        assert!(
            leftovers(dir.path()).is_empty(),
            "{:?}",
            leftovers(dir.path())
        );
    }

    #[test]
    fn a_download_fetches_each_file_from_the_pinned_revision() {
        let files: &'static [ModelFile] = Box::leak(Box::new([
            pinned("model.onnx", b"abc"),
            pinned("tokenizer.json", b"{}"),
        ]));
        let model = PinnedModel {
            repo: "org/tiny",
            revision: "0123abcd",
            files,
        };
        assert_eq!(model.total_size(), 5);
        let dir = tempfile::tempdir().unwrap();
        let fetched = std::sync::Mutex::new(Vec::new());
        download_into(&model, dir.path(), |url, path| {
            fetched.lock().unwrap().push(url.to_string());
            let bytes: &[u8] = if url.ends_with("tokenizer.json") {
                b"{}"
            } else {
                b"abc"
            };
            fs::write(path, bytes).map_err(|e| e.to_string())
        })
        .unwrap();
        assert_eq!(
            *fetched.lock().unwrap(),
            [
                "https://huggingface.co/org/tiny/resolve/0123abcd/model.onnx",
                "https://huggingface.co/org/tiny/resolve/0123abcd/tokenizer.json",
            ]
        );
        // An install of the pinned bytes is hashed on the next run, never
        // fetched again.
        download_into(&model, dir.path(), never_fetch).unwrap();
    }
}
