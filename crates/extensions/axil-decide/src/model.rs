//! Pinned model files: where they come from, how they are checked, where they live.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::{DecideError, Result};

/// Hugging Face base URL for model files.
const HF_BASE: &str = "https://huggingface.co";

/// One file of a pinned model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelFile {
    /// Path inside the Hugging Face repo.
    pub remote: &'static str,
    /// File name in the local model directory.
    pub local: &'static str,
    /// Expected sha256 of the file, lowercase hex.
    pub sha256: &'static str,
    /// Expected size in bytes, checked on every load (hashing is not).
    pub size: u64,
}

/// A pair-classifier model pinned to one Hugging Face revision.
///
/// [`PairModel::download`] fetches the files from that revision and accepts
/// them only if their sha256 matches; [`PairModel::is_available`] (and
/// therefore every load) checks presence and size, and
/// [`PairModel::verify`] re-hashes on demand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairModel {
    /// Local name, also the model directory's name.
    pub name: &'static str,
    /// Hugging Face repo id.
    pub repo: &'static str,
    /// Git revision (commit sha) every file is fetched from.
    pub revision: &'static str,
    /// The ONNX graph.
    pub onnx: ModelFile,
    /// The `tokenizer.json` that matches it.
    pub tokenizer: ModelFile,
    /// Output labels, in logit order.
    pub labels: &'static [&'static str],
    /// Maximum sequence length; longer pairs are truncated longest-first.
    pub max_len: usize,
    /// Whether a pair's logits are the same alone and in a padded batch.
    ///
    /// False for dynamically quantized graphs: they quantize activations with
    /// one scale for the whole input tensor, so other pairs and padding in a
    /// batch change every pair's result (on the U1 model, by up to 0.27 in
    /// probability). Such models score one pair per run.
    pub batch_invariant: bool,
}

/// `cross-encoder/nli-deberta-v3-xsmall`, dynamic int8 (87 MB, Apache-2.0).
///
/// The `model_qint8_arm64.onnx` file is byte-identical to the repo's
/// `avx512` and `avx512_vnni` variants (same sha256): one file for every CPU.
pub const NLI_DEBERTA_V3_XSMALL_QINT8: PairModel = PairModel {
    name: "nli-deberta-v3-xsmall-qint8",
    repo: "cross-encoder/nli-deberta-v3-xsmall",
    revision: "a150876415327c80daeff35ca6f68f5ed8cf5c24",
    onnx: ModelFile {
        remote: "onnx/model_qint8_arm64.onnx",
        local: "model.onnx",
        sha256: "ef41fe8474f6070373f97fbe8362461d55108f387723280fcdb64f8015133db2",
        size: 87_377_068,
    },
    tokenizer: ModelFile {
        remote: "tokenizer.json",
        local: "tokenizer.json",
        sha256: "5124ef2ead1a10a717703bc436de7f353da76d6340e4587719b42b1693707964",
        size: 8_656_624,
    },
    labels: &["contradiction", "entailment", "neutral"],
    max_len: 512,
    batch_invariant: false,
};

impl PairModel {
    /// The model's files.
    pub fn files(&self) -> [ModelFile; 2] {
        [self.onnx, self.tokenizer]
    }

    /// `~/.axil/models/<name>/`, next to the embedding models.
    pub fn default_dir(&self) -> Option<PathBuf> {
        std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(|home| {
                PathBuf::from(home)
                    .join(".axil")
                    .join("models")
                    .join(self.name)
            })
    }

    /// Pinned download URL of one file.
    pub fn url(&self, file: &ModelFile) -> String {
        format!(
            "{HF_BASE}/{}/resolve/{}/{}",
            self.repo, self.revision, file.remote
        )
    }

    /// Index of `label` in the model's logits.
    pub fn label_index(&self, label: &str) -> Option<usize> {
        self.labels.iter().position(|l| *l == label)
    }

    /// Whether every file is in `dir` with its pinned size.
    pub fn is_available(&self, dir: &Path) -> bool {
        self.files().iter().all(|f| {
            fs::metadata(dir.join(f.local))
                .map(|m| m.len() == f.size)
                .unwrap_or(false)
        })
    }

    /// Fetch every missing file into `dir` from the pinned revision.
    ///
    /// Each file downloads to a temporary name, is hashed as it streams, and
    /// is renamed into place only if its sha256 matches the pin. A file
    /// already present with the pinned size is kept.
    pub fn download(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir)
            .map_err(|e| DecideError::Model(format!("create {}: {e}", dir.display())))?;
        for file in self.files() {
            let dest = dir.join(file.local);
            if fs::metadata(&dest)
                .map(|m| m.len() == file.size)
                .unwrap_or(false)
            {
                continue;
            }
            download_verified(&self.url(&file), &dest, file.sha256)?;
        }
        Ok(())
    }

    /// Re-hash every file in `dir` against its pin.
    pub fn verify(&self, dir: &Path) -> Result<()> {
        for file in self.files() {
            let path = dir.join(file.local);
            let actual = sha256_file(&path)?;
            if actual != file.sha256 {
                return Err(DecideError::Model(format!(
                    "{}: sha256 {actual} does not match the pinned {}",
                    path.display(),
                    file.sha256
                )));
            }
        }
        Ok(())
    }
}

/// Download `url` to `dest`, keeping it only if its sha256 is `expected`.
/// Retries transient failures twice; a hash mismatch is not retried.
fn download_verified(url: &str, dest: &Path, expected: &str) -> Result<()> {
    const ATTEMPTS: u32 = 3;
    let tmp = dest.with_extension("part");
    let mut last = String::new();
    for attempt in 1..=ATTEMPTS {
        match fetch(url, &tmp) {
            Ok(actual) if actual == expected => {
                return fs::rename(&tmp, dest).map_err(|e| {
                    let _ = fs::remove_file(&tmp);
                    DecideError::Model(format!("rename to {}: {e}", dest.display()))
                });
            }
            Ok(actual) => {
                let _ = fs::remove_file(&tmp);
                return Err(DecideError::Model(format!(
                    "{url}: sha256 {actual} does not match the pinned {expected}"
                )));
            }
            Err(e) => {
                let _ = fs::remove_file(&tmp);
                last = e;
                if attempt < ATTEMPTS {
                    std::thread::sleep(std::time::Duration::from_secs(2u64.pow(attempt)));
                }
            }
        }
    }
    Err(DecideError::Model(format!(
        "{url}: {last} (after {ATTEMPTS} attempts)"
    )))
}

/// Stream `url` into `path`; returns the sha256 of what was written.
fn fetch(url: &str, path: &Path) -> std::result::Result<String, String> {
    let response = ureq::get(url).call().map_err(|e| e.to_string())?;
    let mut reader = response.into_body().into_reader();
    let mut out = fs::File::create(path).map_err(|e| e.to_string())?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = reader.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        out.write_all(&buf[..n]).map_err(|e| e.to_string())?;
    }
    out.sync_all().map_err(|e| e.to_string())?;
    Ok(hex(&hasher.finalize()))
}

/// sha256 of a file, lowercase hex.
fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)
        .map_err(|e| DecideError::Model(format!("open {}: {e}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| DecideError::Model(format!("read {}: {e}", path.display())))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY: PairModel = PairModel {
        name: "tiny",
        repo: "org/tiny",
        revision: "0123abcd",
        onnx: ModelFile {
            remote: "onnx/model.onnx",
            local: "model.onnx",
            // sha256("abc")
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            size: 3,
        },
        tokenizer: ModelFile {
            remote: "tokenizer.json",
            local: "tokenizer.json",
            // sha256("")
            sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            size: 0,
        },
        labels: &["no", "yes"],
        max_len: 16,
        batch_invariant: true,
    };

    #[test]
    fn urls_pin_the_revision() {
        assert_eq!(
            TINY.url(&TINY.onnx),
            "https://huggingface.co/org/tiny/resolve/0123abcd/onnx/model.onnx"
        );
        let u1 = NLI_DEBERTA_V3_XSMALL_QINT8;
        assert!(u1
            .url(&u1.onnx)
            .contains(&format!("/resolve/{}/", u1.revision)));
        assert!(!u1.url(&u1.tokenizer).contains("/resolve/main/"));
    }

    #[test]
    fn availability_checks_presence_and_size() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!TINY.is_available(dir.path()));
        fs::write(dir.path().join("model.onnx"), b"abc").unwrap();
        assert!(!TINY.is_available(dir.path()), "tokenizer still missing");
        fs::write(dir.path().join("tokenizer.json"), b"").unwrap();
        assert!(TINY.is_available(dir.path()));
        fs::write(dir.path().join("model.onnx"), b"abcd").unwrap();
        assert!(!TINY.is_available(dir.path()), "wrong size");
    }

    #[test]
    fn verify_rehashes_against_the_pin() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model.onnx"), b"abc").unwrap();
        fs::write(dir.path().join("tokenizer.json"), b"").unwrap();
        TINY.verify(dir.path()).unwrap();
        // Same size, different bytes: only the hash catches it.
        fs::write(dir.path().join("model.onnx"), b"abd").unwrap();
        assert!(TINY.is_available(dir.path()));
        let err = TINY.verify(dir.path()).unwrap_err().to_string();
        assert!(err.contains("does not match the pinned"), "{err}");
    }

    #[test]
    fn label_lookup() {
        let u1 = NLI_DEBERTA_V3_XSMALL_QINT8;
        assert_eq!(u1.label_index("contradiction"), Some(0));
        assert_eq!(u1.label_index("neutral"), Some(2));
        assert_eq!(u1.label_index("relevant"), None);
    }

    #[test]
    fn default_dir_sits_under_axil_models() {
        if let Some(dir) = NLI_DEBERTA_V3_XSMALL_QINT8.default_dir() {
            assert!(dir.ends_with(".axil/models/nli-deberta-v3-xsmall-qint8"));
        }
    }
}
