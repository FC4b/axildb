//! Pinned model files: where they come from, how they are checked, where they live.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
/// [`PairModel::download`] fetches the files from that revision and installs
/// them only if their size and sha256 match. [`PairModel::is_available`] is
/// the cheap check (presence and size); [`PairModel::verify`] re-hashes, and
/// every [`crate::PairClassifier`] load runs it.
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
    /// batch change every pair's result. On the U1 model, padded batches of 8
    /// move logits by up to 1.4 and flip 4 of 249 verdicts (`batch_check` in
    /// `benchmarks/results/typed-decisions-u1-calibration-2026-10-04.json`).
    /// Such models score one pair per run.
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

    /// Whether every file is in `dir` with its pinned size. A cheap check
    /// that misses a corrupt file of the right size; [`PairModel::download`]
    /// verifies and repairs, and a load verifies.
    pub fn is_available(&self, dir: &Path) -> bool {
        self.files().iter().all(|f| {
            fs::metadata(dir.join(f.local))
                .map(|m| m.len() == f.size)
                .unwrap_or(false)
        })
    }

    /// Make `dir` hold the pinned files: verify each one present (size and
    /// sha256), and fetch from the pinned revision any that is missing or
    /// fails. With every file intact it only hashes them, so it is the usual
    /// step before a load, and it repairs a corrupt cache.
    ///
    /// Safe to run from several processes at once: each download writes its
    /// own temporary file, which must match the pinned size and sha256 before
    /// it is renamed into place, and a file another process already installed
    /// and verified is kept.
    pub fn download(&self, dir: &Path) -> Result<()> {
        self.download_with(dir, fetch)
    }

    /// [`PairModel::download`] with `fetch(url, path)` writing each file.
    fn download_with<F>(&self, dir: &Path, fetch: F) -> Result<()>
    where
        F: Fn(&str, &Path) -> std::result::Result<(), String>,
    {
        fs::create_dir_all(dir)
            .map_err(|e| DecideError::Model(format!("create {}: {e}", dir.display())))?;
        for file in self.files() {
            let url = self.url(&file);
            install(dir, &file, |tmp| fetch(&url, tmp))?;
        }
        Ok(())
    }

    /// Check every file in `dir` against its pinned size and sha256.
    pub fn verify(&self, dir: &Path) -> Result<()> {
        self.files()
            .iter()
            .try_for_each(|file| check(&dir.join(file.local), file))
    }
}

/// Attempts per file; a transfer error is retried, a verification failure is not.
const ATTEMPTS: u32 = 3;

/// Install `file` into `dir`, with `fetch` writing its bytes to the path it
/// is given.
///
/// Each attempt writes to its own temporary file in `dir` (process id, time
/// and a counter in the name), so concurrent downloaders never share one.
/// That file's size and sha256 are checked on disk, after it is fully
/// written, and only then is it renamed over the final name, which is atomic
/// within one directory. A final file that already verifies is kept: it is
/// either from an earlier run or from a concurrent downloader that finished
/// first, and both copies are the pinned bytes.
fn install<F>(dir: &Path, file: &ModelFile, fetch: F) -> Result<()>
where
    F: Fn(&Path) -> std::result::Result<(), String>,
{
    let dest = dir.join(file.local);
    if verifies(&dest, file) {
        return Ok(());
    }
    let mut last = String::new();
    for attempt in 1..=ATTEMPTS {
        let tmp = temp_path(dir, file.local);
        if let Err(e) = fetch(&tmp) {
            let _ = fs::remove_file(&tmp);
            last = e;
            if attempt < ATTEMPTS {
                std::thread::sleep(Duration::from_secs(2u64.pow(attempt)));
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
            Ok(()) => Ok(()),
            Err(e) => {
                let _ = fs::remove_file(&tmp);
                if verifies(&dest, file) {
                    Ok(())
                } else {
                    Err(DecideError::Model(format!(
                        "install {}: {e}",
                        dest.display()
                    )))
                }
            }
        };
    }
    Err(DecideError::Model(format!(
        "{}: {last} (after {ATTEMPTS} attempts)",
        file.remote
    )))
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
fn check(path: &Path, file: &ModelFile) -> Result<()> {
    let size = fs::metadata(path)
        .map_err(|e| DecideError::Model(format!("{}: {e}", path.display())))?
        .len();
    if size != file.size {
        return Err(DecideError::Model(format!(
            "{}: {size} bytes, pinned {} for {}",
            path.display(),
            file.size,
            file.remote
        )));
    }
    let actual = sha256_file(path)?;
    if actual != file.sha256 {
        return Err(DecideError::Model(format!(
            "{}: sha256 {actual} does not match the pinned {} for {}",
            path.display(),
            file.sha256,
            file.remote
        )));
    }
    Ok(())
}

/// Stream `url` into `path`.
fn fetch(url: &str, path: &Path) -> std::result::Result<(), String> {
    let response = ureq::get(url).call().map_err(|e| e.to_string())?;
    let mut reader = response.into_body().into_reader();
    let mut out = fs::File::create(path).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = reader.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(|e| e.to_string())?;
    }
    out.sync_all().map_err(|e| e.to_string())
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

    /// A pinned file for `bytes`, as the tests' download payload.
    fn pinned(bytes: &[u8]) -> ModelFile {
        ModelFile {
            remote: "onnx/model.onnx",
            local: "model.onnx",
            sha256: Box::leak(hex(&Sha256::digest(bytes)).into_boxed_str()),
            size: bytes.len() as u64,
        }
    }

    /// Writes `bytes` to the path in small chunks, yielding between them, so
    /// concurrent writers interleave.
    fn slow_writer(bytes: &[u8]) -> impl Fn(&Path) -> std::result::Result<(), String> + '_ {
        move |path| {
            let mut out = fs::File::create(path).map_err(|e| e.to_string())?;
            for chunk in bytes.chunks(4096) {
                out.write_all(chunk).map_err(|e| e.to_string())?;
                std::thread::sleep(Duration::from_micros(200));
            }
            Ok(())
        }
    }

    fn leftovers(dir: &Path) -> Vec<String> {
        fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".part"))
            .collect()
    }

    #[test]
    fn concurrent_downloads_install_one_verified_file() {
        let payload: Vec<u8> = (0..256 * 1024u32).map(|i| (i * 31 % 251) as u8).collect();
        let file = pinned(&payload);
        let dir = tempfile::tempdir().unwrap();
        let start = std::sync::Barrier::new(4);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        start.wait();
                        install(dir.path(), &file, slow_writer(&payload))
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
    fn a_download_that_fails_verification_is_never_installed() {
        let file = pinned(b"pinned bytes");
        let dir = tempfile::tempdir().unwrap();
        // Right size, wrong bytes: only the hash catches it.
        let err = install(dir.path(), &file, slow_writer(b"forged bytes")).unwrap_err();
        assert!(
            err.to_string().contains("does not match the pinned"),
            "{err}"
        );
        let err = install(dir.path(), &file, slow_writer(b"short")).unwrap_err();
        assert!(err.to_string().contains("bytes, pinned"), "{err}");
        assert!(!dir.path().join("model.onnx").exists());
        assert!(leftovers(dir.path()).is_empty());
    }

    #[test]
    fn a_verified_file_is_kept_and_a_corrupt_one_replaced() {
        let file = pinned(b"pinned bytes");
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("model.onnx");
        fs::write(&dest, b"pinned bytes").unwrap();
        install(dir.path(), &file, |_| {
            panic!("a verified file must not be fetched")
        })
        .unwrap();

        fs::write(&dest, b"forged bytes").unwrap();
        install(dir.path(), &file, slow_writer(b"pinned bytes")).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"pinned bytes");
        assert!(leftovers(dir.path()).is_empty());
    }

    #[test]
    fn download_repairs_a_same_size_corrupt_cache() {
        let dir = tempfile::tempdir().unwrap();
        let serve = |url: &str, path: &Path| -> std::result::Result<(), String> {
            let bytes: &[u8] = if url.ends_with("model.onnx") {
                b"abc"
            } else {
                b""
            };
            fs::write(path, bytes).map_err(|e| e.to_string())
        };
        // Right sizes, wrong bytes: looks available, fails verification.
        fs::write(dir.path().join("model.onnx"), b"xyz").unwrap();
        fs::write(dir.path().join("tokenizer.json"), b"").unwrap();
        assert!(TINY.is_available(dir.path()));
        assert!(TINY.verify(dir.path()).is_err());

        TINY.download_with(dir.path(), serve).unwrap();
        TINY.verify(dir.path()).unwrap();
        // Intact files are only hashed, never fetched again.
        TINY.download_with(dir.path(), |_, _| {
            panic!("intact files must not be fetched")
        })
        .unwrap();
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
