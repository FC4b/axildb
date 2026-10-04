//! The ONNX pair classifier: tokenize two texts together, run the
//! cross-encoder, calibrate the chosen label's logit.

use std::path::Path;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::Mutex;

use crate::calibration::Calibration;
use crate::model::PairModel;
use crate::{DecideError, Result};

/// Largest number of pairs a batch-invariant model runs in one call.
const MAX_BATCH: usize = 8;

/// The session's graph optimisation level (`Level3`), named as onnxruntime
/// and the calibration files name it.
pub(crate) const OPTIMIZATION_LEVEL: &str = "ORT_ENABLE_ALL";

/// Session settings for [`PairClassifier::load_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadOptions {
    /// ONNX intra-op threads. On the U1 model, 1, 2, 4 and 8 threads give
    /// identical logits (`session_settings_check` in
    /// `benchmarks/results/typed-decisions-u1-calibration-2026-10-04.json`).
    pub intra_threads: usize,
}

impl Default for LoadOptions {
    /// Up to 4 threads: the model is small, and the judge runs beside other work.
    fn default() -> Self {
        Self {
            intra_threads: std::thread::available_parallelism().map_or(4, |n| n.get().min(4)),
        }
    }
}

/// One scored pair.
#[derive(Debug, Clone, PartialEq)]
pub struct PairScore {
    /// Raw logits, one per model label.
    pub logits: Vec<f32>,
    /// Tokens the pair took after truncation, special tokens included.
    pub tokens: usize,
    /// Calibrated probability of the calibration's label.
    pub probability: f64,
    /// Whether `probability` reached the calibration's threshold.
    pub flagged: bool,
}

/// A loaded cross-encoder pair classifier with its calibration.
///
/// Scores match Python onnxruntime 1.22 (the version this build's `ort`
/// bundles) at the same graph optimisation level, which is what the bundled
/// calibrations were fitted on. The session is CPU-only at `Level3`: on the
/// U1 model, the basic level moves logits by up to 0.95 and flips 4 of 249
/// verdicts (`session_settings_check` in
/// `benchmarks/results/typed-decisions-u1-calibration-2026-10-04.json`).
pub struct PairClassifier {
    session: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    /// Whether the graph declares `token_type_ids`. DeBERTa-v3 exports do
    /// not, and ONNX rejects an input the graph does not declare.
    accepts_token_type_ids: bool,
    labels: usize,
    label_index: usize,
    /// Pairs per inference call: 1 unless both the model and the
    /// calibration allow batching (see [`batch_limit`]).
    max_batch: usize,
    calibration: Calibration,
    name: String,
}

impl PairClassifier {
    /// Load `model` from `dir` with default [`LoadOptions`].
    pub fn load(model: &PairModel, dir: &Path, calibration: Calibration) -> Result<Self> {
        Self::load_with(model, dir, calibration, LoadOptions::default())
    }

    /// Load `model` from `dir`, which must hold its pinned files.
    /// [`PairModel::download`] puts them there, and repairs any that fail
    /// their hash, so calling it first is the usual recipe.
    ///
    /// Fails if `calibration` has unusable values or does not fit this model
    /// and the onnxruntime actually loaded ([`Calibration::check_fit`]), and
    /// if any file in `dir` is missing or does not match its pinned size and
    /// sha256. Hashing the files is part of every load.
    pub fn load_with(
        model: &PairModel,
        dir: &Path,
        calibration: Calibration,
        options: LoadOptions,
    ) -> Result<Self> {
        let runtime = loaded_runtime_version()?;
        Self::load_on_runtime(model, dir, calibration, options, &runtime)
    }

    /// [`PairClassifier::load_with`] given the loaded onnxruntime's version.
    fn load_on_runtime(
        model: &PairModel,
        dir: &Path,
        calibration: Calibration,
        options: LoadOptions,
        runtime: &str,
    ) -> Result<Self> {
        calibration.validate()?;
        let label_index = calibration.check_fit(model, runtime, OPTIMIZATION_LEVEL)?;
        model.verify(dir).map_err(|e| {
            DecideError::Model(format!(
                "{e} (PairModel::download fetches missing files and repairs bad ones)"
            ))
        })?;

        let session = Session::builder()
            .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
            .and_then(|b| b.with_intra_threads(options.intra_threads.max(1)))
            .and_then(|b| b.with_inter_threads(1))
            .and_then(|b| b.commit_from_file(dir.join(model.onnx.local)))
            .map_err(|e| DecideError::Model(format!("load {}: {e}", model.name)))?;
        let accepts_token_type_ids = session.inputs.iter().any(|i| i.name == "token_type_ids");

        let mut tokenizer = tokenizers::Tokenizer::from_file(dir.join(model.tokenizer.local))
            .map_err(|e| DecideError::Model(format!("tokenizer: {e}")))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: model.max_len,
                ..Default::default()
            }))
            .map_err(|e| DecideError::Model(format!("tokenizer truncation: {e}")))?;
        tokenizer.with_padding(None);

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            accepts_token_type_ids,
            labels: model.labels.len(),
            label_index,
            max_batch: batch_limit(model.batch_invariant, &calibration),
            calibration,
            name: format!("{}@{}", model.name, model.revision),
        })
    }

    /// The calibration verdicts are cut with.
    pub fn calibration(&self) -> &Calibration {
        &self.calibration
    }

    /// Score `(premise, hypothesis)` pairs; one result per pair, in input order.
    ///
    /// Each pair runs alone, as calibration scores are normally produced.
    /// Only a batch-invariant model with a calibration fitted on batched
    /// scores runs up to 8 pairs per call, grouped by length so little is
    /// padding.
    pub fn score_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<PairScore>> {
        let encodings = pairs
            .iter()
            .map(|&(premise, hypothesis)| {
                self.tokenizer
                    .encode((premise, hypothesis), true)
                    .map_err(|e| DecideError::Inference(format!("tokenize: {e}")))
            })
            .collect::<Result<Vec<_>>>()?;

        let lengths: Vec<usize> = encodings.iter().map(|e| e.get_ids().len()).collect();
        let mut logits: Vec<Vec<f32>> = vec![Vec::new(); pairs.len()];
        for group in plan_batches(&lengths, self.max_batch) {
            let rows: Vec<&tokenizers::Encoding> = group.iter().map(|&i| &encodings[i]).collect();
            for (&i, row) in group.iter().zip(self.run(&rows)?) {
                logits[i] = row;
            }
        }

        Ok(logits
            .into_iter()
            .zip(lengths)
            .map(|(logits, tokens)| {
                let probability = self.calibration.probability(&logits, self.label_index);
                PairScore {
                    flagged: self.calibration.flags(probability),
                    logits,
                    tokens,
                    probability,
                }
            })
            .collect())
    }

    /// One ONNX call over `rows`, padded to the longest; one logit row each.
    fn run(&self, rows: &[&tokenizers::Encoding]) -> Result<Vec<Vec<f32>>> {
        let (width, ids, mask, types) = pad_rows(rows);
        let shape = [rows.len(), width];
        let tensor = |data: Vec<i64>| {
            Tensor::<i64>::from_array((shape, data))
                .map_err(|e| DecideError::Inference(format!("input tensor: {e}")))
        };
        let (ids, mask) = (tensor(ids)?, tensor(mask)?);

        let mut session = self.session.lock();
        let outputs = if self.accepts_token_type_ids {
            session.run(ort::inputs![
                "input_ids" => ids,
                "attention_mask" => mask,
                "token_type_ids" => tensor(types)?,
            ])
        } else {
            session.run(ort::inputs![
                "input_ids" => ids,
                "attention_mask" => mask,
            ])
        }
        .map_err(|e| DecideError::Inference(format!("run: {e}")))?;

        let (_, output) = outputs
            .iter()
            .next()
            .ok_or_else(|| DecideError::Inference("model produced no outputs".into()))?;
        let (out_shape, data) = output
            .try_extract_tensor::<f32>()
            .map_err(|e| DecideError::Inference(format!("logits: {e}")))?;
        if out_shape.len() != 2
            || out_shape[0] as usize != rows.len()
            || out_shape[1] as usize != self.labels
        {
            return Err(DecideError::Inference(format!(
                "expected logits of shape [{}, {}], got {out_shape:?}",
                rows.len(),
                self.labels
            )));
        }
        Ok(data.chunks(self.labels).map(<[f32]>::to_vec).collect())
    }
}

impl axil_core::PairJudge for PairClassifier {
    fn judge(&self, pairs: &[(&str, &str)]) -> axil_core::Result<Vec<axil_core::PairVerdict>> {
        let scores = self
            .score_pairs(pairs)
            .map_err(|e| axil_core::AxilError::plugin(e.to_string()))?;
        Ok(scores
            .into_iter()
            .map(|s| axil_core::PairVerdict {
                probability: s.probability,
                flagged: s.flagged,
            })
            .collect())
    }

    fn threshold(&self) -> f64 {
        self.calibration.threshold
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Pairs per inference call. Batching changes scores, slightly even for a
/// batch-invariant model, so pairs are batched only when the model is
/// batch-invariant and the calibration was fitted on batched scores; a
/// calibration fitted one pair per call gets one pair per call.
fn batch_limit(batch_invariant: bool, calibration: &Calibration) -> usize {
    if batch_invariant && !calibration.one_pair_per_call {
        MAX_BATCH
    } else {
        1
    }
}

/// The loaded onnxruntime's version, from its build info (official builds
/// name their release branch, `git-branch=rel-1.22.0`). Read through the
/// loaded library, so a shared runtime other than the one this build was
/// compiled against reports its own version.
pub(crate) fn loaded_runtime_version() -> Result<String> {
    let info = std::panic::catch_unwind(ort::info)
        .map_err(|_| DecideError::Model("onnxruntime failed to initialise".into()))?;
    version_from_build_info(info)
        .map(str::to_string)
        .ok_or_else(|| {
            DecideError::Calibration(format!(
                "cannot tell the onnxruntime version from its build info ({info}), so no \
             calibration can be checked against it"
            ))
        })
}

/// The release version in an onnxruntime build-info string, if it names one.
fn version_from_build_info(info: &str) -> Option<&str> {
    let start = info.find("git-branch=rel-")? + "git-branch=rel-".len();
    let rest = &info[start..];
    let end = rest
        .find(|c: char| c == ',' || c.is_whitespace())
        .unwrap_or(rest.len());
    Some(&rest[..end]).filter(|v| !v.is_empty())
}

/// Split pairs, given their token lengths, into calls of at most `max_batch`.
/// Pairs are taken shortest first so each call pads little; every index
/// appears in exactly one call. With `max_batch` 1 each pair runs alone, in
/// input order.
fn plan_batches(lengths: &[usize], max_batch: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..lengths.len()).collect();
    if max_batch > 1 {
        order.sort_by_key(|&i| lengths[i]);
    }
    order
        .chunks(max_batch.max(1))
        .map(<[usize]>::to_vec)
        .collect()
}

/// Flatten encodings into row-major `[rows, width]` input buffers, padding
/// each row to the longest with id 0 and attention 0. Returns
/// `(width, input_ids, attention_mask, token_type_ids)`.
fn pad_rows(rows: &[&tokenizers::Encoding]) -> (usize, Vec<i64>, Vec<i64>, Vec<i64>) {
    let width = rows.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
    let mut ids = vec![0i64; rows.len() * width];
    let mut mask = vec![0i64; rows.len() * width];
    let mut types = vec![0i64; rows.len() * width];
    for (r, enc) in rows.iter().enumerate() {
        let base = r * width;
        for (j, ((&id, &m), &t)) in enc
            .get_ids()
            .iter()
            .zip(enc.get_attention_mask())
            .zip(enc.get_type_ids())
            .enumerate()
        {
            ids[base + j] = id as i64;
            mask[base + j] = m as i64;
            types[base + j] = t as i64;
        }
    }
    (width, ids, mask, types)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoding(ids: &[u32], types: &[u32]) -> tokenizers::Encoding {
        tokenizers::Encoding::new(
            ids.to_vec(),
            types.to_vec(),
            vec![String::new(); ids.len()],
            vec![None; ids.len()],
            vec![(0, 0); ids.len()],
            vec![0; ids.len()],
            vec![1; ids.len()],
            Vec::new(),
            Default::default(),
        )
    }

    #[test]
    fn the_u1_calibration_scores_one_pair_per_call() {
        let u1 = Calibration::u1();
        assert!(u1.one_pair_per_call);
        // Even on a model whose scores do not change in a batch.
        assert_eq!(batch_limit(true, &u1), 1);
        assert_eq!(batch_limit(false, &u1), 1);
        let batched = Calibration {
            one_pair_per_call: false,
            ..Calibration::u1()
        };
        assert_eq!(batch_limit(true, &batched), MAX_BATCH);
        assert_eq!(batch_limit(false, &batched), 1);
    }

    #[test]
    fn runtime_version_comes_from_the_build_info() {
        let real = "ORT Build Info: git-branch=rel-1.22.0, git-commit-id=f217402, \
                    build type=Release, cmake cxx flags: -DCPUINFO_SUPPORTED";
        assert_eq!(version_from_build_info(real), Some("1.22.0"));
        assert_eq!(
            version_from_build_info("git-branch=rel-1.30.1 build type=Release"),
            Some("1.30.1")
        );
        assert_eq!(
            version_from_build_info("ORT Build Info: git-branch=main, x"),
            None
        );
        assert_eq!(version_from_build_info("git-branch=rel-, x"), None);
        assert_eq!(version_from_build_info(""), None);
        // The runtime this crate links reports the version U1 was fitted on.
        assert!(loaded_runtime_version().unwrap().starts_with("1.22."));
    }

    #[test]
    fn load_refuses_a_runtime_the_calibration_was_not_measured_on() {
        let dir = tempfile::tempdir().unwrap();
        let model = crate::NLI_DEBERTA_V3_XSMALL_QINT8;
        for runtime in ["1.23.0", "1.30.0", "1.21.0"] {
            let err = PairClassifier::load_on_runtime(
                &model,
                dir.path(),
                Calibration::u1(),
                LoadOptions::default(),
                runtime,
            )
            .err()
            .expect("other onnxruntime");
            assert!(
                matches!(err, DecideError::Calibration(_)) && err.to_string().contains(runtime),
                "{runtime}: {err}"
            );
        }
    }

    #[test]
    fn one_pair_per_call_keeps_input_order() {
        assert_eq!(
            plan_batches(&[30, 5, 12], 1),
            vec![vec![0], vec![1], vec![2]]
        );
        assert!(plan_batches(&[], 1).is_empty());
    }

    #[test]
    fn batches_group_by_length_and_cover_every_pair() {
        let lengths = [40, 3, 17, 3, 90, 12, 8, 60, 5, 22];
        let groups = plan_batches(&lengths, 4);
        assert!(groups.iter().all(|g| !g.is_empty() && g.len() <= 4));
        let mut seen: Vec<usize> = groups.concat();
        seen.sort_unstable();
        assert_eq!(seen, (0..lengths.len()).collect::<Vec<_>>());
        // Shortest first: each call's longest is no longer than the next call's shortest.
        for pair in groups.windows(2) {
            let longest = pair[0].iter().map(|&i| lengths[i]).max().unwrap();
            let shortest = pair[1].iter().map(|&i| lengths[i]).min().unwrap();
            assert!(longest <= shortest);
        }
    }

    #[test]
    fn padding_fills_to_the_longest_row() {
        let a = encoding(&[1, 7, 2], &[0, 0, 1]);
        let b = encoding(&[1, 9, 9, 9, 2], &[0, 0, 1, 1, 1]);
        let (width, ids, mask, types) = pad_rows(&[&a, &b]);
        assert_eq!(width, 5);
        assert_eq!(ids, vec![1, 7, 2, 0, 0, 1, 9, 9, 9, 2]);
        assert_eq!(mask, vec![1, 1, 1, 0, 0, 1, 1, 1, 1, 1]);
        assert_eq!(types, vec![0, 0, 1, 0, 0, 0, 0, 1, 1, 1]);
    }

    #[test]
    fn single_row_is_not_padded() {
        let a = encoding(&[1, 7, 2], &[0, 0, 1]);
        let (width, ids, mask, _) = pad_rows(&[&a]);
        assert_eq!(width, 3);
        assert_eq!(ids, vec![1, 7, 2]);
        assert_eq!(mask, vec![1, 1, 1]);
    }

    #[test]
    fn load_refuses_a_calibration_for_another_model() {
        let dir = tempfile::tempdir().unwrap();
        let model = crate::NLI_DEBERTA_V3_XSMALL_QINT8;
        let mut other = Calibration::u1();
        other.revision = "main".into();
        let err = PairClassifier::load(&model, dir.path(), other)
            .err()
            .expect("revision mismatch");
        assert!(matches!(err, DecideError::Calibration(_)), "{err}");

        let mut unknown = Calibration::u1();
        unknown.label = "relevant".into();
        let err = PairClassifier::load(&model, dir.path(), unknown)
            .err()
            .expect("unknown label");
        assert!(matches!(err, DecideError::Calibration(_)), "{err}");

        // Same name and revision, a different pinned graph: the calibration
        // was not fitted on its scores.
        let regraphed = PairModel {
            onnx: crate::ModelFile {
                sha256: "7105da41f625c42eca24e9465ec99150d02a80e644659d7a1daa93a6357155d4",
                ..model.onnx
            },
            ..model
        };
        let err = PairClassifier::load(&regraphed, dir.path(), Calibration::u1())
            .err()
            .expect("graph mismatch");
        assert!(matches!(err, DecideError::Calibration(_)), "{err}");
    }

    #[test]
    fn load_refuses_unusable_calibration_values() {
        let dir = tempfile::tempdir().unwrap();
        let model = crate::NLI_DEBERTA_V3_XSMALL_QINT8;
        let bad = [
            (0.0, 0.5),
            (-1.0, 0.5),
            (f64::NAN, 0.5),
            (f64::INFINITY, 0.5),
            (1.0, f64::NAN),
            (1.0, 0.0),
            (1.0, 1.0),
            (1.0, 1.5),
            (1.0, -0.1),
        ];
        for (temperature, threshold) in bad {
            let calibration = Calibration {
                temperature,
                threshold,
                ..Calibration::u1()
            };
            let err = PairClassifier::load(&model, dir.path(), calibration)
                .err()
                .expect("unusable calibration");
            assert!(
                matches!(err, DecideError::Calibration(_)),
                "T = {temperature}, threshold = {threshold}: {err}"
            );
        }
    }

    #[test]
    fn load_refuses_files_that_fail_their_hash() {
        // Pinned sizes, wrong bytes: the size check passes, the hash does not.
        let dir = tempfile::tempdir().unwrap();
        let model = crate::NLI_DEBERTA_V3_XSMALL_QINT8;
        for file in model.files() {
            let f = std::fs::File::create(dir.path().join(file.local)).unwrap();
            f.set_len(file.size).unwrap();
        }
        assert!(model.is_available(dir.path()));
        let err = PairClassifier::load(&model, dir.path(), Calibration::u1())
            .err()
            .expect("hash mismatch");
        assert!(
            err.to_string().contains("does not match the pinned"),
            "{err}"
        );
    }

    #[test]
    fn load_reports_missing_files() {
        let dir = tempfile::tempdir().unwrap();
        let err = PairClassifier::load(
            &crate::NLI_DEBERTA_V3_XSMALL_QINT8,
            dir.path(),
            Calibration::u1(),
        )
        .err()
        .expect("empty dir");
        assert!(matches!(err, DecideError::Model(_)), "{err}");
    }
}
