use parking_lot::RwLock;
use std::collections::HashMap;
#[cfg(feature = "embed")]
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::models::EmbeddingModel;
#[cfg(feature = "embed")]
use crate::models::PoolingStrategy;

/// Environment variable: override ONNX session pool size.
///
/// Each pooled session owns an independent ONNX runtime state — including
/// its own GPU memory allocation — so N sessions = N concurrent embeds at
/// the cost of N × model-size VRAM.
///
/// Defaults to 1. **Only raise this on CPU workloads.** Empirically,
/// pool > 1 *hurts* on a single CUDA device: the GPU is already the
/// real bottleneck, and multiple sessions on the same device just add
/// memory pressure and context-switch overhead on top of an already-
/// serialized hardware queue. On CPU, where N worker threads can truly
/// run N forward passes in parallel, a pool matching the worker count
/// (e.g. `AXIL_EMBED_SESSIONS=8`) removes the single-mutex bottleneck.
#[cfg(feature = "embed")]
const EMBED_POOL_ENV: &str = "AXIL_EMBED_SESSIONS";

/// Text-to-vector embedder.
///
/// When the `embed` feature is enabled, loads an ONNX model + tokenizer and
/// runs local inference. Without the feature, all methods return an error
/// directing users to provide pre-computed vectors.
///
/// Holds a pool of ONNX sessions sized by the `AXIL_EMBED_SESSIONS` env var
/// (default 1). Each call to `embed`/`embed_batch` takes exclusive use of
/// one session via `acquire_session` — round-robin with try-lock fallback,
/// so N parallel workers can run N concurrent forward passes on a pool of
/// N sessions with no contention.
pub struct Embedder {
    model: EmbeddingModel,
    #[cfg(feature = "embed")]
    sessions: Vec<parking_lot::Mutex<ort::session::Session>>,
    #[cfg(feature = "embed")]
    next_session: AtomicUsize,
    #[cfg(feature = "embed")]
    tokenizer: tokenizers::Tokenizer,
    /// Whether the loaded ONNX model declares a `token_type_ids` input.
    /// BERT-family models (BGE) do; ModernBERT (gte-modernbert-base) does
    /// not — binding an undeclared input causes `Invalid input name`.
    #[cfg(feature = "embed")]
    accepts_token_type_ids: bool,
}

impl Embedder {
    /// Create a new embedder for the given model.
    ///
    /// With `embed` feature: loads the ONNX session(s) and tokenizer.
    /// Without: succeeds immediately (embed() will return a clear error at call time).
    ///
    /// The session pool size is controlled by the `AXIL_EMBED_SESSIONS` env
    /// var (default 1). Set to N > 1 to allow N concurrent embeds at the
    /// cost of N × model-size GPU memory.
    pub fn new(model: EmbeddingModel) -> Result<Self, String> {
        #[cfg(feature = "embed")]
        {
            let pool_size = env_pool_size();
            Self::new_with_pool(model, pool_size)
        }

        #[cfg(not(feature = "embed"))]
        {
            // Without the embed feature, no model files are needed.
            // embed() will return a clear error directing users to
            // rebuild with the feature or use pre-computed vectors.
            Ok(Self { model })
        }
    }

    /// Create a new embedder with an explicit session-pool size.
    ///
    /// Bypasses the `AXIL_EMBED_SESSIONS` env var — useful in tests and
    /// benchmarks where you want a fixed, reproducible pool size.
    #[cfg(feature = "embed")]
    pub fn new_with_pool(model: EmbeddingModel, pool_size: usize) -> Result<Self, String> {
        let pool_size = pool_size.max(1);

        let (model_file, tokenizer_file) = if let EmbeddingModel::Custom { ref path, .. } = model {
            let dir = path.parent().ok_or_else(|| {
                format!(
                    "cannot determine directory for custom model: {}",
                    path.display()
                )
            })?;
            (path.clone(), dir.join("tokenizer.json"))
        } else {
            let dir = model
                .model_dir()
                .ok_or_else(|| "cannot resolve model directory".to_string())?;
            (dir.join("model.onnx"), dir.join("tokenizer.json"))
        };

        if !model_file.exists() {
            return Err(format!(
                "model file not found: {}. Run `axil model-download {}`",
                model_file.display(),
                model.name()
            ));
        }

        if !tokenizer_file.exists() {
            return Err(format!(
                "tokenizer.json not found at {}",
                tokenizer_file.display()
            ));
        }

        // Build `pool_size` independent sessions. Each owns its own GPU
        // memory, so multiple workers can run forward passes concurrently
        // without contending on a shared mutex.
        //
        // The first session logs whether CUDA was available; subsequent
        // ones re-use the same decision path silently. If CUDA fails on
        // session 1 we fall back to CPU for the whole pool to avoid a
        // half-GPU half-CPU surprise at runtime.
        let mut sessions: Vec<parking_lot::Mutex<ort::session::Session>> =
            Vec::with_capacity(pool_size);
        let mut accepts_token_type_ids = false;
        for i in 0..pool_size {
            let session = build_session(&model_file, i == 0)?;
            if i == 0 {
                accepts_token_type_ids = session
                    .inputs
                    .iter()
                    .any(|input| input.name == "token_type_ids");
            }
            sessions.push(parking_lot::Mutex::new(session));
        }

        let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| format!("failed to load tokenizer: {e}"))?;

        // Enforce truncation to the model's max sequence length.
        let max_len = model.max_seq_len();
        let truncation = tokenizers::TruncationParams {
            max_length: max_len,
            ..Default::default()
        };
        tokenizer
            .with_truncation(Some(truncation))
            .map_err(|e| format!("failed to set truncation: {e}"))?;

        if pool_size > 1 {
            eprintln!("[axil-vector] session pool size = {pool_size}");
        }

        Ok(Self {
            model,
            sessions,
            next_session: AtomicUsize::new(0),
            tokenizer,
            accepts_token_type_ids,
        })
    }

    /// Return one of the pooled sessions with exclusive access. Tries each
    /// slot with `try_lock` once; if every slot is busy, blocks on the
    /// round-robin target so the pool degrades gracefully under contention
    /// rather than starving.
    #[cfg(feature = "embed")]
    fn acquire_session(&self) -> parking_lot::MutexGuard<'_, ort::session::Session> {
        if self.sessions.len() == 1 {
            return self.sessions[0].lock();
        }
        let start = self.next_session.fetch_add(1, Ordering::Relaxed) % self.sessions.len();
        for i in 0..self.sessions.len() {
            let idx = (start + i) % self.sessions.len();
            if let Some(guard) = self.sessions[idx].try_lock() {
                return guard;
            }
        }
        // All busy — block on the round-robin target.
        self.sessions[start].lock()
    }

    /// Output vector dimensions for the loaded model.
    pub fn dimensions(&self) -> usize {
        self.model.dimensions()
    }

    /// Model name.
    pub fn model_name(&self) -> &str {
        self.model.name()
    }

    /// Embed a single text string into a vector.
    ///
    /// Pipeline: tokenize (with truncation) → ONNX inference → pool → L2 normalize.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        #[cfg(feature = "embed")]
        {
            self.embed_impl(text)
        }

        #[cfg(not(feature = "embed"))]
        {
            let _ = text;
            Err(format!(
                "local embedding requires the `embed` feature. \
                 Rebuild with `cargo build --features axil-vector/embed`, \
                 or provide pre-computed vectors via add_vector(). Model: {}",
                self.model.name()
            ))
        }
    }

    #[cfg(feature = "embed")]
    fn embed_impl(&self, text: &str) -> Result<Vec<f32>, String> {
        use ort::value::Tensor;

        // Tokenize (truncated to model's max_seq_len).
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("tokenization failed: {e}"))?;

        let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let token_type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();

        let seq_len = token_ids.len();

        let input_ids = Tensor::<i64>::from_array(([1, seq_len], token_ids))
            .map_err(|e| format!("input_ids tensor error: {e}"))?;
        let attention = Tensor::<i64>::from_array(([1, seq_len], attention_mask.clone()))
            .map_err(|e| format!("attention_mask tensor error: {e}"))?;

        // Run ONNX inference. Copy output data while lock is held,
        // then release before pooling (pure CPU work). ModernBERT-family
        // models reject `token_type_ids` (no segment embeddings); only
        // bind it when the model declared the input.
        let data: Vec<f32> = {
            let mut session = self.acquire_session();
            let outputs = if self.accepts_token_type_ids {
                let type_ids = Tensor::<i64>::from_array(([1, seq_len], token_type_ids))
                    .map_err(|e| format!("token_type_ids tensor error: {e}"))?;
                session.run(ort::inputs![
                    "input_ids" => input_ids,
                    "attention_mask" => attention,
                    "token_type_ids" => type_ids,
                ])
            } else {
                session.run(ort::inputs![
                    "input_ids" => input_ids,
                    "attention_mask" => attention,
                ])
            }
            .map_err(|e| format!("ONNX inference failed: {e}"))?;

            let output = outputs.iter().next().ok_or("model produced no outputs")?.1;
            let (_shape, tensor_data): (_, &[f32]) = output
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("failed to extract output tensor: {e}"))?;
            tensor_data.to_vec()
        }; // session lock released here

        let dims = self.model.dimensions();
        let total = data.len();
        if total < dims {
            return Err(format!(
                "output tensor too small: {total} elements, expected at least {dims}"
            ));
        }

        // Pool hidden states into a single vector based on model's strategy.
        let mut embedding = match self.model.pooling() {
            PoolingStrategy::Cls => {
                // First token's hidden state: data[0..dims].
                data[..dims].to_vec()
            }
            PoolingStrategy::Mean => {
                // Average all token hidden states, weighted by attention mask.
                // data layout: [token_0[0..dims], token_1[0..dims], ...]
                let mut avg = vec![0.0f32; dims];
                let mut count = 0.0f32;
                for (t, &mask) in attention_mask.iter().enumerate() {
                    if mask == 1 {
                        let offset = t * dims;
                        if offset + dims <= total {
                            for (i, v) in avg.iter_mut().enumerate() {
                                *v += data[offset + i];
                            }
                            count += 1.0;
                        }
                    }
                }
                if count > 0.0 {
                    for v in &mut avg {
                        *v /= count;
                    }
                }
                avg
            }
        };

        // L2 normalize in place.
        l2_normalize(&mut embedding);

        Ok(embedding)
    }

    /// Batch embed multiple texts in a single ONNX inference call.
    ///
    /// Pads all sequences to the longest in the batch and runs one forward pass.
    /// 5-10x faster than sequential embedding for bulk operations.
    #[cfg(feature = "embed")]
    pub fn embed_batch_impl(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        if texts.len() == 1 {
            return self.embed_impl(texts[0]).map(|v| vec![v]);
        }

        use ort::value::Tensor;

        // Batch tokenize all texts
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| format!("batch tokenization failed: {e}"))?;

        let batch_size = encodings.len();
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);
        let dims = self.model.dimensions();

        // Pad to max_len and flatten into batch tensors
        let mut all_ids = vec![0i64; batch_size * max_len];
        let mut all_mask = vec![0i64; batch_size * max_len];
        let mut all_types = vec![0i64; batch_size * max_len];

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let types = enc.get_type_ids();
            let seq_len = ids.len();
            let offset = i * max_len;
            for j in 0..seq_len {
                all_ids[offset + j] = ids[j] as i64;
                all_mask[offset + j] = mask[j] as i64;
                all_types[offset + j] = types[j] as i64;
            }
        }

        let input_ids = Tensor::<i64>::from_array(([batch_size, max_len], all_ids))
            .map_err(|e| format!("batch input_ids tensor error: {e}"))?;
        let attention = Tensor::<i64>::from_array(([batch_size, max_len], all_mask.clone()))
            .map_err(|e| format!("batch attention_mask tensor error: {e}"))?;

        // Single ONNX inference call for entire batch. ModernBERT-family
        // models reject `token_type_ids`; only bind it when the model
        // declared the input.
        let data: Vec<f32> = {
            let mut session = self.acquire_session();
            let outputs = if self.accepts_token_type_ids {
                let type_ids = Tensor::<i64>::from_array(([batch_size, max_len], all_types))
                    .map_err(|e| format!("batch token_type_ids tensor error: {e}"))?;
                session.run(ort::inputs![
                    "input_ids" => input_ids,
                    "attention_mask" => attention,
                    "token_type_ids" => type_ids,
                ])
            } else {
                session.run(ort::inputs![
                    "input_ids" => input_ids,
                    "attention_mask" => attention,
                ])
            }
            .map_err(|e| format!("batch ONNX inference failed: {e}"))?;

            let output = outputs.iter().next().ok_or("model produced no outputs")?.1;
            let (_shape, tensor_data): (_, &[f32]) = output
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("failed to extract batch output tensor: {e}"))?;
            tensor_data.to_vec()
        }; // session lock released here

        // Pool each sample in the batch (pure CPU, no lock needed)
        let stride = max_len * dims; // elements per sample
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let sample_offset = i * stride;
            let mask_offset = i * max_len;

            if sample_offset + stride > data.len() {
                return Err(format!(
                    "batch output too small: sample {i} needs offset {} but only {} elements",
                    sample_offset + stride,
                    data.len()
                ));
            }

            let sample_data = &data[sample_offset..sample_offset + stride];
            let sample_mask = &all_mask[mask_offset..mask_offset + max_len];

            let mut embedding = match self.model.pooling() {
                PoolingStrategy::Cls => sample_data[..dims].to_vec(),
                PoolingStrategy::Mean => {
                    let mut avg = vec![0.0f32; dims];
                    let mut count = 0.0f32;
                    for (t, &mask) in sample_mask.iter().enumerate() {
                        if mask == 1 {
                            let off = t * dims;
                            if off + dims <= stride {
                                for (j, v) in avg.iter_mut().enumerate() {
                                    *v += sample_data[off + j];
                                }
                                count += 1.0;
                            }
                        }
                    }
                    if count > 0.0 {
                        for v in &mut avg {
                            *v /= count;
                        }
                    }
                    avg
                }
            };

            l2_normalize(&mut embedding);
            results.push(embedding);
        }

        Ok(results)
    }
}

/// L2-normalize a vector in place.
#[cfg(feature = "embed")]
fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Resolve the session-pool size from `AXIL_EMBED_SESSIONS` or fall back to 1.
///
/// Invalid values (zero, negative, non-numeric) silently fall back to the
/// default so a typo in the env doesn't break the embedder on startup.
#[cfg(feature = "embed")]
fn env_pool_size() -> usize {
    std::env::var(EMBED_POOL_ENV)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(1)
}

/// Print the execution-provider banner once per process, and only
/// when `AXIL_VERBOSE=1`. `recall-across` opens N sibling DBs in a
/// single invocation, each spinning up its own embedder — the banner
/// is noisy to agent output and useless on the 2nd+ open.
#[cfg(feature = "embed")]
fn log_execution_provider_once(msg: &str) {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    axil_core::util::log_once_if_verbose(&LOGGED, &format!("[axil-vector] {msg}"));
}

/// Build one ONNX session for `model_file`. Tries each compile-time-
/// enabled GPU execution provider in priority order (CUDA → DirectML),
/// falling back to CPU on the first one that registers. `log_ep` is the
/// banner-flush switch — only the first call in a multi-session pool
/// emits the "using X EP" line so we don't spam the agent's stderr.
#[cfg(feature = "embed")]
fn build_session(
    model_file: &std::path::Path,
    log_ep: bool,
) -> Result<ort::session::Session, String> {
    fn fresh_builder() -> Result<ort::session::builder::SessionBuilder, String> {
        ort::session::Session::builder()
            .map_err(|e| format!("ONNX session builder error: {e}"))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| format!("optimization level error: {e}"))?
            .with_intra_threads(std::thread::available_parallelism().map_or(4, |n| n.get().min(8)))
            .map_err(|e| format!("thread config error: {e}"))
    }

    // Priority list of GPU EP attempts, populated from compile-time features.
    // Each entry is (label, closure that consumes a fresh builder and tries
    // to attach the EP). Returning Ok means the EP attached; Err means try
    // the next entry. Pure CPU is the unconditional final fallback.
    #[allow(unused_mut)]
    let mut attempts: Vec<(
        &'static str,
        Box<
            dyn Fn(
                ort::session::builder::SessionBuilder,
            ) -> Result<ort::session::builder::SessionBuilder, ()>,
        >,
    )> = Vec::new();

    #[cfg(feature = "cuda")]
    attempts.push((
        "CUDA",
        Box::new(|b| {
            // `error_on_failure` makes EP registration fail loudly when the CUDA
            // runtime libs can't load (e.g. cuDNN 9.x missing — ORT 1.22 links
            // `cudnn64_9.dll`). Without it, ORT silently falls back to CPU and
            // the "CUDA execution provider enabled" banner below would lie.
            // On failure we return Err and the loop drops through to the next
            // attempt (DirectML, then CPU).
            let ep = ort::execution_providers::cuda::CUDAExecutionProvider::default()
                .build()
                .error_on_failure();
            b.with_execution_providers([ep]).map_err(|_| ())
        }),
    ));

    #[cfg(feature = "directml")]
    attempts.push((
        "DirectML",
        Box::new(|b| {
            let ep = ort::execution_providers::directml::DirectMLExecutionProvider::default()
                .build()
                .error_on_failure();
            b.with_execution_providers([ep]).map_err(|_| ())
        }),
    ));

    for (label, attach) in &attempts {
        let Ok(builder) = fresh_builder() else {
            continue;
        };
        if let Ok(gpu_builder) = attach(builder) {
            if log_ep {
                eprintln!("[axil-vector] {label} execution provider enabled");
            }
            return gpu_builder
                .commit_from_file(model_file)
                .map_err(|e| format!("failed to load model {}: {e}", model_file.display()));
        }
    }

    if log_ep {
        if attempts.is_empty() {
            eprintln!("[axil-vector] CPU execution provider (no GPU feature compiled in)");
        } else {
            eprintln!("[axil-vector] CPU execution provider (GPU EPs attempted but did not register — check driver/runtime libs)");
        }
    }
    fresh_builder()?
        .commit_from_file(model_file)
        .map_err(|e| format!("failed to load model {}: {e}", model_file.display()))
}

// ── MultiEmbedder ─────────────────────────────────────────────────────

/// Interior state for [`MultiEmbedder`], protected by a single lock.
struct MultiEmbedderInner {
    models: HashMap<String, Embedder>,
    active: String,
}

/// A multi-model embedder that holds multiple ONNX models simultaneously,
/// keyed by name. Supports hot-swapping the active model without restarting.
///
/// Uses a single `RwLock` to avoid TOCTOU races between the active model
/// name and the model map.
pub struct MultiEmbedder {
    inner: RwLock<MultiEmbedderInner>,
}

impl MultiEmbedder {
    /// Create a new MultiEmbedder with an initial model.
    pub fn new(model: EmbeddingModel) -> Result<Self, String> {
        let name = model.name().to_string();
        let embedder = Embedder::new(model)?;
        let mut models = HashMap::new();
        models.insert(name.clone(), embedder);
        Ok(Self {
            inner: RwLock::new(MultiEmbedderInner {
                models,
                active: name,
            }),
        })
    }

    /// Load an additional model by type. Does nothing if already loaded.
    pub fn load_model(&self, model: EmbeddingModel) -> Result<(), String> {
        let name = model.name().to_string();
        let mut inner = self.inner.write();
        if inner.models.contains_key(&name) {
            return Ok(());
        }
        let embedder = Embedder::new(model)?;
        inner.models.insert(name, embedder);
        Ok(())
    }

    /// Load a custom model from a user-provided path.
    pub fn load_custom(&self, name: &str, model: EmbeddingModel) -> Result<(), String> {
        let embedder = Embedder::new(model)?;
        self.inner.write().models.insert(name.to_string(), embedder);
        Ok(())
    }

    /// Switch the active model to a different loaded model.
    /// Returns an error if the model is not loaded.
    pub fn switch_active(&self, name: &str) -> Result<(), String> {
        let mut inner = self.inner.write();
        if !inner.models.contains_key(name) {
            let available: Vec<_> = inner.models.keys().cloned().collect();
            return Err(format!(
                "model '{name}' not loaded. Available: {}",
                available.join(", ")
            ));
        }
        inner.active = name.to_string();
        Ok(())
    }

    /// Get the name of the currently active model.
    pub fn active_model(&self) -> String {
        self.inner.read().active.clone()
    }

    /// List all loaded model names with their dimensions.
    pub fn loaded_models(&self) -> Vec<(String, usize)> {
        self.inner
            .read()
            .models
            .iter()
            .map(|(name, e)| (name.clone(), e.dimensions()))
            .collect()
    }

    /// Embed text using the active model.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let inner = self.inner.read();
        let embedder = inner
            .models
            .get(&inner.active)
            .ok_or_else(|| format!("active model '{}' not found", inner.active))?;
        embedder.embed(text)
    }

    /// Embed text using a specific named model (not the active one).
    pub fn embed_with(&self, text: &str, model_name: &str) -> Result<Vec<f32>, String> {
        let inner = self.inner.read();
        let embedder = inner
            .models
            .get(model_name)
            .ok_or_else(|| format!("model '{model_name}' not loaded"))?;
        embedder.embed(text)
    }

    /// Unload a model from memory. Cannot unload the active model.
    pub fn unload(&self, name: &str) -> Result<(), String> {
        let mut inner = self.inner.write();
        if inner.active == name {
            return Err("cannot unload the active model — switch to another first".into());
        }
        if inner.models.remove(name).is_none() {
            return Err(format!("model '{name}' not loaded"));
        }
        Ok(())
    }

    /// Dimensions of the active model.
    pub fn dimensions(&self) -> usize {
        let inner = self.inner.read();
        inner
            .models
            .get(&inner.active)
            .map(|e| e.dimensions())
            .unwrap_or(0)
    }
}
