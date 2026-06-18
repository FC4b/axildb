use std::path::PathBuf;

/// Pooling strategy used to reduce token-level hidden states to a single vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Use the [CLS] token's hidden state (position 0). Used by BGE models.
    Cls,
    /// Average all token hidden states, weighted by attention mask. Used by Nomic.
    Mean,
}

/// Supported embedding models.
#[derive(Debug, Clone)]
pub enum EmbeddingModel {
    /// bge-small-en-v1.5: 384 dimensions, ~33MB ONNX.
    BgeSmall,
    /// bge-small-en-v1.5 (int8 quantized): 384 dimensions, ~17MB ONNX (8b.7).
    /// Same quality (<1% MTEB loss), 4x smaller, 2-3x faster inference.
    BgeSmallInt8,
    /// bge-base-en-v1.5: 768 dimensions, ~110MB ONNX.
    BgeBase,
    /// nomic-embed-text-v1.5: 768 dimensions, ~500MB ONNX (Matryoshka).
    Nomic,
    /// BAAI/bge-m3: 1024 dimensions, ~2.2GB ONNX, 8192-token context.
    ///
    /// Multilingual, multi-functionality: supports dense retrieval,
    /// multi-vector retrieval, and sparse retrieval from one checkpoint.
    /// Strongest general-purpose embedder in the BGE family; use it when
    /// retrieval quality matters more than insert throughput and disk.
    /// XLM-RoBERTa backbone → uses mean pooling like Nomic.
    BgeM3,
    /// Alibaba-NLP/gte-modernbert-base: 768 dimensions, ~600MB ONNX, 8192-token context.
    ///
    /// ModernBERT backbone, Apache-2.0, MTEB 64.38 (vs bge-small 62.17).
    /// 16× longer context than bge-small without truncation. Mean pooling.
    /// Registered as an additional option in Phase 15 (P0.2) — bge-small
    /// remains the default; switch via `axil.toml` `model = "gte-modernbert-base"`.
    GteModernbertBase,
    /// User-provided ONNX model.
    Custom {
        path: PathBuf,
        dimensions: usize,
        pooling: PoolingStrategy,
        max_seq_len: usize,
    },
}

impl EmbeddingModel {
    /// Output vector dimensions for this model.
    pub fn dimensions(&self) -> usize {
        match self {
            Self::BgeSmall | Self::BgeSmallInt8 => 384,
            Self::BgeBase => 768,
            Self::Nomic => 768,
            Self::BgeM3 => 1024,
            Self::GteModernbertBase => 768,
            Self::Custom { dimensions, .. } => *dimensions,
        }
    }

    /// Human-readable model name.
    pub fn name(&self) -> &str {
        match self {
            Self::BgeSmall => "bge-small-en-v1.5",
            Self::BgeSmallInt8 => "bge-small-en-v1.5-int8",
            Self::BgeBase => "bge-base-en-v1.5",
            Self::Nomic => "nomic-embed-text-v1.5",
            Self::BgeM3 => "bge-m3",
            Self::GteModernbertBase => "gte-modernbert-base",
            Self::Custom { path, .. } => path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("custom"),
        }
    }

    /// Parse a model name string into an EmbeddingModel variant.
    /// Returns None for unrecognized names.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "bge-small" | "bge-small-en-v1.5" => Some(Self::BgeSmall),
            "bge-small-int8" | "bge-small-en-v1.5-int8" => Some(Self::BgeSmallInt8),
            "bge-base" | "bge-base-en-v1.5" => Some(Self::BgeBase),
            "nomic" | "nomic-embed-text-v1.5" => Some(Self::Nomic),
            "bge-m3" | "bgem3" => Some(Self::BgeM3),
            "gte-modernbert" | "gte-modernbert-base" | "modernbert" => {
                Some(Self::GteModernbertBase)
            }
            _ => None,
        }
    }

    /// Approximate download size for display purposes.
    pub fn approx_size(&self) -> &'static str {
        match self {
            Self::BgeSmall => "~33MB",
            Self::BgeSmallInt8 => "~17MB",
            Self::BgeBase => "~110MB",
            Self::Nomic => "~500MB",
            Self::BgeM3 => "~2.2GB",
            Self::GteModernbertBase => "~600MB",
            Self::Custom { .. } => "unknown",
        }
    }

    /// Maximum input sequence length (in tokens) for this model.
    pub fn max_seq_len(&self) -> usize {
        match self {
            Self::BgeSmall | Self::BgeSmallInt8 | Self::BgeBase => 512,
            Self::Nomic => 8192,
            Self::BgeM3 => 8192,
            Self::GteModernbertBase => 8192,
            Self::Custom { max_seq_len, .. } => *max_seq_len,
        }
    }

    /// Pooling strategy for this model.
    pub fn pooling(&self) -> PoolingStrategy {
        match self {
            Self::BgeSmall | Self::BgeSmallInt8 | Self::BgeBase => PoolingStrategy::Cls,
            // BGE-M3 is XLM-RoBERTa-based; Xenova/bge-m3 ONNX exports use the
            // [CLS] token consistent with the reference implementation of
            // FlagEmbedding's dense retrieval head.
            Self::BgeM3 => PoolingStrategy::Cls,
            Self::Nomic => PoolingStrategy::Mean,
            // gte-modernbert-base uses mean pooling per the model card and
            // upstream sentence-transformers config; CLS is unprojected and
            // does not produce a useful sentence vector for this model.
            Self::GteModernbertBase => PoolingStrategy::Mean,
            Self::Custom { pooling, .. } => *pooling,
        }
    }

    /// Whether this model supports Matryoshka Representation Learning (8b.6).
    pub fn supports_mrl(&self) -> bool {
        matches!(self, Self::Nomic)
    }

    /// Resolve the model directory path.
    ///
    /// For built-in models: `~/.axil/models/<model_name>/`
    /// For custom models: the provided path's parent directory.
    pub fn model_dir(&self) -> Option<PathBuf> {
        match self {
            Self::Custom { path, .. } => path.parent().map(|p| p.to_path_buf()),
            Self::BgeSmallInt8 => {
                dirs_or_home().map(|base| base.join("models").join("bge-small-en-v1.5-int8"))
            }
            _ => dirs_or_home().map(|base| base.join("models").join(self.name())),
        }
    }
}

/// Resolve `~/.axil/` base directory.
fn dirs_or_home() -> Option<PathBuf> {
    home_dir().map(|h| h.join(".axil"))
}

/// Platform home directory (Unix: $HOME, Windows: %USERPROFILE%).
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

impl std::fmt::Display for EmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for EmbeddingModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bge-small" | "bge-small-en-v1.5" => Ok(Self::BgeSmall),
            "bge-small-int8" | "bge-small-en-v1.5-int8" => Ok(Self::BgeSmallInt8),
            "bge-base" | "bge-base-en-v1.5" => Ok(Self::BgeBase),
            "nomic" | "nomic-embed-text-v1.5" => Ok(Self::Nomic),
            "bge-m3" | "bgem3" => Ok(Self::BgeM3),
            "gte-modernbert" | "gte-modernbert-base" | "modernbert" => Ok(Self::GteModernbertBase),
            other => {
                // Check the custom model registry.
                if let Some(model) = load_custom_model(other) {
                    return Ok(model);
                }
                Err(format!("unknown model: {other}"))
            }
        }
    }
}

// ── Custom model registry ─────────────────────────────────────────────

/// Path to the custom models registry file.
fn custom_models_path() -> Option<PathBuf> {
    dirs_or_home().map(|base| base.join("custom_models.json"))
}

/// A serializable custom model entry.
#[derive(serde::Serialize, serde::Deserialize)]
struct CustomModelEntry {
    path: String,
    dimensions: usize,
    pooling: String,
    max_seq_len: usize,
}

/// Save a custom model to the persistent registry at `~/.axil/custom_models.json`.
pub fn save_custom_model(
    name: &str,
    path: &std::path::Path,
    dimensions: usize,
    pooling: PoolingStrategy,
    max_seq_len: usize,
) -> Result<(), String> {
    let registry_path =
        custom_models_path().ok_or_else(|| "cannot resolve ~/.axil directory".to_string())?;

    let mut registry: std::collections::HashMap<String, CustomModelEntry> =
        if registry_path.exists() {
            let data = std::fs::read_to_string(&registry_path)
                .map_err(|e| format!("failed to read custom models registry: {e}"))?;
            serde_json::from_str(&data)
                .map_err(|e| format!("failed to parse custom models registry: {e}"))?
        } else {
            std::collections::HashMap::new()
        };

    registry.insert(
        name.to_string(),
        CustomModelEntry {
            path: path.display().to_string(),
            dimensions,
            pooling: match pooling {
                PoolingStrategy::Cls => "cls".to_string(),
                PoolingStrategy::Mean => "mean".to_string(),
            },
            max_seq_len,
        },
    );

    // Ensure ~/.axil/ exists.
    if let Some(parent) = registry_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("failed to create ~/.axil: {e}"))?;
    }

    let json = serde_json::to_string_pretty(&registry)
        .map_err(|e| format!("failed to serialize custom models: {e}"))?;
    std::fs::write(&registry_path, json)
        .map_err(|e| format!("failed to write custom models registry: {e}"))?;

    Ok(())
}

/// Load a custom model by name from the persistent registry.
fn load_custom_model(name: &str) -> Option<EmbeddingModel> {
    let registry_path = custom_models_path()?;
    if !registry_path.exists() {
        return None;
    }
    let data = std::fs::read_to_string(&registry_path).ok()?;
    let registry: std::collections::HashMap<String, CustomModelEntry> =
        serde_json::from_str(&data).ok()?;
    let entry = registry.get(name)?;
    let pooling = match entry.pooling.as_str() {
        "mean" => PoolingStrategy::Mean,
        _ => PoolingStrategy::Cls,
    };
    Some(EmbeddingModel::Custom {
        path: PathBuf::from(&entry.path),
        dimensions: entry.dimensions,
        pooling,
        max_seq_len: entry.max_seq_len,
    })
}

/// List all registered custom models.
pub fn list_custom_models() -> Vec<(String, EmbeddingModel)> {
    let Some(registry_path) = custom_models_path() else {
        return Vec::new();
    };
    if !registry_path.exists() {
        return Vec::new();
    }
    let Ok(data) = std::fs::read_to_string(&registry_path) else {
        return Vec::new();
    };
    let Ok(registry) =
        serde_json::from_str::<std::collections::HashMap<String, CustomModelEntry>>(&data)
    else {
        return Vec::new();
    };
    registry
        .into_iter()
        .filter_map(|(name, entry)| {
            let pooling = match entry.pooling.as_str() {
                "mean" => PoolingStrategy::Mean,
                _ => PoolingStrategy::Cls,
            };
            Some((
                name,
                EmbeddingModel::Custom {
                    path: PathBuf::from(entry.path),
                    dimensions: entry.dimensions,
                    pooling,
                    max_seq_len: entry.max_seq_len,
                },
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bge_m3_metadata() {
        let m = EmbeddingModel::BgeM3;
        assert_eq!(m.dimensions(), 1024);
        assert_eq!(m.name(), "bge-m3");
        assert_eq!(m.max_seq_len(), 8192);
        assert_eq!(m.pooling(), PoolingStrategy::Cls);
        assert!(!m.supports_mrl());
    }

    #[test]
    fn bge_m3_parses_from_name() {
        assert!(matches!(
            EmbeddingModel::from_name("bge-m3"),
            Some(EmbeddingModel::BgeM3)
        ));
        assert!(matches!(
            EmbeddingModel::from_name("BGEM3"),
            Some(EmbeddingModel::BgeM3)
        ));
    }

    #[test]
    fn from_name_unknown_returns_none() {
        assert!(EmbeddingModel::from_name("not-a-real-model").is_none());
    }

    #[test]
    fn gte_modernbert_metadata() {
        let m = EmbeddingModel::GteModernbertBase;
        assert_eq!(m.dimensions(), 768);
        assert_eq!(m.name(), "gte-modernbert-base");
        assert_eq!(m.max_seq_len(), 8192);
        assert_eq!(m.pooling(), PoolingStrategy::Mean);
        assert!(!m.supports_mrl());
    }

    #[test]
    fn gte_modernbert_parses_from_name() {
        assert!(matches!(
            EmbeddingModel::from_name("gte-modernbert-base"),
            Some(EmbeddingModel::GteModernbertBase)
        ));
        assert!(matches!(
            EmbeddingModel::from_name("MODERNBERT"),
            Some(EmbeddingModel::GteModernbertBase)
        ));
    }
}
