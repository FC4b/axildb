//! Which embedding model built a vector store, and whether the model a
//! database is configured with may search it and add to it.
//!
//! Two models of the same width make vectors that cannot be compared with
//! each other, yet a store sized for one opens under the other without a
//! dimension error: bge-small-en-v1.5 and granite-embedding-small-english-r2
//! both write 384 dimensions, and three registered models write 768. So the
//! default vector store records the model whose vectors it holds, under
//! [`MODEL_KEY`] in its metadata table beside `dimensions`, and every open
//! that attaches a model to it checks that record first
//! ([`check_store_model`]).
//!
//! A store that holds another model's vectors is refused: the database opens
//! without its vector engine, so recall runs on its other signals and nothing
//! new is embedded, and `axil reembed` rebuilds the store under the
//! configured model.
//!
//! Stores written before model records name no model. Such a store is taken
//! to hold the configured model's vectors ([`ModelCheck::Legacy`]) when it has
//! that model's width and the model is one of [`PRE_IDENTITY_MODELS`], the
//! built-in models a store could have been built with then: the configured
//! model is the best evidence there is, and refusing every such store would
//! leave every database configured for a non-default model without vector
//! search until rebuilt. Taking over a store for any model but the default
//! prints a notice, once per thread, in case another model built it. Granite
//! is not on the list: it came after model records, so no store
//! from before them holds its vectors, and a store taken for its own would be
//! exactly the silent switch this check exists to catch. A store that holds
//! no vectors is taken over by any model of its width ([`ModelCheck::Empty`]):
//! there is nothing to mix.
//!
//! Every open checks the model it is given against the database's
//! `axil.toml` too (`check_store_model_at`). When `axil.toml` names a model,
//! a store from before model records is taken to hold that model's vectors,
//! so a library caller that opens it with any other model is refused it: the
//! CLI's hooks embed with the configured model, and taking the store for the
//! caller's would write its vectors beside theirs. A custom model `axil.toml`
//! names is refused such a store as well: until model records, the CLI and
//! the MCP server did not load a custom model named there, and embedded with
//! bge-small in its place, so the store may hold either model's vectors.
//! When `axil.toml` names no model, the model passed in is the best evidence
//! there is, and a custom one takes over such a store of its width as a
//! built-in does.
//!
//! Only the configured model takes over such a store. A command that names
//! another model (`--model`) gets a store from before model records refused
//! ([`check_store_model_for`]): the configured model is the evidence for who
//! built it, so a store taken for the named model would carry the configured
//! model's vectors into one that records another.
//!
//! bge-small-en-v1.5-int8 has an identity of its own. It is a dynamically
//! quantized export of the same checkpoint, so its vectors approximate
//! bge-small's, but how closely has not been measured; keeping the two apart
//! costs one re-embed when switching between them, where treating them as one
//! could mix vectors that rank differently without anything saying so.
//!
//! Named vector spaces (`<db>.vec.<name>`) hold vectors the caller supplies,
//! made by no model Axil knows of, and record none.

use std::cell::RefCell;
use std::collections::HashSet;
use std::path::Path;

use crate::models::EmbeddingModel;

/// Metadata-table key naming the model whose vectors a store holds: its
/// [`EmbeddingModel::store_identity`], as UTF-8.
pub(crate) const MODEL_KEY: &str = "model";

/// How [`EmbeddingModel::store_identity`] begins for a custom model.
const CUSTOM_PREFIX: &str = "custom:";

/// The built-in models a vector store could be built with before stores
/// recorded their model, by [`EmbeddingModel::store_identity`]. A store from
/// then is taken to hold the configured model's vectors only when that model
/// is on this list. A model registered later stays off it, so a store from
/// before can never pass for one of its own.
pub const PRE_IDENTITY_MODELS: &[&str] = &[
    "bge-small-en-v1.5",
    "bge-small-en-v1.5-int8",
    "bge-base-en-v1.5",
    "nomic-embed-text-v1.5",
    "bge-m3",
    "gte-modernbert-base",
];

/// What a vector store's metadata says about the vectors it holds.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct StoredModel {
    /// Vector dimensions.
    pub dimensions: usize,
    /// The model the store records ([`EmbeddingModel::store_identity`]), or
    /// `None` for a store written before model records.
    pub model: Option<String>,
    /// Vectors the store holds, loadable or not.
    pub vectors: usize,
}

/// Whether a configured model may use a store; see [`check_store_model`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelCheck {
    /// The store records the configured model.
    Recorded,
    /// The store holds no vectors and has the configured model's width, so
    /// the model takes it over. Its next vector write records the model.
    Empty,
    /// The store predates model records and the configured model is taken to
    /// have built it (see the [module docs](self)). Its next vector write
    /// records the model, so an open that only reads writes nothing.
    Legacy,
    /// The store holds vectors of another model, or of one it does not name.
    Refused(ModelMismatch),
}

/// Whether `model` is one a store from before model records may hold the
/// vectors of: a built-in on [`PRE_IDENTITY_MODELS`].
fn predates_model_records(model: &EmbeddingModel) -> bool {
    !matches!(model, EmbeddingModel::Custom { .. })
        && PRE_IDENTITY_MODELS.contains(&model.store_identity().as_str())
}

/// Decide whether `configured`, the model `axil.toml` configures for the
/// database, may search the store `stored` describes and add vectors to it.
///
/// A custom model never takes over a store from before model records here;
/// one a caller passes to an open may, when `axil.toml` names no model
/// (`check_store_model_at`).
pub fn check_store_model(stored: &StoredModel, configured: &EmbeddingModel) -> ModelCheck {
    let identity = configured.store_identity();
    let same_width = stored.dimensions == configured.dimensions();
    match &stored.model {
        Some(model) if *model == identity && same_width => return ModelCheck::Recorded,
        // Nothing to mix: an empty store of the right width serves any model.
        _ if same_width && stored.vectors == 0 => return ModelCheck::Empty,
        None if same_width && predates_model_records(configured) => return ModelCheck::Legacy,
        _ => {}
    }
    ModelCheck::Refused(ModelMismatch {
        stored: stored.model.clone(),
        stored_dimensions: stored.dimensions,
        vectors: stored.vectors,
        model: identity,
        model_dimensions: configured.dimensions(),
        configured: None,
    })
}

/// Decide whether `model`, passed to an open of the default store of the
/// database at `db_path`, may use the store `stored` describes: every open
/// goes through this one (see the [module docs](self)).
///
/// When the database's `axil.toml` names a model, read as the CLI and the MCP
/// server read it, this is [`check_store_model_for`] with that model as the
/// configured one: a store from before model records is taken to hold its
/// vectors, so another model passed in, built-in or custom, is refused the
/// store, and a custom model `axil.toml` names is refused it too. When
/// `axil.toml` names none, it is [`check_store_model`] with the model passed
/// in, except that a custom model takes over such a store of its width.
pub(crate) fn check_store_model_at(
    db_path: &Path,
    stored: &StoredModel,
    model: &EmbeddingModel,
) -> ModelCheck {
    if let Some(configured) = configured_model(db_path) {
        return check_store_model_for(stored, model, &configured);
    }
    let check = check_store_model(stored, model);
    let adopt = matches!(check, ModelCheck::Refused(_))
        && matches!(model, EmbeddingModel::Custom { .. })
        && stored.model.is_none()
        && stored.dimensions == model.dimensions();
    if adopt {
        ModelCheck::Legacy
    } else {
        check
    }
}

/// The model the `axil.toml` of the database at `db_path` names, resolved as
/// the CLI and the MCP server resolve it (a name neither knows stands for
/// bge-small, which both fall back to); `None` when it names none.
fn configured_model(db_path: &Path) -> Option<EmbeddingModel> {
    let dir = db_path.parent().unwrap_or(Path::new("."));
    let name = axil_core::load_config_from(dir)
        .ok()?
        .database
        .embedding_model?;
    Some(EmbeddingModel::from_config_name(&name).unwrap_or(EmbeddingModel::BgeSmall))
}

/// Decide whether `model`, named by a command (`--model`), may add vectors to
/// the store `stored` describes, or have its vectors carried into a store of
/// `model`'s, in a database configured with `configured`.
///
/// For the configured model this is [`check_store_model`]. Any other model
/// may use a store that records it, or an empty one of its width, and
/// nothing else: a store from before model records is taken to hold the
/// configured model's vectors, never the named one's.
pub fn check_store_model_for(
    stored: &StoredModel,
    model: &EmbeddingModel,
    configured: &EmbeddingModel,
) -> ModelCheck {
    let identity = model.store_identity();
    let configured = configured.store_identity();
    if identity == configured {
        return check_store_model(stored, model);
    }
    let same_width = stored.dimensions == model.dimensions();
    match &stored.model {
        Some(recorded) if *recorded == identity && same_width => ModelCheck::Recorded,
        _ if same_width && stored.vectors == 0 => ModelCheck::Empty,
        _ => ModelCheck::Refused(ModelMismatch {
            stored: stored.model.clone(),
            stored_dimensions: stored.dimensions,
            vectors: stored.vectors,
            model: identity,
            model_dimensions: model.dimensions(),
            configured: Some(configured),
        }),
    }
}

/// A store that holds the vectors of another model than the one it was
/// checked for, or of one it does not name.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ModelMismatch {
    /// The model the store records, or `None` when it names none.
    pub stored: Option<String>,
    /// The store's vector dimensions.
    pub stored_dimensions: usize,
    /// Vectors the store holds.
    pub vectors: usize,
    /// The model the store was checked for
    /// ([`EmbeddingModel::store_identity`]): the configured model, unless
    /// `configured` says otherwise.
    pub model: String,
    /// That model's vector dimensions.
    pub model_dimensions: usize,
    /// The configured model, when the store was checked for another one a
    /// command named or a caller passed in ([`check_store_model_for`]);
    /// `None` when `model` is the configured model.
    pub configured: Option<String>,
}

impl ModelMismatch {
    /// What the store holds, naming its model when it records one, and why a
    /// store that records none is not taken for the checked model's.
    fn holdings(&self) -> String {
        if self.vectors == 0 {
            return format!(
                "no vectors, sized for {}-dimension ones",
                self.stored_dimensions
            );
        }
        let vectors = format!(
            "{} {}-dimension vector{}",
            self.vectors,
            self.stored_dimensions,
            if self.vectors == 1 { "" } else { "s" }
        );
        let Some(model) = &self.stored else {
            let why = if let Some(configured) = &self.configured {
                format!(", taken to be the configured model's ({configured})")
            } else if self.model_dimensions != self.stored_dimensions {
                String::new()
            } else if self.model.starts_with(CUSTOM_PREFIX) {
                " (a store that names no model was written before stores recorded their \
                 model, when a custom model named in axil.toml was not loaded, so bge-small \
                 may have built it)"
                    .to_string()
            } else {
                " (a store that names no model was written before stores recorded their \
                 model, and the configured model came after that, so it did not build it)"
                    .to_string()
            };
            return format!("{vectors} from a model it does not name{why}");
        };
        format!("{vectors} from {model}")
    }

    /// Why the store is not used, naming both models.
    pub fn reason(&self, db_path: &Path) -> String {
        let vec_path = crate::vector_db_path(db_path);
        let holdings = self.holdings();
        let same_width = self.model_dimensions == self.stored_dimensions;
        if self.configured.is_some() {
            let model = if same_width {
                format!("{}'s vectors cannot be compared with them", self.model)
            } else {
                format!(
                    "{} makes {}-dimension vectors",
                    self.model, self.model_dimensions
                )
            };
            return format!("{} holds {holdings}, and {model}", vec_path.display());
        }
        let configured = if same_width {
            format!(
                "the configured embedding model is {}, whose vectors cannot be compared \
                 with them",
                self.model
            )
        } else {
            format!(
                "the configured embedding model {} makes {}-dimension vectors",
                self.model, self.model_dimensions
            )
        };
        format!(
            "{} holds {holdings}, but {configured}, so recall runs without vectors and \
             nothing is embedded until the store is rebuilt",
            vec_path.display(),
        )
    }

    /// The command that rebuilds the store under the configured model, and,
    /// when the store names its model, the way back to it. For a model a
    /// command named or a caller passed in, the two ways forward: use the
    /// configured one, or configure it and rebuild.
    pub fn fix(&self, db_path: &Path) -> String {
        if let Some(configured) = &self.configured {
            return format!(
                "use the configured model, {configured} (drop --model, or open the \
                 database with it), or set `[database] embedding_model = \"{}\"` in \
                 axil.toml and run `axil --db {} reembed` to re-embed every record with it",
                self.model,
                db_path.display()
            );
        }
        let rebuild = format!(
            "run `axil --db {} reembed` to re-embed every record with {}",
            db_path.display(),
            self.model
        );
        match self.stored.as_deref() {
            Some(model) => match model.strip_prefix(CUSTOM_PREFIX) {
                Some(path) => format!("{rebuild}, or configure the custom model at {path} again"),
                None => format!(
                    "{rebuild}, or set `[database] embedding_model = \"{model}\"` in axil.toml \
                     to use the store as it is"
                ),
            },
            None => rebuild,
        }
    }

    /// The [`axil_core::DegradedEngine`] an open that refuses the store
    /// records, for `doctor`, `boot` and recall's warning.
    pub fn degraded(&self, db_path: &Path) -> axil_core::DegradedEngine {
        axil_core::DegradedEngine {
            engine: "vector".to_string(),
            reason: self.reason(db_path),
            fix: self.fix(db_path),
        }
    }
}

/// Print `line` through the diagnostic sink unless this thread printed it
/// before: a command that opens its database more than once says it once.
///
/// Once per thread rather than per process, because a long-lived process
/// answers on threads of its own: the daemon runs each request on a new
/// thread, so every request that meets a refused store (a workspace member's
/// included) carries the line on its own stderr, not only the first.
fn say_once(line: String) {
    thread_local! {
        static SAID: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
    }
    if SAID.with(|said| said.borrow_mut().insert(line.clone())) {
        axil_core::diagln!("{line}");
    }
}

/// Say on stderr that the store at `db_path` was refused, once per thread
/// for each store and pair of models.
pub(crate) fn warn_refused(db_path: &Path, mismatch: &ModelMismatch) {
    let degraded = mismatch.degraded(db_path);
    say_once(format!(
        "[axil vector] warning: {}. Fix: {}",
        degraded.summary(),
        degraded.fix
    ));
}

/// The notice that the store at `db_path`, which records no model, was taken
/// to hold `model`'s vectors ([`ModelCheck::Legacy`]); `None` for the default
/// model, which every store from before model records was built with unless
/// `axil.toml` named another.
pub fn adoption_notice(db_path: &Path, model: &EmbeddingModel) -> Option<String> {
    if matches!(model, EmbeddingModel::BgeSmall) {
        return None;
    }
    let name = model.store_identity();
    let configured = configured_model(db_path).is_some_and(|c| c.store_identity() == name);
    if !configured {
        // Passed in by a caller with no axil.toml naming a model (see
        // `check_store_model_at`), so `axil reembed`, which rebuilds with the
        // configured model, bge-small, is not the way back.
        return Some(format!(
            "[axil vector] note: {} predates model records, so it is taken to hold {name} \
             vectors, the model it was opened with; the next vector written records that. \
             If another model built it, re-embed every record with {name} first",
            crate::vector_db_path(db_path).display(),
        ));
    }
    Some(format!(
        "[axil vector] note: {} predates model records, so it is taken to hold {name} \
         vectors, the configured model's; the next vector written records that. If \
         another model built it, run `axil --db {} reembed` to rebuild it with {name}",
        crate::vector_db_path(db_path).display(),
        db_path.display(),
    ))
}

/// Say [`adoption_notice`] on stderr, once per thread for each store and
/// model.
pub(crate) fn note_adopted(db_path: &Path, model: &EmbeddingModel) {
    if let Some(notice) = adoption_notice(db_path, model) {
        say_once(notice);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn stored(dimensions: usize, model: Option<&str>, vectors: usize) -> StoredModel {
        StoredModel {
            dimensions,
            model: model.map(str::to_string),
            vectors,
        }
    }

    fn custom(dimensions: usize) -> EmbeddingModel {
        EmbeddingModel::Custom {
            path: PathBuf::from("/models/mine/model.onnx"),
            dimensions,
            pooling: crate::models::PoolingStrategy::Cls,
            max_seq_len: 512,
        }
    }

    const BGE: &str = "bge-small-en-v1.5";
    const GRANITE: &str = "granite-embedding-small-english-r2";

    #[test]
    fn a_store_recording_the_configured_model_is_used() {
        let check = check_store_model(
            &stored(384, Some(GRANITE), 10),
            &EmbeddingModel::GraniteSmallR2,
        );
        assert_eq!(check, ModelCheck::Recorded);
    }

    #[test]
    fn a_store_recording_another_model_of_the_same_width_is_refused() {
        let ModelCheck::Refused(m) =
            check_store_model(&stored(384, Some(BGE), 10), &EmbeddingModel::GraniteSmallR2)
        else {
            panic!("bge-small vectors must not serve granite");
        };
        assert_eq!(m.stored.as_deref(), Some(BGE));
        assert_eq!((m.model.as_str(), m.configured.as_deref()), (GRANITE, None));
        let db = Path::new("/p/memory.axil");
        let reason = m.reason(db);
        assert!(reason.contains(BGE) && reason.contains(GRANITE), "{reason}");
        let fix = m.fix(db);
        assert!(fix.contains("axil --db /p/memory.axil reembed"), "{fix}");
        assert!(
            fix.contains(&format!("embedding_model = \"{BGE}\"")),
            "{fix}"
        );

        // And the other way round.
        assert!(matches!(
            check_store_model(&stored(384, Some(GRANITE), 10), &EmbeddingModel::BgeSmall),
            ModelCheck::Refused(_)
        ));
    }

    #[test]
    fn bge_small_and_its_int8_export_are_different_models() {
        for (store, configured) in [
            (BGE, EmbeddingModel::BgeSmallInt8),
            ("bge-small-en-v1.5-int8", EmbeddingModel::BgeSmall),
        ] {
            assert!(
                matches!(
                    check_store_model(&stored(384, Some(store), 3), &configured),
                    ModelCheck::Refused(_)
                ),
                "{store} vectors must not serve {configured}"
            );
        }
    }

    #[test]
    fn a_store_of_another_width_is_refused_whatever_it_records() {
        for model in [Some(BGE), None] {
            let ModelCheck::Refused(m) =
                check_store_model(&stored(384, model, 4), &EmbeddingModel::BgeBase)
            else {
                panic!("a 384-dimension store must not serve bge-base");
            };
            assert_eq!((m.stored_dimensions, m.model_dimensions), (384, 768));
        }
        // Even empty: its width is wrong for every vector the model makes.
        assert!(matches!(
            check_store_model(&stored(384, None, 0), &EmbeddingModel::BgeBase),
            ModelCheck::Refused(_)
        ));
    }

    #[test]
    fn an_empty_store_of_the_right_width_serves_any_model() {
        for model in [None, Some(BGE)] {
            assert_eq!(
                check_store_model(&stored(384, model, 0), &EmbeddingModel::GraniteSmallR2),
                ModelCheck::Empty,
                "{model:?}"
            );
        }
        assert_eq!(
            check_store_model(&stored(384, None, 0), &custom(384)),
            ModelCheck::Empty
        );
    }

    #[test]
    fn a_legacy_store_is_taken_for_a_pre_identity_model_of_its_width() {
        for (dimensions, configured) in [
            (384, EmbeddingModel::BgeSmall),
            (384, EmbeddingModel::BgeSmallInt8),
            (768, EmbeddingModel::BgeBase),
            (768, EmbeddingModel::Nomic),
            (768, EmbeddingModel::GteModernbertBase),
            (1024, EmbeddingModel::BgeM3),
        ] {
            assert_eq!(
                check_store_model(&stored(dimensions, None, 500), &configured),
                ModelCheck::Legacy,
                "{configured}"
            );
        }
    }

    #[test]
    fn a_legacy_store_is_refused_to_granite_and_to_a_custom_model() {
        let db = Path::new("m.axil");
        let ModelCheck::Refused(m) =
            check_store_model(&stored(384, None, 500), &EmbeddingModel::GraniteSmallR2)
        else {
            panic!("granite came after model records: it built no unrecorded store");
        };
        assert_eq!(m.stored, None);
        let reason = m.reason(db);
        assert!(
            reason.contains("does not name") && reason.contains("came after that"),
            "{reason}"
        );
        assert!(!m.fix(db).contains("embedding_model"));

        let check = check_store_model(&stored(384, None, 500), &custom(384));
        let ModelCheck::Refused(m) = check else {
            panic!("a custom model must not take over an unrecorded store");
        };
        let reason = m.reason(db);
        assert!(reason.contains("custom model"), "{reason}");
        assert!(m.fix(db).contains("reembed"), "{}", m.fix(db));
    }

    #[test]
    fn pre_identity_models_are_every_built_in_but_granite() {
        let built_in = [
            EmbeddingModel::BgeSmall,
            EmbeddingModel::BgeSmallInt8,
            EmbeddingModel::BgeBase,
            EmbeddingModel::Nomic,
            EmbeddingModel::BgeM3,
            EmbeddingModel::GteModernbertBase,
            EmbeddingModel::GraniteSmallR2,
        ];
        for model in &built_in {
            let listed = PRE_IDENTITY_MODELS.contains(&model.store_identity().as_str());
            let granite = matches!(model, EmbeddingModel::GraniteSmallR2);
            assert_eq!(listed, !granite, "{model}");
        }
        assert_eq!(PRE_IDENTITY_MODELS.len(), built_in.len() - 1);
    }

    #[test]
    fn only_a_non_default_adoption_gives_notice() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("memory.axil");
        assert_eq!(adoption_notice(&db, &EmbeddingModel::BgeSmall), None);

        // Configured, so `axil reembed` rebuilds with it.
        configure(dir.path(), "bge-base");
        let notice = adoption_notice(&db, &EmbeddingModel::BgeBase).unwrap();
        assert!(notice.contains("bge-base-en-v1.5"), "{notice}");
        assert!(notice.contains("the configured model's"), "{notice}");
        assert!(
            notice.contains(&format!("axil --db {} reembed", db.display())),
            "{notice}"
        );
        assert!(
            !notice.contains("warning"),
            "a notice, not a warning: {notice}"
        );

        // Passed in with no model configured: `axil reembed` would rebuild
        // with bge-small, so it is not named.
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("memory.axil");
        let notice = adoption_notice(&db, &EmbeddingModel::BgeBase).unwrap();
        assert!(notice.contains("the model it was opened with"), "{notice}");
        assert!(!notice.contains("reembed"), "{notice}");
    }

    #[test]
    fn a_legacy_store_is_refused_to_a_model_other_than_the_configured_one() {
        let db = Path::new("/p/memory.axil");
        // A bge-base database from before model records: nomic is as wide and
        // as old, but a command naming it must not take the store for its own.
        for (dimensions, named, configured) in [
            (768, EmbeddingModel::Nomic, EmbeddingModel::BgeBase),
            (384, EmbeddingModel::BgeSmallInt8, EmbeddingModel::BgeSmall),
        ] {
            let ModelCheck::Refused(m) =
                check_store_model_for(&stored(dimensions, None, 500), &named, &configured)
            else {
                panic!("{named} must not take over a store taken for {configured}");
            };
            let (named, configured) = (named.store_identity(), configured.store_identity());
            assert_eq!(m.configured.as_deref(), Some(configured.as_str()));
            let reason = m.reason(db);
            assert!(
                reason.contains(&named) && reason.contains(&configured),
                "{reason}"
            );
            assert!(!reason.contains("recall runs without vectors"), "{reason}");
            let fix = m.fix(db);
            assert!(fix.contains("drop --model"), "{fix}");
            assert!(
                fix.contains(&format!("embedding_model = \"{named}\"")),
                "{fix}"
            );
        }
        // The configured model itself still takes it over.
        assert_eq!(
            check_store_model_for(
                &stored(768, None, 500),
                &EmbeddingModel::BgeBase,
                &EmbeddingModel::BgeBase
            ),
            ModelCheck::Legacy
        );
    }

    #[test]
    fn a_named_model_may_use_its_own_store_or_an_empty_one() {
        let nomic = EmbeddingModel::Nomic;
        let configured = EmbeddingModel::BgeBase;
        let own = stored(768, Some("nomic-embed-text-v1.5"), 9);
        assert_eq!(
            check_store_model_for(&own, &nomic, &configured),
            ModelCheck::Recorded
        );
        assert_eq!(
            check_store_model_for(&stored(768, None, 0), &nomic, &configured),
            ModelCheck::Empty
        );
        let theirs = stored(768, Some("bge-base-en-v1.5"), 9);
        assert!(matches!(
            check_store_model_for(&theirs, &nomic, &configured),
            ModelCheck::Refused(_)
        ));
    }

    #[test]
    fn an_empty_store_of_another_width_is_not_said_to_predate_anything() {
        let ModelCheck::Refused(m) =
            check_store_model(&stored(768, None, 0), &EmbeddingModel::BgeSmall)
        else {
            panic!("a 768-dimension store must not serve bge-small");
        };
        let reason = m.reason(Path::new("m.axil"));
        assert!(
            reason.contains("no vectors, sized for 768-dimension ones"),
            "{reason}"
        );
        assert!(!reason.contains("model records"), "{reason}");
    }

    #[test]
    fn a_custom_store_points_back_at_its_file() {
        let m = ModelMismatch {
            stored: Some("custom:/models/mine/model.onnx".to_string()),
            stored_dimensions: 384,
            vectors: 1,
            model: BGE.to_string(),
            model_dimensions: 384,
            configured: None,
        };
        let fix = m.fix(Path::new("m.axil"));
        assert!(
            fix.contains("custom model at /models/mine/model.onnx"),
            "{fix}"
        );
        assert!(!fix.contains("embedding_model"), "{fix}");
        // One vector, said so.
        let reason = m.reason(Path::new("m.axil"));
        assert!(reason.contains("1 384-dimension vector from"), "{reason}");
    }

    /// An axil.toml naming `model` beside the database in `dir`.
    fn configure(dir: &Path, model: &str) {
        std::fs::write(
            dir.join("axil.toml"),
            format!("[database]\nembedding_model = \"{model}\"\n"),
        )
        .unwrap();
    }

    /// With no model configured, a custom model passed to an open takes over
    /// a store from before model records of its width, as a built-in does.
    /// (One `axil.toml` names stays refused: the CLI test covers that, since
    /// naming one takes a model registry in `HOME`.)
    #[test]
    fn a_custom_model_passed_in_takes_over_a_store_from_before_model_records() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("memory.axil");
        assert_eq!(
            check_store_model_at(&db, &stored(384, None, 500), &custom(384)),
            ModelCheck::Legacy
        );
        // The rest is check_store_model's: another width, or a store that
        // records another model, is refused.
        assert!(matches!(
            check_store_model_at(&db, &stored(768, None, 500), &custom(384)),
            ModelCheck::Refused(_)
        ));
        assert!(matches!(
            check_store_model_at(&db, &stored(384, Some(BGE), 500), &custom(384)),
            ModelCheck::Refused(_)
        ));
        let notice = adoption_notice(&db, &custom(384)).unwrap();
        assert!(
            notice.contains("custom:/models/mine/model.onnx") && notice.contains("opened with"),
            "{notice}"
        );
        assert!(!notice.contains("reembed"), "{notice}");
    }

    /// When axil.toml names a model, a store from before model records holds
    /// its vectors, as the CLI's hooks embedded them: a custom model passed
    /// in, of the same width, must not take it over and write beside them.
    #[test]
    fn a_custom_model_passed_in_does_not_take_over_the_configured_models_store() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("memory.axil");
        configure(dir.path(), "bge-small");
        let ModelCheck::Refused(m) =
            check_store_model_at(&db, &stored(384, None, 500), &custom(384))
        else {
            panic!("the store is taken to be bge-small's");
        };
        assert_eq!(m.configured.as_deref(), Some(BGE));
        let reason = m.reason(&db);
        assert!(
            reason.contains("custom:/models/mine/model.onnx")
                && reason.contains(&format!("the configured model's ({BGE})")),
            "{reason}"
        );
        assert!(
            m.fix(&db).contains("use the configured model"),
            "{}",
            m.fix(&db)
        );
        // A store that records the custom model, or holds nothing yet, is
        // still the custom model's to use.
        let own = stored(384, Some("custom:/models/mine/model.onnx"), 500);
        assert_eq!(
            check_store_model_at(&db, &own, &custom(384)),
            ModelCheck::Recorded
        );
        assert_eq!(
            check_store_model_at(&db, &stored(384, None, 0), &custom(384)),
            ModelCheck::Empty
        );
    }

    /// A built-in passed in is checked against the model axil.toml names, not
    /// taken for the configured one: bge-small-int8 on a bge-small store, or
    /// nomic on a bge-base one, would mix two models' vectors.
    #[test]
    fn a_built_in_passed_in_is_checked_against_the_configured_model() {
        for (configured, dims, passed) in [
            ("bge-small", 384, EmbeddingModel::BgeSmallInt8),
            ("bge-base", 768, EmbeddingModel::Nomic),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let db = dir.path().join("memory.axil");
            // With no model configured, the model passed in takes it over.
            assert_eq!(
                check_store_model_at(&db, &stored(dims, None, 500), &passed),
                ModelCheck::Legacy,
                "{passed}"
            );
            configure(dir.path(), configured);
            assert!(
                matches!(
                    check_store_model_at(&db, &stored(dims, None, 500), &passed),
                    ModelCheck::Refused(_)
                ),
                "{passed} must not take over {configured}'s store"
            );
            // The configured model still does.
            let own = EmbeddingModel::from_name(configured).unwrap();
            assert_eq!(
                check_store_model_at(&db, &stored(dims, None, 500), &own),
                ModelCheck::Legacy
            );
        }

        // A name the CLI does not know stands for bge-small, its fallback.
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("memory.axil");
        configure(dir.path(), "no-such-model");
        assert!(matches!(
            check_store_model_at(&db, &stored(384, None, 500), &EmbeddingModel::BgeSmallInt8),
            ModelCheck::Refused(_)
        ));
        assert_eq!(
            check_store_model_at(&db, &stored(384, None, 500), &EmbeddingModel::BgeSmall),
            ModelCheck::Legacy
        );
    }
}
