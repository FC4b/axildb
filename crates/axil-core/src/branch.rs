//! Memory branching — create, list, delete, and diff database branches.
//!
//! Branches are file-level copies of the database and all companion files.
//! No merge support (yet) — branches are for experimentation and rollback.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{AxilError, Result};

/// Difference between a branch and its parent database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchDiff {
    /// Branch name.
    pub branch_name: String,
    /// Per-table record counts in the main database.
    pub main_tables: BTreeMap<String, usize>,
    /// Per-table record counts in the branch.
    pub branch_tables: BTreeMap<String, usize>,
    /// Tables that exist only in the branch.
    pub new_tables: Vec<String>,
    /// Tables that exist only in main.
    pub deleted_tables: Vec<String>,
    /// Per-table count difference (branch - main). Positive = branch has more.
    pub count_diff: BTreeMap<String, i64>,
}

/// Known companion file suffixes (same as db.rs COMPANION_SUFFIXES).
const COMPANION_SUFFIXES: &[&str] = &[".vec", ".graph", ".ts"];

/// Enumerate named vector-space companion files (`<db>.vec.<name>`) next to a
/// database path. A branch that skips these loses every named-space index.
fn named_space_files(db_path: &Path) -> Vec<(PathBuf, String)> {
    let parent = db_path.parent().unwrap_or(Path::new("."));
    let db_name = db_path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let prefix = format!("{db_name}.vec.");
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir(parent) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy().to_string();
            if let Some(space) = name_str.strip_prefix(&prefix) {
                // Space names are validated (alnum, hyphen, underscore), so a
                // remainder containing a dot is not a space file.
                if !space.is_empty()
                    && !space.contains('.')
                    && entry.path().is_file()
                {
                    out.push((entry.path(), space.to_string()));
                }
            }
        }
    }
    out.sort_by(|a, b| a.1.cmp(&b.1));
    out
}

/// Serializes branch creation across processes for the whole create.
///
/// The lock is an OS file lock on `<db>.branch-lock`, held from before the
/// first copy until after the branch is published. The OS releases it when
/// its holder exits, so it never expires under a creator that is still
/// copying, however long that takes, and a second creator only ever finds
/// the leftovers of one that died.
///
/// Where the filesystem refuses OS locks, the lock is the file's existence
/// instead, judged stale after five minutes without change: the scheme older
/// axil binaries use. A holder from one of those writes only its pid, so a
/// fresh file not written by an OS-lock holder is honored even when the OS
/// lock is free.
///
/// redb's own file lock cannot be held across the copy — on Windows its
/// byte-range lock makes the file uncopyable while held (verified: OS error 33),
/// and on Unix the copy succeeds but captures a header redb marks "recovery
/// required" while the file is open, so the branch would need a full repair
/// on first open — so the handles must be dropped first. This lock serializes
/// branch creators, but not an unrelated writer that opens the database
/// mid-copy (see the quiesce notes on [`copy_branch_files`]).
struct BranchCopyGuard {
    lock_path: PathBuf,
    /// The OS-locked handle, or `None` when the lock is held by the file's
    /// existence alone.
    file: Option<fs::File>,
    /// What this holder wrote into the file, so that releasing a lock held
    /// by existence never removes a file another holder wrote since.
    body: String,
}

/// How long a creator waits for another to finish before giving up.
const BRANCH_LOCK_WAIT: std::time::Duration = std::time::Duration::from_secs(5);
/// Age after which a lock held by existence alone counts as abandoned.
const BRANCH_LOCK_STALE_SECS: u64 = 300;
/// Content prefix of a lock file written by a holder of the OS lock.
const BRANCH_LOCK_TAG: &str = "flock";

impl BranchCopyGuard {
    fn acquire(db_path: &Path, wait: std::time::Duration) -> std::io::Result<Self> {
        Self::acquire_with(db_path, wait, fs::File::try_lock)
    }

    /// `acquire`, with the OS lock call passed in so tests can stand in for a
    /// filesystem that refuses it.
    fn acquire_with(
        db_path: &Path,
        wait: std::time::Duration,
        try_lock: impl Fn(&fs::File) -> std::result::Result<(), fs::TryLockError>,
    ) -> std::io::Result<Self> {
        let lock_path = crate::companion_path(db_path, ".branch-lock");
        let deadline = std::time::Instant::now() + wait;
        loop {
            if let Some(guard) = Self::try_acquire(&lock_path, &try_lock)? {
                return Ok(guard);
            }
            if std::time::Instant::now() >= deadline {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WouldBlock,
                    "another branch creation is in progress",
                ));
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }

    /// One attempt; `Ok(None)` while someone else holds the lock.
    fn try_acquire(
        lock_path: &Path,
        try_lock: &impl Fn(&fs::File) -> std::result::Result<(), fs::TryLockError>,
    ) -> std::io::Result<Option<Self>> {
        let (file, created) = match open_lock_file(lock_path, true) {
            Ok(file) => (file, true),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                match open_lock_file(lock_path, false) {
                    Ok(file) => (file, false),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                    Err(e) => return Err(e),
                }
            }
            Err(e) => return Err(e),
        };
        match try_lock(&file) {
            Ok(()) => {}
            Err(fs::TryLockError::WouldBlock) => return Ok(None),
            Err(fs::TryLockError::Error(_)) => {
                drop(file);
                if created {
                    let body = staging_token();
                    let _ = fs::write(lock_path, &body);
                    return Ok(Some(Self {
                        lock_path: lock_path.to_path_buf(),
                        file: None,
                        body,
                    }));
                }
                if !lock_age_fresh(fs::metadata(lock_path)) {
                    let _ = fs::remove_file(lock_path);
                }
                return Ok(None);
            }
        }
        // A holder unlinks the path before it unlocks, so a lock taken on the
        // file it just unlinked proves nothing: try again on the file the
        // path names now.
        if !same_file(&file, lock_path) {
            return Ok(None);
        }
        // An older axil holds the lock by the file's existence and writes
        // only its pid. An empty file is free: one just created, or one whose
        // holder died before writing.
        let body = read_lock_body(&file);
        if !created
            && !body.is_empty()
            && !body.starts_with(BRANCH_LOCK_TAG)
            && lock_age_fresh(file.metadata())
        {
            return Ok(None);
        }
        let body = format!("{BRANCH_LOCK_TAG} {}", std::process::id());
        {
            use std::io::{Seek as _, Write as _};
            let mut handle = &file;
            let _ = handle.set_len(0);
            let _ = handle.seek(std::io::SeekFrom::Start(0));
            let _ = handle.write_all(body.as_bytes());
        }
        Ok(Some(Self {
            lock_path: lock_path.to_path_buf(),
            file: Some(file),
            body,
        }))
    }

    /// Whether the OS lock is held: no other live creator can run, so what
    /// another creator left behind is safe to remove.
    fn exclusive(&self) -> bool {
        self.file.is_some()
    }
}

impl Drop for BranchCopyGuard {
    fn drop(&mut self) {
        match self.file.take() {
            // Unlink while still locked, so nobody can lock this file after
            // the path stops naming it and believe they hold the lock.
            #[cfg(unix)]
            Some(file) => {
                if same_file(&file, &self.lock_path) {
                    let _ = fs::remove_file(&self.lock_path);
                }
                drop(file);
            }
            // The handle withholds delete sharing, so the path cannot be
            // removed while it is open: close it, then remove the path. That
            // fails harmlessly if another creator opened it in between.
            #[cfg(not(unix))]
            Some(file) => {
                drop(file);
                let _ = fs::remove_file(&self.lock_path);
            }
            None => {
                if fs::read_to_string(&self.lock_path).is_ok_and(|body| body.trim() == self.body) {
                    let _ = fs::remove_file(&self.lock_path);
                }
            }
        }
    }
}

/// Open the branch lock file for reading and writing, creating it
/// exclusively when `create_new` is set.
fn open_lock_file(path: &Path, create_new: bool) -> std::io::Result<fs::File> {
    let mut options = fs::OpenOptions::new();
    options.read(true).write(true).create_new(create_new);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt as _;
        // FILE_SHARE_READ | FILE_SHARE_WRITE without FILE_SHARE_DELETE, so
        // the path cannot be unlinked while a holder has it open.
        options.share_mode(0x1 | 0x2);
    }
    options.open(path)
}

/// Whether `path` still names the file open as `file`.
#[cfg(unix)]
fn same_file(file: &fs::File, path: &Path) -> bool {
    use std::os::unix::fs::MetadataExt as _;
    match (file.metadata(), fs::metadata(path)) {
        (Ok(a), Ok(b)) => a.dev() == b.dev() && a.ino() == b.ino(),
        _ => false,
    }
}

/// Whether `path` still names the file open as `file`: always, since
/// `open_lock_file` withholds delete sharing.
#[cfg(not(unix))]
fn same_file(_file: &fs::File, _path: &Path) -> bool {
    true
}

/// The trimmed content of the lock file, or "" when it cannot be read.
fn read_lock_body(file: &fs::File) -> String {
    use std::io::Read as _;
    let mut body = String::new();
    if (&*file).read_to_string(&mut body).is_err() {
        return String::new();
    }
    body.trim().to_string()
}

/// Whether a lock held by existence alone is still live by its age.
fn lock_age_fresh(meta: std::io::Result<fs::Metadata>) -> bool {
    meta.and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.elapsed().ok())
        .is_none_or(|age| age.as_secs() <= BRANCH_LOCK_STALE_SECS)
}

/// Build the branch base path: `<db_path>.branch.<name>`.
fn branch_path(db_path: &Path, name: &str) -> PathBuf {
    let mut p = db_path.as_os_str().to_owned();
    p.push(format!(".branch.{name}"));
    PathBuf::from(p)
}

/// Create a branch by copying the database and all companion files.
///
/// The branch is stored at `<db_path>.branch.<name>` with companion files
/// at `<db_path>.branch.<name>.vec`, etc.
///
/// This copies the files **off a path** without opening the database, so it is
/// point-in-time consistent only when the database is already quiescent (no
/// writer active). For a guaranteed-consistent branch from a live process,
/// prefer [`Axil::branch_create`](crate::Axil::branch_create), which flushes the
/// engines and closes the handles before copying.
///
/// It also requires that no live [`Axil`](crate::Axil) handle holds the core
/// file open. On Windows redb takes a byte-range lock for the lifetime of the
/// `Database`, so `fs::copy` of a held-open file fails with a lock violation;
/// on Unix the copy succeeds but the branch is flagged for repair (redb marks
/// the on-disk header "recovery required" while the file is open).
pub fn branch_create(db_path: &Path, name: &str) -> Result<PathBuf> {
    validate_branch_name(name)?;
    copy_branch_files(db_path, name)
}

/// `fs::copy` with a short retry: freshly-written companion files are
/// transiently locked on Windows (antivirus/indexer, OS error 32), and a
/// copy that only fails on that passes a moment later.
fn copy_file_retry(src: &Path, dest: &Path) -> std::io::Result<u64> {
    const RETRIES: usize = 4;
    let mut last_err = None;
    for attempt in 0..RETRIES {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_millis(50 * attempt as u64));
        }
        match fs::copy(src, dest) {
            Ok(n) => return Ok(n),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap())
}

/// Copy the core `.axil` and every companion file/dir to the branch location.
///
/// The caller must have already validated `name` and ensured the database is
/// quiescent (no open writer). Shared copy step behind both the free
/// [`branch_create`] function and [`Axil::branch_create`](crate::Axil::branch_create),
/// which flushes the engines and closes the handles before copying.
///
/// The files are copied under staging names owned by this call
/// (`<branch>.staging-<pid>-<nonce>…`) and renamed into place only once every
/// copy has succeeded, companions first and the core last, since a branch
/// exists once its core does. A copy that fails removes its own staged files
/// and whatever it already moved into place, and nothing else: no branch that
/// lists but cannot open is left, and nothing blocks creating the same branch
/// again. A staging name holds a dot, which no branch name can, so
/// [`branch_list`] never shows one.
///
/// Concurrency: creators are serialized by [`BranchCopyGuard`], held for the
/// whole create. Only while it holds the OS lock does a creator remove what
/// an earlier, dead creator left under this branch's names; with the
/// existence-only fallback another holder may still be alive. The core file
/// is copied first, so the only remaining window is an external writer
/// acquiring the database's own lock mid-copy and mutating *companion* files
/// after their core snapshot was taken — `doctor` + `heal --reindex`
/// reconcile that.
pub(crate) fn copy_branch_files(db_path: &Path, name: &str) -> Result<PathBuf> {
    copy_branch_files_with(db_path, name, BRANCH_LOCK_WAIT, &|_| {})
}

/// [`copy_branch_files`] with the lock wait passed in, and `after_step` called
/// after each file or directory is staged (`"core"`, the companion suffix,
/// `".fts"`), so tests can stall a copy partway.
fn copy_branch_files_with(
    db_path: &Path,
    name: &str,
    lock_wait: std::time::Duration,
    after_step: &dyn Fn(&str),
) -> Result<PathBuf> {
    let guard = BranchCopyGuard::acquire(db_path, lock_wait)
        .map_err(|e| AxilError::plugin(format!("branch copy could not start: {e}")))?;
    let dest = branch_path(db_path, name);
    if dest.exists() {
        return Err(AxilError::plugin(format!(
            "branch '{name}' already exists at {}",
            dest.display()
        )));
    }
    if !db_path.exists() {
        return Err(AxilError::plugin(format!(
            "database not found: {}",
            db_path.display()
        )));
    }

    if guard.exclusive() {
        // No other creator is alive, and with no core at `dest` nothing under
        // this branch's names is a usable branch: clear what a creator that
        // died left behind.
        for staging in staging_bases(&dest) {
            remove_branch_files(&staging);
        }
        remove_branch_companions(&dest);
    }

    let staging = crate::companion_path(&dest, &format!(".staging-{}", staging_token()));
    let mut published = Vec::new();
    let result = stage_branch_files(db_path, &staging, after_step).and_then(|()| {
        publish_staged_branch(&staging, &dest, &mut published)
            .map_err(|e| AxilError::plugin(format!("failed to publish branch '{name}': {e}")))
    });
    if result.is_err() {
        remove_branch_files(&staging);
        for path in &published {
            let _ = fs::remove_file(path).or_else(|_| fs::remove_dir_all(path));
        }
    }
    drop(guard);
    result.map(|()| dest)
}

/// A name part no other creator uses: this process's pid and a nonce.
fn staging_token() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or_default();
    format!(
        "{}-{nanos:x}{:x}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

/// The staging bases (`<branch>.staging-<token>`) left beside a branch.
fn staging_bases(dest: &Path) -> Vec<PathBuf> {
    let parent = dest.parent().unwrap_or(Path::new("."));
    let Some(prefix) = dest
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| format!("{n}.staging-"))
    else {
        return Vec::new();
    };
    let mut bases: Vec<PathBuf> = fs::read_dir(parent)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_str()?.to_string();
            let token = name.strip_prefix(&prefix)?.split('.').next()?.to_string();
            Some(crate::companion_path(dest, &format!(".staging-{token}")))
        })
        .collect();
    bases.sort();
    bases.dedup();
    bases
}

/// Copy the database and every companion under the branch base `staging`.
fn stage_branch_files(db_path: &Path, staging: &Path, after_step: &dyn Fn(&str)) -> Result<()> {
    // Copy main database file.
    fs::copy(db_path, staging)
        .map_err(|e| AxilError::plugin(format!("failed to copy database: {e}")))?;
    after_step("core");

    // Copy companion files.
    for suffix in COMPANION_SUFFIXES {
        let src = crate::companion_path(db_path, suffix);
        if src.exists() {
            let companion_dest = crate::companion_path(staging, suffix);
            copy_file_retry(&src, &companion_dest).map_err(|e| {
                AxilError::plugin(format!(
                    "failed to copy companion file {}: {e}",
                    src.display()
                ))
            })?;
            after_step(suffix);
        }
    }

    // Copy named vector-space companions (`<db>.vec.<name>`) — a branch
    // without them loses every named-space index.
    for (src, space) in named_space_files(db_path) {
        let suffix = format!(".vec.{space}");
        let companion_dest = crate::companion_path(staging, &suffix);
        copy_file_retry(&src, &companion_dest).map_err(|e| {
            AxilError::plugin(format!(
                "failed to copy named vector space {}: {e}",
                src.display()
            ))
        })?;
        after_step(&suffix);
    }

    // Copy FTS directory if it exists.
    let fts_src = crate::companion_path(db_path, ".fts");
    if fts_src.is_dir() {
        let fts_dest = crate::companion_path(staging, ".fts");
        copy_dir_stable(&fts_src, &fts_dest)
            .map_err(|e| AxilError::plugin(format!("failed to copy FTS directory: {e}")))?;
        after_step(".fts");
    }
    Ok(())
}

/// Rename a staged branch's files to their final names, the core last,
/// recording each final path in `published` as it is moved.
fn publish_staged_branch(
    staging: &Path,
    dest: &Path,
    published: &mut Vec<PathBuf>,
) -> std::io::Result<()> {
    let mut moves = Vec::new();
    for suffix in COMPANION_SUFFIXES.iter().copied().chain([".fts"]) {
        let from = crate::companion_path(staging, suffix);
        if from.exists() {
            moves.push((from, crate::companion_path(dest, suffix)));
        }
    }
    for (from, space) in named_space_files(staging) {
        moves.push((from, crate::companion_path(dest, &format!(".vec.{space}"))));
    }
    moves.push((staging.to_path_buf(), dest.to_path_buf()));
    for (from, to) in moves {
        fs::rename(from, &to)?;
        published.push(to);
    }
    Ok(())
}

/// Remove a branch's companions under the branch base `base`, best effort:
/// engine files, named vector spaces, and the FTS directory.
fn remove_branch_companions(base: &Path) {
    for suffix in COMPANION_SUFFIXES {
        let _ = fs::remove_file(crate::companion_path(base, suffix));
    }
    for (companion, _space) in named_space_files(base) {
        let _ = fs::remove_file(companion);
    }
    let _ = fs::remove_dir_all(crate::companion_path(base, ".fts"));
}

/// [`remove_branch_companions`], then the core file too.
fn remove_branch_files(base: &Path) {
    remove_branch_companions(base);
    let _ = fs::remove_file(base);
}

/// List all branches for a database.
pub fn branch_list(db_path: &Path) -> Result<Vec<String>> {
    let parent = db_path.parent().unwrap_or(Path::new("."));
    let db_name = db_path.file_name().and_then(|n| n.to_str()).unwrap_or("");

    let prefix = format!("{db_name}.branch.");
    let mut branches = Vec::new();

    if let Ok(entries) = fs::read_dir(parent) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if let Some(branch_name) = name_str.strip_prefix(&prefix) {
                // Exclude companion files of branches (e.g. .branch.foo.vec).
                if !branch_name.contains('.') {
                    branches.push(branch_name.to_string());
                }
            }
        }
    }

    branches.sort();
    Ok(branches)
}

/// Delete a branch and all its companion files.
pub fn branch_delete(db_path: &Path, name: &str) -> Result<()> {
    validate_branch_name(name)?;

    let dest = branch_path(db_path, name);
    if !dest.exists() {
        return Err(AxilError::plugin(format!("branch '{name}' does not exist")));
    }

    // Delete companion files first, then the main branch file.
    remove_branch_companions(&dest);
    fs::remove_file(&dest)
        .map_err(|e| AxilError::plugin(format!("failed to delete branch: {e}")))?;

    Ok(())
}

/// Get the database path for a named branch.
///
/// Returns the path if the branch exists, or an error if not found.
/// Use this path with `AXIL_DB` to switch to the branch.
pub fn branch_switch(db_path: &Path, name: &str) -> Result<PathBuf> {
    validate_branch_name(name)?;

    let bp = branch_path(db_path, name);
    if !bp.exists() {
        return Err(AxilError::plugin(format!("branch '{name}' does not exist")));
    }

    Ok(bp)
}

/// Compare a branch to the main database.
///
/// Opens both databases, compares record counts per table, and reports
/// which tables are new, deleted, or have different counts.
pub fn branch_diff(db_path: &Path, name: &str) -> Result<BranchDiff> {
    validate_branch_name(name)?;

    let branch_db_path = branch_path(db_path, name);
    if !branch_db_path.exists() {
        return Err(AxilError::plugin(format!("branch '{name}' does not exist")));
    }

    // Open both databases.
    let main_db = crate::Axil::open(db_path).build()?;
    let branch_db = crate::Axil::open(&branch_db_path).build()?;

    let main_tc = main_db.tables_with_counts()?;
    let branch_tc = branch_db.tables_with_counts()?;

    let main_tables: BTreeMap<String, usize> = main_tc.into_iter().collect();
    let branch_tables: BTreeMap<String, usize> = branch_tc.into_iter().collect();

    let new_tables: Vec<String> = branch_tables
        .keys()
        .filter(|k| !main_tables.contains_key(*k))
        .cloned()
        .collect();

    let deleted_tables: Vec<String> = main_tables
        .keys()
        .filter(|k| !branch_tables.contains_key(*k))
        .cloned()
        .collect();

    let mut count_diff = BTreeMap::new();
    for table in main_tables.keys().chain(branch_tables.keys()) {
        if count_diff.contains_key(table) {
            continue;
        }
        let main_count = main_tables.get(table).copied().unwrap_or(0) as i64;
        let branch_count = branch_tables.get(table).copied().unwrap_or(0) as i64;
        let diff = branch_count - main_count;
        if diff != 0 {
            count_diff.insert(table.clone(), diff);
        }
    }

    Ok(BranchDiff {
        branch_name: name.to_string(),
        main_tables,
        branch_tables,
        new_tables,
        deleted_tables,
        count_diff,
    })
}

/// Validate branch name: alphanumeric, hyphens, underscores only.
pub(crate) fn validate_branch_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(AxilError::InvalidQuery(
            "branch name cannot be empty".to_string(),
        ));
    }
    if name.len() > 64 {
        return Err(AxilError::InvalidQuery(
            "branch name too long (max 64 chars)".to_string(),
        ));
    }
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        return Err(AxilError::InvalidQuery(
            "branch name must contain only alphanumeric, hyphen, or underscore characters"
                .to_string(),
        ));
    }
    Ok(())
}

/// How to resolve conflicts when merging a branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MergeStrategy {
    /// Branch record wins on conflict (default).
    #[default]
    BranchWins,
    /// Main record wins on conflict (skip conflicting branch records).
    MainWins,
    /// Keep both by inserting the branch record as a new record.
    KeepBoth,
}

impl std::str::FromStr for MergeStrategy {
    type Err = ();
    fn from_str(s: &str) -> std::result::Result<Self, ()> {
        match s {
            "main-wins" => Ok(Self::MainWins),
            "keep-both" => Ok(Self::KeepBoth),
            _ => Ok(Self::BranchWins),
        }
    }
}

/// Result of merging a branch into the main database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeReport {
    /// Branch that was merged.
    pub branch_name: String,
    /// Number of new records added from the branch.
    pub records_added: usize,
    /// Number of records updated (conflicts resolved in favor of branch).
    pub records_updated: usize,
    /// Number of records skipped (conflicts resolved in favor of main).
    pub records_skipped: usize,
    /// Tables that were affected.
    pub tables_affected: Vec<String>,
    /// Whether indexes (vector, graph, FTS) need rebuilding after merge.
    pub indexes_need_rebuild: bool,
}

/// Merge a branch back into the main database.
///
/// Opens both databases, iterates all tables in the branch, and for each
/// record either inserts (if new) or applies the merge strategy (if conflict).
///
/// With the off-by-default `cdc` feature, this replays the branch's
/// `_changelog` tape — touching only the records the branch actually changed —
/// instead of the full O(all-records) per-table diff. Without `cdc`, it falls
/// back to the full diff.
///
/// Note: companion indexes (vector, graph, FTS) are NOT merged — after merge,
/// run `axil doctor` and `axil heal` to rebuild indexes if needed.
pub fn branch_merge(db_path: &Path, name: &str, strategy: MergeStrategy) -> Result<MergeReport> {
    validate_branch_name(name)?;

    let branch_db_path = branch_path(db_path, name);
    if !branch_db_path.exists() {
        return Err(AxilError::plugin(format!("branch '{name}' does not exist")));
    }

    let main_db = crate::Axil::open(db_path).build()?;
    let branch_db = crate::Axil::open(&branch_db_path).build()?;

    #[cfg(feature = "cdc")]
    {
        return merge_via_changelog(&main_db, &branch_db, name, strategy);
    }

    #[cfg(not(feature = "cdc"))]
    merge_full_scan(&main_db, &branch_db, name, strategy)
}

/// Full-scan merge: diff every branch table against main. Used when `cdc` is
/// off (and so there is no tape to replay).
#[cfg(not(feature = "cdc"))]
fn merge_full_scan(
    main_db: &crate::Axil,
    branch_db: &crate::Axil,
    name: &str,
    strategy: MergeStrategy,
) -> Result<MergeReport> {
    let branch_tables = branch_db.tables_with_counts()?;
    let mut records_added = 0usize;
    let mut records_updated = 0usize;
    let mut records_skipped = 0usize;
    let mut tables_affected = Vec::new();

    for (table, _count) in &branch_tables {
        let branch_records = branch_db.list(table)?;
        if branch_records.is_empty() {
            continue;
        }

        // Batch-load main records for this table to avoid N+1 get() calls.
        let main_records: BTreeMap<crate::record::RecordId, crate::record::Record> = main_db
            .list(table)
            .unwrap_or_default()
            .into_iter()
            .map(|r| (r.id.clone(), r))
            .collect();

        let mut table_changed = false;

        for branch_record in &branch_records {
            match main_records.get(&branch_record.id) {
                None => {
                    // New record — insert into main.
                    // Use storage directly to preserve the original ID and timestamps.
                    main_db.storage().insert(branch_record)?;
                    records_added += 1;
                    table_changed = true;
                }
                Some(ref main_record) => {
                    // Record exists in both — check if it changed.
                    if main_record.data == branch_record.data
                        && main_record.updated_at == branch_record.updated_at
                    {
                        // Identical — skip.
                        continue;
                    }

                    // Conflict: records differ.
                    match strategy {
                        MergeStrategy::BranchWins => {
                            main_db.update(&branch_record.id, branch_record.data.clone())?;
                            records_updated += 1;
                            table_changed = true;
                        }
                        MergeStrategy::MainWins => {
                            records_skipped += 1;
                        }
                        MergeStrategy::KeepBoth => {
                            // Insert the branch version as a new record.
                            main_db.insert(table, branch_record.data.clone())?;
                            records_added += 1;
                            table_changed = true;
                        }
                    }
                }
            }
        }

        if table_changed {
            tables_affected.push(table.clone());
        }
    }

    Ok(MergeReport {
        branch_name: name.to_string(),
        indexes_need_rebuild: records_added > 0 || records_updated > 0,
        records_added,
        records_updated,
        records_skipped,
        tables_affected,
    })
}

/// Changelog-replay merge (cdc feature): apply only the records the branch
/// actually changed, found by diffing the branch's `_changelog` tape against
/// main's.
///
/// A branch is a file copy that carries main's tape up to the divergence point,
/// then accrues its own post-divergence entries (fresh ULIDs main never sees).
/// The branch-only change IDs therefore identify exactly the touched records —
/// no full table scan. For each touched record we apply the branch's *final*
/// state, so the result matches the full-scan path's last-writer semantics.
///
/// Falls back to [`merge_full_scan`] semantics implicitly when the tape can't
/// cover the divergence: the caller's branch was created before `cdc` existed,
/// or the branch's earliest retained entry is newer than main's newest (the
/// whole branch tape is "branch-only"), both of which still replay correctly
/// here because branch-only IDs are simply every branch entry in that case.
#[cfg(feature = "cdc")]
fn merge_via_changelog(
    main_db: &crate::Axil,
    branch_db: &crate::Axil,
    name: &str,
    strategy: MergeStrategy,
) -> Result<MergeReport> {
    use std::collections::HashSet;

    // Build the set of change IDs main already knows about. Bounded by the
    // retained tape; a branch-only ID is one absent from this set.
    let mut main_ids: HashSet<String> = HashSet::new();
    let mut cursor: Option<String> = None;
    loop {
        let batch = main_db.changes_since(cursor.as_deref(), 10_000)?;
        if batch.is_empty() {
            break;
        }
        cursor = batch.last().map(|e| e.change_id.clone());
        for e in batch {
            main_ids.insert(e.change_id);
        }
    }

    // Walk the branch tape, collecting the records whose change IDs main has
    // never seen. Preserve first-seen order for deterministic replay.
    let mut touched: Vec<(String, String)> = Vec::new(); // (record_id, table)
    let mut seen_records: HashSet<String> = HashSet::new();
    let mut cursor: Option<String> = None;
    loop {
        let batch = branch_db.changes_since(cursor.as_deref(), 10_000)?;
        if batch.is_empty() {
            break;
        }
        cursor = batch.last().map(|e| e.change_id.clone());
        for e in batch {
            if main_ids.contains(&e.change_id) {
                continue;
            }
            if seen_records.insert(e.record_id.clone()) {
                touched.push((e.record_id, e.table));
            }
        }
    }

    let mut records_added = 0usize;
    let mut records_updated = 0usize;
    let mut records_skipped = 0usize;
    let mut tables_affected: Vec<String> = Vec::new();
    let mut affected_set: HashSet<String> = HashSet::new();
    let mark = |table: &str, set: &mut HashSet<String>, out: &mut Vec<String>| {
        if set.insert(table.to_string()) {
            out.push(table.to_string());
        }
    };

    for (record_id, table) in touched {
        let rid = crate::record::RecordId(record_id);
        let branch_record = branch_db.storage().get(&rid)?;
        let main_record = main_db.storage().get(&rid)?;

        match (branch_record, main_record) {
            // Branch deleted it; main still has it.
            (None, Some(_)) => match strategy {
                MergeStrategy::BranchWins => {
                    main_db.delete(&rid)?;
                    records_updated += 1;
                    mark(&table, &mut affected_set, &mut tables_affected);
                }
                MergeStrategy::MainWins | MergeStrategy::KeepBoth => {
                    records_skipped += 1;
                }
            },
            // Branch deleted it and main never had it — nothing to do.
            (None, None) => {}
            // New in branch — insert into main, preserving ID and timestamps.
            (Some(branch_record), None) => {
                main_db.storage().insert(&branch_record)?;
                records_added += 1;
                mark(&table, &mut affected_set, &mut tables_affected);
            }
            // Exists in both — apply the conflict strategy if it differs.
            (Some(branch_record), Some(main_record)) => {
                if main_record.data == branch_record.data
                    && main_record.updated_at == branch_record.updated_at
                {
                    continue;
                }
                match strategy {
                    MergeStrategy::BranchWins => {
                        main_db.update(&rid, branch_record.data.clone())?;
                        records_updated += 1;
                        mark(&table, &mut affected_set, &mut tables_affected);
                    }
                    MergeStrategy::MainWins => {
                        records_skipped += 1;
                    }
                    MergeStrategy::KeepBoth => {
                        main_db.insert(&table, branch_record.data.clone())?;
                        records_added += 1;
                        mark(&table, &mut affected_set, &mut tables_affected);
                    }
                }
            }
        }
    }

    Ok(MergeReport {
        branch_name: name.to_string(),
        indexes_need_rebuild: records_added > 0 || records_updated > 0,
        records_added,
        records_updated,
        records_skipped,
        tables_affected,
    })
}

/// Recursively copy a directory.
/// Copy a directory that something may still be changing, retrying until one
/// copy starts and ends with the same files in place.
///
/// The FTS index's merge threads write new segment files, rewrite
/// `meta.json`, and delete the segments they replaced.
/// [`Axil::branch_create`](crate::Axil::branch_create) waits for them before
/// closing the handle, so the first attempt normally stands. The retry covers
/// a copy taken some other way: a file listed and then deleted fails the copy
/// outright, and a copy taken while the files changed can hold a `meta.json`
/// that names segments the copy lacks, so neither attempt is kept.
fn copy_dir_stable(src: &Path, dest: &Path) -> std::io::Result<()> {
    const ATTEMPTS: u64 = 5;
    let mut last_err = None;
    for attempt in 0..ATTEMPTS {
        if attempt > 0 {
            let _ = fs::remove_dir_all(dest);
            std::thread::sleep(std::time::Duration::from_millis(50 * attempt));
        }
        let before = dir_state(src)?;
        match copy_dir_recursive(src, dest) {
            Ok(()) if dir_state(src)? == before => return Ok(()),
            Ok(()) => {
                last_err = Some(std::io::Error::other(
                    "its files changed while it was being copied",
                ))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => last_err = Some(e),
            Err(e) => return Err(e),
        }
    }
    Err(last_err.expect("at least one attempt ran"))
}

/// Every file under `dir` with its length and modification time, sorted. A
/// file deleted while it is listed is left out.
fn dir_state(dir: &Path) -> std::io::Result<Vec<(PathBuf, u64, Option<std::time::SystemTime>)>> {
    fn walk(
        dir: &Path,
        out: &mut Vec<(PathBuf, u64, Option<std::time::SystemTime>)>,
    ) -> std::io::Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let meta = match entry.metadata() {
                Ok(meta) => meta,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => return Err(e),
            };
            if meta.is_dir() {
                walk(&entry.path(), out)?;
            } else {
                out.push((entry.path(), meta.len(), meta.modified().ok()));
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    walk(dir, &mut out)?;
    out.sort();
    Ok(out)
}

fn copy_dir_recursive(src: &Path, dest: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dest)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let dest_path = dest.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&entry.path(), &dest_path)?;
        } else {
            fs::copy(entry.path(), &dest_path)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Axil;
    use serde_json::json;

    fn temp_db() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.axil");
        let db = Axil::open(&db_path).build().unwrap();
        // Insert a record so the DB file exists.
        db.insert("notes", json!({"text": "hello"})).unwrap();
        drop(db);
        (dir, db_path)
    }

    #[test]
    fn create_and_list_branch() {
        let (_dir, db_path) = temp_db();

        let branch_path = branch_create(&db_path, "experiment").unwrap();
        assert!(branch_path.exists());

        let branches = branch_list(&db_path).unwrap();
        assert_eq!(branches, vec!["experiment"]);
    }

    #[test]
    fn branch_copies_the_fts_directory_tree() {
        let (_dir, db_path) = temp_db();
        let fts = crate::companion_path(&db_path, ".fts");
        fs::create_dir_all(fts.join("nested")).unwrap();
        fs::write(fts.join("meta.json"), "{}").unwrap();
        fs::write(fts.join("nested").join("seg.idx"), "segment bytes").unwrap();

        let bp = branch_create(&db_path, "fts-copy").unwrap();
        let copied = crate::companion_path(&bp, ".fts");
        assert_eq!(fs::read_to_string(copied.join("meta.json")).unwrap(), "{}");
        assert_eq!(
            fs::read_to_string(copied.join("nested").join("seg.idx")).unwrap(),
            "segment bytes"
        );
        let names = |dir: &Path| -> Vec<(PathBuf, u64)> {
            dir_state(dir)
                .unwrap()
                .into_iter()
                .map(|(p, len, _)| (p.strip_prefix(dir).unwrap().to_path_buf(), len))
                .collect()
        };
        assert_eq!(names(&copied), names(&fts));
    }

    /// A copy that fails partway publishes nothing: no branch lists, no
    /// branch file is left, and the same name can be created afterwards.
    #[cfg(unix)]
    #[test]
    fn a_failed_copy_leaves_no_branch_behind() {
        use std::os::unix::fs::PermissionsExt;
        let (dir, db_path) = temp_db();
        let fts = crate::companion_path(&db_path, ".fts");
        fs::create_dir_all(&fts).unwrap();
        fs::write(fts.join("meta.json"), "{}").unwrap();
        let unreadable = fts.join("segment.idx");
        fs::write(&unreadable, "bytes").unwrap();
        fs::set_permissions(&unreadable, fs::Permissions::from_mode(0o000)).unwrap();
        if fs::read(&unreadable).is_ok() {
            // Running as root: permissions cannot make the copy fail.
            return;
        }
        // Leftovers of a creator that died mid-copy.
        let stale = crate::companion_path(&branch_path(&db_path, "exp"), ".staging-999-dead");
        fs::write(&stale, "stale core").unwrap();
        fs::write(crate::companion_path(&stale, ".vec"), "stale").unwrap();

        let err = branch_create(&db_path, "exp").unwrap_err();
        assert!(
            err.to_string().contains("failed to copy FTS directory"),
            "{err}"
        );
        assert!(branch_list(&db_path).unwrap().is_empty());
        let db_name = db_path.file_name().unwrap().to_str().unwrap().to_string();
        let left: Vec<String> = fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with(&format!("{db_name}.branch")))
            .collect();
        assert!(left.is_empty(), "{left:?}");

        fs::set_permissions(&unreadable, fs::Permissions::from_mode(0o644)).unwrap();
        let bp = branch_create(&db_path, "exp").unwrap();
        assert_eq!(branch_list(&db_path).unwrap(), ["exp"]);
        let copied = crate::companion_path(&bp, ".fts");
        assert_eq!(
            fs::read_to_string(copied.join("segment.idx")).unwrap(),
            "bytes"
        );
        assert!(staging_bases(&bp).is_empty());
        Axil::open(&bp).build().unwrap();
    }

    fn set_age(path: &Path, secs: u64) {
        let when = std::time::SystemTime::now() - std::time::Duration::from_secs(secs);
        fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(when)
            .unwrap();
    }

    /// A copy that stalls keeps its lock however old the lock file looks,
    /// and a second creator neither starts nor touches the first one's
    /// staged files.
    #[test]
    fn a_stalled_copy_keeps_its_lock_and_its_files() {
        use std::sync::mpsc;
        let (_dir, db_path) = temp_db();
        fs::write(crate::companion_path(&db_path, ".ts"), "series").unwrap();
        let dest = branch_path(&db_path, "exp");

        let (staged, staged_rx) = mpsc::channel();
        let (resume, resume_rx) = mpsc::channel::<()>();
        let first = std::thread::spawn({
            let db_path = db_path.clone();
            move || {
                let stall = move |step: &str| {
                    if step == "core" {
                        staged.send(()).unwrap();
                        resume_rx.recv().unwrap();
                    }
                };
                copy_branch_files_with(&db_path, "exp", BRANCH_LOCK_WAIT, &stall)
            }
        });
        staged_rx.recv().unwrap();

        // Older than the existence-only rule would allow.
        let lock = crate::companion_path(&db_path, ".branch-lock");
        set_age(&lock, 2 * BRANCH_LOCK_STALE_SECS);
        let staged_files = staging_bases(&dest);
        assert_eq!(staged_files.len(), 1, "{staged_files:?}");

        let second = copy_branch_files_with(
            &db_path,
            "exp",
            std::time::Duration::from_millis(300),
            &|_| {},
        );
        let err = second.unwrap_err().to_string();
        assert!(err.contains("in progress"), "{err}");
        assert!(
            staged_files[0].exists(),
            "the second creator removed staged files"
        );

        resume.send(()).unwrap();
        assert_eq!(first.join().unwrap().unwrap(), dest);
        assert_eq!(branch_list(&db_path).unwrap(), ["exp"]);
        assert_eq!(
            fs::read_to_string(crate::companion_path(&dest, ".ts")).unwrap(),
            "series"
        );
        assert!(staging_bases(&dest).is_empty());
        assert!(!lock.exists());
        Axil::open(&dest).build().unwrap();
    }

    /// An older axil holds the lock by the file's existence and writes its
    /// pid. Its lock is honored until it goes stale.
    #[test]
    fn an_older_axils_lock_is_honored_until_stale() {
        let (_dir, db_path) = temp_db();
        let lock = crate::companion_path(&db_path, ".branch-lock");
        fs::write(&lock, "4242").unwrap();
        let wait = std::time::Duration::from_millis(200);
        let held = BranchCopyGuard::acquire(&db_path, wait).err().unwrap();
        assert_eq!(held.kind(), std::io::ErrorKind::WouldBlock);

        set_age(&lock, 2 * BRANCH_LOCK_STALE_SECS);
        let guard = BranchCopyGuard::acquire(&db_path, wait).unwrap();
        assert!(guard.exclusive());
        drop(guard);
        assert!(!lock.exists());
    }

    /// Where OS locks are refused, the lock is the file's existence, broken
    /// once stale.
    #[test]
    fn without_os_locks_the_lock_is_the_file() {
        let (_dir, db_path) = temp_db();
        let refuse = |_: &fs::File| {
            Err(fs::TryLockError::Error(std::io::Error::other(
                "locks unsupported",
            )))
        };
        let wait = std::time::Duration::from_millis(200);
        let lock = crate::companion_path(&db_path, ".branch-lock");

        let first = BranchCopyGuard::acquire_with(&db_path, wait, refuse).unwrap();
        assert!(!first.exclusive());
        assert!(BranchCopyGuard::acquire_with(&db_path, wait, refuse).is_err());
        set_age(&lock, 2 * BRANCH_LOCK_STALE_SECS);
        let second = BranchCopyGuard::acquire_with(&db_path, wait, refuse).unwrap();
        // The first holder's release leaves the second holder's file alone.
        drop(first);
        assert!(lock.exists());
        drop(second);
        assert!(!lock.exists());
    }

    #[test]
    fn delete_branch() {
        let (_dir, db_path) = temp_db();

        branch_create(&db_path, "to-delete").unwrap();
        assert!(!branch_list(&db_path).unwrap().is_empty());

        branch_delete(&db_path, "to-delete").unwrap();
        assert!(branch_list(&db_path).unwrap().is_empty());
    }

    #[test]
    fn duplicate_branch_errors() {
        let (_dir, db_path) = temp_db();

        branch_create(&db_path, "dup").unwrap();
        let result = branch_create(&db_path, "dup");
        assert!(result.is_err());
    }

    #[test]
    fn delete_nonexistent_branch_errors() {
        let (_dir, db_path) = temp_db();
        let result = branch_delete(&db_path, "nope");
        assert!(result.is_err());
    }

    #[test]
    fn diff_branch() {
        let (_dir, db_path) = temp_db();

        // Create branch.
        let bp = branch_create(&db_path, "diff-test").unwrap();

        // Add a record to the branch.
        let branch_db = Axil::open(&bp).build().unwrap();
        branch_db
            .insert("notes", json!({"text": "branch record"}))
            .unwrap();
        drop(branch_db);

        let diff = branch_diff(&db_path, "diff-test").unwrap();
        assert_eq!(diff.branch_name, "diff-test");
        // Branch should have 1 more record in "notes".
        assert_eq!(diff.count_diff.get("notes"), Some(&1));
    }

    #[test]
    fn merge_branch_adds_new_records() {
        let (_dir, db_path) = temp_db();

        let bp = branch_create(&db_path, "feature").unwrap();

        // Add a record only in the branch.
        let branch_db = Axil::open(&bp).build().unwrap();
        branch_db
            .insert("tasks", json!({"text": "branch-only task"}))
            .unwrap();
        drop(branch_db);

        let report = branch_merge(&db_path, "feature", MergeStrategy::BranchWins).unwrap();
        assert_eq!(report.records_added, 1);
        assert!(report.tables_affected.contains(&"tasks".to_string()));

        // Verify the record exists in main.
        let main_db = Axil::open(&db_path).build().unwrap();
        let tasks = main_db.list("tasks").unwrap();
        assert!(tasks.iter().any(|r| r.data["text"] == "branch-only task"));
    }

    #[test]
    fn merge_branch_wins_updates_conflicts() {
        let (_dir, db_path) = temp_db();

        // Get the existing record ID.
        let main_db = Axil::open(&db_path).build().unwrap();
        let records = main_db.list("notes").unwrap();
        let record_id = records[0].id.clone();
        drop(main_db);

        let bp = branch_create(&db_path, "fix").unwrap();

        // Modify the record in the branch.
        let branch_db = Axil::open(&bp).build().unwrap();
        branch_db
            .update(&record_id, json!({"text": "updated in branch"}))
            .unwrap();
        drop(branch_db);

        let report = branch_merge(&db_path, "fix", MergeStrategy::BranchWins).unwrap();
        assert_eq!(report.records_updated, 1);

        let main_db = Axil::open(&db_path).build().unwrap();
        let r = main_db.get(&record_id).unwrap().unwrap();
        assert_eq!(r.data["text"], "updated in branch");
    }

    #[test]
    fn merge_main_wins_skips_conflicts() {
        let (_dir, db_path) = temp_db();

        let main_db = Axil::open(&db_path).build().unwrap();
        let records = main_db.list("notes").unwrap();
        let record_id = records[0].id.clone();
        drop(main_db);

        let bp = branch_create(&db_path, "skip").unwrap();

        let branch_db = Axil::open(&bp).build().unwrap();
        branch_db
            .update(&record_id, json!({"text": "branch change"}))
            .unwrap();
        drop(branch_db);

        let report = branch_merge(&db_path, "skip", MergeStrategy::MainWins).unwrap();
        assert_eq!(report.records_skipped, 1);
        assert_eq!(report.records_updated, 0);

        // Main record unchanged.
        let main_db = Axil::open(&db_path).build().unwrap();
        let r = main_db.get(&record_id).unwrap().unwrap();
        assert_eq!(r.data["text"], "hello");
    }

    #[test]
    fn merge_keep_both_duplicates() {
        let (_dir, db_path) = temp_db();

        let main_db = Axil::open(&db_path).build().unwrap();
        let records = main_db.list("notes").unwrap();
        let record_id = records[0].id.clone();
        drop(main_db);

        let bp = branch_create(&db_path, "both").unwrap();

        let branch_db = Axil::open(&bp).build().unwrap();
        branch_db
            .update(&record_id, json!({"text": "branch version"}))
            .unwrap();
        drop(branch_db);

        let report = branch_merge(&db_path, "both", MergeStrategy::KeepBoth).unwrap();
        assert_eq!(report.records_added, 1);

        // Main should now have 2 records in notes.
        let main_db = Axil::open(&db_path).build().unwrap();
        let notes = main_db.list("notes").unwrap();
        assert_eq!(notes.len(), 2);
    }

    #[test]
    fn merge_nonexistent_branch_errors() {
        let (_dir, db_path) = temp_db();
        let result = branch_merge(&db_path, "nope", MergeStrategy::BranchWins);
        assert!(result.is_err());
    }

    #[test]
    fn invalid_branch_names() {
        let (_dir, db_path) = temp_db();

        assert!(branch_create(&db_path, "").is_err());
        assert!(branch_create(&db_path, "has.dot").is_err());
        assert!(branch_create(&db_path, "has space").is_err());
        assert!(branch_create(&db_path, "ok-name_123").is_ok());
    }

    #[test]
    fn list_empty_branches() {
        let (_dir, db_path) = temp_db();
        let branches = branch_list(&db_path).unwrap();
        assert!(branches.is_empty());
    }

    /// Whether redb asks to repair `path` on open (the file was not cleanly
    /// closed). Opens the file directly with redb, bypassing `Axil`.
    fn redb_needs_repair(path: &Path) -> bool {
        let repaired = std::rc::Rc::new(std::cell::Cell::new(false));
        let flag = repaired.clone();
        let db = redb::Database::builder()
            .set_repair_callback(move |_| flag.set(true))
            .open(path)
            .unwrap();
        drop(db);
        repaired.get()
    }

    /// Probe: can the core file be copied while a live redb handle holds it?
    /// If it could, `Axil::branch_create` could hold the single-writer lock
    /// across the copy instead of dropping it. It cannot, on any platform:
    /// Windows refuses the copy outright (redb's byte-range lock, OS error 33
    /// `ERROR_LOCK_VIOLATION`), and on Unix the copy succeeds but snapshots a
    /// header redb flags "recovery required" for as long as the file is open,
    /// so the branch would only open through a full repair.
    #[test]
    fn fs_copy_of_open_redb_file_is_never_a_clean_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("probe.axil");
        let db = Axil::open(&db_path).build().unwrap();
        db.insert("notes", json!({"probe": true})).unwrap();
        let dest = dir.path().join("probe-copy.axil");
        let copied = std::fs::copy(&db_path, &dest);
        drop(db);

        if cfg!(windows) {
            let err = copied.expect_err("Windows must refuse copying a redb file held open");
            assert_eq!(err.raw_os_error(), Some(33), "unexpected copy error: {err}");
        } else {
            copied.expect("Unix allows copying a redb file held open");
            assert!(
                redb_needs_repair(&dest),
                "a copy taken while the handle is open must be flagged for repair"
            );
        }
    }

    /// `Axil::branch_create` flushes and drops the handle before copying, so
    /// the branch is a cleanly-closed file that opens without a repair — on
    /// every platform, which is why that path never holds the lock across the
    /// copy.
    #[test]
    fn live_branch_create_copies_a_cleanly_closed_file() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("live.axil");
        let db = Axil::open(&db_path).build().unwrap();
        db.insert("notes", json!({"live": true})).unwrap();

        let branch = db.branch_create("clean").unwrap();
        assert!(
            !redb_needs_repair(&branch),
            "a branch copied from a quiesced handle must not need repair"
        );
        let branch_db = Axil::open(&branch).build().unwrap();
        assert_eq!(branch_db.list("notes").unwrap().len(), 1);
    }

    #[test]
    fn branch_copy_includes_named_vector_space_files() {
        let (_dir, db_path) = temp_db();

        // Simulate two named-space companion files next to the database.
        let space_a = crate::companion_path(&db_path, ".vec.alpha");
        let space_b = crate::companion_path(&db_path, ".vec.beta");
        fs::write(&space_a, b"alpha-vec-data").unwrap();
        fs::write(&space_b, b"beta-vec-data").unwrap();

        let bp = branch_create(&db_path, "spaces").unwrap();
        assert!(crate::companion_path(&bp, ".vec.alpha").exists());
        assert!(crate::companion_path(&bp, ".vec.beta").exists());

        // And they go away with the branch.
        branch_delete(&db_path, "spaces").unwrap();
        assert!(!crate::companion_path(&bp, ".vec.alpha").exists());
        assert!(!crate::companion_path(&bp, ".vec.beta").exists());
    }

    #[test]
    fn branch_copy_lock_marker_is_released() {
        let (_dir, db_path) = temp_db();

        branch_create(&db_path, "locked").unwrap();

        let mut lock = db_path.as_os_str().to_owned();
        lock.push(".branch-lock");
        assert!(
            !PathBuf::from(&lock).exists(),
            "lock marker must be removed after a successful copy"
        );
    }

    /// The changelog-replay merge path (cdc feature) propagates a branch delete
    /// back to main under BranchWins — something the full-scan path can't see.
    #[cfg(feature = "cdc")]
    #[test]
    fn merge_replays_branch_delete() {
        let (_dir, db_path) = temp_db();

        let main_db = Axil::open(&db_path).build().unwrap();
        let record_id = main_db.list("notes").unwrap()[0].id.clone();
        drop(main_db);

        let bp = branch_create(&db_path, "del").unwrap();
        let branch_db = Axil::open(&bp).build().unwrap();
        branch_db.delete(&record_id).unwrap();
        drop(branch_db);

        let report = branch_merge(&db_path, "del", MergeStrategy::BranchWins).unwrap();
        assert_eq!(report.records_updated, 1);

        let main_db = Axil::open(&db_path).build().unwrap();
        assert!(main_db.get(&record_id).unwrap().is_none());
    }
}
