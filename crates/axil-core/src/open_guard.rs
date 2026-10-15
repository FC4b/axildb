//! A check that runs before a database's files are opened.
//!
//! Only one process can have a database open. A host whose databases may be
//! held by a process of its own elsewhere (the CLI's warm daemon) installs a
//! guard with [`set_open_guard`], which can ask that process to let go first.
//! [`Axil::open`](crate::Axil::open) calls it with the database's path before
//! anything touches the database's files, and so do
//! [`Storage::open`](crate::storage::Storage::open) and
//! [`Storage::open_read_only`](crate::storage::Storage::open_read_only), for an
//! open that does not begin with `Axil::open`. Code that reads a database's
//! files before opening it, such as an engine probing its companion file,
//! calls [`open_guard`] itself first.
//!
//! With no guard installed, every call is a no-op.

use std::cell::Cell;
use std::path::Path;
use std::sync::OnceLock;

/// Runs with a database's path before the database is opened.
pub type OpenGuard = fn(&Path);

static GUARD: OnceLock<OpenGuard> = OnceLock::new();

thread_local! {
    /// How many [`without_open_guard`] calls this thread is inside.
    static BYPASSED: Cell<usize> = const { Cell::new(0) };
}

/// Run `guard` before every database open from now on. A process sets it
/// once: this returns `false`, and changes nothing, when one is already set.
pub fn set_open_guard(guard: OpenGuard) -> bool {
    GUARD.set(guard).is_ok()
}

/// Run the installed guard for the database at `path`, unless this thread is
/// inside [`without_open_guard`].
pub fn open_guard(path: &Path) {
    if BYPASSED.with(Cell::get) > 0 {
        return;
    }
    if let Some(guard) = GUARD.get() {
        guard(path);
    }
}

/// Run `f` with the guard skipped for every open this thread makes in it:
/// for the process the guard would otherwise ask to let go, opening the
/// database for itself.
pub fn without_open_guard<T>(f: impl FnOnce() -> T) -> T {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            BYPASSED.with(|b| b.set(b.get() - 1));
        }
    }
    BYPASSED.with(|b| b.set(b.get() + 1));
    let _restore = Restore;
    f()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::path::PathBuf;

    thread_local! {
        static SEEN: RefCell<Option<Vec<PathBuf>>> = const { RefCell::new(None) };
    }

    /// Records the paths of a thread that asked for them; the other
    /// threads of the test binary open databases unwatched.
    fn record(path: &Path) {
        SEEN.with(|s| {
            if let Some(seen) = s.borrow_mut().as_mut() {
                seen.push(path.to_path_buf());
            }
        });
    }

    fn seen() -> Vec<PathBuf> {
        SEEN.with(|s| s.borrow_mut().replace(Vec::new()).unwrap_or_default())
    }

    #[test]
    fn every_open_runs_the_guard_first_unless_bypassed() {
        assert!(set_open_guard(record));
        assert!(!set_open_guard(|_| panic!("replaced the guard")));
        SEEN.with(|s| *s.borrow_mut() = Some(Vec::new()));
        let tmp = tempfile::tempdir().unwrap();
        let db = tmp.path().join("memory.axil");

        // The builder alone, before anything is opened.
        let builder = crate::Axil::open(&db);
        assert_eq!(seen(), [db.clone()]);
        // Building opens the core store, which runs it again.
        drop(builder.build().unwrap());
        assert_eq!(seen(), [db.clone()]);
        drop(crate::storage::Storage::open(&db).unwrap());
        drop(crate::storage::Storage::open_read_only(&db).unwrap());
        assert_eq!(seen(), [db.clone(), db.clone()]);

        let built = without_open_guard(|| {
            // Nested, and through a panic caught inside, it stays skipped
            // until the outermost call returns.
            without_open_guard(|| open_guard(&db));
            let _ = std::panic::catch_unwind(|| without_open_guard(|| panic!("inside")));
            crate::Axil::open(&db).build()
        });
        drop(built.unwrap());
        assert_eq!(seen(), Vec::<PathBuf>::new());
        open_guard(&db);
        assert_eq!(seen(), [db]);
        SEEN.with(|s| *s.borrow_mut() = None);
    }
}
