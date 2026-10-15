//! The hook queue, drained inside the daemon.
//!
//! Hooks queue their writes as job files in the database's own queue,
//! `<db dir>/hook-queue/` for `memory.axil` (see [`hook_brain::queue_dir`]).
//! While the daemon serves the database, its drain thread runs them on the
//! open handle, so a session's writes neither open the database nor make the
//! daemon let go of it.
//!
//! The thread keeps the queue's protocol exactly as `hook drain` does:
//! - It holds `drain.lock` ([`DrainLock`], an OS lock on that very file)
//!   while it drains, from the moment it first takes it until the daemon
//!   stops. A one-shot drainer locks the same file, so the two never drain
//!   at once, and a drainer from before OS locks finds the file there and
//!   fresh, since the thread touches it at least once a minute.
//! - Each time it takes the lock it sets aside, into `interrupted/`, any
//!   `.running` job a drainer that died left behind, never running it again.
//! - It claims a job by renaming it from `.job` to `.running`, runs it, and
//!   removes it, whether or not its steps succeeded; jobs run in the order
//!   their names sort, one at a time.
//!
//! It wakes when a hook pokes it ([`super::proto::Op::Poke`]), when the
//! database opens again, and every 2 s. A job is started only while the
//! daemon serves: during a lease, a reload or the opening, the queue waits
//! at the job boundary. Between steps, writes clients sent go first.
//!
//! A step runs on the open handle when the daemon runs that command at all
//! ([`exec::warm_route`]). One it doesn't runs in a process of its own, as
//! `hook drain` runs it, with `AXIL_DAEMON=off`: the thread first has the
//! daemon let go of the database as for a lease ([`Server::yield_db`]),
//! keeps it closed for the steps that follow while they too run in their
//! own process, and has it opened again once a step the daemon runs comes,
//! or the queue is empty.
//!
//! A job queued by a hook the daemon does not serve (another build, another
//! environment; see [`hook_brain::for_own_drainer`]) is left to that hook's
//! own drainer: the thread lets go of the lock at that job, and once that
//! drainer has taken it, takes the queue back when it is free. When no
//! drainer takes it within a few seconds, the thread runs the job itself
//! rather than leave the queue stuck.
//!
//! When the daemon stops, the thread finishes the job it is on and lets go
//! of the lock; the daemon then starts `hook drain` for the jobs left, once
//! it has closed the database. A step of that job the closing database can
//! no longer run runs in a process of its own, as `hook drain` runs it.
//!
//! The thread is the only one in the daemon that starts processes, and the
//! only one that lets go of the lock, so no process it starts can hold a
//! copy of the lock's handle at the moment the lock is let go (a forked
//! child keeps the lock until it execs; see [`DrainLock`]). A drainer that
//! still finds the lock held at that moment leaves its job to the thread,
//! which runs it once no drainer took the queue.

use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::exec;
use super::server::{Queued, Server};
use crate::hook_brain::{
    self, drain_log, for_own_drainer, next_job, project_dir_of, queue_dir, run_job_file,
    set_aside_interrupted, step_failed, DrainLock, JobRunner, SubprocessRunner,
};

/// How often the thread looks at the queue without being poked.
const POLL: Duration = Duration::from_secs(2);

/// How often the thread touches `drain.lock` while it waits, so a drainer
/// that judges it by age finds it fresh (it gives up on one after 30
/// minutes).
const TOUCH_EVERY: Duration = Duration::from_secs(60);

/// How long the thread waits, once it let go of the queue at a job queued
/// by a hook it does not serve, for that hook's drainer to take it.
const HANDOVER_GRACE: Duration = Duration::from_secs(5);

/// The stack the thread runs on: `main`'s, since it parses command lines.
const STACK: usize = 32 * 1024 * 1024;

/// What the queue has come to since the daemon started, for `daemon status`.
#[derive(Default)]
pub struct QueueStats {
    /// The thread holds the queue's lock.
    holding: AtomicBool,
    /// Jobs run, failed or not.
    jobs: AtomicU64,
    /// Jobs whose run failed.
    failed: AtomicU64,
    /// Steps run on the open handle.
    warm_steps: AtomicU64,
    /// Steps run in a process of their own.
    own_process_steps: AtomicU64,
    /// Times the daemon let go of the database for such steps.
    yields: AtomicU64,
    /// Times the thread left the queue to the drainer of a hook it does
    /// not serve.
    handed_over: AtomicU64,
}

impl QueueStats {
    /// Whether the drain thread holds the queue's lock.
    pub fn holding(&self) -> bool {
        self.holding.load(Ordering::Acquire)
    }

    /// The figures, as `daemon status` reports them.
    pub fn report(&self) -> Value {
        let n = |c: &AtomicU64| c.load(Ordering::Acquire);
        json!({
            "draining": self.holding.load(Ordering::Acquire),
            "jobs": n(&self.jobs),
            "failed": n(&self.failed),
            "warm_steps": n(&self.warm_steps),
            "own_process_steps": n(&self.own_process_steps),
            "yields": n(&self.yields),
            "handed_over": n(&self.handed_over),
        })
    }

    /// One more job run, which failed when `failed`.
    fn ran(&self, failed: bool) {
        self.jobs.fetch_add(1, Ordering::AcqRel);
        if failed {
            self.failed.fetch_add(1, Ordering::AcqRel);
        }
    }
}

/// Start the drain thread for `server`'s database.
pub fn spawn(server: Arc<Server>) -> io::Result<JoinHandle<()>> {
    thread::Builder::new()
        .name("axil-drain".into())
        .stack_size(STACK)
        .spawn(move || run(&server))
}

/// The thread: drain the queue until the daemon stops.
fn run(server: &Arc<Server>) {
    let db = server.key().to_path_buf();
    let dir = queue_dir(&db);
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(e) => {
            super::server::log(format!("not draining the hook queue: {e}"));
            return;
        }
    };
    let cwd = project_dir_of(&db);
    let mut runner = WarmRunner {
        server,
        db: db.clone(),
        cwd,
        exe: exe.clone(),
        yielded: None,
        lock: dir.join("drain.lock"),
        touched: Instant::now(),
    };
    let mut lock: Option<DrainLock> = None;
    // The job last handed over that no drainer took: run here next time.
    let mut not_taken: Option<PathBuf> = None;
    while !server.stopping() {
        let Some(held) = lock.as_ref() else {
            lock = take(&dir, server);
            if lock.is_some() {
                runner.touched = Instant::now();
            } else {
                server.wait_for_wake(POLL);
            }
            continue;
        };
        runner.keep_lock_fresh();
        let Some(job) = next_job(&dir) else {
            runner.end_yield();
            server.wait_for_wake(POLL);
            continue;
        };
        // A job starts only while the daemon serves, or while it has let go
        // of the database for the queue's own steps.
        if runner.yielded.is_none() && !server.serving() {
            server.wait_for_wake(POLL);
            continue;
        }
        if for_own_drainer(&job) && not_taken.as_deref() != Some(job.as_path()) {
            runner.end_yield();
            server.queue().handed_over.fetch_add(1, Ordering::AcqRel);
            server.queue().holding.store(false, Ordering::Release);
            let name = job_name(&job);
            drain_log(
                &dir,
                &format!("daemon: leaving job {name} and the ones after it to the drainer of the hook that queued it"),
            );
            lock = None;
            match hand_over(&dir, &job, server) {
                None => {}
                Some(back) => {
                    drain_log(
                        &dir,
                        &format!("daemon: no drainer took the queue; running job {name} here"),
                    );
                    set_aside_interrupted(&dir);
                    server.queue().holding.store(true, Ordering::Release);
                    lock = Some(back);
                    not_taken = Some(job);
                }
            }
            continue;
        }
        held.touch();
        runner.touched = Instant::now();
        let ran = run_job_file(&mut runner, &exe, &db, &dir, &job);
        server.queue().ran(!ran);
    }
    runner.end_yield();
    server.queue().holding.store(false, Ordering::Release);
    drop(lock);
}

/// Take the queue's lock, when the queue exists and no drainer holds it,
/// and set aside what a drainer that died left running.
fn take(dir: &Path, server: &Server) -> Option<DrainLock> {
    if !dir.is_dir() {
        return None;
    }
    let lock = DrainLock::acquire(dir)?;
    set_aside_interrupted(dir);
    server.queue().holding.store(true, Ordering::Release);
    Some(lock)
}

/// Leave the queue, let go of a moment ago at `job`, to another drainer:
/// `None` once one holds it or has claimed `job`, the lock back when none
/// did within [`HANDOVER_GRACE`]. It is let go of for most of each tenth of
/// a second, and taken back between, to find out.
fn hand_over(dir: &Path, job: &Path, server: &Server) -> Option<DrainLock> {
    let deadline = Instant::now() + HANDOVER_GRACE;
    loop {
        thread::sleep(Duration::from_millis(100));
        if !job.exists() {
            return None;
        }
        let lock = DrainLock::acquire(dir)?;
        if Instant::now() >= deadline || server.stopping() {
            return Some(lock);
        }
    }
}

fn job_name(job: &Path) -> String {
    job.file_stem()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Runs a job's steps on the daemon's open handle, or in a process of their
/// own when the daemon doesn't run the command.
struct WarmRunner<'a> {
    server: &'a Arc<Server>,
    db: PathBuf,
    /// The project's directory, where a step runs.
    cwd: PathBuf,
    exe: PathBuf,
    /// The lease the daemon holds for the queue's own steps, while it has
    /// let go of the database for them.
    yielded: Option<u64>,
    /// `drain.lock`, which the thread holds while it runs a job.
    lock: PathBuf,
    /// When `drain.lock` was last touched.
    touched: Instant,
}

impl WarmRunner<'_> {
    /// `axil --db <db> <args>`, or `None` when the database's path is not
    /// text, which a request must be.
    fn argv(&self, args: &[&str]) -> Option<Vec<String>> {
        let db = self.db.to_str()?;
        let mut argv = vec!["axil".to_string(), "--db".to_string(), db.to_string()];
        argv.extend(args.iter().map(|a| a.to_string()));
        Some(argv)
    }

    /// Touch `drain.lock` when [`TOUCH_EVERY`] has passed since it last
    /// was, however long the queue waits, so a drainer that judges it by
    /// age never finds it stale. Only its time changes: the holder's tag
    /// and pid stay.
    fn keep_lock_fresh(&mut self) {
        if self.touched.elapsed() < TOUCH_EVERY {
            return;
        }
        self.touched = Instant::now();
        if let Ok(file) = std::fs::File::options().append(true).open(&self.lock) {
            let _ = file.set_modified(std::time::SystemTime::now());
        }
    }

    /// Wait for the writes clients sent, which go first between steps.
    fn let_writers_go_first(&self) {
        while self.server.writers_waiting() > 0 && !self.server.stopping() {
            thread::sleep(Duration::from_millis(2));
        }
    }

    /// Have the daemon let go of the database for steps run in their own
    /// process. False when it could not, its requests still running.
    fn start_yield(&mut self) -> bool {
        if self.yielded.is_some() {
            return true;
        }
        match self.server.yield_db() {
            Ok(id) => {
                self.yielded = Some(id);
                self.server.queue().yields.fetch_add(1, Ordering::AcqRel);
                true
            }
            Err(why) => {
                super::server::log(format!(
                    "the queue could not have the database let go of: {why}"
                ));
                false
            }
        }
    }

    /// Have the daemon open the database again, if it let go of it for the
    /// queue.
    fn end_yield(&mut self) {
        if let Some(id) = self.yielded.take() {
            self.server.unyield(id);
        }
    }

    fn subprocess(&self) -> SubprocessRunner<'_> {
        SubprocessRunner::new(&self.exe, &self.db)
            .with_env("AXIL_DAEMON", "off")
            .in_dir(&self.cwd)
    }
}

impl JobRunner for WarmRunner<'_> {
    fn write(
        &mut self,
        args: &[String],
        stdin: Option<&str>,
        env: Option<&[(String, String)]>,
    ) -> Result<String, String> {
        let args_ref: Vec<&str> = args.iter().map(String::as_str).collect();
        let argv = self.argv(&args_ref);
        if self.yielded.is_some() {
            match &argv {
                Some(argv) if exec::routes(argv) => self.end_yield(),
                _ => {
                    self.server
                        .queue()
                        .own_process_steps
                        .fetch_add(1, Ordering::AcqRel);
                    return self.subprocess().write(args, stdin, env);
                }
            }
        }
        let Some(argv) = argv else {
            return self.own_process(args, stdin, env, "the database's path is not text");
        };
        let request_env = deps_env_of(env);
        loop {
            self.let_writers_go_first();
            let piped = stdin.unwrap_or_default().as_bytes();
            match self
                .server
                .run_queued(&argv, &self.cwd, piped, &request_env)
            {
                Queued::Ran(response) => {
                    self.server
                        .queue()
                        .warm_steps
                        .fetch_add(1, Ordering::AcqRel);
                    return if response.exit == 0 {
                        Ok(String::from_utf8_lossy(&response.stdout).into_owned())
                    } else {
                        Err(step_failed(
                            args,
                            &String::from_utf8_lossy(&response.stderr),
                        ))
                    };
                }
                Queued::NotServed(why) => {
                    return self.own_process(args, stdin, env, &why.to_string());
                }
                Queued::Wait if !self.server.stopping() => {
                    self.keep_lock_fresh();
                    self.server.wait_for_wake(POLL);
                }
                Queued::Wait | Queued::Stopped => {
                    return self.own_process(args, stdin, env, "the daemon is stopping");
                }
            }
        }
    }

    fn read(&mut self, args: &[&str], stdin: &[u8]) -> Option<String> {
        // It opens no database: when the daemon can't run it now, a process
        // of its own runs it without the daemon letting go of anything.
        if self.yielded.is_none() {
            if let Some(argv) = self.argv(args) {
                if let Queued::Ran(response) = self.server.run_queued(&argv, &self.cwd, stdin, &[])
                {
                    self.server
                        .queue()
                        .warm_steps
                        .fetch_add(1, Ordering::AcqRel);
                    return (response.exit == 0)
                        .then(|| String::from_utf8_lossy(&response.stdout).into_owned());
                }
            }
        }
        self.server
            .queue()
            .own_process_steps
            .fetch_add(1, Ordering::AcqRel);
        self.subprocess().read(args, stdin)
    }
}

impl WarmRunner<'_> {
    /// Run a step the daemon does not run in a process of its own, once the
    /// daemon has let go of the database. Once the daemon is stopping, the
    /// database is closing, if not closed: the step runs without waiting
    /// for it, as `hook drain` runs it, waiting out the close.
    fn own_process(
        &mut self,
        args: &[String],
        stdin: Option<&str>,
        env: Option<&[(String, String)]>,
        why: &str,
    ) -> Result<String, String> {
        super::server::log(format!(
            "running queued `axil {}` in a process of its own: {why}",
            args.first().map(String::as_str).unwrap_or("")
        ));
        while !self.server.stopping() && !self.start_yield() {
            self.keep_lock_fresh();
            self.server.wait_for_wake(POLL);
        }
        self.server
            .queue()
            .own_process_steps
            .fetch_add(1, Ordering::AcqRel);
        self.subprocess().write(args, stdin, env)
    }
}

/// The [`crate::DEPS_ENV`] values a queued step's request carries: the ones
/// its job recorded, else the daemon's own, which a process of the
/// daemon's would run the step with.
fn deps_env_of(recorded: Option<&[(String, String)]>) -> Vec<(OsString, OsString)> {
    match recorded {
        Some(env) => env
            .iter()
            .map(|(name, value)| (OsString::from(name), OsString::from(value)))
            .collect(),
        None => crate::DEPS_ENV
            .iter()
            .filter_map(|name| Some((OsString::from(name), std::env::var_os(name)?)))
            .collect(),
    }
}

/// After the daemon stopped and closed the database: start `hook drain` for
/// the jobs left in the queue, if any.
pub fn hand_back(key: &Path) {
    let dir = queue_dir(key);
    if next_job(&dir).is_none() {
        return;
    }
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    drain_log(&dir, "daemon: stopped with jobs queued; starting a drainer");
    hook_brain::spawn_drainer(&exe, key, &project_dir_of(key));
}
