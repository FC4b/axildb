# Warm Daemon

`axil daemon` runs one long-lived process per database. It keeps the database
open, with the embedding model and the graph loaded, and runs the hot commands
(`recall`, `boot`, `recall-for-file`, `code-search`, `fts`, `since`, `store`,
`checkpoint`) sent to it over a local socket, so a command neither opens the
database nor loads a model on each call. Unix only.

> **Started by hand only, and nothing forwards to it yet.** The CLI and the
> hooks still run every command in their own process, and each one first
> leases the database from the daemon ([below](#leasing-the-database)), so
> the daemon is cold again after every command: start one to try it, and stop
> it when you are done.

## Commands

```bash
axil daemon start [--idle-timeout 30m] [--wait 30]   # start in the background
axil daemon status [--json]                         # what it is doing, or how it ended
axil daemon logs [-n 50]                            # the end of its log
axil daemon stop [--wait 15]                        # stop it and wait until it has exited
```

`start` does nothing when a daemon already serves the database, and waits for
one that is still starting, or for one still shutting down to go. It prints
the daemon's pid and build id as JSON. `stop` waits for the daemon to exit and
says whether it closed the database (`"clean"`); one that couldn't (see below)
leaves the next open to repair it. A daemon that holds the lock but answers
nothing is sent SIGTERM, once `stop` has checked that the process is this
database's daemon. `status` works whether or not a daemon runs: it reads what
the last one recorded and probes the lock and the socket. While one runs it
adds the idle time left, the requests in flight, each command's p50/p95
latency over its last 256 requests, the reloads and why, the processes
holding a lease, and the state of the engines and the graph.

`AXIL_DAEMON=off` keeps `start` from starting one.

## Leasing the database

Only one process can have a database open, and a running daemon has its
database open. So every other `axil` process asks the daemon to let go before
it opens the database (or reads the vector store ahead of the open): the
daemon stops serving, answering commands sent meanwhile `leased`, waits for
the requests it is running and for its warm-up, closes the database, and the
process goes on to open it as if no daemon ran. A command that needs a store
the daemon's handle lacks (`fts` with no full-text index yet, `since` with no
time-series store, `store --embed` with no vector store) is handed back
unrun; run in its own process, it leases the database and creates the store,
and the daemon serves the command once it has opened the database again.

The lease lasts as long as the process. It holds a connection to the socket
that it never reads again and that the programs it starts don't inherit, so
it ends when the process exits, however it exits: killed with SIGKILL, say.
250 ms after the last lease ends, the daemon opens the database again and
warms it up, unless another process asked for a lease meanwhile. `axil mcp`
and `axil serve` hold theirs for their whole life; `daemon status` lists the
holders (`leased to: pid 4242 for 30s`).

While a lease is held, the process holding it has the database open, as
without a daemon: a hook's recall finds it held and returns no context at
once, and another command waits for it as it would for any process.

A daemon that can't let go within 10 s, because a request it runs takes
longer, says so (`lease_timeout`) and serves on; the process prints one line
saying the daemon has the database open and that `axil daemon stop` frees
it, then opens the database as it would beside any process that has it
open: a write waits up to 10 s for the lock, and a read retries briefly and
then reports it busy. A process asks once per database while its lease
holds: again once the daemon that granted it has exited (another may have
started since), 30 s after a refusal, and at every open while no daemon
runs.

A lease runs nothing for the process that asks, so it is granted whatever the
asker's build, features or environment, with the same socket checks as a
command (see below). A daemon too old to know leases is asked to stop
instead. With `AXIL_DAEMON=off` no lease is asked for, and a command finds a
running daemon's database busy. `compact --drop-engine`, `reembed` and
`branch delete` move or remove files before anything opens the database, so
they ask first; a store removed by anything else is found gone, and the
daemon reloads.

## Files

Everything sits in a directory of the database's own under `run/` beside it
(`.axil/run/memory.axil/`), so two databases in one directory, such as a
branch beside its trunk, each have their own daemon. Both directories are
created `0700` and refused if another user owns them:

| File | What it is |
|------|------------|
| `daemon.lock` | Held by the daemon for its whole life. Of several started at once, one gets it; the others exit. |
| `axil.sock` | The socket (`0600`). When its path would be 100 bytes or more, too long for a socket address, it moves to `axil-<uid>/<hash>.sock` in this user's own temporary directory on macOS (the one `getconf DARWIN_USER_TEMP_DIR` prints), in `/run/user/<uid>` on Linux when that is the user's and private, and in `/tmp` otherwise. None of it comes from the environment, so every client finds the same socket. |
| `daemon.json` | What the daemon last recorded about itself: pid, build, state (`starting`, `ready`, `reloading`, `leasing`, `leased`, `stopping`, `stopped`), and why it stopped. |
| `daemon.log` | Its own output, rotated to `daemon.log.1` at 1 MiB. A command's output goes to the client that sent it; only the few warnings printed inside recall's ranking (such as `recall ran without vector search`) land here instead. |
| `daemon.blocked` | Why it would not serve the database. `start` tries again; a daemon that then serves removes it. |
| `spawn.stamp` | When `start` last spawned one. |

## What it serves, and to whom

A client connects only to a socket in a directory of its own user's that no
one else can enter, and only to a daemon running as that user; the daemon
drops a connection from anyone else. A command is served only to a client of the exact same binary (path, inode,
size and modification time), with the same compiled features and the same
values of the environment variables that change what a command prints or
reads: `AXIL_ENC_KEY`, `AXIL_ENC_KEY_FILE` (and the key file's size and
modification time), `AXIL_LLM_API_KEY`, `AXIL_VERBOSE`, `AXIL_EMBED_SESSIONS`,
`HOME`, `USERPROFILE`, `XDG_CONFIG_HOME`, `APPDATA`, `ORT_DYLIB_PATH`,
`CARGO_HOME`, `GOPATH` and `GOMODCACHE`. The daemon clears `AXIL_VERBOSE` and
`AXIL_EMBED_SESSIONS` from its own environment, so a client with either set
runs the command itself. Anything else, or a command the daemon doesn't run,
is handed back to the client unrun.

Vector search ranks as the one-shot command does: up to the same store size it
scans every vector instead of building the approximate graph a long-lived
process could build once.

## When it reloads, and when it stops

Before a request, again before a write once it is next in line, and once a
second, the daemon checks whether the files behind its open handle changed
without it: the config file it reads, the inode of the database or of any
companion store (a backup renamed over it, a store removed by `compact
--drop-engine` or created by another command), and the plugins directory. Any
change, or a graph that failed to load, makes it finish the requests running
on the old handle and open the database again. Until it has, every command is
handed back unrun, so a write never lands in a file that is no longer the
database.

A store written over in place, its inode kept (`cp backup.axil
.axil/memory.axil`), can't be reopened: the daemon stops at the next check
without closing the database, since closing would write its view of the old
file over the copy. The check compares each store's size and modification
time while no command runs, so a copy made while one runs can be missed, and
a read that comes within 250 ms of the last check may run on the copy before
it is found. **Stop the daemon before restoring a backup in place**
(renaming one over the database is always noticed).

It stops by itself after its idle timeout with no command to run (counting
time the machine slept), on SIGTERM, SIGINT or SIGHUP, and when its
executable changes on disk: a new build installed over it, or the path it was
started by, a symlink say, now leading to another build. Stopping records
`stopping`, removes the socket, waits up to 10 s for the requests running and
for a warm-up or reload in progress, and closes the database. If something
still holds it then, the daemon exits without closing it, and says so in
`daemon.json` and to `stop`; the next open repairs it.

It refuses to serve, and says why in `daemon.blocked`, a database whose open
prints a warning, opens with an engine missing or read-only, or whose warm-up
prints one: each command then runs in its own process and shows the warning
every time, until it is fixed. A store left needing repair by a killed process
is repaired at open, and the repair goes to the daemon's log. After a command
panics the daemon exits without closing the database, as a crashed one-shot
command would; the next open repairs it.
