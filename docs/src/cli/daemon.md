# Warm Daemon

`axil daemon` runs one long-lived process per database. It keeps the database
open, with the embedding model and the graph loaded, and runs the hot commands
(`recall`, `boot`, `recall-for-file`, `code-search`, `fts`, `since`, `store`,
`checkpoint`) sent to it over a local socket, so a command neither opens the
database nor loads a model on each call. Unix only.

> **Started by hand only, and nothing forwards to it yet.** The CLI and the
> hooks still run every command in their own process. While a daemon runs it
> holds the database's writer lock, so those commands find the database busy:
> start one to try it, and stop it when you are done.

## Commands

```bash
axil daemon start [--idle-timeout 30m] [--wait 30]   # start in the background
axil daemon status [--json]                         # what it is doing, or how it ended
axil daemon logs [-n 50]                            # the end of its log
axil daemon stop [--wait 15]                        # stop it and wait until it has
```

`start` does nothing when a daemon already serves the database, and waits for
one that is still starting. It prints the daemon's pid and build id as JSON.
`status` works whether or not a daemon runs: it reads what the last one
recorded and probes the lock and the socket. While one runs it adds the idle
time left, the requests in flight, each command's p50/p95 latency over its
last 256 requests, the reloads and why, and the state of the engines and the
graph.

`AXIL_DAEMON=off` keeps `start` from starting one.

## Files

Everything sits in a directory of the database's own under `run/` beside it
(`.axil/run/memory.axil/`), so two databases in one directory, such as a
branch beside its trunk, each have their own daemon. Both directories are
created `0700` and refused if another user owns them:

| File | What it is |
|------|------------|
| `daemon.lock` | Held by the daemon for its whole life. Of several started at once, one gets it; the others exit. |
| `axil.sock` | The socket (`0600`). When its path would be 100 bytes or more, too long for a socket address, it moves to `axil-<uid>/<hash>.sock` in this user's own temporary directory on macOS (the one `getconf DARWIN_USER_TEMP_DIR` prints), in `/run/user/<uid>` on Linux when that is the user's and private, and in `/tmp` otherwise. None of it comes from the environment, so every client finds the same socket. |
| `daemon.json` | What the daemon last recorded about itself: pid, build, state, and why it stopped. |
| `daemon.log` | Its own output, rotated to `daemon.log.1` at 1 MiB. A command's output goes to the client that sent it; only the few warnings printed inside recall's ranking (such as `recall ran without vector search`) land here instead. |
| `daemon.blocked` | Why it would not serve the database. `start` clears it and tries again. |
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

Before a request, and once a second, the daemon checks whether the files
behind its open handle changed without it: the config file it reads, the
inode of the database or of any companion store (a backup renamed over it, a
store removed by `compact --drop-engine` or created by another command), and
the plugins directory. Any change, or a graph that failed to load, makes it
finish the requests running on the old handle and open the database again.

It stops by itself after its idle timeout with no command to run, on
SIGTERM, SIGINT or SIGHUP, and when its own executable changes on disk (a new
build installed over it). Stopping removes the socket first, finishes the
requests running, and closes every store cleanly.

It refuses to serve, and says why in `daemon.blocked`, a database whose open
prints a warning, opens with an engine missing or read-only, or whose warm-up
prints one: each command then runs in its own process and shows the warning
every time, until it is fixed. A store left needing repair by a killed process
is repaired at open, and the repair goes to the daemon's log. After a command
panics the daemon exits without closing the database, as a crashed one-shot
command would; the next open repairs it.
