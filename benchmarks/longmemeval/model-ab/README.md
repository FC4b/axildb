# Embedding-model A/B: the LongMemEval arms run on a second machine

This runs the remaining LongMemEval arms of the embedding-model A/B:
bge-small-en-v1.5 (the default), granite-embedding-small-english-r2 and
gte-modernbert-base, all unprompted, `recall-qtc`, top-k 5, one worker
thread per run. The dogfood eval and the three s20 arms ran on the
development Mac. Their results, and this run's once it is back, go into
`benchmarks/results/embedding-model-ab-2026-10-15.json`.

| Arm | Questions | Default |
|---|---|---|
| `m5-bge`, `m5-granite` | the first 5 of the `m` split (all single-session-assistant) | run |
| `s60-bge`, `s60-granite` | the 60 ids in `s60-question-ids.json`: 10 per question type of the `s` split | run |
| `m5-gte` | the first 5 of the `m` split | only with `--with-gte-m5` |

`m5-gte` runs only when it is asked for. Whoever reads the s20 results
decides whether gte earned it and says whether to pass the flag.

Every arm of one comparison (m5, s60) runs on this one machine, so its
results compare only with each other. Timings from here are never set next to
the Mac's.

## Prerequisites: Ubuntu under WSL2

Keep the clone and the data inside the Linux file system (`~/axildb`), not
under `/mnt/c`: reading the 2.7 GB `m` file through the Windows mount is
slow.

The `m` arms hold the whole `m` file in memory while it loads, about 6 GB at
peak. The script runs them one at a time, but WSL2 gives Linux only half the
Windows RAM by default. Check `free -g`. If it shows less than 12 GB, raise
the limit in `%UserProfile%\.wslconfig` and run `wsl --shutdown`:

```ini
[wsl2]
memory=12GB
```

The load average the script records is the Linux VM's. Windows programs
running beside WSL do not show up in it, so keep the Windows side idle while
it runs.

```bash
sudo apt update
sudo apt install -y build-essential pkg-config libssl-dev curl git ca-certificates time
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
rustup toolchain install 1.92.0     # the toolchain the Mac arms were built with
```

`time` is GNU time: with it, each `.err` file ends with the run's peak RSS.

## 1. Check out the commit

```bash
cd ~/axildb
git fetch origin feat/model-ab
git checkout 5bf1cfd8338c4c5d0c5750d2ac3daf854222a02f
git status        # must be clean: `check` refuses tracked-file changes
```

`5bf1cfd` is the commit that added this runner. Run from it even when the
branch has moved on: it is the code `check` records in every `.meta`. Read
this README from the branch tip, which can carry corrections; this line was
added after the runner itself.

## 2. Fetch the data and the models

```bash
benchmarks/longmemeval/model-ab/run.sh fetch
```

This downloads the files below and installs each one only when its size and
sha256 match.

Data, from the Hugging Face dataset `xiaowu0162/longmemeval-cleaned` at
revision `98d7416c24c778c2fee6e6f3006e7a073259d48f` (MIT), into
`benchmarks/longmemeval/data/`:

| File | Bytes | sha256 |
|---|---|---|
| `longmemeval_s_cleaned.json` | 277,383,467 | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| `longmemeval_m_cleaned.json` | 2,737,100,077 | `9d79e5524794a2e6900a3aa9cb7d9152c5a3e8319c9a87c25494ba1eacee495f` |

The s60 sample is no separate file. `s60-question-ids.json` lists its ids
and the harness takes them from the `s` file (`--question-ids`). The Mac's
copy of that sample holds exactly those 60 questions, unchanged.

Models go into `~/.axil/models/<name>/`, which is where the harness looks.
Each file is fetched from the revision `crates/engines/axil-vector/src/download.rs`
pins, the same bytes `axil model-download <model>` installs. `check`
confirms every sha256 below is still pinned in `download.rs`:

| Model dir | Repo @ revision | File | Bytes | sha256 |
|---|---|---|---|---|
| `bge-small-en-v1.5` | `BAAI/bge-small-en-v1.5` @ `5c38ec7c` | `onnx/model.onnx` | 133,093,490 | `828e1496…940cf35` |
| | | `tokenizer.json` | 711,396 | `d241a60d…1f9e5c66` |
| `granite-embedding-small-english-r2` | `onnx-community/granite-embedding-small-english-r2-ONNX` @ `1dc7835b` | `onnx/model.onnx` | 182,504 | `cddb145c…26c98179e` |
| | | `onnx/model.onnx_data` | 194,844,160 | `86a3a705…7222834b` |
| | | `tokenizer.json` | 2,128,614 | `feeb8334…c2927149` |
| `gte-modernbert-base` | `Alibaba-NLP/gte-modernbert-base` @ `e7f32e3c` | `onnx/model.onnx` | 596,392,315 | `947f31df…e0d24d78` |
| | | `tokenizer.json` | 3,583,228 | `6c8aaa9a…3c08d30` |

The full revisions and sha256s are in `run.sh` (`MODEL_FILES`). The gte
files are needed only for `--with-gte-m5`, but `fetch` and `check` cover
them anyway.

All told, the download is 3.01 GB of data and 0.93 GB of models (0.33 GB
without gte), plus the crates and the ONNX Runtime binaries the first build
fetches.

## 3. Build

```bash
benchmarks/longmemeval/model-ab/run.sh build
```

This copies `Cargo.lock.pinned` to `benchmarks/longmemeval/Cargo.lock`, where
the build reads it. That path is gitignored, so the lockfile lives here under
another name. The script then builds the harness with `cargo +1.92.0 build
--release --locked`, which fails rather than resolve other versions. The
pinned lockfile is the one the Mac's s20 harness was built with
(sha256 `2da16fc4…1bca6e0`). `ort` downloads the ONNX Runtime binaries
during the build, so the build needs network access.

## 4. Check

```bash
benchmarks/longmemeval/model-ab/run.sh check
```

`check` verifies:
- the commit and a clean tree;
- every data and model file by size and sha256;
- that the model pins match `download.rs`;
- that the lockfile is the pinned one;
- that the harness exists, and prints its sha256.

`run` runs `check` first and stops if it fails.

## 5. Run

```bash
benchmarks/longmemeval/model-ab/run.sh run                  # m5 and s60 for bge-small and granite
benchmarks/longmemeval/model-ab/run.sh run --with-gte-m5    # the same, plus m5-gte
```

At most two runs go at once, in two lanes: `m5-bge`, `m5-granite`, then
`m5-gte` when asked for; and `s60-bge`, `s60-granite`. Each arm runs:

```text
longmemeval-bench --variant <m|s> --limit <5|0> --strategy recall-qtc --top-k 5 \
  --model <bge-small|granite-small|gte-modernbert-base> --data-dir benchmarks/longmemeval/data \
  --jobs 1 [--question-ids benchmarks/longmemeval/model-ab/s60-question-ids.json]
```

An arm whose `.json` already exists is skipped, so after an interruption you
can delete the half-written arm's files and run again.

### Expected runtime

This is an estimate on an unloaded 8-core machine, scaled from the Mac's
timings: an M1 Pro at load averages of 40 to 60, halved for the load. Nothing
was measured on x86.
- An `s` question took 48 s with bge-small, 209 s with granite and 519 s
  with gte there, so expect about 25 s, 105 s and 260 s here.
- An `m` question has about ten times as many sessions and costs ten times as
  much.

That comes to about 2.5 to 3 hours without gte, two runs sharing the cores.
`m5-gte` adds about 3.5 to 4 hours.

## 6. Outputs and sending them back

Everything lands in `benchmarks/results/model-ab-pc/`:

| File | What |
|---|---|
| `<arm>.json` | the harness report (`<arm>.failed.json` if it exited non-zero) |
| `<arm>.err` | progress and per-question `[timing]` lines; GNU time's peak RSS at the end |
| `<arm>.meta` | commit, harness and lockfile sha256, toolchain, command, machine (CPU model, cores, RAM, OS, kernel, WSL), start and end time with the 1-minute load average |
| `<arm>.load` | the 1-minute load average every 30 s during the run |

Commit that directory on `feat/model-ab` and push:

```bash
git add benchmarks/results/model-ab-pc
git commit -m "bench(longmemeval): embedding-model A/B arms from <machine>"
git push origin feat/model-ab
```
