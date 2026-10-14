#!/usr/bin/env python3
"""Where the U1 judge can run: load time, memory and time per pair.

Drives the `u1_footprint` example of `axil-decide` in two modes and,
optionally, the real `axil worker run` review pass:

* one-shot: a fresh process loads the model, scores one pair and exits, the
  way a CLI command that judged in the foreground would. Run several times.
* host: one process loads the model once and scores every U1 pair, twice,
  the way a long-lived host (the MCP server, a daemon) would.
* worker (with --axil, a build with the `decide` feature): the same database
  of U1 records, `axil worker run` with `[decide] review = true` against a copy
  with it off. The difference is what the review pass costs in the process
  that runs it.

Wall time is measured around each process; peak resident memory comes from
`/usr/bin/time` (`-l` on macOS, `-v` on Linux), and on macOS also the peak
memory footprint, which counts compressed pages that resident size drops
when other processes press on memory. The 1-minute load average is recorded
before and after each step, because a busy machine moves every number here.

    cargo build --release -p axil-decide --example u1_footprint
    cargo build --release -p axildb --features decide      # for --axil
    python3 benchmarks/typed-decisions/measure_u1_footprint.py \\
        --probe target/release/examples/u1_footprint \\
        --axil target/release/axil --out results.json
"""

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def loadavg():
    try:
        return round(os.getloadavg()[0], 1)
    except OSError:
        return None


def timed(cmd, cwd=None):
    """Run cmd under /usr/bin/time; return (stdout, wall_ms, peak_kb), where
    peak_kb holds `rss` and, on macOS, `footprint`."""
    darwin = sys.platform == "darwin"
    wrapper = ["/usr/bin/time", "-l" if darwin else "-v"]
    started = time.perf_counter()
    proc = subprocess.run(wrapper + cmd, capture_output=True, text=True, cwd=cwd)
    wall_ms = (time.perf_counter() - started) * 1e3
    if proc.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)} failed:\n{proc.stderr[-2000:]}")
    peak_kb = {}
    if darwin:
        m = re.search(r"(\d+)\s+maximum resident set size", proc.stderr)
        peak_kb["rss"] = int(m.group(1)) // 1024 if m else None
        m = re.search(r"(\d+)\s+peak memory footprint", proc.stderr)
        peak_kb["footprint"] = int(m.group(1)) // 1024 if m else None
    else:
        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", proc.stderr)
        peak_kb["rss"] = int(m.group(1)) if m else None
    return proc.stdout, wall_ms, peak_kb


def summary(values):
    values = sorted(values)
    return {
        "n": len(values),
        "median": round(statistics.median(values), 1),
        "min": round(values[0], 1),
        "max": round(values[-1], 1),
    }


def one_shot(probe, runs, threads):
    rows = []
    for _ in range(runs):
        before = loadavg()
        out, wall_ms, peak_kb = timed([probe, "one-shot", "--threads", str(threads)])
        row = json.loads(out.strip().splitlines()[-1])
        row.update(
            process_wall_ms=round(wall_ms, 1),
            peak_rss_kb=peak_kb["rss"],
            peak_footprint_kb=peak_kb.get("footprint"),
            loadavg_1m=[before, loadavg()],
        )
        rows.append(row)
    return {
        "what": "fresh process: load (sha256 check + session + tokenizer), one pair, exit",
        "runs": rows,
        "process_wall_ms": summary([r["process_wall_ms"] for r in rows]),
        "load_ms": summary([r["load_ms"] for r in rows]),
        "first_pair_ms": summary([r["first_pair_ms"] for r in rows]),
        "peak_rss_mb": summary([r["peak_rss_kb"] / 1024 for r in rows]),
        "peak_footprint_mb": summary(
            [r["peak_footprint_kb"] / 1024 for r in rows if r["peak_footprint_kb"]] or [0]
        ),
        "rss_start_mb": summary([r["rss_start_kb"] / 1024 for r in rows]),
    }


def host(probe, rounds, threads):
    before = loadavg()
    out, wall_ms, peak_kb = timed(
        [probe, "host", "--rounds", str(rounds), "--threads", str(threads)]
    )
    row = json.loads(out.strip().splitlines()[-1])
    row.update(
        process_wall_ms=round(wall_ms, 1),
        peak_rss_kb=peak_kb["rss"],
        peak_footprint_kb=peak_kb.get("footprint"),
        loadavg_1m=[before, loadavg()],
    )
    row["what"] = (
        "one process: load once, 5 warm-up pairs, then every U1 pair one at a "
        "time per round; RSS read with ps after each round"
    )
    return row


def store(axil, db, table, text):
    subprocess.run(
        [axil, "--db", str(db), "store", table, json.dumps({"summary": text})],
        check=True,
        capture_output=True,
    )


def worker(axil, sets, max_records, settle_s):
    """`axil worker run` with review on, against a copy with it off. Storing
    the records runs one embedding process per record, so the timed runs wait
    `settle_s` seconds for the load that adds to fade."""
    rows = []
    for name in sets:
        path = HERE / "sets" / name
        rows += [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    olds = list(dict.fromkeys(r["old"] for r in rows))
    news = [r["new"] for r in rows]
    tmp = Path(tempfile.mkdtemp(prefix="u1-worker-", dir=os.environ.get("SCRATCH")))
    try:
        base = tmp / "base"
        (base / ".axil").mkdir(parents=True)
        db = base / ".axil" / "memory.axil"
        subprocess.run([axil, "init", str(db)], check=True, capture_output=True)
        for text in olds + news:
            store(axil, db, "decisions", text)
        time.sleep(settle_s)
        results = {}
        for review in (False, True):
            copy = tmp / ("on" if review else "off")
            shutil.copytree(base, copy)
            (copy / "axil.toml").write_text(
                "[decide]\n"
                f"review = {'true' if review else 'false'}\n"
                f"max_records = {max_records}\n"
            )
            before = loadavg()
            out, wall_ms, peak_kb = timed(
                [axil, "--db", str(copy / ".axil" / "memory.axil"), "worker", "run"],
                cwd=copy,
            )
            report = json.loads(out)
            results["review_on" if review else "review_off"] = {
                "process_wall_ms": round(wall_ms, 1),
                "peak_rss_kb": peak_kb["rss"],
                "peak_footprint_kb": peak_kb.get("footprint"),
                "loadavg_1m": [before, loadavg()],
                "contradiction_review": report.get("contradiction_review"),
            }
        on, off = results["review_on"], results["review_off"]
        results["review_cost"] = {
            "wall_ms": round(on["process_wall_ms"] - off["process_wall_ms"], 1),
            "peak_rss_mb": round((on["peak_rss_kb"] - off["peak_rss_kb"]) / 1024, 1),
        }
        if on["peak_footprint_kb"] and off["peak_footprint_kb"]:
            results["review_cost"]["peak_footprint_mb"] = round(
                (on["peak_footprint_kb"] - off["peak_footprint_kb"]) / 1024, 1
            )
        results["database"] = {
            "sets": sets,
            "records": len(olds) + len(news),
            "max_records": max_records,
            "what": "every distinct U1 old text, then every new text, stored as decisions with "
            "the default embedder; both runs on byte-identical copies",
        }
        return results
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--probe", required=True, help="the u1_footprint example binary")
    ap.add_argument("--axil", help="an axil built with the decide feature")
    ap.add_argument("--runs", type=int, default=7, help="one-shot runs")
    ap.add_argument("--rounds", type=int, default=2, help="host rounds")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--worker-sets", default="u1_repo.jsonl")
    ap.add_argument("--max-records", type=int, default=40)
    ap.add_argument(
        "--modes", default="one-shot,host,worker", help="comma-separated subset to run"
    )
    ap.add_argument(
        "--settle-s", type=int, default=0, help="worker: wait after storing, before timing"
    )
    ap.add_argument("--note", default="", help="caveat recorded with the numbers")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report = {
        "benchmark": "typed-decisions-u1-where-it-runs",
        "date": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}, "
        f"{os.cpu_count()} cores",
        "note": args.note,
        "threads": args.threads,
    }
    modes = args.modes.split(",")
    if "one-shot" in modes:
        report["one_shot"] = one_shot(args.probe, args.runs, args.threads)
    if "host" in modes:
        report["host"] = host(args.probe, args.rounds, args.threads)
    if "worker" in modes and args.axil:
        report["worker"] = worker(
            args.axil, args.worker_sets.split(","), args.max_records, args.settle_s
        )
    text = json.dumps(report, indent=2)
    Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
