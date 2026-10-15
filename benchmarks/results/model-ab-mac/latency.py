"""One-shot CLI recall latency per model snapshot: the prompt hook's call
(context-block, budget 2000, top-k 5) for each dogfood question, after one
warm-up, on a temp copy. Interleaves the models question by question so load
swings hit every arm alike. usage: latency.py <axil> <out.json> <label=snapdir>..."""
import json, os, shutil, statistics, subprocess, sys, tempfile, time
from pathlib import Path

axil, out = sys.argv[1], sys.argv[2]
arms = dict(a.split("=", 1) for a in sys.argv[3:])
questions = [json.loads(l) for l in open(Path(__file__).resolve().parents[3] / "benchmarks/dogfood-recall/questions.jsonl") if l.strip()]
env = dict(os.environ, AXIL_SLOW_QUERY_LOG="0")

def load():
    return os.getloadavg()

with tempfile.TemporaryDirectory() as tmp:
    dbs = {}
    for label, snap in arms.items():
        d = Path(tmp) / label
        shutil.copytree(snap, d)
        dbs[label] = d / "memory.axil"
    def run(db, q):
        t = time.perf_counter()
        r = subprocess.run([axil, "--db", str(db), "recall", q, "--recall-format", "context-block",
                            "--budget", "2000", "--top-k", "5"], capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise SystemExit(r.stderr[-300:])
        return (time.perf_counter() - t) * 1000
    for label, db in dbs.items():
        run(db, questions[0]["question"])
    times = {label: [] for label in dbs}
    loads = [load()]
    for q in questions:
        for label, db in dbs.items():
            times[label].append(run(db, q["question"]))
        loads.append(load())
report = {"command": "axil recall <q> --recall-format context-block --budget 2000 --top-k 5 (one-shot)",
          "questions": len(questions),
          "load_1m": {"min": min(l[0] for l in loads), "max": max(l[0] for l in loads),
                      "mean": statistics.mean(l[0] for l in loads)},
          "ms": {label: {"p50": statistics.median(t), "p90": sorted(t)[int(0.9 * (len(t) - 1))],
                         "mean": statistics.mean(t), "min": min(t), "max": max(t)} for label, t in times.items()},
          "per_question_ms": times}
Path(out).write_text(json.dumps(report, indent=1) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "per_question_ms"}, indent=1))
