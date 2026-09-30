#!/usr/bin/env python3
"""boot-tokens.py — how big is what `axil boot` injects, old binary vs new?

Runs each boot surface agents see on a copy of a database and records the
output size in bytes and in estimated tokens, `ceil(bytes / 4)`. That is the
same bytes/4 heuristic Axil's own budgets use; it is an estimate, not a
tokenizer count.

Surfaces:
  narrative   axil boot --boot-format narrative
  json        axil boot
  schema      axil boot --schema v2   (--schema v1 on a binary that predates schema 2)
  hook        axil boot --boot-format narrative --budget 800   (the SessionStart hook's exact call)
  mcp_boot    MCP tools/call boot {}                            (text of the tool result)

Each binary reads its own fresh copy of the snapshot, so neither run can see
the other's writes and the snapshot itself is never opened.

Usage:
  python3 scripts/boot-tokens.py \\
      --snapshot benchmarks/dogfood-recall/data/snap-prefix \\
      --old /path/to/old/axil --new ./target/release/axil \\
      --out benchmarks/results/boot-tokens-<date>.json
"""
import argparse
import datetime
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SURFACES = {
    "narrative": ["boot", "--boot-format", "narrative"],
    "json": ["boot"],
    "schema": ["boot", "--schema", "<latest>"],
    "hook": ["boot", "--boot-format", "narrative", "--budget", "800"],
}

MCP_FRAMES = "\n".join(
    [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "boot", "arguments": {}},
            }
        ),
    ]
) + "\n"


def est_tokens(n_bytes: int) -> int:
    return math.ceil(n_bytes / 4)


def copy_snapshot(src: str, dst: str) -> None:
    """Clone on APFS (`cp -c`) when possible; fall back to a plain copy."""
    if sys.platform == "darwin":
        if subprocess.run(["cp", "-c", "-R", src, dst], capture_output=True).returncode == 0:
            return
    shutil.copytree(src, dst)


def find_db(snapshot_dir: str) -> str:
    for name in sorted(os.listdir(snapshot_dir)):
        if name.endswith(".axil"):
            return name
    sys.exit(f"no *.axil file in {snapshot_dir}")


def budget_fields(stdout: str) -> dict:
    """The budget bookkeeping a JSON boot reports about itself, if any."""
    try:
        v = json.loads(stdout)
    except json.JSONDecodeError:
        return {}
    if not isinstance(v, dict):
        return {}
    keys = (
        "schema_version",
        "token_budget",
        "token_budget_used",
        "dropped_sections",
        "omitted_items",
        "omitted_by_section",
    )
    return {k: v[k] for k in keys if k in v}


def latest_schema(axil: str) -> str:
    """The newest `--schema` value this binary accepts: `v2` since schema 2,
    `v1` before it (v1 is refused once v2 exists)."""
    p = subprocess.run([axil, "boot", "--help"], capture_output=True, text=True)
    in_schema = False
    for line in p.stdout.splitlines():
        if line.strip().startswith("--"):
            in_schema = line.strip().startswith("--schema")
        elif in_schema and "[possible values:" in line:
            values = line.split("[possible values:", 1)[1].strip(" ]").split(", ")
            return max(values)
    return "v1"


def measure(axil: str, db: str, cwd: str) -> dict:
    out = {}
    schema = latest_schema(axil)
    for name, args in SURFACES.items():
        args = [schema if a == "<latest>" else a for a in args]
        p = subprocess.run(
            [axil, "--db", db, *args], cwd=cwd, capture_output=True, timeout=300
        )
        text = p.stdout.decode("utf-8", errors="replace")
        n = len(p.stdout)
        out[name] = {
            "cmd": "axil " + " ".join(args),
            "exit": p.returncode,
            "bytes": n,
            "est_tokens": est_tokens(n),
            **budget_fields(text),
        }

    p = subprocess.run(
        [axil, "--db", db, "mcp"],
        cwd=cwd,
        input=MCP_FRAMES.encode(),
        capture_output=True,
        timeout=300,
    )
    text = ""
    for line in p.stdout.decode("utf-8", errors="replace").splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("id") == 2:
            content = msg.get("result", {}).get("content") or [{}]
            text = content[0].get("text", "")
    n = len(text.encode())
    out["mcp_boot"] = {
        "cmd": "MCP tools/call boot {}",
        "exit": p.returncode,
        "bytes": n,
        "est_tokens": est_tokens(n),
        **budget_fields(text),
    }
    return out


def version(axil: str) -> str:
    p = subprocess.run([axil, "--version"], capture_output=True, text=True)
    return p.stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--snapshot",
        default=os.path.join(ROOT, "benchmarks", "dogfood-recall", "data", "snap-prefix"),
        help="directory holding the database and its companion files",
    )
    ap.add_argument(
        "--snapshot-label",
        help="name to record for the snapshot (default: its path relative to the repo, or its basename)",
    )
    ap.add_argument("--old", required=True, help="baseline axil binary")
    ap.add_argument(
        "--new", default=os.path.join(ROOT, "target", "release", "axil"), help="axil binary under test"
    )
    ap.add_argument("--out", help="write the JSON result here as well as to stdout")
    args = ap.parse_args()

    snapshot = os.path.abspath(args.snapshot)
    db_name = find_db(snapshot)
    runs = {}
    with tempfile.TemporaryDirectory(prefix="boot-tokens-") as tmp:
        for label, axil in (("old", args.old), ("new", args.new)):
            copy = os.path.join(tmp, label)
            copy_snapshot(snapshot, copy)
            runs[label] = {
                "binary": version(axil),
                "surfaces": measure(os.path.abspath(axil), os.path.join(copy, db_name), copy),
            }

    change = {
        name: {
            "old_est_tokens": runs["old"]["surfaces"][name]["est_tokens"],
            "new_est_tokens": runs["new"]["surfaces"][name]["est_tokens"],
        }
        for name in runs["new"]["surfaces"]
    }
    result = {
        "what": "size of each `axil boot` surface on one database, old binary vs new",
        "date": datetime.date.today().isoformat(),
        "snapshot": args.snapshot_label
        or (os.path.relpath(snapshot, ROOT) if snapshot.startswith(ROOT) else os.path.basename(snapshot)),
        "token_estimate": "ceil(bytes / 4) of the output (Axil's bytes/4 budget heuristic), not a tokenizer count",
        "runs": runs,
        "est_tokens": change,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
