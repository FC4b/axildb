#!/usr/bin/env python3
"""mcp-tools-tokens.py — how big is the MCP server's `tools/list`, per tool profile?

Starts `axil mcp` on a fresh scratch database, sends `initialize` and
`tools/list`, and records the size of the `tools/list` response line as the
client receives it, in bytes and in estimated tokens, `ceil(bytes / 4)`. That
is the same bytes/4 heuristic Axil's own budgets use; it is an estimate, not a
tokenizer count. A client that does not cache tool definitions pays this on
every request; the `initialize` instructions are paid once per session and are
recorded separately.

Profiles measured: `core` (the default) and `full` (`--tools full`). A binary
that predates tool profiles is measured once, as `unprofiled`.

Usage:
  python3 scripts/mcp-tools-tokens.py --new ./target/release/axil \\
      [--old /path/to/baseline/axil] \\
      --out benchmarks/results/mcp-tools-tokens-<date>.json
"""
import argparse
import datetime
import json
import math
import os
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FRAMES = "\n".join(
    json.dumps(frame)
    for frame in (
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )
) + "\n"


def est_tokens(n_bytes: int) -> int:
    return math.ceil(n_bytes / 4)


def clean_env() -> dict:
    """The caller's environment without the variables that would pick the
    database or the profile behind the script's back."""
    env = dict(os.environ)
    for key in ("AXIL_DB", "AXIL_MCP_TOOLS"):
        env.pop(key, None)
    return env


def supports_profiles(axil: str) -> bool:
    p = subprocess.run([axil, "mcp", "--help"], capture_output=True, text=True, env=clean_env())
    return "--tools" in p.stdout


def measure(axil: str, db: str, profile) -> dict:
    args = [axil, "--db", db, "mcp"]
    if profile:
        args += ["--tools", profile]
    p = subprocess.run(
        args, input=FRAMES.encode(), capture_output=True, timeout=300, env=clean_env()
    )
    lines = {}
    for raw in p.stdout.splitlines():
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        lines[msg.get("id")] = (raw, msg)
    if 1 not in lines or 2 not in lines:
        raise SystemExit(f"{' '.join(args)}: no initialize/tools/list reply\n{p.stderr.decode()}")
    list_raw, list_msg = lines[2]
    instructions = lines[1][1]["result"].get("instructions") or ""
    tools = list_msg["result"]["tools"]
    per_tool = {
        t["name"]: len(json.dumps(t, ensure_ascii=False, separators=(",", ":")).encode())
        for t in tools
    }
    return {
        "cmd": " ".join(["axil", "--db", "<scratch>", *args[3:]]),
        "exit": p.returncode,
        "tools": len(tools),
        "names": [t["name"] for t in tools],
        "tools_list_bytes": len(list_raw),
        "tools_list_est_tokens": est_tokens(len(list_raw)),
        "instructions_bytes": len(instructions.encode()),
        "instructions_est_tokens": est_tokens(len(instructions.encode())),
        "per_tool_bytes": per_tool,
    }


def version(axil: str) -> str:
    p = subprocess.run([axil, "--version"], capture_output=True, text=True)
    return p.stdout.strip()


def run_binary(axil: str, tmp: str, label: str) -> dict:
    db = os.path.join(tmp, label, "m.axil")
    os.makedirs(os.path.dirname(db))
    init = subprocess.run([axil, "init", db], capture_output=True, env=clean_env())
    if init.returncode != 0:
        raise SystemExit(f"axil init failed: {init.stderr.decode()}")
    if supports_profiles(axil):
        profiles = {name: measure(axil, db, name) for name in ("core", "full")}
    else:
        profiles = {"unprofiled": measure(axil, db, None)}
    return {"binary": version(axil), "profiles": profiles}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--new", default=os.path.join(ROOT, "target", "release", "axil"), help="axil binary under test"
    )
    ap.add_argument("--old", help="baseline axil binary (optional)")
    ap.add_argument("--out", help="write the JSON result here as well as to stdout")
    args = ap.parse_args()

    runs = {}
    with tempfile.TemporaryDirectory(prefix="mcp-tools-tokens-") as tmp:
        for label, axil in (("old", args.old), ("new", args.new)):
            if axil:
                runs[label] = run_binary(os.path.abspath(axil), tmp, label)

    result = {
        "what": "size of the MCP `tools/list` response per tool profile, on a fresh `axil init` database",
        "date": datetime.date.today().isoformat(),
        "token_estimate": "ceil(bytes / 4) of the response line as the client receives it "
        "(Axil's bytes/4 budget heuristic), not a tokenizer count",
        "runs": runs,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
