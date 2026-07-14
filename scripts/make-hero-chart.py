#!/usr/bin/env python3
"""make-hero-chart.py — render the README hero benchmark figure.

Three self-contained panels, each a single-metric magnitude comparison
(no dual-axis), measured on axil v2.1.1:

  1. Fewer context tokens  — % reduction to reach the SAME correct answer,
     per repo, sorted by size (benchmarks/results/context-ab/...).
  2. Fewer steps to finish — % fewer tool roundtrips, same A/B.
  3. Finds the right code — per-repo retrieval recall on the same tasks:
     ground-truth answer file surfaced by the agents' recorded Axil lookups
     (mechanical replay, index cap 512 KB).

Every bar traces to a committed artifact (numbers-integrity policy):
  - tokens/steps: benchmarks/results/context-ab/context-ab-2026-07-13.json
  - recall replay: benchmarks/results/context-ab/code-recall-agent-queries-512k-2026-07-13.json
    (default-config run + verbatim-question diagnostic committed alongside)

Renders light + dark PNGs for a GitHub <picture> element.
Palette = dataviz skill validated defaults (blue slot-1, red slot-6 diverging).
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "benchmarks/results/context-ab/context-ab-2026-07-13.json")
RECALL_ART = os.path.join(
    ROOT, "benchmarks/results/context-ab/code-recall-agent-queries-methodproxies-2026-07-14.json")
OUT_DIR = os.path.join(ROOT, "assets")

# --- data (traceable to committed artifacts) ---
art = json.load(open(ART))
order = ["flask", "fastapi", "django"]  # small -> large: bars grow with repo size
by = {r["corpus"]: r for r in art["per_repo"]}
repo_label = {"flask": "Flask\n24 files",
              "fastapi": "FastAPI\nmid-size",
              "django": "Django\n906 files"}
tokens = [(repo_label[c], by[c]["token_reduction_pct"]) for c in order]
steps = [(repo_label[c], by[c]["step_reduction_pct"]) for c in order]
# Per-repo retrieval recall on the SAME tasks: did the ground-truth answer
# file surface in the agents' recorded Axil lookups (mechanical replay)?
recall_art = json.load(open(RECALL_ART))
recall_by = {c["corpus"]: c for c in recall_art["per_corpus"]}
recall = [(repo_label[c], recall_by[c]["recall"] * 100) for c in order]
recall_n = " / ".join(str(recall_by[c]["tasks_scored"]) for c in order)

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", sec="#52514e", muted="#898781",
                  grid="#e1e0d9", accent="#2a78d6", loss="#e34948", neutral="#b6b5ae"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", sec="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", accent="#3987e5", loss="#e66767", neutral="#55554f"),
}


def panel(ax, data, t, heading, subtitle, *, xmin, xmax,
          highlight=None, decimals=0):
    labels = [d[0] for d in data]
    vals = [d[1] for d in data]
    y = list(range(len(data)))[::-1]  # first item on top
    colors = []
    for i, v in enumerate(vals):
        if highlight is not None:
            colors.append(t["accent"] if labels[i] == highlight else t["neutral"])
        else:
            colors.append(t["loss"] if v < 0 else t["accent"])
    ax.barh(y, vals, height=0.60, color=colors, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10, color=t["sec"])
    ax.set_xlim(xmin, xmax)
    ax.axvline(0, color=t["muted"], lw=1.0, zorder=2)
    ax.xaxis.grid(True, color=t["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", colors=t["muted"], labelsize=8, length=0)
    ax.tick_params(axis="y", length=0)
    ax.set_xticklabels([])  # values are direct-labeled; axis ticks are noise

    span = xmax - xmin
    off = span * 0.02
    for yi, v in zip(y, vals):
        txt = f"−{abs(v):.{decimals}f}%" if v < 0 else f"{v:.{decimals}f}%"
        if v >= 0:
            # label just past the bar tip
            ax.text(v + off, yi, txt, va="center", ha="left",
                    fontsize=12, fontweight="bold", color=t["ink"])
        else:
            # negative bar grows left; place label in the clear space right of 0
            ax.text(off, yi, txt, va="center", ha="left",
                    fontsize=12, fontweight="bold", color=t["loss"])

    ax.text(0.0, 1.15, heading, transform=ax.transAxes, fontsize=12.5,
            fontweight="bold", color=t["ink"], ha="left")
    ax.text(0.0, 1.04, subtitle, transform=ax.transAxes, fontsize=8.2,
            color=t["muted"], ha="left")


def render(mode):
    t = THEMES[mode]
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))
    fig.patch.set_facecolor(t["surface"])
    for ax in axes:
        ax.set_facecolor(t["surface"])

    panel(axes[0], tokens, t, "①  Fewer tokens",
          "% fewer context tokens for the same answer  ·  pooled over two runs",
          xmin=0, xmax=108)
    panel(axes[1], steps, t, "②  Fewer steps",
          "% fewer tool roundtrips to finish  ·  Axil wins on every repo",
          xmin=0, xmax=55)
    panel(axes[2], recall, t, "③  Finds the right code",
          "answer file surfaced by the agent's Axil lookups (replay)",
          xmin=0, xmax=118)

    fig.suptitle("What Axil buys your agent — measured on v2.1.x",
                 x=0.012, y=0.975, ha="left", fontsize=14.5, fontweight="bold",
                 color=t["ink"])
    n_str = " / ".join(str(by[c]["tasks_counted"]) for c in order)
    fig.text(0.012, 0.055,
             "①② Equal-correctness coding A/B on v2.1.1 (2026-07-13): grep-only vs Axil-only Opus agents, identical "
             f"sandboxes, same questions; only both-correct tasks counted (n = {n_str}).",
             fontsize=7.4, color=t["muted"], ha="left")
    fig.text(0.012, 0.022,
             f"③ Mechanical replay of the agents' recorded queries (n = {recall_n}) on the released index "
             "(axil-indexer 2.2.0: method-level proxies, 512 KB default, ignore-boundary fix).   "
             "Method + data: benchmarks/context-ab/ · benchmarks/results/",
             fontsize=7.4, color=t["muted"], ha="left")

    fig.subplots_adjust(left=0.075, right=0.985, top=0.74, bottom=0.16, wspace=0.5)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"hero-benchmarks-{mode}.png")
    fig.savefig(out, dpi=200, facecolor=t["surface"])
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    for m in ("light", "dark"):
        render(m)
