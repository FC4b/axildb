#!/usr/bin/env python3
"""Render the LongMemEval recall bar chart for the README.

Axil's two numbers are read live from the committed 500-question baselines
(benchmarks/results/{qtc-500,fusion-500}.json) so the chart can never drift
from the committed benchmark. Competitor figures are cited as published
(LongMemEval landscape, Apr 2026) — not measured by Axil.

Usage:  python scripts/plot-longmemeval.py
Output: assets/longmemeval-recall.png  (+ .svg)
Deps:   matplotlib, numpy
"""
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def axil_recall(name):
    d = json.loads((ROOT / "benchmarks" / "results" / name).read_text())
    return round(d["overall"]["avg_recall"] * 100, 1)


AXIL_QTC = axil_recall("qtc-500.json")
AXIL_FUS = axil_recall("fusion-500.json")

# (label, recall %, group) — competitor figures as published (Apr 2026).
rows = [
    ("MemPalace",            96.6,     "local"),
    ("Axil\nRecall-QTC",     AXIL_QTC, "axil"),
    ("Axil\nRecall-fusion",  AXIL_FUS, "axil"),
    ("Hindsight",            91.4,     "llm"),
    ("Memvid",               85.7,     "local"),
    ("Mem0",                 68.4,     "llm"),
    ("Zep",                  66.0,     "llm"),
]

cmaps = {
    "axil":  LinearSegmentedColormap.from_list("axil",  ["#6d28d9", "#a78bfa"]),
    "local": LinearSegmentedColormap.from_list("local", ["#0f766e", "#5eead4"]),
    "llm":   LinearSegmentedColormap.from_list("llm",   ["#475569", "#cbd5e1"]),
}
solid = {"axil": "#7c3aed", "local": "#14b8a6", "llm": "#64748b"}

fig, ax = plt.subplots(figsize=(9.6, 5.4))
n = len(rows)
bw = 0.64
grad = np.linspace(0, 1, 256).reshape(-1, 1)
for i, (label, v, g) in enumerate(rows):
    ax.imshow(grad, extent=[i - bw / 2, i + bw / 2, 0, v], origin="lower",
              aspect="auto", cmap=cmaps[g], zorder=2, interpolation="bicubic")
    ax.text(i, v + 1.6, f"{v:.1f}", ha="center", va="bottom", fontsize=11.5,
            fontweight="bold" if g == "axil" else "normal",
            color="#5b21b6" if g == "axil" else "#334155")

ax.set_xlim(-0.6, n - 0.4)
ax.set_ylim(0, 105)
ax.set_xticks(range(n))
ax.set_xticklabels([r[0] for r in rows], fontsize=9.8)
for lbl, (_, _, g) in zip(ax.get_xticklabels(), rows):
    if g == "axil":
        lbl.set_fontweight("bold")
        lbl.set_color("#5b21b6")
ax.set_ylabel("Recall @ top-5  (%)", fontsize=10.5, color="#334155")
ax.set_yticks(range(0, 101, 20))
ax.grid(axis="y", color="#e5e7eb", linewidth=1, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#cbd5e1")
ax.tick_params(colors="#64748b")

ax.set_title("LongMemEval — Retrieval Recall @ top-5  (500 questions)",
             fontsize=14.5, fontweight="bold", pad=34, color="#1e293b", loc="left")
ax.text(0, 1.045, "Axil runs with no LLM and no server  ·  higher is better",
        transform=ax.transAxes, fontsize=9.8, color="#64748b")

legend = [
    Patch(facecolor=solid["axil"],  label="Axil — no LLM, no server"),
    Patch(facecolor=solid["local"], label="Local, no LLM (MemPalace, Memvid)"),
    Patch(facecolor=solid["llm"],   label="Needs LLM / server (Hindsight, Mem0, Zep)"),
]
ax.legend(handles=legend, loc="upper right", frameon=False, fontsize=9, handlelength=1.2)
fig.text(0.012, 0.005,
         "Source: in-tree benchmarks/longmemeval — 500-Q LongMemEval-S, bge-small, top-k 5 "
         "(committed baselines qtc-500.json / fusion-500.json, RTX 3080). "
         "Competitor figures as published, Apr 2026.",
         fontsize=6.7, color="#94a3b8")

plt.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(ROOT / "assets" / "longmemeval-recall.png", bbox_inches="tight", facecolor="white", dpi=200)
fig.savefig(ROOT / "assets" / "longmemeval-recall.svg", bbox_inches="tight", facecolor="white")
print(f"saved assets/longmemeval-recall.png (Axil QTC={AXIL_QTC} fusion={AXIL_FUS})")
