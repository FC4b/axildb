#!/usr/bin/env python3
"""Render the LongMemEval memory-system landscape chart for the README.

Two metrics live on this chart and they are NOT interchangeable:
  * retrieval recall@5  — did the answer-bearing session land in the top-5?
                          (no LLM needed; what Axil, MemPalace, Memvid report)
  * end-to-end QA accuracy — did the system answer the question correctly?
                          (needs an LLM; what Hindsight, Mem0, Zep report)
Recall is always >= QA accuracy, so the two groups are drawn separately and the
QA-accuracy bars are hatched. Compare within a group, not across.

Axil's two numbers are read live from the committed 500-question baselines so the
chart can never drift. Competitor figures are cited as published (Apr 2026).

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
plt.rcParams["hatch.linewidth"] = 0.9


def axil_recall(name):
    d = json.loads((ROOT / "benchmarks" / "results" / name).read_text())
    return round(d["overall"]["avg_recall"] * 100, 1)


AXIL_QTC = axil_recall("qtc-500.json")
AXIL_FUS = axil_recall("fusion-500.json")

# (label, value %, color-group, metric) — grouped by metric, sorted within group.
# Competitor figures as published (Apr 2026). recall = retrieval recall@5,
# qa = end-to-end QA accuracy (requires an LLM).
rows = [
    ("MemPalace",            96.6,     "local", "recall"),
    ("Axil\nRecall-QTC",     AXIL_QTC, "axil",  "recall"),
    ("Axil\nRecall-fusion",  AXIL_FUS, "axil",  "recall"),
    ("Memvid",               85.7,     "local", "recall"),
    ("Hindsight",            91.4,     "llm",   "qa"),
    ("Mem0",                 68.4,     "llm",   "qa"),
    ("Zep",                  66.0,     "llm",   "qa"),
]
# x positions: 4 recall bars, a gap, then 3 QA bars.
xs = [0, 1, 2, 3, 5, 6, 7]

cmaps = {
    "axil":  LinearSegmentedColormap.from_list("axil",  ["#6d28d9", "#a78bfa"]),
    "local": LinearSegmentedColormap.from_list("local", ["#0f766e", "#5eead4"]),
    "llm":   LinearSegmentedColormap.from_list("llm",   ["#475569", "#cbd5e1"]),
}
solid = {"axil": "#7c3aed", "local": "#14b8a6", "llm": "#64748b"}

fig, ax = plt.subplots(figsize=(10.2, 5.6))
bw = 0.66
grad = np.linspace(0, 1, 256).reshape(-1, 1)

# subtle group backgrounds
ax.axvspan(-0.62, 3.62, color="#7c3aed", alpha=0.04, zorder=0)
ax.axvspan(4.38, 7.62, color="#64748b", alpha=0.05, zorder=0)

for x, (label, v, g, metric) in zip(xs, rows):
    ax.imshow(grad, extent=[x - bw / 2, x + bw / 2, 0, v], origin="lower",
              aspect="auto", cmap=cmaps[g], zorder=2, interpolation="bicubic")
    if metric == "qa":  # hatch QA-accuracy bars to flag the different metric
        ax.bar(x, v, width=bw, facecolor="none", edgecolor="white",
               hatch="////", linewidth=0, zorder=3)
    ax.text(x, v + 1.6, f"{v:.1f}", ha="center", va="bottom", fontsize=11.5,
            fontweight="bold" if g == "axil" else "normal",
            color="#5b21b6" if g == "axil" else "#334155")

# group header labels
ax.text(1.5, 105, "retrieval recall@5  ·  no LLM, no server", ha="center",
        va="bottom", fontsize=10, fontweight="bold", color="#6d28d9")
ax.text(6.0, 105, "end-to-end QA accuracy  ·  LLM + server", ha="center",
        va="bottom", fontsize=10, fontweight="bold", color="#475569")

ax.set_xlim(-0.7, 7.7)
ax.set_ylim(0, 116)
ax.set_xticks(xs)
ax.set_xticklabels([r[0] for r in rows], fontsize=9.8)
for lbl, (_, _, g, _) in zip(ax.get_xticklabels(), rows):
    if g == "axil":
        lbl.set_fontweight("bold")
        lbl.set_color("#5b21b6")
ax.set_ylabel("score  (%)", fontsize=10.5, color="#334155")
ax.set_yticks(range(0, 101, 20))
ax.grid(axis="y", color="#e5e7eb", linewidth=1, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#cbd5e1")
ax.tick_params(colors="#64748b")

ax.set_title("LongMemEval — memory-system landscape  (500 questions)",
             fontsize=14.5, fontweight="bold", pad=30, color="#1e293b", loc="left")
ax.text(0, 1.05,
        "Two metrics — recall@5 (always higher) vs QA accuracy (hatched). "
        "Purple = Axil. Compare within a group, not across.",
        transform=ax.transAxes, fontsize=9.4, color="#64748b")

fig.text(0.012, 0.005,
         "Axil: in-tree benchmarks/longmemeval, 500-Q LongMemEval-S, bge-small, top-k 5 "
         "(committed qtc-500.json / fusion-500.json, RTX 3080). Competitor figures as published, Apr 2026; "
         "MemPalace 96.6 is a verbatim-text + ChromaDB recall config near the retrieval ceiling.",
         fontsize=6.5, color="#94a3b8")

plt.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(ROOT / "assets" / "longmemeval-recall.png", bbox_inches="tight", facecolor="white", dpi=200)
fig.savefig(ROOT / "assets" / "longmemeval-recall.svg", bbox_inches="tight", facecolor="white")
print(f"saved (Axil QTC={AXIL_QTC} fusion={AXIL_FUS})")
