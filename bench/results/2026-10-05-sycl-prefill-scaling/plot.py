#!/usr/bin/env python3
"""Render recorded measurements; no inference or timing is performed."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
wall = json.loads((root / "wall/run.json").read_text())
profile = json.loads((root / "profile/run.json").read_text())
preloaded = json.loads((root / "preloaded/run.json").read_text())
x = [m["tokens"] for m in wall["medians"]]
y = [m["wall_ms"] / 1000 for m in wall["medians"]]
low, high = [], []
for n, median in zip(x, y):
    values = [r["wall_ms"] / 1000 for r in wall["runs"] if r["tokens"] == n]
    low.append(median - min(values))
    high.append(max(values) - median)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "svg.hashsalt": "strata-prefill-scaling"})
fig, (latency, transfer) = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
latency.errorbar(x, y, yerr=[low, high], marker="o", capsize=4,
                 color="#2166ac", label="Normal prefill: median and range of 3 runs")
latency.plot(x, [m["wall_ms"] / 1000 for m in preloaded["medians"]],
             "s--", color="#b35806", label="Diagnostic GPU/transfer stage: 1 profiled run")
latency.set(xlabel="Input tokens", ylabel="Elapsed time (s)", ylim=(0, 28),
            title="Latency growth with input length")
latency.text(0.04, 0.97,
             "4,096 to 8,087 tokens\nNormal: 1.815 ms/additional token\nDiagnostic: 0.975 ms/additional token",
             transform=latency.transAxes, va="top", fontsize=9)
latency.legend(loc="lower right", fontsize=8)

dma = [m["dma_active_ms"] / 1000 for m in profile["medians"]]
size = [m["expert_bytes"] / 1e9 for m in profile["medians"]]
transfer.plot(x, dma, "o-", color="#2166ac")
transfer.set(xlabel="Input tokens", ylabel="Active DMA (s)", ylim=(0, 9),
             title="Expert-weight transfers: 1 profiled run")
bytes_axis = transfer.twinx()
bytes_axis.plot(x, size, "s--", color="#b35806")
bytes_axis.set(ylabel="Transferred bytes (GB, decimal)", ylim=(0, 54))
bytes_axis.spines["top"].set_visible(False)
for n, seconds, gb in zip(x, dma, size):
    transfer.annotate(f"{seconds:.3f} s", (n, seconds), xytext=(0, -16),
                      textcoords="offset points", ha="center", color="#2166ac", fontsize=9)
    bytes_axis.annotate(f"{gb:.3f} GB", (n, gb), xytext=(0, 8),
                        textcoords="offset points", ha="center", color="#b35806", fontsize=9)
for axis in (latency, transfer):
    axis.set_xlim(1024, 9216)
    axis.set_xticks(x, [f"{n:,}" for n in x])
    axis.grid(axis="y", alpha=0.2)

fig.suptitle("Arc B570 10 GB / Ryzen 5 5600X / 128 GB RAM / native IQ3_S", fontsize=12)
fig.supxlabel("Fixed 8,192-token context and chunk; 128 expert slots; one chunk; MTP off.\n"
              "Diagnostic timing excludes the separately gathered PLE input. DMA and compute overlap; do not add them.",
              fontsize=8)
fig.savefig(root / "prefill-scaling.png", dpi=180)
svg_path = root / "prefill-scaling.svg"
fig.savefig(svg_path, metadata={"Date": None})
svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
