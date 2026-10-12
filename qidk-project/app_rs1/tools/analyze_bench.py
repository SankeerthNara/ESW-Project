#!/usr/bin/env python3
"""Plot an RS-1 bench summary as the proposal's Fig. 3: estimate error against
latency, one dot per configuration, and print the Pareto front.

    adb pull /sdcard/Android/data/com.example.qidk.rs1/files/results/ results/
    python3 tools/analyze_bench.py results/<stamp>_summary.csv   # -> <stamp>_summary.png
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROC_COLORS = {"NPU/NPU": "#1f77b4", "GPU/GPU": "#d62728", "NPU/GPU": "#9467bd",
               "GPU/NPU": "#ff7f0e", "CPU/CPU": "#7f7f7f", "CPU/NPU": "#2ca02c"}
VARIANT_MARKERS = {"FPS": "o", "BALANCE": "s", "QUALITY": "^"}


def num(row, key):
    v = row.get(key, "")
    return float(v) if v not in ("", None) else float("nan")


def pareto(rows, err_key, lat_key):
    """Configurations no other configuration beats on both error and latency."""
    front = []
    for r in rows:
        e, l = num(r, err_key), num(r, lat_key)
        if not any(num(o, err_key) <= e and num(o, lat_key) <= l and (num(o, err_key) < e or num(o, lat_key) < l)
                   for o in rows):
            front.append(r)
    return sorted(front, key=lambda r: num(r, lat_key))


def main(path):
    rows = [r for r in csv.DictReader(open(path)) if r["status"] == "ok"]
    for r in rows:
        r["seg_err"] = str(1 - num(r, "seg_miou"))
    ref = rows[0]["reference"] if rows else "?"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    panels = [("depth_rmse", "depth error (RMSE, normalized disparity)", "depth_variant"),
              ("seg_err", "segmentation error (1 - mask IoU)", "seg_variant")]
    for ax, (key, label, variant_key) in zip(axes, panels):
        for r in rows:
            procs = f'{r["depth_proc"]}/{r["seg_proc"]}'
            seq = r["parallel"] == "false"
            color = PROC_COLORS.get(procs, "k")
            x, y = num(r, "total_p50"), num(r, key)
            ax.errorbar(x, y, xerr=[[0], [max(0.0, num(r, "total_p99") - x)]], fmt="none", ecolor=color, alpha=0.35, lw=1)
            ax.scatter(x, y, s=55, marker=VARIANT_MARKERS.get(r[variant_key], "x"),
                       facecolors="none" if seq else color, edgecolors=color, linewidths=1.5)
        front = pareto(rows, key, "total_p99")
        ax.plot([num(r, "total_p50") for r in front], [num(r, key) for r in front], "k--", lw=0.8, alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("pipeline latency, ms (dot = p50, whisker to p99; log scale)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, which="both")

    handles = [plt.Line2D([], [], color=c, marker="o", ls="", label=p) for p, c in PROC_COLORS.items()]
    handles += [plt.Line2D([], [], color="k", marker=m, ls="", mfc="none", label=f"{v} (that panel's model)") for v, m in VARIANT_MARKERS.items()]
    handles += [plt.Line2D([], [], color="k", marker="o", ls="", mfc="none", label="hollow = sequential")]
    fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle(f"RS-1 configurations vs reference {ref}  (proc legend: depth/seg)", y=0.02, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    out = Path(path).with_suffix(".png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")

    print(f"\n{'config':32} {'p50':>7} {'p99':>7} {'D exec':>7} {'S exec':>7} {'D rmse':>7} {'S IoU':>6}")
    for r in sorted(rows, key=lambda r: num(r, "total_p50")):
        print(f'{r["config"]:32} {num(r,"total_p50"):7.1f} {num(r,"total_p99"):7.1f} {num(r,"depth_exec_p50"):7.1f} '
              f'{num(r,"seg_exec_p50"):7.1f} {num(r,"depth_rmse"):7.4f} {num(r,"seg_miou"):6.3f}')
    for key, name in (("depth_rmse", "depth"), ("seg_err", "seg")):
        print(f"\nPareto front ({name} error vs p99):", ", ".join(r["config"] for r in pareto(rows, key, "total_p99")))


if __name__ == "__main__":
    main(sys.argv[1])
