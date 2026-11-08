"""Plot all fixed boundary-verification traces; no solver or case selection."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    styles = [
        ("small_nbl0", "Small box, undamped", "#9b3030", "-"),
        ("small_nbl20", "Small box + 20 damping cells", "#1764a0", "-"),
        ("matched_exterior_20_undamped", "Same exterior as 20 cells, undamped", "#1764a0", "--"),
        ("small_nbl40", "Small box + 40 damping cells", "#d07611", "-"),
        ("matched_exterior_40_undamped", "Same exterior as 40 cells, undamped", "#d07611", "--"),
        ("reference_241", "Enlarged reference: 241 cells + padding", "#777777", ":"),
        ("reference_321", "Enlarged reference: 321 cells + padding", "#1e7549", "-"),
    ]
    traces = {}
    for key, *_ in styles:
        with np.load(args.data / (key + "_raw.npz"), allow_pickle=False) as data:
            times = data["times_before_ms"]
            traces[key] = data["receiver"][:, 0]
    ref = traces["reference_321"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.3), sharex=True)
    for key, label, color, linestyle in styles:
        axes[0].plot(times, traces[key], color=color, ls=linestyle, lw=1.25, label=label)
        if key != "reference_321":
            axes[1].plot(times, traces[key] - ref, color=color, ls=linestyle, lw=1.1)
    for ax in axes:
        ax.axvline(250., color="#888888", ls=":", lw=.8)
        ax.grid(alpha=.18)
        ax.set_xlim(0, 1100)
    axes[0].set_ylabel("Receiver response\n(arbitrary units)")
    axes[1].set_ylabel("Difference from largest reference\n(same units)")
    axes[1].set_xlabel("Time (ms)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, 1.53), ncol=2, fontsize=8.5,
                   frameon=False)
    fig.suptitle("Boundary truncation: all seven fixed software-verification cases", y=.985,
                 fontsize=13)
    fig.text(.5, .932, "Homogeneous 2-D acoustics: 2 km/s · 5 Hz Ricker · 10 m grid · 1 ms time step",
             ha="center", fontsize=9.5)
    fig.text(.5, .015, "The 250 ms line is the fixed diagnostic split, not a causal guarantee. "
             "Same-backend references; no physics pass threshold.", ha="center", fontsize=8.5)
    fig.subplots_adjust(top=.72, bottom=.09, left=.12, right=.98, hspace=.18)
    for suffix in ("png", "svg"):
        fig.savefig(args.output / ("boundary_traces." + suffix), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
