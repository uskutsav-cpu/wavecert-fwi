"""Existing scalar receipts only; no NPZ, solver or project-code imports.

Rendering is held until root confirms the BP native process is terminal.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - Agg must be selected before pyplot.
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter  # noqa: E402


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    work = here.parents[1]
    receipt_path = work/"initial_linearization_probe_run_01/complete_receipt.json"
    receipt_sha = "6d7189953db68a59e3be220c81d86115f100d8f57442ef0ee8357c9b3d112563"
    original_path = work/"wavecert-fwi/results/paid_merit_initial_direction_verification/run_01/original_protocol.json"
    original_sha = "fb6b304fd719441be8cd6eb8d101ca864d0856254f40618583d8d986fd1c945c"
    if checksum(receipt_path) != receipt_sha or checksum(original_path) != original_sha:
        raise ValueError("frozen scalar receipt or original identity metadata changed")
    receipt = json.loads(receipt_path.read_bytes())
    identities = [r["slice_id"] for r in json.loads(original_path.read_bytes())["identities"]]
    if (not receipt["numerical_integrity_pass"] or [r["case_index"] for r in receipt["cases"]] != list(range(8))
            or any(r["status"] != "complete" for r in receipt["cases"])):
        raise ValueError("all eight fixed completed cases required")
    outputs = ["initial_linearization_all8.png", "initial_linearization_all8.svg",
               "all8_band_scalars.csv", "caption.md", "provenance.json"]
    if any((here/name).exists() for name in outputs):
        raise FileExistsError("preserve existing rendering before a new version")
    metrics = ["remainder_relative_to_data_change", "data_change_norm", "tangent_norm", "remainder_norm",
               "real_alignment", "g_data_dot_delta"]
    rows = []
    for case, identity in zip(receipt["cases"], identities, strict=True):
        row = dict(case_index=case["case_index"], slice_id=identity)
        for band in ("lowband", "originalband"):
            if case["bands"][band]["status"] != "complete":
                raise ValueError("partial bands cannot be plotted as complete")
            row.update({band+"_"+key: case["bands"][band][key] for key in metrics})
        rows.append(row)
    with (here/"all8_band_scalars.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(8.7, 5.7))
    fig.subplots_adjust(left=.31, right=.97, bottom=.19, top=.79)
    for row in rows:
        index = row["case_index"]
        lower, original = (row[k+"_remainder_relative_to_data_change"] for k in ("lowband", "originalband"))
        ax.plot([lower, original], [index-.075, index+.075], color="#C8C8C8", linewidth=1.25, zorder=1)
        ax.scatter(lower, index-.075, s=38, marker="o", facecolor="#627D8A", edgecolor="white", linewidth=.5,
                   zorder=3, label="Lower: 0.25, 0.5, 1, 2 Hz" if index == 0 else None)
        ax.scatter(original, index+.075, s=37, marker="s", facecolor="white", edgecolor="#605F61", linewidth=1.15,
                   zorder=3, label="Original: 2.5, 3, 3.5, 4 Hz" if index == 0 else None)
    ax.set_xscale("log")
    ax.set_xlim(.08, 3.3)
    ax.xaxis.set_major_locator(FixedLocator([.1, .2, .5, 1., 2., 3.]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(range(8), [f"{r['case_index']}   {r['slice_id']}" for r in rows])
    ax.tick_params(axis="y", length=0, pad=9, labelsize=8.7)
    ax.tick_params(axis="x", which="minor", length=2)
    ax.set_ylim(7.65, -.65)
    ax.set_xlabel("Relative remainder  ‖b − Jδm‖₂ / ‖b‖₂  (log scale)", labelpad=9)
    ax.legend(loc="lower left", bbox_to_anchor=(-.015, 1.04), frameon=False, ncol=2, fontsize=8.5,
              handletextpad=.4, columnspacing=1.4)
    ax.annotate("Case 2 exception\nLower 1.173; original 1.134", xy=(rows[2]["lowband_remainder_relative_to_data_change"], 1.925),
                xytext=(1.6, 2.75), fontsize=8.2, color="#424242", ha="left", va="center",
                arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": .75})
    fig.text(.055, .955, "Initial linearization remainder: all eight reused cases", fontsize=13, weight="semibold")
    fig.text(.055, .908, "Fixed bands • same initial-to-truth direction • receiver data only", fontsize=9.4, color="#515151")
    fig.text(.055, .052, "Eight reused development slices; SEAM labels share an ancestor. No independent-site claim.", fontsize=7.7, color="#515151")
    fig.text(.055, .024, "Descriptive initial linearization diagnostic; no physical pass threshold, frequency selection or tested remedy.", fontsize=7.7, color="#515151")
    fig.savefig(here/outputs[0], dpi=180, facecolor="white")
    fig.savefig(here/outputs[1], facecolor="white")
    plt.close(fig)
    caption = """Paired initial-linearization remainder ratios for all eight fixed reused cases. Filled circles show the fixed lower band [0.25,0.5,1,2]Hz; open squares show the original band [2.5,3,3.5,4]Hz. Small vertical offsets distinguish paired markers and carry no numerical meaning. For each band b=d_truth−d_initial and Jδm is the initial-model tangent along the fixed full correction δm=m_truth−m_initial. The horizontal axis is ||b−Jδm||₂/||b||₂ on a logarithmic scale. All four frequencies and six-source/twenty-receiver samples enter each ratio. No physical pass threshold is drawn.

All8/8 cases,64/64 case/frequency rows,128 LU factorizations and1152 RHS solves completed; there were no optimizer calls. Lower-band ratios are smaller in seven cases. Case2 remains explicit: lower ratio1.173215 exceeds original ratio1.133888. All-eight pooled ratios are1.143304 lower and2.057289 original. Pooling includes case2 and is not a statistic over independent sites. Full-precision per-case norms, alignment and contractions accompany the figure in CSV; no favorable frequency or case subset is selected.

These are the old49×49 frequency-domain model and noiseless self-consistent synthetic observations, with unchanged graph, damping, geometry and source normalization. All eight inputs were previously reused for development; nominal SEAM identities share an ancestor. This AI-assisted post-outcome software diagnosis does not demonstrate optimization performance, a multiscale remedy, a cycle-skipping cause, physical accuracy, new geology, independent-cohort generalization, certificate transfer, novelty or student-authored conclusions. Frequency continuation is prior FWI work. No new wave solve, NPZ decoding, fitted curve or data adjustment occurred during figure creation.
"""
    (here/"caption.md").write_text(caption)
    if checksum(receipt_path) != receipt_sha or checksum(original_path) != original_sha:
        raise ValueError("input metadata changed during rendering")
    provenance = dict(role="retained-scalar scientific figure only", complete_receipt_sha256=receipt_sha,
        original_identity_protocol_sha256=original_sha, renderer_sha256=checksum(Path(__file__)),
        producer_sha256=receipt["source_sha256"], protocol_sha256=receipt["protocol_sha256"],
        production_source_sha256="9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da",
        files_sha256={name:checksum(here/name) for name in outputs if name != "provenance.json"},
        cases=8, plotted_ratios=16, transformations="log axis and consistent vertical marker separation only; numeric values unchanged",
        raw_NPZ_reads=0, new_wave_solves=0, optimizer_calls=0, physical_acceptance=None,
        input_hashes_checked_before_after=True, independent_sample_claim=False)
    (here/"provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True)+"\n")
    print(json.dumps(dict(output=str(here), file_bytes=sum(p.stat().st_size for p in here.iterdir() if p.is_file()),
                         output_hashes=provenance["files_sha256"]), indent=2))


if __name__ == "__main__":
    main()
