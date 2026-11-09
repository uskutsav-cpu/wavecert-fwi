"""Plot immutable stored Green-comparison arrays and recorded metrics only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RECEIPT_SHA = "91281a563841fea5d34359ebd7400e9cbe7036387d2321816edb5d8ec9412b04"
AUDIT_SHA = "279c25532df3f3a1d008496353aca61d7220abc5734af93c9505e7c61e5b79ab"
RUNNER_SHA = "ba7b03027d837f6e834e47ebf13c79768287d6d2963b1df3b898a6cbce822b69"
PROTOCOL_SHA = "729e23e1a8a5373ba5492a47f2259d0bcb9a0618b25ddc87f3d73e0ea44337e5"
CASES = [
    ("small_nbl0", "61×61; nbl=0", "#c44e52"),
    ("small_nbl20", "61×61; nbl=20", "#dd8452"),
    ("small_nbl40", "61×61; nbl=40", "#55a868"),
    ("matched_exterior_20_undamped", "101×101; undamped exterior control", "#4c72b0"),
    ("matched_exterior_40_undamped", "141×141; undamped exterior control", "#8172b3"),
    ("reference_241", "241×241; nbl=40", "#64b5cd"),
    ("reference_321", "321×321; nbl=40 (primary)", "#937860"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--independent-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.run.resolve()):
        raise ValueError("new publication directory outside immutable run required")
    if sha(args.run / "complete_receipt.json") != RECEIPT_SHA or sha(args.independent_audit) != AUDIT_SHA:
        raise ValueError("actual run or independent-audit pin mismatch")
    receipt = json.loads((args.run / "complete_receipt.json").read_bytes())
    audit = json.loads(args.independent_audit.read_bytes())
    if (receipt["status"] != "complete_analytical_diagnostic" or not receipt["source_and_original_unchanged"]
            or not receipt["all_input_software_gates_pass"] or not receipt["internal_numeric_gates_pass"]
            or receipt["source_before"]["compare_green.py"] != RUNNER_SHA
            or receipt["source_before"]["protocol.json"] != PROTOCOL_SHA
            or audit["result"] != "CLEAR_INDEPENDENT_GREEN_POSTAUDIT" or audit["checks"] != 228 or audit["issues"]
            or receipt["physical_quality_pass"] is not None or audit["physical_quality_pass"] is not None):
        raise ValueError("completed diagnostic/source/scope identity mismatch")
    records = {row["case_id"]: row for row in receipt["cases"]}
    if list(records) != [row[0] for row in CASES]:
        raise ValueError("all seven cases in fixed order are required")
    files = [args.run / "complete_receipt.json", args.independent_audit, args.run / "analytic_order_512.npz"]
    files += [args.run / records[case]["artifact"] for case, _, _ in CASES]
    inputs_before = {str(path.resolve()): sha(path) for path in files}
    own_before = sha(Path(__file__))
    order512 = next(row for row in receipt["orders"] if row["order"] == 512)
    if sha(args.run / order512["artifact"]) != order512["artifact_sha256"]:
        raise ValueError("stored analytical NPZ hash mismatch")
    with np.load(args.run / "analytic_order_512.npz", allow_pickle=False) as data:
        times, valid, analytic = (data[name].copy() for name in ["times_ms", "valid_mask", "analytic"])
    if (times.shape != (1101,) or not np.array_equal(times, np.arange(1101.))
            or not np.array_equal(valid, (np.arange(1101) >= 1) & (np.arange(1101) <= 1099))
            or not np.isfinite(analytic).all()):
        raise ValueError("stored time/mask/analytical shape mismatch")
    traces, rows = [], []
    for case, label, _color in CASES:
        row = records[case]
        path = args.run / row["artifact"]
        if row["status"] != "complete" or sha(path) != row["artifact_sha256"]:
            raise ValueError("case artifact mismatch: " + case)
        with np.load(path, allow_pickle=False) as data:
            receiver = data["receiver"].copy()
            if (not np.array_equal(data["times_ms"], times) or not np.array_equal(data["valid_mask"], valid)
                    or not np.array_equal(data["analytic512"], analytic)
                    or receiver.shape != (1101, 1) or not np.isfinite(receiver).all()):
                raise ValueError("case sampling/shape mismatch: " + case)
        traces.append(receiver[:, 0])
        value = row["metrics"]["full_relative"]["value"]
        if value is None or not np.isfinite(value) or value <= 0:
            raise ValueError("positive recorded relative error required for log plot")
        record = dict(case_id=case, label=label, primary_case=row["primary_case"],
                      full_relative_L2=value, full_relative_L2_percent=100 * value,
                      full_numerator=row["metrics"]["full_relative"]["numerator"],
                      full_denominator=row["metrics"]["full_relative"]["denominator"],
                      early_relative_L2=row["metrics"]["early_relative"]["value"],
                      late_mismatch_over_full_analytic=row["metrics"]["late_mismatch_over_full_analytic"]["value"],
                      valid_samples=1099, first_valid_index=1, last_valid_index=1099,
                      physical_quality_pass="null", metric_source="recorded complete_receipt; no metric re-evaluation")
        rows.append(record)
    args.output.mkdir()
    with (args.output / "all_seven_recorded_metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, (wave, bars) = plt.subplots(2, 1, figsize=(13, 10.5), constrained_layout=True,
                                    gridspec_kw={"height_ratios": [1.55, 1.25]})
    for index, ((_, label, color), trace) in enumerate(zip(CASES, traces, strict=True)):
        options = dict(color=color, linewidth=1.25, label=label)
        if index >= 5:
            options.update(marker="x" if index == 5 else "o", markersize=4,
                           markevery=(50 if index == 5 else 95, 130), fillstyle="none", linewidth=1.5)
        wave.plot(times[valid], trace[valid], **options)
    wave.plot(times[valid], analytic[valid], color="#111111", linewidth=1.4,
              linestyle="--", label="Infinite-domain analytical solution (GL512)", zorder=9)
    wave.set(xlim=(0, 1100), xlabel="Time (ms)", ylabel="u (native solver amplitude units)",
             title="Stored absolute traces — no amplitude or time fitting")
    wave.grid(alpha=.18)
    wave.text(.01, .03, "The two enlarged-reference traces overlap at this scale.", transform=wave.transAxes,
              fontsize=9, bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
    percentages = [r["full_relative_L2_percent"] for r in rows]
    bars.barh(range(7), percentages, color=[c[2] for c in CASES], height=.62)
    bars.set_xscale("log")
    bars.set_yticks(range(7), [c[1] for c in CASES], fontsize=10)
    bars.invert_yaxis()
    bars.set_xlim(min(percentages) / 2, max(percentages) * 3)
    for index, value in enumerate(percentages):
        bars.text(value * 1.12, index, f"{value:.6g}%", va="center", fontsize=10)
    bars.set(xlabel="Relative L2 difference (%) — logarithmic scale; recorded samples 1–1099",
             title="Recorded error against the analytical solution; no physical pass threshold")
    bars.grid(axis="x", alpha=.18, which="both")
    handles, labels = wave.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    fig.suptitle("Homogeneous 2D forward-wave comparison: all seven cases\n"
                 "c = 2 m/ms; r = 50 m; point strength = 100 m²; f₀ = 0.005/ms; delay = 200 ms", fontsize=14)
    fig.savefig(args.output / "analytic_green_all_seven.png", dpi=200)
    fig.savefig(args.output / "analytic_green_all_seven.svg")
    plt.close(fig)
    caption = (
        "All seven retained Devito receiver traces and the causal infinite-domain 2D Green/Ricker solution are shown in native amplitude units, without amplitude normalization or time fitting. "
        "The source-node forcing has no 1/h² factor; matching its integrated strength gives A = h² = 100 m² at h = 10 m. c = 2 m/ms, separation = 50 m, Ricker frequency = 0.005/ms and delay = 200 ms are fixed. "
        "Only actually recorded receiver indices 1–1099 are plotted and scored. Discrete indices 0/1100 were uncomputed zero sentinels; all 1101-sample original and analytical arrays remain retained. "
        "The log bars use the existing complete-receipt relative L2 values, not recomputed or fitted metrics. Both enlarged references differ from the analytical solution by 0.0305142%; the small 61×61/nbl40 case differs by 24.9504%. "
        "An independent post-outcome alternate-variable QUADPACK check agreed with GL512 at 1.8515e-15 relative L2 over the valid samples. This is floating-point numerical agreement, not a formal enclosure or physical pass criterion. "
        "Continuum zero initial state/Ricker H(t) versus the discrete u0 = u1 = 0 and unused nonzero q0 creates a known startup mismatch. Finite-grid source, space/time discretization and finite-domain effects are not separately isolated. "
        "The earlier approximately 25% boundary diagnostic remains unchanged. All seven cases are one homogeneous scene, not independent geological environments; no damping/adjoint/certificate transfer, inversion, goal2 completion or student scientific contribution follows.\n"
    )
    (args.output / "caption.txt").write_text(caption)
    if inputs_before != {str(path.resolve()): sha(path) for path in files} or own_before != sha(Path(__file__)):
        raise RuntimeError("immutable publication inputs/source changed")
    manifest = dict(role="AI publication rendering of stored outcomes only; no analytical or PDE re-evaluation",
                    created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=own_before,
                    input_sha256_before=inputs_before, input_sha256_after=inputs_before,
                    expected_runner_sha256=RUNNER_SHA, expected_protocol_sha256=PROTOCOL_SHA,
                    plot_valid_indices=[1, 1099], normalization_or_fitting=False,
                    cases=[r[0] for r in CASES], physical_quality_pass=None,
                    independently_audited_GL512_relative_L2=next(r["alternate_curve_relative_L2"]["value"] for r in audit["orders"] if r["order"] == 512),
                    output_sha256={p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file()})
    (args.output / "render_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(status="rendered_stored_outcomes_only", output=str(args.output.resolve()),
                         manifest_sha256=sha(args.output / "render_manifest.json"), source_sha256=own_before)))


if __name__ == "__main__":
    main()
