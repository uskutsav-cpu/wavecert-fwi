#!/usr/bin/env python3
"""Plot saved WaveCert study evidence; never load models or recompute predictions.

Usage: python plot-strengthening-evidence.py --repository /path/to/repo
       --output /new/output/directory/outside/repo

Requires only NumPy and Matplotlib. All statistics are descriptive. The optional
runtime figure is produced only from an existing complete actual runtime study.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import io
import json
import math
import os
import platform
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

LABELS = ("legacy_forward", "legacy_jvp", "development_winner", "classical_pod", "coarse_corrected_pod")
MODEL_NAMES = ("Legacy FNO: forward", "Legacy FNO: JVP", "New FNO: validation selected",
               "POD64: uncorrected", "POD64 + coarse96: corrected")
MODEL_COLORS = ("#5175a3", "#a47042", "#8564a1", "#198c98", "#218258")
FAMILIES = ("FlatVel_A", "CurveFault_A")
PROOF_METHODS = ("uncorrected_generic_M", "uncorrected_oriented", "coarse_corrected_oriented")
PROOF_NAMES = ("Uncorrected\ngeneric bound", "Uncorrected\noriented bound", "Corrected\noriented bound")
PARENT_COLORS = ("#2166ac", "#b35806", "#1b8a5a", "#8e63a8")
RUNTIME_METHODS = ("best_exact_objective_gradient", "raw_POD_objective_gradient", "corrected_POD_directional_certificate")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    result = float(value)
    require(math.isfinite(result), "nonfinite input number")
    return result


def optional_number(value):
    return None if value is None or value == "" else number(value)


def boolean(value):
    require(value in {"True", "False"}, "expected an explicit CSV boolean")
    return value == "True"


def stats(values):
    values = np.asarray(values, dtype=float)
    require(values.size > 0 and np.all(np.isfinite(values)), "invalid descriptive sample")
    return dict(n=int(values.size), mean=float(np.mean(values)), median=float(np.median(values)),
                minimum=float(np.min(values)), maximum=float(np.max(values)),
                q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)))


class Inputs:
    def __init__(self, root):
        self.root, self.manifest = root, {}

    def read(self, relative):
        path = (self.root / relative).resolve()
        require(path.is_relative_to(self.root), "input escapes repository")
        payload = path.read_bytes()
        self.manifest[relative] = dict(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
        return payload

    def json(self, relative):
        return json.loads(self.read(relative))

    def csv(self, relative):
        return list(csv.DictReader(io.StringIO(self.read(relative).decode())))

    def completed(self, relative):
        summary = self.json(relative)
        require(summary.get("status") == "complete", f"study is not complete: {relative}")
        if "checks" in summary:
            require(all(value is True for value in summary["checks"].values()), f"study checks failed: {relative}")
        return summary

    def verify_unchanged(self):
        for relative, record in self.manifest.items():
            require(hashlib.sha256((self.root / relative).read_bytes()).hexdigest() == record["sha256"],
                    f"input changed while plotting: {relative}")


def load_fresh(book):
    summary = book.completed("results/gradient_confirmation/summary.json")
    require(summary["model_labels"] == list(LABELS) and summary["n_model_checkpoint_evaluations"] == 1000
            and summary["n_blocks"] == 4000, "fresh five-model cohort changed")
    plan = book.json("configs/gradient_confirmation_plan.json")
    planned = {f["name"]: set(f["confirmation_selected_indices"]) for f in plan["families"]}
    require(set(planned) == set(FAMILIES) and all(len(indices) == 100 for indices in planned.values()),
            "fresh family/index selection changed")
    paths = (
        "results/gradient_confirmation/first_four_models/three_fno/models.csv",
        "results/gradient_confirmation/first_four_models/classical_pod/models.csv",
        "results/gradient_confirmation/coarse_corrected_pod/models.csv",
    )
    rows = [row for path in paths for row in book.csv(path)]
    require(len(rows) == 1000, "fresh aggregate CSVs must contain 1000 rows")
    groups, identities = defaultdict(list), {}
    for row in rows:
        family, label, index = row["family"], row["checkpoint"], int(row["original_index"])
        require(family in FAMILIES and label in LABELS and int(row["n_blocks"]) == 4, "fresh row scope changed")
        require(index in planned[family], "unplanned fresh original index")
        value = number(row["aggregate_gradient_cosine"])
        require(abs(value) <= 1 + 1e-12, "invalid gradient cosine")
        groups[family, label].append(row)
        identity = (row["model_id"], row["raw_model_sha256"])
        key = family, index
        require(key not in identities or identities[key] == identity, "fresh model identities differ across controls")
        identities[key] = identity
    numerical = {}
    for family in FAMILIES:
        numerical[family] = {}
        for label in LABELS:
            group = groups[family, label]
            require(len(group) == 100 and {int(r["original_index"]) for r in group} == planned[family],
                    "each fresh family/model must retain all 100 indices")
            result = stats([number(r["aggregate_gradient_cosine"]) for r in group])
            descent_field = "aggregate_approximate_true_descent" if label == LABELS[-1] else "aggregate_neural_true_descent"
            result["true_descent_count"] = sum(boolean(r[descent_field]) for r in group)
            if label == LABELS[-1]:
                result.update(certificate_available_count=sum(boolean(r["certificate_available"]) for r in group),
                              certified_descent_count=sum(boolean(r["certified_descent"]) for r in group),
                              false_certification_count=sum(boolean(r["aggregate_deterministic_false_acceptance"]) for r in group))
            numerical[family][label] = result
    return groups, numerical


def save_figure(fig, output, stem, caption, artifacts, plt):
    for extension in ("png", "svg"):
        path = output / f"{stem}.{extension}"
        fig.savefig(path, dpi=190, facecolor="white", metadata={"Creator": "plot-strengthening-evidence.py"})
        artifacts[path.name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    (output / f"{stem}.caption.txt").write_text(caption + "\n")
    artifacts[f"{stem}.caption.txt"] = dict(sha256=hashlib.sha256((caption + "\n").encode()).hexdigest())
    plt.close(fig)


def fresh_figure(groups, numerical, output, artifacts, plt):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.6), sharey=True)
    fig.subplots_adjust(left=.235, right=.975, top=.79, bottom=.25, wspace=.13)
    fig.suptitle("Fresh known-batch confirmation: aggregate gradient alignment", x=.51, y=.97,
                 fontsize=17, fontweight="bold")
    fig.text(.51, .905, "Five frozen models  ·  100 previously unused indices per family  ·  four blocks per index",
             ha="center", fontsize=11.5, color="#444444")
    for axis, family in zip(axes, FAMILIES, strict=True):
        samples = [np.array([number(row["aggregate_gradient_cosine"]) for row in sorted(groups[family, label],
                    key=lambda r: int(r["original_index"]))]) for label in LABELS]
        orientation = ({"orientation": "horizontal"} if "orientation" in inspect.signature(axis.boxplot).parameters
                       else {"vert": False})
        boxes = axis.boxplot(samples, **orientation, positions=np.arange(5), widths=.44,
                             whis=(0, 100), showfliers=False, patch_artist=True,
                             medianprops=dict(color="black", linewidth=1.6),
                             whiskerprops=dict(linewidth=.9), capprops=dict(linewidth=.9))
        for i, (values, color, box) in enumerate(zip(samples, MODEL_COLORS, boxes["boxes"], strict=True)):
            box.set(facecolor=color, alpha=.16, edgecolor=color, linewidth=1.1)
            # Fixed, deterministic visual offsets; these do not change data.
            jitter = np.random.default_rng(2026110731 + i).uniform(-.115, .115, len(values))
            axis.scatter(values, i + jitter, s=10, color=color, alpha=.45, linewidths=0, zorder=3)
            axis.scatter(np.mean(values), i, s=40, marker="D", color=color,
                         edgecolor="black", linewidth=.65, zorder=5)
        axis.axvline(0, color="#777777", linestyle="--", linewidth=.9, zorder=0)
        axis.axvline(1, color="#b7b7b7", linestyle=":", linewidth=.9, zorder=0)
        axis.set(xlim=(-1.04, 1.045), ylim=(4.6, -.6), yticks=np.arange(5),
                 xticks=[-1, -.5, 0, .5, 1], xlabel="Cosine with exact aggregate gradient")
        axis.set_title(f"{family}  (100 indices)", fontsize=13, pad=11)
        axis.grid(axis="x", color="#e6e6e6", linewidth=.7)
        axis.tick_params(axis="both", labelsize=10.5)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
    axes[0].set_yticklabels(MODEL_NAMES, fontsize=11)
    axes[0].tick_params(axis="y", length=0, pad=12)
    axes[1].tick_params(axis="y", length=0)
    fig.text(.235, .16, "Point = selected original index    Diamond = sample mean    Box = median / IQR    Whiskers = full range",
             fontsize=10.2, color="#333333")
    caption = (
        "Fresh indices from known OpenFWI velocity batches: 100 selected slots from each fixed 400-index complement; "
        "all five controls use the same slots. These are synthetic discrete tests, not unseen families or field observations. "
        "Each point uses the actual sum of four block gradients at a shared model. Distributions and means are descriptive: "
        "no IID, simultaneous, or population confidence intervals are shown. The corrected method is a physical proposal, "
        "not the derivative of a corrected-ROM objective. Ordinary floating point."
    )
    fig.text(.235, .075, "Known batches and synthetic physics; no unseen-family or field-data claim.\n"
             "Descriptive distributions only; no IID or simultaneous confidence intervals. Ordinary floating point.",
             fontsize=10, color="#555555", linespacing=1.55)
    save_figure(fig, output, "fresh_gradient_alignment", caption, artifacts, plt)


def validation_figure(book, output, artifacts, plt):
    summary = book.completed("results/coarse_adjoint_validation/summary.json")
    protocol = book.json("results/coarse_adjoint_validation/protocol.json")
    rows = book.csv("results/coarse_adjoint_validation/aggregate_cases.csv")
    require(len(rows) == 24 and protocol["methods"] == list(PROOF_METHODS), "validation proof cohort changed")
    identities = {(r["parent_id"], r["slice_id"]) for r in protocol["identities"]}
    parents = sorted({parent for parent, _ in identities})
    require(len(identities) == 8 and len(parents) == 4, "validation must retain eight slices/four parents")
    groups = {method: [row for row in rows if row["method"] == method] for method in PROOF_METHODS}
    for method, group in groups.items():
        require(len(group) == 8 and {(r["parent_id"], r["slice_id"]) for r in group} == identities,
                "a validation method omitted or duplicated a slice")
        require(summary["methods"][method]["n_slices"] == 8 and summary["methods"][method]["n_parents"] == 4,
                "validation summary disagrees with cohort")
    required = {"bound_to_approximate_descent", "bound_effectivity"}
    if not required <= set(rows[0]):
        return {"status": "not plotted", "reason": "saved CSV lacks ratio/effectivity fields"}
    numerical = {}
    for method, group in groups.items():
        ratio = [optional_number(r["bound_to_approximate_descent"]) for r in group]
        effectivity = [optional_number(r["bound_effectivity"]) for r in group]
        for r, value, effect in zip(group, ratio, effectivity, strict=True):
            center, bound, error = (optional_number(r[key]) for key in (
                "approximate_directional_derivative", "directional_error_bound", "actual_directional_error"))
            if value is not None:
                require(center < 0 and bound >= 0 and math.isclose(value, bound / -center, rel_tol=1e-12),
                        "saved bound/margin ratio disagrees with saved components")
            if effect is not None:
                require(error > 0 and math.isclose(effect, bound / error, rel_tol=1e-12),
                        "saved effectivity disagrees with saved components")
        numerical[method] = dict(n_slices=8, n_parents=4,
            available_count=sum(boolean(r["available"]) for r in group),
            certified_descent_count=sum(boolean(r["certified_descent"]) for r in group),
            bound_to_margin=stats([v for v in ratio if v is not None]),
            bound_effectivity=stats([v for v in effectivity if v is not None]),
            undefined_margin_ratios=sum(v is None for v in ratio),
            undefined_effectivities=sum(v is None for v in effectivity))
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.7))
    fig.subplots_adjust(left=.075, right=.98, top=.79, bottom=.32, wspace=.27)
    fig.suptitle("Development validation: directional bound size and observed error", y=.97,
                 fontsize=17, fontweight="bold")
    fig.text(.5, .91, "Eight correlated slices from four parents  ·  all 24 blocks per slice  ·  three fixed methods",
             ha="center", fontsize=11.5, color="#444444")
    parent_colors = dict(zip(parents, PARENT_COLORS, strict=True))
    for parent, slice_id in sorted(identities):
        selected = [next(r for r in groups[m] if (r["parent_id"], r["slice_id"]) == (parent, slice_id))
                    for m in PROOF_METHODS]
        marker = "o" if sorted(s for p, s in identities if p == parent).index(slice_id) == 0 else "s"
        for axis, field in zip(axes, ("bound_to_approximate_descent", "bound_effectivity"), strict=True):
            values = [optional_number(r[field]) for r in selected]
            require(all(v is None or v > 0 for v in values), "log plot needs positive defined ratios")
            y = [np.nan if v is None else v for v in values]
            axis.plot(range(3), y, color=parent_colors[parent], linewidth=1.1, alpha=.65,
                      marker=marker, markersize=5.5, markeredgecolor="white", markeredgewidth=.5)
    for axis in axes:
        axis.set_yscale("log")
        axis.axhline(1, color="#a13d3d", linewidth=1.2, linestyle="--")
        axis.set_xticks(range(3), PROOF_NAMES, fontsize=10.5)
        axis.set_xlim(-.16, 2.16)
        axis.grid(axis="y", which="major", color="#e7e7e7", linewidth=.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="y", labelsize=10.5)
    axes[0].set_ylabel("Bound / approximate descent margin", fontsize=11)
    axes[1].set_ylabel("Bound / observed directional error", fontsize=11)
    axes[0].set_title("Below 1: strict descent can be certified", fontsize=12, pad=11)
    axes[1].set_title("Above 1: saved directional error is covered", fontsize=12, pad=11)
    axes[1].set_ylim(bottom=.75)
    axes[0].set_xticklabels([
        f"{name}\n{numerical[method]['certified_descent_count']}/8 certified"
        for name, method in zip(PROOF_NAMES, PROOF_METHODS, strict=True)], fontsize=10.5)
    handles = [plt.Line2D([0], [0], color=parent_colors[p], marker="o", label=p, linewidth=1.2)
               for p in parents]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .875), ncol=4,
               frameon=False, fontsize=10.5, handlelength=2)
    caption = (
        "Previously used development validation: all eight slices/four parents and all 24 blocks, with paired slice lines. "
        "Left is the deterministic directional bound divided by the negative approximate directional derivative; "
        "strict certification requires an available upper derivative below zero. Right divides the bound by the observed "
        "absolute directional error. Line at one is a threshold, not a confidence interval. Colors are parent groups; "
        "circles/squares distinguish their two slices. Outcomes were used during method development, not independent confirmation. "
        "Exact-arithmetic inequalities are evaluated in ordinary floating point with an unverified cancellation allowance."
    )
    fig.text(.075, .13, "Lines pair the same development slice; colors mark parent groups. These outcomes were used in method development.\n"
             "The dashed line is a numerical threshold, not a confidence interval. No IID-eight inference.\n"
             "Ordinary floating point with an unverified cancellation allowance; no formal roundoff enclosure.",
             fontsize=10, color="#555555", linespacing=1.55, va="top")
    save_figure(fig, output, "validation_directional_bounds", caption, artifacts, plt)
    return numerical


def historical_table(book, output, artifacts):
    summary = book.completed("results/historical_transfer/summary.json")
    require(summary["n_slice_model_aggregates"] == 100 and summary["n_block_model_rows"] == 2400
            and summary["n_parent_model_summaries"] == 45, "historical full cohort changed")
    protocol = book.json("results/historical_transfer_freeze/protocol.json")
    rows = book.csv("results/historical_transfer/parents.csv")
    models = book.csv("results/historical_transfer/models.csv")
    require(len(rows) == 45 and len(models) == 100, "historical CSV counts changed")
    expected = {(role, parent, slice_id, label) for role, cfg in protocol["dataset"]["roles"].items()
                for parent, slices in cfg["parents"].items() for slice_id in slices for label in LABELS}
    require({(r["role"], r["model_id"], r["slice_id"], r["checkpoint"]) for r in models} == expected
            and all(int(r["n_blocks"]) == 24 for r in models), "historical model/slice identities changed")
    result = []
    for label, name in zip(LABELS, MODEL_NAMES, strict=True):
        item = dict(model=label, display_name=name)
        for role, prefix, count in (("test_id", "ID", 8), ("test_ood", "OOD", 1)):
            group = [r for r in rows if r["role"] == role and r["checkpoint"] == label]
            require(len(group) == count and {r["model_id"] for r in group} == set(protocol["dataset"]["roles"][role]["parents"]),
                    "historical parent identity/count changed")
            for r in group:
                selected = [m for m in models if m["checkpoint"] == label and m["role"] == role and m["model_id"] == r["model_id"]]
                require(len(selected) == int(r["n_slices"]) and math.isclose(
                    np.mean([number(m["aggregate_gradient_cosine"]) for m in selected]),
                    number(r["mean_actual_aggregate_gradient_cosine"]), rel_tol=1e-12, abs_tol=1e-12),
                    "historical parent average differs from saved slice averages")
            item[f"{prefix}_n_parents"] = count
            item[f"{prefix}_n_slices"] = sum(int(r["n_slices"]) for r in group)
            item[f"{prefix}_equal_parent_mean_cosine"] = float(np.mean([number(r["mean_actual_aggregate_gradient_cosine"]) for r in group]))
            item[f"{prefix}_equal_parent_true_descent_fraction"] = float(np.mean([number(r["true_descent_slice_fraction"]) for r in group]))
        result.append(item)
    path = output / "historical_parent_comparison.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    text = ["# Historical test reuse: descriptive parent comparison", "",
            "Prior inspected roles; 16 ID slices / 8 parents and 4 OOD slices / 1 parent. All five controls, all 24 blocks.", "",
            "| Fixed model | ID: equal-parent mean cosine | OOD: one-parent mean cosine |",
            "| --- | ---: | ---: |"]
    text += [f"| {r['display_name']} | {r['ID_equal_parent_mean_cosine']:.4f} | {r['OOD_equal_parent_mean_cosine']:.4f} |" for r in result]
    text += ["", "Parent means average each parent's own slices first. No IID intervals, ID/OOD pooling, new-population inference, or nominal conformal claim. One OOD parent cannot establish broad OOD reliability. Ordinary floating point."]
    markdown = output / "historical_parent_comparison.md"
    markdown.write_text("\n".join(text) + "\n")
    for target in (path, markdown):
        artifacts[target.name] = dict(sha256=hashlib.sha256(target.read_bytes()).hexdigest(), bytes=target.stat().st_size)
    return result


def optional_runtime_figure(book, output, artifacts, plt):
    relative = "results/coarse_runtime_controls/summary.json"
    if not (book.root / relative).is_file():
        return dict(status="not plotted", reason="no actual runtime summary exists; no timing values invented")
    summary = book.json(relative)
    if summary.get("status") != "complete":
        return dict(status="not plotted", reason="actual runtime study is not complete")
    require(all(summary["checks"].values()), "runtime study checks failed")
    rows = book.csv("results/coarse_runtime_controls/cold_calls.csv")
    expected = {(rep, index, method) for rep in range(3) for index in range(8) for method in RUNTIME_METHODS}
    require(len(rows) == 72 and {(int(r["repetition"]), int(r["case_index"]), r["method"]) for r in rows} == expected,
            "actual runtime cohort must contain all 72 cold calls")
    values, numerical = [], {}
    for method in RUNTIME_METHODS:
        medians = []
        for index in range(8):
            group = [r for r in rows if r["method"] == method and int(r["case_index"]) == index]
            require(len(group) == 3, "runtime must retain all three repetitions")
            for r in group:
                setup, query, total = (number(r[key]) for key in ("setup_seconds", "query_seconds", "total_cold_seconds"))
                require(setup >= 0 and query >= 0 and total > 0 and math.isclose(setup + query, total, abs_tol=1e-9),
                        "runtime timing decomposition changed")
            medians.append(float(np.median([number(r["total_cold_seconds"]) for r in group])))
        values.append(medians)
        numerical[method] = dict(n_slices=8, repetitions_per_slice=3,
                                 per_slice_median_total_cold_seconds=medians,
                                 descriptive_across_slice_medians=stats(medians))
    fig, axis = plt.subplots(figsize=(10.6, 6.2))
    fig.subplots_adjust(left=.12, right=.97, top=.80, bottom=.31)
    names = ("Exact objective\n+ full gradient", "Raw POD64 objective\n+ reduced-model gradient",
             "Corrected POD64 proposal\n+ directional certificate")
    label_option = {"tick_labels" if "tick_labels" in inspect.signature(axis.boxplot).parameters else "labels": names}
    axis.boxplot(values, **label_option,
                 whis=(0, 100), widths=.38, showfliers=False,
                 medianprops=dict(color="black", linewidth=1.5))
    for i, sample in enumerate(values, start=1):
        axis.scatter(i + np.linspace(-.09, .09, 8), sample, s=35,
                     c=[PARENT_COLORS[j // 2] for j in range(8)], edgecolor="white", linewidth=.5)
    axis.set_yscale("log")
    axis.set_ylabel("Total cold setup + query wall time (seconds)", fontsize=11.5)
    axis.grid(axis="y", color="#e7e7e7")
    axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Actual cold-query costs: three different information tasks", y=.96, fontsize=16, fontweight="bold")
    fig.text(.5, .885, "Eight existing validation slices / four parents  ·  point = median of three cold visits",
             ha="center", fontsize=11)
    caption = ("Actual measured complete runtime study only. New per-model solver caches; constructor setup and all current-model "
               "query work included. One point is a three-visit median for a previously used validation slice. Input I/O, historical "
               "teacher/basis development and offline exact verification are separately paid in the study, outside these core times. "
               "Different tasks and approximation qualities: no equal-accuracy, inversion, time-to-target or field-scale speedup claim. "
               "Descriptive values only, not independent geological samples or confidence intervals. See original load/context receipts.")
    fig.text(.12, .13, "Cold per-model solver caches, not cold processes or OS caches. Tasks and approximation qualities differ.\n"
             "Historical development and offline verification are separately paid; see original study receipts.\n"
             "No equal-accuracy, inversion, time-to-target, or field-scale speedup claim.", fontsize=10, color="#555555", linespacing=1.5)
    save_figure(fig, output, "actual_cold_runtime", caption, artifacts, plt)
    return dict(status="plotted from actual complete study", methods=numerical,
                compute_context=summary.get("compute_context"), source_sha256=summary.get("source_sha256_at_end"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.repository.resolve(), args.output.resolve()
    require(root.is_dir(), "repository does not exist")
    require(not output.is_relative_to(root), "plot outputs must stay outside the repository")
    require(not output.exists() or (output.is_dir() and all(
        p.name == ".matplotlib" and p.is_dir() and not p.is_symlink() for p in output.iterdir())),
        "output must be new or empty (a task-local Matplotlib cache is allowed)")
    output.mkdir(parents=True, exist_ok=True)
    # Keep even Matplotlib's font cache inside this task's output directory.
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "axes.labelcolor": "#333333",
                         "text.color": "#222222", "axes.edgecolor": "#999999"})
    book, artifacts = Inputs(root), {}
    try:
        groups, fresh = load_fresh(book)
        fresh_figure(groups, fresh, output, artifacts, plt)
        validation = validation_figure(book, output, artifacts, plt)
        historical = historical_table(book, output, artifacts)
        runtime = optional_runtime_figure(book, output, artifacts, plt)
        book.verify_unchanged()
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
        manifest = dict(status="complete", created_utc=datetime.now(timezone.utc).isoformat(),
            repository=str(root), repository_revision_observed=revision,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
            model_label_order=list(LABELS), input_hashes=book.manifest, inputs_unchanged_during_plotting=True,
            outputs=artifacts, fresh_known_batch_confirmation=fresh,
            existing_development_validation=validation, historical_test_reuse=historical,
            optional_actual_runtime=runtime,
            scope="Saved outcomes only; no model loading, prediction, training, timing study or new inferential interval. No novelty or inversion acceleration claim.")
        (output / "numeric_and_input_hash_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(dict(status="complete", output=str(output), figures=[name for name in artifacts if name.endswith(".png")],
                              runtime=runtime["status"]), indent=2))
    except Exception as exc:
        (output / "plotting_failure.json").write_text(json.dumps(dict(status="failed", exception=type(exc).__name__,
            message=str(exc), inputs_read=book.manifest), indent=2, allow_nan=False) + "\n")
        raise


if __name__ == "__main__":
    main()
