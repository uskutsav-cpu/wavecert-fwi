"""Recompute an evidence audit without rerunning or rewriting historical studies."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from wavecert.data.splits import audit_split_manifest
from wavecert.statistics.inference import binomial_failure_summary, cluster_bootstrap_mean


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _artifact(path: Path, root: Path) -> dict:
    exists = path.is_file()
    return {
        "path": str(path.relative_to(root)),
        "exists": exists,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if exists else None,
        "bytes": path.stat().st_size if exists else None,
    }


def run_research_audit(
    *,
    repo_root: str | Path = ".",
    output_dir: str | Path = "results/research_audit",
    seed: int = 20260912,
    n_resamples: int = 4000,
) -> dict:
    """Audit archived slice independence, effect uncertainty, and missing provenance.

    Computations use archived tables only. No new physics experiment is implied.
    Parent IDs are read from the committed manifest, never inferred from names.
    """

    root = Path(repo_root).resolve()
    manifest_path = root / "configs/final_subsurfacegen_split.csv"
    manifest = _read(manifest_path)
    split_audit = audit_split_manifest(manifest)
    parent_for_slice = {row["slice_id"]: row["model_id"] for row in manifest}
    policy_path = root / "results/final_subsurfacegen/policy_cases.csv"
    matched_path = root / "results/final_subsurfacegen_matched_mc/matched_random_cases.csv"
    policy_rows = _read(policy_path)
    matched_rows = _read(matched_path)
    policies = {}
    for split in sorted({row["split"] for row in policy_rows}):
        policies[split] = {}
        for policy in sorted({row["policy"] for row in policy_rows if row["split"] == split}):
            selected = [row for row in policy_rows if row["split"] == split and row["policy"] == policy]
            grouped = defaultdict(list)
            for row in selected:
                grouped[parent_for_slice[row["slice_id"]]].append(row)
            any_non_descent = np.asarray([
                any(row["true_descent"].lower() != "true" for row in group)
                for group in grouped.values()
            ])
            any_false_acceptance = np.asarray([
                any(row["false_certification"].lower() == "true" for row in group)
                for group in grouped.values()
            ])
            policies[split][policy] = {
                "n_slices": len(selected),
                "n_parent_models": len(grouped),
                "observed_slice_descent_rate": float(np.mean([row["true_descent"].lower() == "true" for row in selected])),
                "parent_any_non_descent": binomial_failure_summary(any_non_descent),
                "parent_any_false_acceptance": binomial_failure_summary(any_false_acceptance),
                "interpretation": "Parent-any-event intervals require independent parent sampling; they are not per-slice probability intervals.",
            }
    effects = {}
    for split in sorted({row["split"] for row in matched_rows}):
        selected = [row for row in matched_rows if row["split"] == split]
        effects[split] = cluster_bootstrap_mean(
            np.asarray([float(row["wavecert_minus_random_median_cosine"]) for row in selected]),
            np.asarray([parent_for_slice[row["slice_id"]] for row in selected]),
            seed=seed, n_resamples=n_resamples,
        )
        effects[split]["quantity"] = "per-slice WaveCert cosine minus median of 100 matched-cost random-repair cosines"
        effects[split]["scope"] = "descriptive reanalysis of archived final cases; not a new held-out test"
    final_summary_path = root / "results/final_subsurfacegen/summary.json"
    final_summary = json.loads(final_summary_path.read_text())
    protocol_path = root / "configs/final_subsurfacegen_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    checkpoint_path = root / final_summary["checkpoint"]
    dataset_path = root / final_summary["dataset"]
    artifacts = [_artifact(path, root) for path in [
        manifest_path, protocol_path, policy_path, matched_path, final_summary_path, checkpoint_path,
    ]]
    dataset_files = ["train.npz", "validation.npz", "calibration.npz", "test_id.npz", "test_ood.npz"]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        commit = None
    versions = {}
    for name in ["numpy", "scipy", "pandas", "torch"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    summary = {
        "audit_kind": "archived-results provenance and statistical-unit audit",
        "source_commit": commit,
        "seed": int(seed),
        "environment_versions": versions,
        "manifest_sha256_matches_recorded_protocol": hashlib.sha256(manifest_path.read_bytes()).hexdigest() == protocol.get("manifest_sha256"),
        "split_audit": split_audit,
        "archived_policy_parent_summaries": policies,
        "matched_cost_cosine_effects": effects,
        "artifacts": artifacts,
        "final_dataset": {
            "path": str(dataset_path.relative_to(root)),
            "directory_available": dataset_path.is_dir(),
            "files_available": {name: (dataset_path / name).is_file() for name in dataset_files},
            "checkpoint_available": checkpoint_path.is_file(),
            "checkpoint_training_manifest_recorded_in_archived_summary": "training_manifest_sha256" in final_summary,
            "model_selection_calibration_disjointness": "unverified: archived summary does not identify a separate checkpoint-selection dataset and its parent IDs",
        },
        "limitations": [
            "A correct manifest hash proves file identity, not a timestamped preregistration or absence of test-set inspection.",
            "No nominal conformal coverage statement is established by observed descent or zero false acceptances alone.",
            "Correlated slices and 100 Monte Carlo repeats per slice are not independent geological replications.",
            "The single OOD parent supports a stress test; it cannot estimate uncertainty across new OOD parent models.",
            "Bootstrap intervals are descriptive and assume the observed parent groups represent independent draws.",
        ],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
