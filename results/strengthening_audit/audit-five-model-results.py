"""Separate raw-table/vector audit; no model fitting, PDE solves or timing claims."""
import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def close(a, b):
    require(math.isclose(float(a), float(b), rel_tol=2e-9, abs_tol=2e-10), f"arithmetic mismatch: {a}, {b}")


def flag(value):
    require(value in ("True", "False"), "invalid boolean table entry")
    return value == "True"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository.resolve()
    study = root / "results/gradient_confirmation"
    plan = json.loads((root / "configs/gradient_confirmation_plan.json").read_text())
    selected = {f["name"]: set(f["confirmation_selected_indices"]) for f in plan["families"]}
    require(all(len(ids) == 100 for ids in selected.values()), "fixed cohorts changed")
    outer = json.loads((study / "summary.json").read_text())
    require(outer["status"] == "complete" and outer["n_blocks"] == 4000
            and outer["n_model_checkpoint_evaluations"] == 1000, "joint study incomplete")
    definitions = [
        ("first_four_models/three_fno", 600, "neural_gradients", None),
        ("first_four_models/classical_pod", 200, "neural_gradients", "classical_pod"),
        ("coarse_corrected_pod", 200, "approximate_gradients", "coarse_corrected_pod"),
    ]
    expected_labels = {"legacy_forward", "legacy_jvp", "development_winner", "classical_pod", "coarse_corrected_pod"}
    expected_blocks = {(s, f) for s in (54, 65) for f in (3., 4.)}
    all_models, exact_by_case, groups = set(), {}, defaultdict(list)
    manifests, source_revisions, source_digests, counters = [], set(), set(), {}
    for relative, count, proposal_key, constant_label in definitions:
        directory = study / relative
        summary = json.loads((directory / "summary.json").read_text())
        require(summary["status"] == "complete" and summary["source_unchanged_during_run"] is True
                and summary["input_files_unchanged_during_run"] is True, "missing immutable-run evidence")
        provenance = summary["provenance"]
        manifests.append(provenance["source_files"])
        source_revisions.add(provenance["git_revision"])
        source_digests.add(provenance["source_sha256"])
        models = list(csv.DictReader((directory / "models.csv").open(newline="")))
        blocks = list(csv.DictReader((directory / "blocks.csv").open(newline="")))
        require(len(models) == count and len(blocks) == 4 * count, "raw Cartesian size mismatch")
        by_model = defaultdict(list)
        for row in blocks:
            by_model[row["family"], row["checkpoint"], int(row["original_index"])].append(row)
        with np.load(directory / "aggregate_gradients.npz", allow_pickle=False) as arrays:
            require(arrays["exact_gradients"].shape == arrays[proposal_key].shape == (count, 576), "vector dimensions changed")
            for i, row in enumerate(models):
                family, label, index = row["family"], row["checkpoint"], int(row["original_index"])
                key = family, label, index
                require(label in expected_labels and index in selected[family] and key not in all_models, "missing/duplicate cohort unit")
                all_models.add(key)
                require(str(arrays["family"][i]) == family and int(arrays["original_index"][i]) == index
                        and str(arrays["model_id"][i]) == row["model_id"], "CSV/vector identity mismatch")
                require(constant_label == label if constant_label else str(arrays["checkpoint"][i]) == label, "vector model label changed")
                exact, proposal = arrays["exact_gradients"][i], arrays[proposal_key][i]
                require(np.all(np.isfinite(exact)) and np.all(np.isfinite(proposal)), "nonfinite retained gradient")
                en, pn = float(np.linalg.norm(exact)), float(np.linalg.norm(proposal))
                cosine = float(np.clip(np.dot(exact, proposal)/(en*pn), -1, 1)) if min(en, pn) > 1e-14 else 0.
                direction = -proposal/pn if pn > 1e-14 else np.zeros_like(proposal)
                exact_dd = float(np.dot(exact, direction))
                close(row["aggregate_gradient_cosine"], cosine)
                close(row["aggregate_exact_gradient_norm"], en)
                gradient_name = "aggregate_approximate_gradient_norm" if constant_label == "coarse_corrected_pod" else "aggregate_neural_gradient_norm"
                descent_name = "aggregate_approximate_true_descent" if constant_label == "coarse_corrected_pod" else "aggregate_neural_true_descent"
                close(row[gradient_name], pn)
                require(flag(row[descent_name]) == (exact_dd < 0), "true descent flag mismatch")
                case = family, index
                if case in exact_by_case:
                    require(np.allclose(exact_by_case[case], exact, rtol=2e-11, atol=2e-12), "methods used different exact objectives/gradients")
                else:
                    exact_by_case[case] = exact.copy()
                block_group = by_model[key]
                require(len(block_group) == 4 and {(int(b["source_index"]), float(b["frequency_hz"])) for b in block_group} == expected_blocks, "block identity mismatch")
                close(row["aggregate_exact_objective"], sum(float(b["exact_objective"]) for b in block_group))
                if constant_label == "coarse_corrected_pod":
                    require(np.allclose(arrays["common_directions"][i], direction, rtol=1e-11, atol=1e-12), "shared direction mismatch")
                    close(row["aggregate_approximate_directional_derivative"], np.dot(proposal, direction))
                    close(row["aggregate_exact_directional_derivative"], exact_dd)
                    close(sum(float(b["approximate_directional_derivative"]) for b in block_group), np.dot(proposal, direction))
                    close(sum(float(b["exact_directional_derivative"]) for b in block_group), exact_dd)
                    available = all(flag(b["certificate_available"]) for b in block_group)
                    require(flag(row["certificate_available"]) == available, "availability changed")
                    if available:
                        bound = sum(float(b["deterministic_bound"]) for b in block_group)
                        upper = float(np.dot(proposal, direction)) + bound
                        close(row["aggregate_deterministic_bound"], bound)
                        close(row["aggregate_certified_upper_derivative"], upper)
                        require(abs(exact_dd - np.dot(proposal, direction)) <= bound + 1e-12, "observed aggregate error exceeds bound")
                        require(flag(row["certified_descent"]) == (upper < 0), "certificate acceptance changed")
                    else:
                        require(row["aggregate_deterministic_bound"] == row["aggregate_certified_upper_derivative"] == ""
                                and not flag(row["certified_descent"]), "unavailable aggregate lost abstention")
                    require(flag(row["exact_fallback_required"]) == (not flag(row["certified_descent"])), "fallback flag changed")
                    require(not (flag(row["certified_descent"]) and exact_dd >= 0), "observed false certification")
                    cost = json.loads(row["online_cost_json"])
                    for category in ("reference_solver", "coarse_stability"):
                        for name, value in cost[category].items():
                            if (category == "reference_solver" and (name == "factorizations" or "rhs" in name or "svd" in name)) or name.startswith("fine_grid_"):
                                require(value == 0, "hidden reported online fine solve/SVD")
                groups[family, label].append((cosine, exact_dd < 0, flag(row["certified_descent"]) if "certified_descent" in row else None))
        counters[relative] = {"model_rows": len(models), "block_rows": len(blocks)}
    require(len(all_models) == 1000 and all_models == {(family, label, index) for family, ids in selected.items() for label in expected_labels for index in ids}, "five-model full cohort mismatch")
    require(len(source_revisions) == len(source_digests) == 1 and all(m == manifests[0] for m in manifests), "methods did not share frozen source")
    revision, manifest = next(iter(source_revisions)), manifests[0]
    digest = hashlib.sha256()
    for name, checksum in sorted(manifest.items()):
        historical = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=root)
        require(hashlib.sha256(historical).hexdigest() == checksum == sha(root/name), f"frozen source mismatch: {name}")
        digest.update(name.encode() + b"\0" + checksum.encode() + b"\0")
    require(digest.hexdigest() == next(iter(source_digests)), "frozen source digest mismatch")
    probability = lambda failures: math.prod((300-i)/(400-i) for i in range(failures))
    zero_failure_upper = max(k for k in range(301) if probability(k) >= .05)
    require(zero_failure_upper == 10, "finite-population zero-failure cross-check changed")
    receipt = {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_script_sha256": sha(__file__), "source_revision": revision, "source_sha256": digest.hexdigest(),
        "source_files_verified_against_git_and_current": len(manifest), "counters": counters,
        "all_model_label_rows": len(all_models), "all_original_model_slots": len(exact_by_case),
        "zero_false_failure_95pct_upper_per_fixed400_family": zero_failure_upper/400,
        "groups": [{"family": family, "model": label, "n": len(values), "mean_gradient_cosine": float(np.mean([v[0] for v in values])),
                    "true_descent_models": sum(v[1] for v in values), "certified_models": None if values[0][2] is None else sum(v[2] for v in values)}
                   for (family, label), values in sorted(groups.items())],
        "artifact_sha256": {str(p.relative_to(root)): sha(p) for p in sorted(study.rglob("*")) if p.is_file()},
        "limits": ["Separate AI-assisted arithmetic/provenance audit, not independent human reproduction.",
                   "Saved full aggregate vectors are checked; per-block full gradients were checked at execution time and are not independently reconstructed here.",
                   "Reported operation counters are checked for forbidden fine work, not reconstructed from every executed operation.",
                   "No PDE re-solve, formal roundoff enclosure, trajectory, runtime or field-data claim.",
                   "The 2.5% upper bound is individual per named fixed400 complement under uniform100-index sampling, not simultaneous or universal."]}
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: receipt[k] for k in ("status", "all_model_label_rows", "all_original_model_slots", "source_files_verified_against_git_and_current")}))


if __name__ == "__main__":
    main()
