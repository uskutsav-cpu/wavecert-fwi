import csv
import json
import runpy
from pathlib import Path

import pytest

pytest.importorskip("torch")

from wavecert.experiments.evidence_study import EvidenceConfig, run_evidence_study


def test_evidence_pipeline_records_real_costs_and_verified_trajectories(tmp_path):
    config = EvidenceConfig(
        seeds=(731,),
        cases_per_seed=1,
        iterations=1,
        calibration_cases=2,
        alpha=0.5,
        scenarios=("clean",),
        policies=("exact", "random-repair", "conformal-selective", "analytic-selective"),
    )
    result = run_evidence_study(output_dir=tmp_path, config=config)
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["configuration"]["checkpoint_sha256"]
    assert result["comparisons"]["clean"]["exact"]["mean_online_adjoint_rhs"] > 0
    assert result["clean_initial_pathwise_coverage"]["failure_uncertainty"]["n_units"] == 1
    rows = list(csv.DictReader((tmp_path / "cases.csv").open()))
    assert len(rows) == 4
    assert all(int(row["objective_increases"]) == 0 for row in rows)
    assert all(float(row["validation_seconds"]) > 0 for row in rows)
    assert all(
        float(row["final_objective"]) <= float(row["initial_objective"]) + 1e-12 for row in rows
    )
    audit_study = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/audit_evidence_study.py")
    )["audit_study"]
    audit = audit_study(tmp_path, expected_models=1, expected_iterations=1)
    assert audit["audit_passed"] and audit["trajectory_rows"] == 8
    # A plausible saved summary must not hide an omitted policy result.
    case_file = tmp_path / "cases.csv"
    original = case_file.read_text()
    case_file.write_text("\n".join(original.splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="case table is incomplete"):
        audit_study(tmp_path)
    case_file.write_text(original)
    trajectory_file = tmp_path / "trajectories.csv"
    records = list(csv.DictReader(trajectory_file.open()))
    records[0]["iteration"] = records[1]["iteration"]
    with trajectory_file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with pytest.raises(ValueError, match="incomplete or duplicated iterations"):
        audit_study(tmp_path)


def test_duplicate_scenarios_cannot_inflate_experiment_sample_count(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        run_evidence_study(output_dir=tmp_path, config=EvidenceConfig(scenarios=("clean", "clean")))
