import csv
import json

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


def test_duplicate_scenarios_cannot_inflate_experiment_sample_count(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        run_evidence_study(output_dir=tmp_path, config=EvidenceConfig(scenarios=("clean", "clean")))
