import csv

import numpy as np
import pytest

from wavecert.experiments.certificate_calibration import _split_quantile
from wavecert.experiments.real_wavecert import _conformal_quantile
from wavecert.statistics import (
    binomial_failure_summary,
    cluster_bootstrap_mean,
    error_ratios,
    scale_error_bounds,
    split_conformal_quantile,
)


def test_small_calibration_cannot_claim_unsupported_finite_coverage():
    scores = np.arange(5.0)
    for quantile in (split_conformal_quantile, _split_quantile, _conformal_quantile):
        assert np.isposinf(quantile(scores, 0.1))
    assert np.isfinite(split_conformal_quantile(np.arange(9.0), 0.1))
    assert np.isposinf(split_conformal_quantile(np.array([np.inf]), 0.5))


def test_order_statistic_has_requested_exchangeable_rank_coverage():
    # Each point is equally likely to be the new exchangeable observation.
    population = np.arange(10.0)
    covered = []
    for new in range(len(population)):
        calibration = np.delete(population, new)
        covered.append(population[new] <= split_conformal_quantile(calibration, 0.1))
    assert np.mean(covered) == 0.9


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, np.nan, np.inf, True])
def test_invalid_miscoverage_rejected(alpha):
    with pytest.raises(ValueError):
        split_conformal_quantile(np.arange(10.0), alpha)


@pytest.mark.parametrize("scores", [[], [[1.0, 2.0]], [np.nan], [-np.inf]])
def test_invalid_calibration_scores_rejected(scores):
    with pytest.raises(ValueError):
        split_conformal_quantile(np.asarray(scores), 0.1)


def test_zero_reference_bound_does_not_hide_a_positive_error():
    ratios = error_ratios(np.array([0.0, 1.0, 4.0]), np.array([0.0, 0.0, 2.0]))
    np.testing.assert_array_equal(ratios, [0.0, np.inf, 2.0])
    bounds = scale_error_bounds(np.array([0.0, 1.0]), np.inf)
    assert np.all(np.isposinf(bounds))
    assert not np.any(np.isnan(bounds))


def test_no_observed_failures_still_has_positive_probability_upper_limit():
    summary = binomial_failure_summary(np.zeros(30, dtype=bool))
    assert summary["one_sided_failure_probability_upper"] == pytest.approx(1 - 0.05 ** (1 / 30))
    assert binomial_failure_summary(np.array([False]))["one_sided_failure_probability_upper"] == pytest.approx(0.95)
    assert binomial_failure_summary(np.array([], dtype=bool))["one_sided_failure_probability_upper"] == 1.0


def test_correlated_slice_repetition_cannot_create_independent_parents():
    single = cluster_bootstrap_mean(np.arange(55.0), np.repeat("one-earth-model", 55))
    assert single["n_parent_groups"] == 1
    assert single["interval"] is None
    values = np.array([0.0, 1.0, 2.0])
    parents = np.array(["a", "b", "c"])
    first = cluster_bootstrap_mean(values, parents, n_resamples=200, seed=1)
    repeated = cluster_bootstrap_mean(np.repeat(values, 10), np.repeat(parents, 10), n_resamples=200, seed=1)
    assert first["interval"] == repeated["interval"]
    assert repeated["n_parent_groups"] == 3


def test_missing_or_duplicate_block_rows_cannot_inflate_simultaneous_coverage(tmp_path):
    from wavecert.experiments.certificate_calibration import run_certificate_calibration

    cases = [{"case_id": i} for i in range(2)]
    blocks = [{"case_id": 0, "label": "a"}, {"case_id": 0, "label": "b"}, {"case_id": 1, "label": "a"}]
    for name, rows in [("cases.csv", cases), ("blocks.csv", blocks)]:
        with (tmp_path / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with pytest.raises(ValueError, match="exactly one row per case/label"):
        run_certificate_calibration(phase4_case_csv=tmp_path / "cases.csv", phase4_block_csv=tmp_path / "blocks.csv", output_dir=tmp_path / "out")
