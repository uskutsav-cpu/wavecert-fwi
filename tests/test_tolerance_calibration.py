import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import beta, binom, multinomial

from wavecert.statistics.tolerance import pac_tolerance_rank, pac_tolerance_threshold


@pytest.mark.parametrize("n,expected", [
    (0, None), (12, None), (28, None), (29, 29), (30, 30), (60, 59), (100, 96),
])
def test_conventional_tolerance_request_requires_supported_calibration_rank(n, expected):
    assert pac_tolerance_rank(n, alpha=0.1, delta=0.05) == expected


@pytest.mark.parametrize("n,alpha,delta", [(5, 0.5, 0.1), (30, 0.1, 0.05), (100, 0.1, 0.05)])
def test_selected_rank_satisfies_continuous_order_statistic_coverage_law(n, alpha, delta):
    rank = pac_tolerance_rank(n, alpha=alpha, delta=delta)
    assert rank is not None
    # F(S_(k)) has a Beta(k,n+1-k) law for continuous IID scores. This
    # independently checks both coverage and minimality of the chosen rank.
    assert beta.cdf(1 - alpha, rank, n + 1 - rank) <= delta + 1e-14
    if rank > 1:
        assert beta.cdf(1 - alpha, rank - 1, n + 2 - rank) > delta


@pytest.mark.parametrize("probabilities", [
    (0.49, 0.50, 0.01), (0.69, 0.01, 0.30), (0.2, 0.6, 0.2), (0.999, 0, 0.001),
])
def test_tied_discrete_and_infinite_scores_have_conservative_tolerance_coverage(probabilities):
    n, alpha, delta = 5, 0.3, 0.2
    failure_probability = 0.0
    coverage_by_value = {0.0: probabilities[0], 1.0: sum(probabilities[:2]), np.inf: 1.0}
    # Exhaust the sampling distribution of scores on {0,1,+infinity}; no
    # simulation, distinct-score assumption, or asserted IID empirical proof.
    for zeros in range(n + 1):
        for ones in range(n - zeros + 1):
            counts = [zeros, ones, n - zeros - ones]
            scores = np.repeat([0.0, 1.0, np.inf], counts)
            result = pac_tolerance_threshold(scores, alpha=alpha, delta=delta)
            if coverage_by_value[result.threshold] < 1 - alpha:
                failure_probability += multinomial.pmf(counts, n, probabilities)
    assert failure_probability <= delta + 1e-14


def test_rank_support_and_an_infinite_selected_score_remain_distinct_in_strict_json():
    unsupported = pac_tolerance_threshold(np.arange(12), alpha=0.1, delta=0.05)
    assert unsupported.rank is None and np.isposinf(unsupported.threshold)
    selected_infinite = pac_tolerance_threshold(np.r_[np.zeros(29), np.inf], alpha=0.1, delta=0.05)
    assert selected_infinite.rank == 30 and np.isposinf(selected_infinite.threshold)
    for result, supported in [(unsupported, False), (selected_infinite, True)]:
        saved = json.loads(json.dumps(result.to_dict(), allow_nan=False))
        assert saved["finite_rank_supported"] is supported
        assert saved["threshold"] is None and not saved["finite_threshold"]
    assert unsupported.to_dict()["threshold_kind"] != selected_infinite.to_dict()["threshold_kind"]
    finite = pac_tolerance_threshold(np.r_[np.arange(96), np.full(4, np.inf)], alpha=0.1, delta=0.05)
    assert finite.rank == 96 and finite.threshold == 95


def test_tiny_delta_does_not_round_confidence_to_one_and_silently_support_a_rank():
    # At n=400, even the maximum leaves >1e-20 calibration failure risk.
    # A naive CDF comparison against float(1-1e-20)==1 would accept a rank.
    assert 1 - 1e-20 == 1
    assert binom.cdf(399, 400, 0.9) == 1
    assert pac_tolerance_rank(400, alpha=0.1, delta=1e-20) is None
    assert pac_tolerance_rank(438, alpha=0.1, delta=1e-20) == 438


def test_extreme_tail_does_not_use_an_underflowed_binomial_sf_as_zero_failure_risk():
    n, delta = 7000, 1e-320
    # A zero from a direct numerical SF at an intermediate rank must not
    # silently make a request supported. At rank n-1 the final two binomial
    # outcomes alone exceed delta; the maximum rank is the first valid one.
    final_two_log_probability = np.logaddexp(
        binom.logpmf(n - 1, n, 0.9), binom.logpmf(n, n, 0.9),
    )
    assert final_two_log_probability > np.log(delta)
    assert binom.logpmf(n, n, 0.9) <= np.log(delta)
    result = pac_tolerance_threshold(np.arange(n), alpha=0.1, delta=delta)
    assert result.rank == n and result.threshold == n - 1
    assert result.rank_failure_probability_upper > 0
    assert np.isfinite(result.rank_failure_log_probability_upper)
    json.dumps(result.to_dict(), allow_nan=False)


def test_zero_quantile_is_not_replaced_by_a_deployment_scale_in_the_math_helper():
    result = pac_tolerance_threshold(np.zeros(30), alpha=0.1, delta=0.05)
    assert result.threshold == 0 and result.to_dict()["threshold"] == 0


@pytest.mark.parametrize("scores", [[], [[0, 1]], [0, np.nan], [0, -np.inf]])
def test_invalid_score_arrays_are_rejected(scores):
    with pytest.raises(ValueError, match="scores"):
        pac_tolerance_threshold(np.asarray(scores), alpha=0.1, delta=0.05)


@pytest.mark.parametrize("n,alpha,delta", [
    (-1, 0.1, 0.05), (1.5, 0.1, 0.05), (True, 0.1, 0.05),
    (30, True, 0.05), (30, np.nan, 0.05), (30, 0, 0.05),
    (30, 0.1, False), (30, 0.1, np.inf), (30, 0.1, 1),
])
def test_invalid_sample_counts_and_probability_requests_are_rejected(n, alpha, delta):
    with pytest.raises(ValueError):
        pac_tolerance_rank(n, alpha=alpha, delta=delta)


def test_saved_score_diagnostic_preserves_original_results_and_labels_posthoc_comparison(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = runpy.run_path(str(root / "scripts/diagnose_tolerance_calibration.py"))
    source = root / "results/evidence_study"
    original = {name: (source / name).read_bytes() for name in script["INPUT_NAMES"]}
    result = script["run_diagnostic"](source, tmp_path / "diagnostic", repository_root=root)
    assert result["status"] == "complete"
    assert result["marginal"]["rank"] == 28
    assert result["marginal"]["observed_held_out"]["covered"] == 45
    assert result["pac_tolerance"]["rank"] == 30
    assert result["pac_tolerance"]["observed_held_out"]["covered"] == 52
    assert result["pac_tolerance"]["threshold"] > result["marginal"]["original_deployed_scale"]
    assert "Post-hoc" in result["analysis_status"]
    assert all((source / name).read_bytes() == data for name, data in original.items())
    assert json.loads((tmp_path / "diagnostic/status.json").read_text())["status"] == "complete"
    # Preserve the source-study directory even if a user specifies it as output.
    with pytest.raises(ValueError, match="outside"):
        script["run_diagnostic"](source, source / "diagnostic", repository_root=root)
    changed = tmp_path / "tampered_study"
    changed.mkdir()
    for name in script["INPUT_NAMES"]:
        shutil.copyfile(source / name, changed / name)
    calibration = changed / "calibration.csv"
    lines = calibration.read_text().splitlines()
    calibration.write_text("\n".join([*lines[:-1], lines[1]]) + "\n")
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        script["run_diagnostic"](changed, tmp_path / "rejected", repository_root=root)
    assert json.loads((tmp_path / "rejected/status.json").read_text())["status"] == "failed"
