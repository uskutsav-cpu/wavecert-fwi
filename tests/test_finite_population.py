import numpy as np
import pytest
from scipy.stats import hypergeom

from wavecert.statistics.finite_population import finite_population_failure_summary


@pytest.mark.parametrize("size,sample,confidence", [(9, 4, 0.8), (12, 5, 0.95), (8, 8, 0.9)])
def test_hypergeometric_bound_has_at_least_declared_coverage_for_every_failure_count(size, sample, confidence):
    bounds = [finite_population_failure_summary(
        np.array([1] * observed + [0] * (sample - observed)), population_size=size,
        confidence=confidence,
    )["one_sided_failure_count_upper"] for observed in range(sample + 1)]
    for actual in range(size + 1):
        noncoverage = sum(hypergeom.pmf(observed, size, actual, sample)
                          for observed in range(sample + 1) if bounds[observed] < actual)
        assert noncoverage <= 1 - confidence + 1e-14
        for observed, bound in enumerate(bounds):
            if hypergeom.pmf(observed, size, actual, sample) == 0:
                continue
            assert hypergeom.cdf(observed, size, bound, sample) >= 1-confidence-1e-14
            if bound < size - sample + observed:
                assert hypergeom.cdf(observed, size, bound+1, sample) < 1-confidence+1e-14


def test_hypergeometric_zero_all_failures_and_census_are_explicit():
    zero = finite_population_failure_summary(np.zeros(100), population_size=500)
    assert 0 < zero["one_sided_failure_count_upper"] < 500
    all_fail = finite_population_failure_summary(np.ones(100), population_size=500)
    assert all_fail["one_sided_failure_fraction_upper"] == 1
    census = finite_population_failure_summary(np.array([0, 1, 0, 1]), population_size=4)
    assert census["one_sided_failure_count_upper"] == 2


@pytest.mark.parametrize("failures,size", [([], 3), ([0, 1], 1), ([0.5], 4), ([[0, 1]], 4), ([0], True)])
def test_hypergeometric_bound_rejects_invalid_sampling_inputs(failures, size):
    with pytest.raises(ValueError):
        finite_population_failure_summary(np.asarray(failures), population_size=size)
