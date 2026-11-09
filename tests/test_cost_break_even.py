import pytest

from wavecert.experiments.cost_break_even import measure_receiver_map_cost, receiver_map_cost_model


@pytest.mark.parametrize("sources,receivers", [(2, 4), (2, 2), (6, 2)])
def test_counter_measurements_match_initial_map_break_even(sources, receivers):
    result = measure_receiver_map_cost(sources=sources, receivers=receivers, shape=(12, 12))
    assert result["observed_initial_map_rhs"] == 4 * receivers
    assert result["observed_full_gradient_rhs"] == 4 * sources
    assert result["certificate_to_gradient_rhs_ratio"] == pytest.approx(receivers / sources)
    assert result["rhs_saving_impossible_even_with_zero_repairs"] == (receivers >= sources)
    assert result["certificate_costs"]["svd_evaluations"] == 4
    assert result["exact_gradient_costs"]["svd_evaluations"] == 0
    assert result["minimum_synthetic_residual_norm"] > 0
    assert result["exact_objective"] > 0
    assert result["observed_counts_match_formulas"]


@pytest.mark.parametrize("kwargs", [
    {"sources": 0}, {"receivers": -1}, {"frequencies": 1.5}, {"sources": True},
])
def test_cost_model_rejects_invalid_count_inputs(kwargs):
    options = dict(sources=2, receivers=4, frequencies=2)
    options.update(kwargs)
    with pytest.raises(ValueError):
        receiver_map_cost_model(**options)
