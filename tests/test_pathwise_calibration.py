import numpy as np

from wavecert.experiments.synthetic import build_synthetic_problem
from wavecert.repair.adaptive import adaptive_hybrid_direction
from wavecert.statistics.pathwise import pathwise_case_score_details
from wavecert.surrogates.low_fidelity import SmoothedHelmholtzSurrogate


def test_complete_path_score_covers_changed_directions_until_acceptance():
    problem = build_synthetic_problem(
        shape=(8, 8), frequencies_hz=(3.0, 3.5), n_sources=2, n_receivers=6
    )
    surrogate = SmoothedHelmholtzSurrogate(problem.physics, smoothing_sigma=0.8, model_scale=1.015)
    score = pathwise_case_score_details(
        physics=problem.physics, surrogate=surrogate, m=problem.m0,
        blocks=problem.blocks, receiver_indices=problem.geometry.receiver_indices,
    )
    assert score.certificates_scored == 10  # 4+3+2+1 on the complete four-block path.
    assert set(score.repaired_order) == {block.label for block in problem.blocks}
    assert np.isfinite(score.score) and score.score >= 0
    result = adaptive_hybrid_direction(
        physics=problem.physics, surrogate=surrogate, m=problem.m0,
        blocks=problem.blocks, receiver_indices=problem.geometry.receiver_indices,
        stability_mode="directional",
        bound_scales={block.label: max(score.score * (1 + 1e-12), 1e-12) for block in problem.blocks},
    )
    gradient = np.sum([
        problem.physics.objective_and_gradient(
            problem.m0, problem.physics.source(block.source_index), block.observed,
            problem.geometry.receiver_indices, block.frequency_hz,
        )[1]
        for block in problem.blocks
    ], axis=0)
    assert result.certified_descent
    assert float(np.dot(gradient, result.direction)) < 0
