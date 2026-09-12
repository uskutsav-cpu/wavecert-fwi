from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wavecert.experiments.end_to_end import _blocks_for_model, _make_problem
from wavecert.experiments.synthetic import SyntheticProblem, velocity_to_m
from wavecert.metrics import cosine_similarity, relative_l2
from wavecert.physics.helmholtz import Helmholtz2D, SurveyGeometry
from wavecert.repair.adaptive import adaptive_hybrid_direction
from wavecert.surrogates.fno import TrainedFNOWavefieldSurrogate


def _normalised_negative(gradient: np.ndarray) -> np.ndarray:
    gradient = np.asarray(gradient, dtype=float)
    norm = float(np.linalg.norm(gradient))
    return np.zeros_like(gradient) if norm <= 1e-14 else -gradient / norm


def _make_baseline_case(
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> SyntheticProblem:
    """Create one reproducible random baseline case.

    This intentionally reuses the same random problem family as the Phase-7
    end-to-end trajectory study. `build_synthetic_problem()` is a separate,
    deterministic reference problem and does not accept a seed.
    """

    physics = Helmholtz2D(
        shape=shape,
        spacing=0.05,
        damping_width=max(3, min(shape) // 8),
        damping_strength=2.0,
    )
    geometry = SurveyGeometry.from_grid(
        shape,
        n_sources=6,
        n_receivers=min(20, shape[1] - 4),
        source_depth=2,
        receiver_depth=2,
        margin=2,
    )
    velocity_true, velocity0 = _make_problem(shape, rng)
    m_true = velocity_to_m(velocity_true).reshape(-1)
    m0 = velocity_to_m(velocity0).reshape(-1)
    blocks = _blocks_for_model(physics, geometry, m_true)
    return SyntheticProblem(
        physics=physics,
        geometry=geometry,
        m_true=m_true,
        m0=m0,
        blocks=blocks,
    )


def _exact_block_gradient(problem: SyntheticProblem, m: np.ndarray, block) -> np.ndarray:
    q = problem.physics.source(block.source_index)
    _, grad, _, _ = problem.physics.objective_and_gradient(
        m,
        q,
        block.observed,
        problem.geometry.receiver_indices,
        block.frequency_hz,
    )
    return np.asarray(grad, dtype=float)


def _neural_block_gradient(
    surrogate: TrainedFNOWavefieldSurrogate,
    problem: SyntheticProblem,
    m: np.ndarray,
    block,
) -> tuple[np.ndarray, float]:
    q = problem.physics.source(block.source_index)
    _, grad, u_neural, _ = surrogate.objective_and_gradient(
        m,
        q,
        block.observed,
        problem.geometry.receiver_indices,
        block.frequency_hz,
    )
    u_exact = problem.physics.solve_state(m, q, block.frequency_hz)
    exact_rec = problem.physics.restrict(
        u_exact,
        problem.geometry.receiver_indices,
    )
    neural_rec = problem.physics.restrict(
        u_neural,
        problem.geometry.receiver_indices,
    )
    error = float(relative_l2(exact_rec, neural_rec))
    return np.asarray(grad, dtype=float), error


def run_heuristic_baselines(
    *,
    checkpoint_path: str | Path = "checkpoints/fno_phase2.pt",
    conformal_scales_path: str | Path = "results/phase5/conformal_scales.json",
    output_dir: str | Path = "results/heuristic_baselines",
    n_cases: int = 80,
    random_repairs: int = 1,
    periodic_interval: int = 4,
    forward_error_threshold: float = 0.20,
    seed: int = 20261201,
) -> dict:
    """Compare simple fallback heuristics with WaveCert on matched cases.

    Costs are exact source/frequency gradient blocks. The forward-error
    threshold is an oracle diagnostic baseline because its gate needs the exact
    reference receiver field; it is not a deployable policy.
    """

    surrogate = TrainedFNOWavefieldSurrogate.load(checkpoint_path)
    calibration = json.loads(Path(conformal_scales_path).read_text())
    block_scales = {
        key: float(value)
        for key, value in calibration["block_scales_by_label"].items()
    }
    rng = np.random.default_rng(seed)

    policies: dict[str, list[dict]] = {
        "neural-only": [],
        "random-repair": [],
        "periodic-exact": [],
        "forward-error-threshold": [],
        "wavecert-selective": [],
        "exact": [],
    }

    for _case_id in range(n_cases):
        problem = _make_baseline_case(surrogate.shape, rng)
        blocks = problem.blocks
        m = np.asarray(problem.m0, dtype=float)

        exact_blocks = [
            _exact_block_gradient(problem, m, block)
            for block in blocks
        ]
        neural_pairs = [
            _neural_block_gradient(surrogate, problem, m, block)
            for block in blocks
        ]
        neural_blocks = [pair[0] for pair in neural_pairs]
        forward_errors = [pair[1] for pair in neural_pairs]

        exact_gradient = np.sum(exact_blocks, axis=0)
        neural_gradient = np.sum(neural_blocks, axis=0)

        def score(
            direction: np.ndarray,
            exact_cost: int,
            reference_gradient: np.ndarray,
        ) -> dict:
            dd = float(np.dot(reference_gradient, direction))
            return {
                "true_descent": dd < 0.0,
                "exact_blocks": exact_cost,
                "cosine_to_exact": float(
                    cosine_similarity(
                        _normalised_negative(reference_gradient),
                        direction,
                    )
                ),
            }

        policies["neural-only"].append(
            score(
                _normalised_negative(neural_gradient),
                0,
                exact_gradient,
            )
        )
        policies["exact"].append(
            score(
                _normalised_negative(exact_gradient),
                len(blocks),
                exact_gradient,
            )
        )

        chosen = {
            int(index)
            for index in rng.choice(
                len(blocks),
                size=min(random_repairs, len(blocks)),
                replace=False,
            )
        }
        random_gradient = np.sum(
            [
                exact_blocks[index]
                if index in chosen
                else neural_blocks[index]
                for index in range(len(blocks))
            ],
            axis=0,
        )
        policies["random-repair"].append(
            score(
                _normalised_negative(random_gradient),
                len(chosen),
                exact_gradient,
            )
        )

        # Periodic-exact is intentionally keyed to the simulation case index,
        # not RNG state, so its cost schedule is completely reproducible.
        case_index = len(policies["periodic-exact"])
        if case_index % max(periodic_interval, 1) == 0:
            periodic_gradient = exact_gradient
            periodic_cost = len(blocks)
        else:
            periodic_gradient = neural_gradient
            periodic_cost = 0
        policies["periodic-exact"].append(
            score(
                _normalised_negative(periodic_gradient),
                periodic_cost,
                exact_gradient,
            )
        )

        flagged = {
            index
            for index, error in enumerate(forward_errors)
            if error > forward_error_threshold
        }
        threshold_gradient = np.sum(
            [
                exact_blocks[index]
                if index in flagged
                else neural_blocks[index]
                for index in range(len(blocks))
            ],
            axis=0,
        )
        policies["forward-error-threshold"].append(
            score(
                _normalised_negative(threshold_gradient),
                len(flagged),
                exact_gradient,
            )
        )

        result = adaptive_hybrid_direction(
            physics=problem.physics,
            surrogate=surrogate,
            m=m,
            blocks=blocks,
            receiver_indices=problem.geometry.receiver_indices,
            stability_mode="directional",
            bound_scales=block_scales,
        )
        policies["wavecert-selective"].append(
            score(
                result.direction,
                result.exact_block_evaluations,
                exact_gradient,
            )
        )

    summary: dict[str, dict] = {}
    for name, rows in policies.items():
        summary[name] = {
            "true_descent_rate": float(
                np.mean([row["true_descent"] for row in rows])
            ),
            "mean_exact_blocks": float(
                np.mean([row["exact_blocks"] for row in rows])
            ),
            "mean_cosine_to_exact": float(
                np.mean([row["cosine_to_exact"] for row in rows])
            ),
        }

    result = {
        "seed": seed,
        "n_cases": n_cases,
        "random_repairs": random_repairs,
        "periodic_interval": periodic_interval,
        "forward_error_threshold": forward_error_threshold,
        "case_generator": "phase7_end_to_end_random_problem_family",
        "policies": summary,
        "semantics": {
            "forward_error_threshold": (
                "diagnostic oracle baseline because it measures exact receiver error"
            ),
            "wavecert_selective": (
                "deployable residual/certificate policy"
            ),
        },
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result
