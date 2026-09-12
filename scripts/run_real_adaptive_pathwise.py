from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi

from wavecert.certificates.directional import certify_direction
from wavecert.data.wavefields import WavefieldDatasetArrays
from wavecert.metrics import cosine_similarity
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock
from wavecert.repair.adaptive import (
    adaptive_hybrid_direction,
    global_fallback_direction,
)
from wavecert.surrogates.fno import TrainedFNOWavefieldSurrogate


@dataclass(frozen=True)
class RealCase:
    slice_id: str
    model_type: str
    m_eval: np.ndarray
    blocks: tuple[SurveyBlock, ...]


@dataclass(frozen=True)
class PolicyRow:
    split: str
    slice_id: str
    model_type: str
    policy: str
    certified: bool
    true_descent: bool
    false_certification: bool
    true_directional_derivative: float
    cosine_to_exact: float
    exact_blocks: int
    exact_fraction: float
    mode: str


def _normalised_negative(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, dtype=float)
    norm = float(np.linalg.norm(g))
    return np.zeros_like(g) if norm <= 1e-14 else -g / norm


def _split_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    if n == 0:
        raise ValueError("empty calibration score array")
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(scores[k - 1])


def _load_cases(
    dataset_dir: Path,
    split: str,
    physics: Helmholtz2D,
    sigma: float,
) -> list[RealCase]:
    arrays = WavefieldDatasetArrays.load(dataset_dir / f"{split}.npz")
    meta = pd.read_csv(dataset_dir / f"{split}_metadata.csv")
    if len(meta) != len(arrays.models):
        raise ValueError(f"{split}: metadata/data length mismatch")

    receiver_indices = tuple(int(x) for x in arrays.receiver_indices)
    cases: list[RealCase] = []

    for slice_id, group in meta.groupby("slice_id", sort=False):
        indices = group.index.to_numpy(dtype=int)
        if len(indices) != 24:
            raise ValueError(f"{split}/{slice_id}: expected 24 blocks, found {len(indices)}")

        m_true = arrays.models[indices[0]].astype(np.float64).reshape(-1)
        velocity = (1.0 / np.sqrt(m_true)).reshape(arrays.shape)
        velocity_eval = ndi.gaussian_filter(velocity, sigma=sigma, mode="reflect")
        m_eval = (1.0 / np.maximum(velocity_eval, 1e-8) ** 2).reshape(-1)

        blocks: list[SurveyBlock] = []
        for index in indices:
            source_index = int(arrays.source_indices[index])
            frequency = float(arrays.frequencies_hz[index])
            stored = arrays.wavefields[index]
            u_true = (stored[..., 0] + 1j * stored[..., 1]).reshape(-1)
            observed = physics.restrict(u_true, receiver_indices)
            blocks.append(
                SurveyBlock(
                    source_index=source_index,
                    frequency_hz=frequency,
                    observed=observed,
                    label=f"s{source_index}_f{frequency:g}",
                )
            )

        labels = [b.label for b in blocks]
        if len(set(labels)) != 24:
            raise ValueError(f"{split}/{slice_id}: block labels are not unique")

        cases.append(
            RealCase(
                slice_id=str(slice_id),
                model_type=str(group["model_type"].iloc[0]),
                m_eval=m_eval,
                blocks=tuple(blocks),
            )
        )

    return cases


def _pathwise_case_score(
    *,
    case: RealCase,
    physics: Helmholtz2D,
    surrogate: TrainedFNOWavefieldSurrogate,
    receiver_indices: tuple[int, ...],
) -> tuple[float, int]:
    """Worst normalized certificate error across the full greedy repair path.

    The repair order is generated with the unscaled deterministic bounds.
    A single uniform conformal multiplier leaves this ranking unchanged, so
    calibrating the maximum score over the complete path covers every stopping
    round that the uniformly scaled adaptive policy may encounter.
    """

    states: list[dict] = []
    for block in case.blocks:
        q = physics.source(block.source_index)
        _, neural_gradient, _, _ = surrogate.objective_and_gradient(
            case.m_eval,
            q,
            block.observed,
            receiver_indices,
            block.frequency_hz,
        )
        states.append(
            {
                "block": block,
                "q": q,
                "neural_gradient": np.asarray(neural_gradient, dtype=float),
                "exact_gradient": None,
            }
        )

    frequencies = sorted({float(s["block"].frequency_hz) for s in states})
    constants: dict[float, tuple[np.ndarray, float]] = {}
    for frequency in frequencies:
        matrix = physics.receiver_resolvent_matrix(
            case.m_eval,
            frequency,
            receiver_indices,
        )
        alpha = physics.receiver_resolvent_norm(
            case.m_eval,
            frequency,
            receiver_indices,
            receiver_resolvent_matrix=matrix,
        )
        constants[frequency] = (matrix, float(alpha))

    worst_ratio = 0.0
    scored_certificates = 0

    while True:
        hybrid_gradient = np.sum(
            [
                state["exact_gradient"]
                if state["exact_gradient"] is not None
                else state["neural_gradient"]
                for state in states
            ],
            axis=0,
        )
        direction = _normalised_negative(hybrid_gradient)
        if np.linalg.norm(direction) <= 1e-14:
            break

        unrepaired = [state for state in states if state["exact_gradient"] is None]
        if not unrepaired:
            break

        kappas: dict[float, float] = {}
        for frequency in frequencies:
            matrix, _alpha = constants[frequency]
            kappas[frequency] = float(
                physics.directional_receiver_tangent_resolvent_norm(
                    case.m_eval,
                    frequency,
                    receiver_indices,
                    direction,
                    receiver_resolvent_matrix=matrix,
                )
            )

        uncertainty: list[tuple[dict, float]] = []
        for state in unrepaired:
            block = state["block"]
            frequency = float(block.frequency_hz)
            matrix, alpha = constants[frequency]

            cert = certify_direction(
                physics=physics,
                surrogate=surrogate,
                m=case.m_eval,
                q=state["q"],
                observed=block.observed,
                receiver_indices=receiver_indices,
                frequency_hz=frequency,
                direction=direction,
                stability_mode="directional",
                receiver_resolvent_matrix=matrix,
                receiver_resolvent_norm=alpha,
                directional_tangent_resolvent_norm=kappas[frequency],
                validate_exact=True,
            )
            realized = float(cert.realized_error or 0.0)
            base = float(cert.directional_error_bound)
            ratio = realized / max(base, 1e-15)
            worst_ratio = max(worst_ratio, ratio)
            scored_certificates += 1
            uncertainty.append((state, base))

        # Uniform scaling does not change this greedy ordering.
        selected = max(uncertainty, key=lambda item: item[1])[0]
        block = selected["block"]
        _, exact_gradient, _, _ = physics.objective_and_gradient(
            case.m_eval,
            selected["q"],
            block.observed,
            receiver_indices,
            block.frequency_hz,
        )
        selected["exact_gradient"] = np.asarray(exact_gradient, dtype=float)

    return float(worst_ratio), int(scored_certificates)


def _block_gradients(
    *,
    case: RealCase,
    physics: Helmholtz2D,
    surrogate: TrainedFNOWavefieldSurrogate,
    receiver_indices: tuple[int, ...],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    exact: list[np.ndarray] = []
    neural: list[np.ndarray] = []
    for block in case.blocks:
        q = physics.source(block.source_index)
        _, eg, _, _ = physics.objective_and_gradient(
            case.m_eval,
            q,
            block.observed,
            receiver_indices,
            block.frequency_hz,
        )
        _, ng, _, _ = surrogate.objective_and_gradient(
            case.m_eval,
            q,
            block.observed,
            receiver_indices,
            block.frequency_hz,
        )
        exact.append(np.asarray(eg, dtype=float))
        neural.append(np.asarray(ng, dtype=float))
    return exact, neural


def _append(
    rows: list[PolicyRow],
    *,
    split: str,
    case: RealCase,
    policy: str,
    direction: np.ndarray,
    certified: bool,
    exact_blocks: int,
    mode: str,
    exact_total: np.ndarray,
    exact_direction: np.ndarray,
) -> None:
    true_dd = float(np.dot(exact_total, direction))
    true_descent = bool(true_dd < 0.0)
    rows.append(
        PolicyRow(
            split=split,
            slice_id=case.slice_id,
            model_type=case.model_type,
            policy=policy,
            certified=bool(certified),
            true_descent=true_descent,
            false_certification=bool(certified and not true_descent),
            true_directional_derivative=true_dd,
            cosine_to_exact=float(cosine_similarity(exact_direction, direction)),
            exact_blocks=int(exact_blocks),
            exact_fraction=float(exact_blocks / len(case.blocks)),
            mode=mode,
        )
    )


def _evaluate_cases(
    *,
    split: str,
    cases: list[RealCase],
    physics: Helmholtz2D,
    surrogate: TrainedFNOWavefieldSurrogate,
    receiver_indices: tuple[int, ...],
    conformal_scale: float,
    seed: int,
) -> list[PolicyRow]:
    rows: list[PolicyRow] = []
    rng = np.random.default_rng(seed)

    for case_index, case in enumerate(cases, start=1):
        exact_blocks, neural_blocks = _block_gradients(
            case=case,
            physics=physics,
            surrogate=surrogate,
            receiver_indices=receiver_indices,
        )
        exact_total = np.sum(exact_blocks, axis=0)
        neural_total = np.sum(neural_blocks, axis=0)
        exact_direction = _normalised_negative(exact_total)
        neural_direction = _normalised_negative(neural_total)

        _append(
            rows,
            split=split,
            case=case,
            policy="neural-only",
            direction=neural_direction,
            certified=False,
            exact_blocks=0,
            mode="unverified",
            exact_total=exact_total,
            exact_direction=exact_direction,
        )

        global_result = global_fallback_direction(
            physics=physics,
            surrogate=surrogate,
            m=case.m_eval,
            blocks=case.blocks,
            receiver_indices=receiver_indices,
            stability_mode="directional",
            bound_scale=conformal_scale,
        )
        _append(
            rows,
            split=split,
            case=case,
            policy="global-fallback",
            direction=global_result.direction,
            certified=global_result.certified_descent,
            exact_blocks=global_result.exact_block_evaluations,
            mode=global_result.mode,
            exact_total=exact_total,
            exact_direction=exact_direction,
        )

        uniform_scales = {block.label: conformal_scale for block in case.blocks}
        selective = adaptive_hybrid_direction(
            physics=physics,
            surrogate=surrogate,
            m=case.m_eval,
            blocks=case.blocks,
            receiver_indices=receiver_indices,
            stability_mode="directional",
            bound_scales=uniform_scales,
        )
        _append(
            rows,
            split=split,
            case=case,
            policy="wavecert-selective",
            direction=selective.direction,
            certified=selective.certified_descent,
            exact_blocks=selective.exact_block_evaluations,
            mode=selective.mode,
            exact_total=exact_total,
            exact_direction=exact_direction,
        )

        # Matched random-repair cost baselines.
        for k in (1, 2, 4, 8, 12, 15, 16, 20):
            chosen = set(
                int(i)
                for i in rng.choice(
                    len(case.blocks),
                    size=min(k, len(case.blocks)),
                    replace=False,
                )
            )
            hybrid = np.sum(
                [
                    exact_blocks[i] if i in chosen else neural_blocks[i]
                    for i in range(len(case.blocks))
                ],
                axis=0,
            )
            _append(
                rows,
                split=split,
                case=case,
                policy=f"random-{k}",
                direction=_normalised_negative(hybrid),
                certified=False,
                exact_blocks=len(chosen),
                mode="random-repair",
                exact_total=exact_total,
                exact_direction=exact_direction,
            )

        _append(
            rows,
            split=split,
            case=case,
            policy="exact",
            direction=exact_direction,
            certified=True,
            exact_blocks=len(case.blocks),
            mode="exact",
            exact_total=exact_total,
            exact_direction=exact_direction,
        )

        print(
            f"{split}: {case_index:02d}/{len(cases)} "
            f"{case.slice_id} selective={selective.exact_block_evaluations}/24 "
            f"mode={selective.mode}"
        )

    return rows


def _aggregate(rows: list[PolicyRow]) -> dict:
    result: dict[str, dict] = {}
    policies = sorted({row.policy for row in rows})
    for policy in policies:
        subset = [row for row in rows if row.policy == policy]
        exact = np.asarray([row.exact_blocks for row in subset], dtype=float)
        desc = np.asarray([row.true_descent for row in subset], dtype=bool)
        cert = np.asarray([row.certified for row in subset], dtype=bool)
        false = np.asarray([row.false_certification for row in subset], dtype=bool)
        cosine = np.asarray([row.cosine_to_exact for row in subset], dtype=float)
        modes = {
            mode: int(sum(row.mode == mode for row in subset))
            for mode in sorted({row.mode for row in subset})
        }
        result[policy] = {
            "n_cases": len(subset),
            "true_descent_rate": float(np.mean(desc)),
            "certified_rate": float(np.mean(cert)),
            "false_certifications": int(np.sum(false)),
            "mean_exact_blocks": float(np.mean(exact)),
            "median_exact_blocks": float(np.median(exact)),
            "p90_exact_blocks": float(np.quantile(exact, 0.90)),
            "mean_exact_fraction": float(np.mean(exact) / 24.0),
            "mean_cosine_to_exact": float(np.mean(cosine)),
            "mode_counts": modes,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="data/external/subsurfacegen-wavecert-v2",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/fno_subsurfacegen_v2.pt",
    )
    parser.add_argument(
        "--output",
        default="results/real_adaptive_pathwise",
    )
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--sigma", type=float, default=1.6)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    surrogate = TrainedFNOWavefieldSurrogate.load(args.checkpoint)
    validation_arrays = WavefieldDatasetArrays.load(dataset_dir / "validation.npz")
    receiver_indices = tuple(int(x) for x in validation_arrays.receiver_indices)
    physics = Helmholtz2D(
        shape=surrogate.shape,
        spacing=validation_arrays.spacing,
        damping_width=max(3, min(surrogate.shape) // 8),
        damping_strength=2.0,
    )

    print("LOADING 24-BLOCK GEOLOGY CASES")
    validation = _load_cases(dataset_dir, "validation", physics, args.sigma)
    test_id = _load_cases(dataset_dir, "test_id", physics, args.sigma)
    test_ood = _load_cases(dataset_dir, "test_ood", physics, args.sigma)

    print(f"validation={len(validation)} ID={len(test_id)} OOD={len(test_ood)}")

    print()
    print("PATHWISE CASE-LEVEL CALIBRATION")
    scores: list[dict] = []
    for i, case in enumerate(validation, start=1):
        score, ncert = _pathwise_case_score(
            case=case,
            physics=physics,
            surrogate=surrogate,
            receiver_indices=receiver_indices,
        )
        scores.append(
            {
                "slice_id": case.slice_id,
                "model_type": case.model_type,
                "pathwise_max_ratio": score,
                "certificates_scored": ncert,
            }
        )
        print(f"calibration {i:02d}/{len(validation)} {case.slice_id}: max_ratio={score:.8g}")

    score_values = np.asarray(
        [row["pathwise_max_ratio"] for row in scores],
        dtype=float,
    )
    q = _split_quantile(score_values, args.alpha)

    with (output / "calibration_scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scores[0].keys()))
        writer.writeheader()
        writer.writerows(scores)

    print()
    print("PATHWISE CONFORMAL SCALE:", q)
    print("alpha:", args.alpha)
    print("target ID coverage:", 1.0 - args.alpha)

    if not np.isfinite(q):
        raise RuntimeError(
            "The finite-sample conformal quantile is infinite. "
            "Use a larger alpha or more independent validation cases."
        )

    print()
    print("EVALUATING ID")
    id_rows = _evaluate_cases(
        split="id",
        cases=test_id,
        physics=physics,
        surrogate=surrogate,
        receiver_indices=receiver_indices,
        conformal_scale=q,
        seed=args.seed + 1,
    )

    print()
    print("EVALUATING OOD PENOBSCOT")
    ood_rows = _evaluate_cases(
        split="ood",
        cases=test_ood,
        physics=physics,
        surrogate=surrogate,
        receiver_indices=receiver_indices,
        conformal_scale=q,
        seed=args.seed + 2,
    )

    all_rows = [*id_rows, *ood_rows]
    with (output / "policy_cases.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(asdict(all_rows[0]).keys()),
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(asdict(row))

    summary = {
        "dataset": str(dataset_dir),
        "checkpoint": str(args.checkpoint),
        "alpha": args.alpha,
        "target_id_pathwise_coverage": 1.0 - args.alpha,
        "calibration_cases": len(validation),
        "pathwise_uniform_scale": q,
        "calibration_score_min": float(np.min(score_values)),
        "calibration_score_median": float(np.median(score_values)),
        "calibration_score_max": float(np.max(score_values)),
        "blocks_per_case": 24,
        "id": _aggregate(id_rows),
        "ood": _aggregate(ood_rows),
        "semantics": {
            "id": (
                "case-level pathwise split-conformal calibration; interpretation "
                "requires exchangeability between validation and ID cases"
            ),
            "ood": (
                "held-out-geology stress test only; nominal conformal coverage "
                "is not claimed under distribution shift"
            ),
            "offline_exact_validation": (
                "exact full gradients used to measure true descent/cosine are "
                "excluded from policy exact-block counts"
            ),
        },
    }

    for split in ("id", "ood"):
        selective = summary[split]["wavecert-selective"]
        summary[split]["wavecert_selective_saving_vs_exact"] = float(
            1.0 - selective["mean_exact_blocks"] / 24.0
        )

    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print()
    print("FINAL SUMMARY")
    print(json.dumps(summary, indent=2))
    print()
    print("WROTE:", output / "summary.json")
    print("WROTE:", output / "policy_cases.csv")
    print("WROTE:", output / "calibration_scores.csv")


if __name__ == "__main__":
    main()
