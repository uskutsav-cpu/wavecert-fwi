from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from run_real_adaptive_pathwise import (
    _block_gradients,
    _load_cases,
    _normalised_negative,
)
from scipy.stats import binomtest, wilcoxon

from wavecert.data.wavefields import WavefieldDatasetArrays
from wavecert.metrics import cosine_similarity
from wavecert.physics.helmholtz import Helmholtz2D
from wavecert.surrogates.fno import TrainedFNOWavefieldSurrogate


def independent_rng(seed: int, split_code: int, case_index: int, k: int, repeat: int):
    ss = np.random.SeedSequence([seed, split_code, case_index, k, repeat])
    return np.random.default_rng(ss)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def evaluate_split(
    split_name: str,
    dataset_split: str,
    split_code: int,
    cases,
    policy_df: pd.DataFrame,
    physics: Helmholtz2D,
    surrogate: TrainedFNOWavefieldSurrogate,
    receiver_indices: tuple[int, ...],
    repeats: int,
    seed: int,
):
    wc = policy_df[
        (policy_df["split"] == split_name) & (policy_df["policy"] == "wavecert-selective")
    ].copy()
    wc_by_slice = {str(r["slice_id"]): r for _, r in wc.iterrows()}
    if len(wc_by_slice) != len(cases):
        raise ValueError(f"{split_name}: {len(wc_by_slice)} WaveCert rows for {len(cases)} cases")

    case_rows, draw_rows = [], []

    for case_index, case in enumerate(cases):
        row = wc_by_slice[case.slice_id]
        k = int(row["exact_blocks"])

        exact_blocks, neural_blocks = _block_gradients(
            case=case,
            physics=physics,
            surrogate=surrogate,
            receiver_indices=receiver_indices,
        )
        exact_total = np.sum(exact_blocks, axis=0)
        exact_direction = _normalised_negative(exact_total)

        descents, cosines = [], []
        for repeat in range(repeats):
            rng = independent_rng(seed, split_code, case_index, k, repeat)
            chosen = (
                set(int(i) for i in rng.choice(len(case.blocks), size=k, replace=False))
                if k
                else set()
            )

            hybrid = np.sum(
                [
                    exact_blocks[i] if i in chosen else neural_blocks[i]
                    for i in range(len(case.blocks))
                ],
                axis=0,
            )
            direction = _normalised_negative(hybrid)
            dd = float(np.dot(exact_total, direction))
            cosine = float(cosine_similarity(exact_direction, direction))
            descent = bool(dd < 0.0)
            descents.append(descent)
            cosines.append(cosine)

            draw_rows.append(
                {
                    "split": split_name,
                    "dataset_split": dataset_split,
                    "slice_id": case.slice_id,
                    "model_type": case.model_type,
                    "case_index": case_index,
                    "repeat": repeat,
                    "matched_exact_blocks": k,
                    "true_descent": descent,
                    "true_directional_derivative": dd,
                    "cosine_to_exact": cosine,
                }
            )

        c = np.asarray(cosines, dtype=float)
        wc_cos = float(row["cosine_to_exact"])
        random_median = float(np.median(c))
        random_p90 = float(np.quantile(c, 0.90))
        case_rows.append(
            {
                "split": split_name,
                "dataset_split": dataset_split,
                "slice_id": case.slice_id,
                "model_type": case.model_type,
                "case_index": case_index,
                "wavecert_exact_blocks": k,
                "wavecert_true_descent": bool(row["true_descent"]),
                "wavecert_cosine_to_exact": wc_cos,
                "random_repeats": repeats,
                "random_true_descent_probability": float(np.mean(descents)),
                "random_cosine_mean": float(np.mean(c)),
                "random_cosine_median": random_median,
                "random_cosine_p10": float(np.quantile(c, 0.10)),
                "random_cosine_p90": random_p90,
                "wavecert_minus_random_median_cosine": wc_cos - random_median,
                "wavecert_beats_random_median": wc_cos > random_median,
                "wavecert_beats_random_p90": wc_cos > random_p90,
            }
        )

        print(
            f"{split_name}: {case_index + 1:02d}/{len(cases)} "
            f"{case.slice_id} k={k:02d} "
            f"WC={wc_cos:.3f} rand_med={random_median:.3f} "
            f"rand_desc={np.mean(descents):.3f}"
        )

    return case_rows, draw_rows


def summarize(case_rows: list[dict], draw_rows: list[dict]) -> dict:
    wc = np.asarray([r["wavecert_cosine_to_exact"] for r in case_rows], dtype=float)
    rm = np.asarray([r["random_cosine_median"] for r in case_rows], dtype=float)
    delta = wc - rm
    wins = delta > 0
    sign = binomtest(int(wins.sum()), len(wins), 0.5, alternative="greater")

    if np.allclose(delta, 0.0):
        w_stat, w_p = None, 1.0
    else:
        w = wilcoxon(wc, rm, alternative="greater", method="auto")
        w_stat, w_p = float(w.statistic), float(w.pvalue)

    return {
        "n_cases": len(case_rows),
        "repeats_per_case": len(draw_rows) // len(case_rows),
        "mean_wavecert_exact_blocks": float(
            np.mean([r["wavecert_exact_blocks"] for r in case_rows])
        ),
        "wavecert_true_descent_rate": float(
            np.mean([r["wavecert_true_descent"] for r in case_rows])
        ),
        "mean_wavecert_cosine_to_exact": float(wc.mean()),
        "matched_random_true_descent_probability": float(
            np.mean([r["true_descent"] for r in draw_rows])
        ),
        "mean_matched_random_cosine_to_exact": float(
            np.mean([r["cosine_to_exact"] for r in draw_rows])
        ),
        "mean_case_random_cosine_median": float(rm.mean()),
        "mean_wavecert_minus_random_median_cosine": float(delta.mean()),
        "median_wavecert_minus_random_median_cosine": float(np.median(delta)),
        "fraction_cases_wavecert_beats_random_median": float(wins.mean()),
        "fraction_cases_wavecert_beats_random_p90": float(
            np.mean([r["wavecert_beats_random_p90"] for r in case_rows])
        ),
        "sign_test_wavecert_greater": {
            "wins": int(wins.sum()),
            "n_cases": len(wins),
            "pvalue_one_sided": float(sign.pvalue),
        },
        "wilcoxon_wavecert_vs_random_median": {
            "statistic": w_stat,
            "pvalue_one_sided": w_p,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/external/subsurfacegen-wavecert-v2")
    p.add_argument("--checkpoint", default="checkpoints/fno_subsurfacegen_v2.pt")
    p.add_argument(
        "--policy-cases",
        default="results/real_adaptive_matched/policy_cases.csv",
    )
    p.add_argument("--output", default="results/real_adaptive_matched_mc")
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--sigma", type=float, default=1.6)
    p.add_argument("--seed", type=int, default=20260911)
    args = p.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    dataset = Path(args.dataset)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    policy_df = pd.read_csv(args.policy_cases)

    surrogate = TrainedFNOWavefieldSurrogate.load(args.checkpoint)
    arrays = WavefieldDatasetArrays.load(dataset / "validation.npz")
    receivers = tuple(int(x) for x in arrays.receiver_indices)
    physics = Helmholtz2D(
        shape=surrogate.shape,
        spacing=arrays.spacing,
        damping_width=max(3, min(surrogate.shape) // 8),
        damping_strength=2.0,
    )

    id_cases = _load_cases(dataset, "test_id", physics, args.sigma)
    ood_cases = _load_cases(dataset, "test_ood", physics, args.sigma)

    id_case_rows, id_draws = evaluate_split(
        "id",
        "test_id",
        1,
        id_cases,
        policy_df,
        physics,
        surrogate,
        receivers,
        args.repeats,
        args.seed,
    )
    ood_case_rows, ood_draws = evaluate_split(
        "ood",
        "test_ood",
        2,
        ood_cases,
        policy_df,
        physics,
        surrogate,
        receivers,
        args.repeats,
        args.seed,
    )

    all_cases = id_case_rows + ood_case_rows
    all_draws = id_draws + ood_draws
    write_csv(out / "matched_random_cases.csv", all_cases)
    write_csv(out / "matched_random_draws.csv", all_draws)

    summary = {
        "dataset": str(dataset),
        "checkpoint": str(args.checkpoint),
        "policy_cases": str(args.policy_cases),
        "base_seed": args.seed,
        "rng": "SeedSequence([seed, split_code, case_index, k, repeat])",
        "repeats_per_case": args.repeats,
        "comparison": (
            "Matched-random repairs exactly the same number of blocks "
            "as WaveCert-selective on each individual case."
        ),
        "id": summarize(id_case_rows, id_draws),
        "ood": summarize(ood_case_rows, ood_draws),
        "semantics": {
            "development_only": ("Uses only the already-open 25 ID and 25 OOD development cases."),
            "random_policy": "Random baselines are uncertified.",
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\nFINAL MATCHED-MC SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nWROTE:", out / "summary.json")
    print("WROTE:", out / "matched_random_cases.csv")
    print("WROTE:", out / "matched_random_draws.csv")


if __name__ == "__main__":
    main()
