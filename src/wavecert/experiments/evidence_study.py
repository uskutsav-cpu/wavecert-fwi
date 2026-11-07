"""Reproducible development evidence with complete online cost accounting.

This is an engineering/research audit, not a preregistered confirmatory study.
The statistical gate is calibrated on independent synthetic initial states;
no nominal coverage is claimed for later iterates or shifted scenarios.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wavecert.experiments.end_to_end import _make_problem
from wavecert.experiments.synthetic import velocity_to_m
from wavecert.optimization.fwi import run_inversion
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock, SurveyGeometry
from wavecert.provenance import environment_record, sha256_file, source_digest
from wavecert.statistics.conformal import split_conformal_quantile
from wavecert.statistics.inference import binomial_failure_summary
from wavecert.statistics.pathwise import pathwise_case_score
from wavecert.surrogates.fno import TrainedFNOWavefieldSurrogate


@dataclass(frozen=True)
class EvidenceConfig:
    seeds: tuple[int, ...] = (731, 1907, 8101)
    cases_per_seed: int = 4
    iterations: int = 12
    calibration_cases: int = 30
    calibration_seed: int = 2026110701
    alpha: float = 0.10
    scenarios: tuple[str, ...] = ("clean", "noise-5pct", "sparse-receivers", "frequency-shift")
    policies: tuple[str, ...] = (
        "exact",
        "neural-only",
        "random-repair",
        "residual-repair",
        "periodic-exact",
        "conformal-selective",
        "conformal-selective-guarded",
        "deterministic-selective",
        "analytic-selective",
    )
    step_acceptance: str = "exact-decrease"
    repair_blocks: int = 2
    periodic_interval: int = 2
    torch_threads: int = 1


def _case(shape: tuple[int, int], seed: int, case_index: int, scenario: str):
    if scenario not in {
        "clean",
        "noise-5pct",
        "sparse-receivers",
        "frequency-shift",
        "initial-bias",
    }:
        raise ValueError(f"unknown scenario: {scenario}")
    # The same model pair is used across scenarios; independent models are the
    # seed/case clusters, not each scenario or individual source/frequency.
    rng = np.random.default_rng(np.random.SeedSequence([seed, case_index]))
    true_velocity, initial_velocity = _make_problem(shape, rng)
    if scenario == "initial-bias":
        initial_velocity = np.clip(initial_velocity + 0.15, 1.55, 3.20)
    physics = Helmholtz2D(
        shape, spacing=0.05, damping_width=max(3, min(shape) // 8), damping_strength=2.0
    )
    geometry = SurveyGeometry.from_grid(shape, n_sources=6, n_receivers=min(20, shape[1] - 4))
    receivers = (
        geometry.receiver_indices[::2]
        if scenario == "sparse-receivers"
        else geometry.receiver_indices
    )
    frequencies = (4.5, 5.0) if scenario == "frequency-shift" else (3.0, 4.0)
    true_model = velocity_to_m(true_velocity).reshape(-1)
    initial_model = velocity_to_m(initial_velocity).reshape(-1)
    blocks = []
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed, case_index, 991]))
    for source in (geometry.source_indices[1], geometry.source_indices[-2]):
        for frequency in frequencies:
            state = physics.solve_state(true_model, physics.source(source), frequency)
            observed = physics.restrict(state, receivers)
            if scenario == "noise-5pct":
                noise = noise_rng.normal(size=len(receivers)) + 1j * noise_rng.normal(
                    size=len(receivers)
                )
                noise *= 0.05 * np.linalg.norm(observed) / max(np.linalg.norm(noise), 1e-15)
                observed = observed + noise
            blocks.append(
                SurveyBlock(int(source), float(frequency), observed, f"s{source}_f{frequency:g}")
            )
    return physics, initial_model, true_model, tuple(blocks), tuple(receivers)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _interval(values: list[float], seed: int = 71) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return {"mean": float(np.mean(values)), "ci95": None, "n_independent_models": len(values)}
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, size=(2000, len(values)), replace=True), axis=1)
    return {
        "mean": float(np.mean(values)),
        "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        "n_independent_models": len(values),
        "method": "model-level percentile bootstrap",
    }


def _run_evidence_study(
    *,
    checkpoint_path: str | Path = "checkpoints/fno_phase2.pt",
    output_dir: str | Path = "results/evidence_study",
    config: EvidenceConfig | None = None,
) -> dict:
    import torch

    cfg = EvidenceConfig() if config is None else config
    if cfg.cases_per_seed < 1 or cfg.iterations < 1 or cfg.calibration_cases < 1:
        raise ValueError("case counts, calibration count and iterations must be positive")
    if not cfg.seeds or len(set(cfg.seeds)) != len(cfg.seeds):
        raise ValueError("evaluation seeds must be nonempty and unique")
    if cfg.calibration_seed in cfg.seeds:
        raise ValueError("calibration and evaluation seeds must be disjoint")
    if not cfg.scenarios or not cfg.policies or cfg.torch_threads < 1:
        raise ValueError("scenarios, policies and thread count must be nonempty/positive")
    if len(set(cfg.scenarios)) != len(cfg.scenarios) or len(set(cfg.policies)) != len(cfg.policies):
        raise ValueError("scenario and policy names must be unique")
    torch.set_num_threads(cfg.torch_threads)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    surrogate = TrainedFNOWavefieldSurrogate.load(checkpoint_path)
    warm_physics, warm_model, _, warm_blocks, warm_receivers = _case(
        surrogate.shape, 91277, 0, "clean"
    )
    warm_block = warm_blocks[0]
    calibration_domain = {
        "frequencies_hz": sorted({float(b.frequency_hz) for b in warm_blocks}),
        "source_indices": sorted({int(b.source_index) for b in warm_blocks}),
        "receiver_indices": [int(i) for i in warm_receivers],
    }
    surrogate.objective_and_gradient(
        warm_model,
        warm_physics.source(warm_block.source_index),
        warm_block.observed,
        warm_receivers,
        warm_block.frequency_hz,
    )
    surrogate.state_and_jvp(
        warm_model,
        np.ones_like(warm_model) / np.sqrt(len(warm_model)),
        warm_physics.source(warm_block.source_index),
        warm_block.frequency_hz,
    )
    run_config = {
        **asdict(cfg),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "shape": list(surrogate.shape),
        "study_status": "development; not preregistered",
        "scope": "synthetic 2-D discrete Helmholtz inversion; not field-scale or measured seismic data",
        "timing_protocol": "warm FNO state/VJP/JVP; fresh policy solver cache; rotated policy order across model/scenario cases; CPU",
        "calibration_acquisition_domain": calibration_domain,
        "provenance": environment_record(root),
    }
    (output / "config.json").write_text(json.dumps(run_config, indent=2, allow_nan=False) + "\n")
    calibration = []
    for case_index in range(cfg.calibration_cases):
        physics, model, _, blocks, receivers = _case(
            surrogate.shape, cfg.calibration_seed, case_index, "clean"
        )
        score = pathwise_case_score(
            physics=physics, surrogate=surrogate, m=model, blocks=blocks, receiver_indices=receivers
        )
        calibration.append({"case_index": case_index, "pathwise_score": score})
        print(f"calibration {case_index + 1}/{cfg.calibration_cases}", flush=True)
    scale = split_conformal_quantile([r["pathwise_score"] for r in calibration], cfg.alpha)
    if not np.isfinite(scale):
        raise ValueError(
            "insufficient independent calibration cases for a finite conformal quantile"
        )
    scale = 1.0 if scale == 0.0 else float(scale)
    _write_csv(output / "calibration.csv", calibration)
    rows, trajectories, held_out_coverage = [], [], []
    for seed in cfg.seeds:
        for case_index in range(cfg.cases_per_seed):
            for scenario_index, scenario in enumerate(cfg.scenarios):
                if scenario == "clean":
                    physics, model, _, blocks, receivers = _case(
                        surrogate.shape, seed, case_index, scenario
                    )
                    score = pathwise_case_score(
                        physics=physics,
                        surrogate=surrogate,
                        m=model,
                        blocks=blocks,
                        receiver_indices=receivers,
                    )
                    held_out_coverage.append(
                        {
                            "seed": seed,
                            "case_index": case_index,
                            "pathwise_score": score,
                            "covered": bool(score <= scale),
                        }
                    )
                offset = (seed + case_index + scenario_index) % len(cfg.policies)
                order = cfg.policies[offset:] + cfg.policies[:offset]
                for policy in order:
                    physics, model, truth, blocks, receivers = _case(
                        surrogate.shape, seed, case_index, scenario
                    )
                    run = run_inversion(
                        policy=policy,
                        physics=physics,
                        surrogate=surrogate,
                        m0=model,
                        m_true=truth,
                        blocks=blocks,
                        receiver_indices=receivers,
                        iterations=cfg.iterations,
                        case_scale=scale,
                        block_scales={b.label: scale for b in blocks},
                        step_acceptance=cfg.step_acceptance,
                        repair_blocks=cfg.repair_blocks,
                        periodic_interval=cfg.periodic_interval,
                        random_seed=seed + 10000 * case_index,
                        calibration_domain=calibration_domain,
                    )
                    initial, final = run.records[0], run.records[-1]
                    objectives = np.asarray([r.true_objective for r in run.records])
                    row = {
                        "seed": seed,
                        "case_index": case_index,
                        "scenario": scenario,
                        "policy": policy,
                        "initial_objective": initial.true_objective,
                        "final_objective": final.true_objective,
                        "objective_ratio": final.true_objective
                        / max(initial.true_objective, 1e-15),
                        "initial_model_error": initial.model_relative_l2,
                        "final_model_error": final.model_relative_l2,
                        "algorithm_seconds": run.total_algorithm_seconds,
                        "validation_seconds": run.total_validation_seconds,
                        "exact_gradient_blocks": run.total_exact_gradient_blocks,
                        "verification_gradient_blocks": run.total_verification_gradient_blocks,
                        "objective_increases": int(
                            np.count_nonzero(
                                np.diff(objectives) > 1e-10 * max(initial.true_objective, 1e-15)
                            )
                        ),
                        "accepted_steps": sum(r.accepted for r in run.records[:-1]),
                        "line_search_trials": sum(r.line_search_trials for r in run.records),
                    }
                    row.update({f"online_{k}": v for k, v in run.total_online_exact_cost.items()})
                    rows.append(row)
                    for record in run.records:
                        rec = asdict(record)
                        rec.pop("online_exact_cost")
                        rec.pop("validation_exact_cost")
                        # Final evaluation has no method objective; use empty
                        # CSV cell rather than serializing JSON NaN.
                        if not np.isfinite(rec["method_objective"]):
                            rec["method_objective"] = ""
                        trajectories.append(
                            {
                                "seed": seed,
                                "case_index": case_index,
                                "scenario": scenario,
                                "policy": policy,
                                **rec,
                            }
                        )
                    _write_csv(output / "cases.csv", rows)
                    _write_csv(output / "trajectories.csv", trajectories)
                    print(
                        f"seed={seed} case={case_index} {scenario} {policy} "
                        f"objective_ratio={row['objective_ratio']:.4f} time={run.total_algorithm_seconds:.3f}s",
                        flush=True,
                    )
    _write_csv(output / "held_out_initial_path_coverage.csv", held_out_coverage)
    summary = {
        "configuration": run_config,
        "pathwise_scale": scale,
        "comparisons": {},
        "limitations": [
            "Conformal interpretation applies only to exchangeable clean initial-state repair paths.",
            "No trajectory-wide or shifted-distribution conformal coverage is claimed.",
            "Exact-decrease acceptance pays online true-objective forward costs and prevents increases by construction; this is not evidence that calibration alone guarantees finite-step safety.",
            "Fixed heuristic budgets are declared in config; these are not matched-cost unless measured total costs match.",
            "Scenarios share seed/case models; uncertainty is computed within scenario over independent generated model pairs.",
            "Timing includes online certification and verification; offline validation uses a separate solver/cache.",
            "This development study does not establish field-scale computational advantage or competition readiness.",
        "The guarded policy detects declared acquisition-support mismatches only; it cannot establish geological exchangeability or detect every kind of OOD shift.",
        ],
    }
    if held_out_coverage:
        summary["clean_initial_pathwise_coverage"] = {
            "observed_coverage": float(np.mean([r["covered"] for r in held_out_coverage])),
            "failure_uncertainty": binomial_failure_summary(
                [not r["covered"] for r in held_out_coverage]
            ),
            "scope": "independent generated clean initial models; simultaneous repair path, not inversion trajectory",
        }
    for scenario in cfg.scenarios:
        exact = {
            (r["seed"], r["case_index"]): r
            for r in rows
            if r["scenario"] == scenario and r["policy"] == "exact"
        }
        for policy in cfg.policies:
            selected = [r for r in rows if r["scenario"] == scenario and r["policy"] == policy]
            entry = {
                "n_models": len(selected),
                "objective_ratio": _interval([r["objective_ratio"] for r in selected]),
                "final_model_error": _interval([r["final_model_error"] for r in selected]),
                "algorithm_seconds": _interval([r["algorithm_seconds"] for r in selected]),
                "objective_increases": sum(r["objective_increases"] for r in selected),
                "mean_exact_gradient_blocks": float(
                    np.mean([r["exact_gradient_blocks"] for r in selected])
                ),
            }
            for key in selected[0]:
                if key.startswith("online_"):
                    entry[f"mean_{key}"] = float(np.mean([r[key] for r in selected]))
            if exact:
                entry["paired_runtime_ratio_to_exact"] = _interval(
                    [
                        r["algorithm_seconds"]
                        / exact[(r["seed"], r["case_index"])]["algorithm_seconds"]
                        for r in selected
                    ]
                )
                entry["paired_final_model_error_difference_to_exact"] = _interval(
                    [
                        r["final_model_error"]
                        - exact[(r["seed"], r["case_index"])]["final_model_error"]
                        for r in selected
                    ]
                )
            for baseline in ("random-repair", "residual-repair"):
                reference = {
                    (r["seed"], r["case_index"]): r
                    for r in rows
                    if r["scenario"] == scenario and r["policy"] == baseline
                }
                if reference:
                    entry[f"paired_model_error_difference_to_{baseline}"] = _interval(
                        [
                            r["final_model_error"]
                            - reference[(r["seed"], r["case_index"])]["final_model_error"]
                            for r in selected
                        ]
                    )
                    entry[f"paired_runtime_ratio_to_{baseline}"] = _interval(
                        [
                            r["algorithm_seconds"]
                            / reference[(r["seed"], r["case_index"])]["algorithm_seconds"]
                            for r in selected
                        ]
                    )
            summary["comparisons"].setdefault(scenario, {})[policy] = entry
    summary["source_sha256_at_end"] = source_digest(root)
    summary["source_unchanged_during_run"] = (
        summary["source_sha256_at_end"] == run_config["provenance"]["source_sha256"]
    )
    if not summary["source_unchanged_during_run"]:
        summary["limitations"].append(
            "Source/configuration changed during execution; rerun a frozen checkout before treating this as the current reproducible result."
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for policy in cfg.policies:
        selected = [r for r in rows if r["scenario"] == "clean" and r["policy"] == policy]
        if not selected:
            continue
        axes[0].scatter(
            np.mean([r["algorithm_seconds"] for r in selected]),
            np.mean([r["final_model_error"] for r in selected]),
            label=policy,
        )
        selected_t = [r for r in trajectories if r["scenario"] == "clean" and r["policy"] == policy]
        curve = [
            np.mean(
                [
                    r["true_objective"]
                    / next(
                        x["initial_objective"]
                        for x in selected
                        if x["seed"] == r["seed"] and x["case_index"] == r["case_index"]
                    )
                    for r in selected_t
                    if r["iteration"] == i
                ]
            )
            for i in range(cfg.iterations + 1)
        ]
        axes[1].plot(range(cfg.iterations + 1), curve, label=policy)
    axes[0].set(
        xlabel="Total online seconds (mean per inversion)",
        ylabel="Final model relative L2 error",
        title="Clean synthetic cases: quality versus total time",
    )
    axes[1].set(
        xlabel="Iteration",
        ylabel="True objective / initial objective",
        title="Complete inversion trajectories",
        yscale="log",
    )
    axes[1].legend(fontsize=7)
    fig.savefig(output / "quality_and_convergence.png", dpi=180)
    plt.close(fig)
    return summary


def run_evidence_study(
    *,
    checkpoint_path: str | Path = "checkpoints/fno_phase2.pt",
    output_dir: str | Path = "results/evidence_study",
    config: EvidenceConfig | None = None,
) -> dict:
    """Execute with a status record; interrupted/failed runs are never complete."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    status = output / "status.json"
    status.write_text(json.dumps({"status": "running"}) + "\n")
    try:
        result = _run_evidence_study(
            checkpoint_path=checkpoint_path, output_dir=output, config=config
        )
    except Exception as error:
        status.write_text(json.dumps({"status": "failed", "error": str(error)}) + "\n")
        raise
    status.write_text(
        json.dumps(
            {
                "status": "complete" if result["source_unchanged_during_run"] else "stale",
                "source_unchanged_during_run": result["source_unchanged_during_run"],
            }
        )
        + "\n"
    )
    return result
