"""Reproduce fresh-process first-call and warm batched/serial FNO API controls."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wavecert.experiments.batched_neural_controls import (
    neural_control_agreement,
    run_neural_control_worker,
)
from wavecert.provenance import environment_record, sha256_file, source_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/fno_parent_grouped.pt")
    parser.add_argument("--output", default="results/batched_neural_controls")
    parser.add_argument("--process-repeats", type=int, default=3)
    parser.add_argument("--warm-repeats", type=int, default=5)
    parser.add_argument("--worker-method", choices=("serial", "batch"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-batch-size", type=int, choices=(4, 12, 40), help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = Path(args.checkpoint).resolve()
    if args.worker_method:
        result = run_neural_control_worker(checkpoint=checkpoint, method=args.worker_method,
            batch_size=args.worker_batch_size, warm_repeats=args.warm_repeats, repository_root=root)
        print(json.dumps(result, allow_nan=False))
        return
    if args.process_repeats < 1 or args.warm_repeats < 1:
        raise ValueError("repetition counts must be positive integers")
    import torch
    torch.set_num_threads(1)
    provenance, checkpoint_hash = environment_record(root), sha256_file(checkpoint)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    workers = []
    status = {"status": "running", "completed_workers": 0, "expected_workers": 6*args.process_repeats,
              "provenance": provenance, "checkpoint_sha256": checkpoint_hash}
    status_path = output / "run_status.json"
    status_path.write_text(json.dumps(status, indent=2, allow_nan=False)+"\n")
    environment = os.environ.copy()
    environment.update({name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")})
    environment["PYTHONPATH"] = str(root / "src") + (os.pathsep+environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    try:
        for batch_size in (4, 12, 40):
            for repetition in range(args.process_repeats):
                methods = ("serial", "batch") if repetition % 2 == 0 else ("batch", "serial")
                for method in methods:
                    completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        "--worker-method", method, "--worker-batch-size", str(batch_size),
                        "--checkpoint", str(checkpoint), "--warm-repeats", str(args.warm_repeats)],
                        cwd=root, env=environment, text=True, capture_output=True, check=True, timeout=300)
                    result = json.loads(completed.stdout)
                    result["process_repetition"] = repetition
                    workers.append(result)
                    (output / "worker_outputs.json").write_text(json.dumps(workers, indent=2, allow_nan=False)+"\n")
                    status.update(completed_workers=len(workers), last_completed={"batch_size": batch_size,
                        "method": method, "repetition": repetition})
                    status_path.write_text(json.dumps(status, indent=2, allow_nan=False)+"\n")
        cases, timings = [], []
        for batch_size in (4, 12, 40):
            selected = [w for w in workers if w["batch_size"] == batch_size]
            reference = next(w for w in selected if w["method"] == "serial")
            agreements = [neural_control_agreement(reference["cold_output"], w[phase])
                          for w in selected for phase in ("cold_output", "warm_output")]
            if any(w["input_record"] != reference["input_record"] for w in selected):
                raise RuntimeError("worker survey inputs differ")
            if not all(a["passes_float32_tolerances"] for a in agreements):
                raise RuntimeError("serial and batched outputs disagree beyond declared float32 tolerances")
            case = {"batch_size": batch_size, "input_record": reference["input_record"],
                "all_numerical_checks_pass": True,
                "maximum_agreement_errors": {key: max(a[key] for a in agreements)
                    for key in agreements[0] if key != "passes_float32_tolerances"}, "methods": {}}
            for method in ("serial", "batch"):
                measured = [w for w in selected if w["method"] == method]
                cold = [w["cold_seconds"] for w in measured]
                warm = [seconds for w in measured for seconds in w["warm_seconds"]]
                case["methods"][method] = {"cold_seconds": cold, "warm_seconds": warm,
                    "median_cold_seconds": float(np.median(cold)), "median_warm_seconds": float(np.median(warm))}
                for worker in measured:
                    timings.append({"batch_size": batch_size, "method": method, "phase": "first-call",
                        "process_repetition": worker["process_repetition"], "warm_repetition": "",
                        "elapsed_seconds": worker["cold_seconds"]})
                    timings.extend({"batch_size": batch_size, "method": method, "phase": "warm",
                        "process_repetition": worker["process_repetition"], "warm_repetition": i,
                        "elapsed_seconds": seconds} for i, seconds in enumerate(worker["warm_seconds"]))
            cases.append(case)
        unchanged = (source_digest(root) == provenance["source_sha256"]
            and sha256_file(checkpoint) == checkpoint_hash
            and all(w["source_sha256"] == provenance["source_sha256"] and w["source_unchanged_during_worker"]
                and w["checkpoint_sha256"] == checkpoint_hash and w["checkpoint_unchanged_during_worker"] for w in workers))
        summary = {"schema_version": 1, "study_status": "complete" if unchanged else "stale-source-or-checkpoint-changed",
            "source_and_checkpoint_unchanged": unchanged, "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash, "provenance": provenance,
            "worker_thread_environment": {key: environment[key] for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "process_repeats_per_method_and_batch_size": args.process_repeats,
            "warm_repeats_per_process": args.warm_repeats, "cases": cases,
            "timing_protocol": [
                "Each method/batch-size/repetition runs in a fresh sequential subprocess with one Torch/BLAS thread.",
                "First-call timing starts after checkpoint loading and exact observation synthesis, before any neural objective evaluation in that process.",
                "The operating-system filesystem cache and hardware caches are not flushed; first-call means a fresh application invocation, not physically cold hardware.",
                "After that first call, one additional invocation warms the method outside timing, followed by the declared warm repetitions.",
                "Methods alternate process order across repetitions; checkpoint loading, worker startup, observation generation, JSON output and numerical comparisons are excluded from invocation times.",
                "Both methods return the summed objective/full gradient and all wavefields/residuals, including their normal CPU NumPy conversion costs.",
            ],
            "limitations": [
                "A single deterministic 24x24 synthetic model and one frozen checkpoint; batch sizes vary acquisition, not independent geological replication.",
                "The comparison checks API equivalence and invocation timing, not surrogate accuracy or useful inversion directions.",
                "Float32 batched kernels and summed reductions may differ numerically; raw output arrays and declared tolerances are retained.",
                "No optimizer policy, training, certificate, exact directional oracle, line search, memory peak, time-to-reconstruction, or full inversion acceleration result is measured.",
                "Larger batches use more activation memory. This API has no automatic chunking or deployment-scale memory guarantee.",
            ]}
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        with (output / "timings.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(timings[0]))
            writer.writeheader()
            writer.writerows(timings)
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4), constrained_layout=True)
        for axis, phase in zip(axes, ("cold", "warm"), strict=True):
            for method, color in (("serial", "#0072B2"), ("batch", "#D55E00")):
                axis.plot([c["batch_size"] for c in cases],
                    [1000*c["methods"][method][f"median_{phase}_seconds"] for c in cases],
                    marker="o", label=method, color=color)
            axis.set_title("Fresh-process first call" if phase == "cold" else "Warm calls")
            axis.set_xlabel("Source/frequency blocks sharing one model")
            axis.set_xticks((4, 12, 40))
            axis.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel("Median objective + full-gradient API time (ms)")
        axes[1].legend(frameon=False)
        fig.suptitle("CPU API invocation controls; no full-inversion result")
        fig.savefig(output / "api_invocation_times.png", dpi=180)
        plt.close(fig)
        status.update(status=summary["study_status"], source_and_checkpoint_unchanged=unchanged)
        status_path.write_text(json.dumps(status, indent=2, allow_nan=False)+"\n")
        if not unchanged:
            raise RuntimeError("source or checkpoint changed during the control benchmark")
        print(json.dumps({"study_status": summary["study_status"], "batch_sizes": [c["batch_size"] for c in cases],
                          "all_numerical_checks_pass": True}, indent=2))
    except Exception as error:
        status.update(status="failed" if not status["status"].startswith("stale") else status["status"],
                      error_type=type(error).__name__, error=str(error), completed_workers=len(workers))
        if isinstance(error, subprocess.CalledProcessError):
            status["failed_worker_stderr"] = (error.stderr or "")[-5000:]
        status_path.write_text(json.dumps(status, indent=2, allow_nan=False)+"\n")
        raise


if __name__ == "__main__":
    main()
