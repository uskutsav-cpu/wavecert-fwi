"""Measure factorization reuse in exact reference gradients, without neural inference."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from wavecert.physics.helmholtz import Helmholtz2D, SurveyGeometry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/reference_cache_benchmark")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for size in (16, 32, 64):
        reference = []
        pair = []
        for cache_size in (0, 8):
            physics = Helmholtz2D(shape=(size, size), factorization_cache_size=cache_size)
            geometry = SurveyGeometry.from_grid(
                physics.shape, n_sources=8, n_receivers=min(20, size - 4),
            )
            model = np.full(physics.n, 0.25)
            sources = [physics.source(index) for index in geometry.source_indices]
            observed = np.zeros(len(geometry.receiver_indices), dtype=complex)
            start = time.perf_counter()
            results = []
            for _ in range(3):
                for frequency in (3.0, 4.0):
                    for q in sources:
                        value, gradient, _, _ = physics.objective_and_gradient(
                            model, q, observed, geometry.receiver_indices, frequency,
                        )
                        results.append((value, gradient))
            seconds = time.perf_counter() - start
            if cache_size == 0:
                reference = results
            unchanged = all(
                np.isclose(a[0], b[0], rtol=1e-12, atol=1e-12)
                and np.allclose(a[1], b[1], rtol=1e-12, atol=1e-12)
                for a, b in zip(reference, results, strict=True)
            )
            pair.append({
                "cache_size": cache_size, "seconds": seconds,
                "costs": physics.cost_snapshot(), "outputs_match_uncached": unchanged,
            })
        rows.append({"shape": [size, size], "runs": pair})
    summary = {
        "schema_version": 1, "shapes": rows,
        "scope": "Single-process microbenchmark of 48 exact reference gradient blocks per run; no neural model or certification. Benefits exact FWI too; not WaveCert-versus-exact speedup.",
        "protocol": {"sources": 8, "frequencies_hz": [3.0, 4.0], "repeats": 3, "spacing": 0.05, "squared_slowness": 0.25},
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(),
        "thread_environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
