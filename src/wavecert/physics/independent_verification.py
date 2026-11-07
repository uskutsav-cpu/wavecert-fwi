"""Analytic and spectral checks independent of the sparse operator assembly."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.fft import dstn, idstn

from wavecert.physics.helmholtz import Helmholtz2D
from wavecert.provenance import environment_record


def verify_physics_independently(
    *, output_dir: str | Path | None = None, sizes: tuple[int, ...] = (15, 31, 63)
) -> dict:
    """Verify Dirichlet Helmholtz solves using a sine basis and continuum solution.

    The DST solves the constant-coefficient *discrete* problem without calling
    the production stiffness/operator assembly. The manufactured solution then
    checks second-order convergence to a *continuous* PDE solution. Neither
    validates absorbing boundary fidelity or variable-coefficient field data.
    """
    if len(sizes) < 2 or any(n < 3 for n in sizes) or list(sizes) != sorted(set(sizes)):
        raise ValueError("sizes must contain at least two increasing distinct integers >= 3")
    rng = np.random.default_rng(20261107)
    rows = []
    frequency, squared_slowness = 0.3, 0.25
    omega2 = (2 * np.pi * frequency) ** 2
    for n in sizes:
        spacing = 1.0 / (n + 1)
        physics = Helmholtz2D((n, n), spacing=spacing, damping_strength=0.0)
        model = np.full(n * n, squared_slowness)
        q = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        k = np.arange(1, n + 1)
        eigenvalues = 4 / spacing**2 * np.sin(np.pi * k / (2 * (n + 1))) ** 2
        denominator = eigenvalues[:, None] + eigenvalues[None, :] - omega2 * squared_slowness
        spectral = idstn(dstn(q, type=1, norm="ortho") / denominator, type=1, norm="ortho")
        sparse = physics.solve_state(model, q.reshape(-1), frequency).reshape(n, n)
        x = np.arange(1, n + 1) * spacing
        solution = np.sin(np.pi * x[:, None]) * np.sin(2 * np.pi * x[None, :])
        forcing = (5 * np.pi**2 - omega2 * squared_slowness) * solution
        approximate = physics.solve_state(model, forcing.reshape(-1), frequency).reshape(n, n)
        rows.append(
            {
                "shape": [n, n],
                "spacing": spacing,
                "spectral_relative_error": float(
                    np.linalg.norm(sparse - spectral) / np.linalg.norm(spectral)
                ),
                "continuum_relative_error": float(
                    np.linalg.norm(approximate - solution) / np.linalg.norm(solution)
                ),
            }
        )
    orders = [
        float(
            np.log(a["continuum_relative_error"] / b["continuum_relative_error"])
            / np.log(a["spacing"] / b["spacing"])
        )
        for a, b in zip(rows[:-1], rows[1:], strict=True)
    ]
    result = {
        "scope": "undamped constant-coefficient Dirichlet discrete/continuous Helmholtz verification",
        "boundary_fidelity_or_field_validation": False,
        "rows": rows,
        "observed_orders": orders,
        "passed": all(r["spectral_relative_error"] < 1e-10 for r in rows) and min(orders) > 1.8,
        "provenance": environment_record(Path(__file__).resolve().parents[3]),
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
