from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

Array = np.ndarray


@dataclass(frozen=True)
class SurveyGeometry:
    """Point-source / point-receiver geometry on a regular 2-D grid."""

    source_indices: tuple[int, ...]
    receiver_indices: tuple[int, ...]

    @classmethod
    def from_grid(
        cls,
        shape: tuple[int, int],
        *,
        n_sources: int = 2,
        n_receivers: int = 20,
        source_depth: int = 2,
        receiver_depth: int = 2,
        margin: int = 2,
    ) -> "SurveyGeometry":
        nz, nx = shape
        xs_src = np.linspace(margin, nx - margin - 1, n_sources).round().astype(int)
        xs_rec = np.linspace(margin, nx - margin - 1, n_receivers).round().astype(int)
        src = tuple(np.ravel_multi_index((source_depth, int(x)), shape) for x in xs_src)
        rec = tuple(np.ravel_multi_index((receiver_depth, int(x)), shape) for x in xs_rec)
        return cls(src, rec)


@dataclass(frozen=True)
class SurveyBlock:
    """One source-frequency block and its observed receiver data."""

    source_index: int
    frequency_hz: float
    observed: Array
    label: str


@dataclass
class Helmholtz2D:
    """Small 2-D acoustic frequency-domain reference solver.

    The state equation is

        A(m, w) u = q,
        A = -Δ - w² diag(m) + i w diag(damping),

    where ``m`` is squared slowness.  This backend is intentionally small and
    transparent: it is designed for mathematical validation of certificates,
    not production-scale seismic modeling.
    """

    shape: tuple[int, int] = (18, 18)
    spacing: float = 0.05
    damping_width: int = 3
    damping_strength: float = 1.5

    @property
    def n(self) -> int:
        return int(np.prod(self.shape))

    @cached_property
    def stiffness(self) -> sp.csr_matrix:
        nz, nx = self.shape
        h2 = self.spacing**2
        tx = sp.diags(
            [-np.ones(nx - 1), 2 * np.ones(nx), -np.ones(nx - 1)],
            [-1, 0, 1],
            format="csr",
        ) / h2
        tz = sp.diags(
            [-np.ones(nz - 1), 2 * np.ones(nz), -np.ones(nz - 1)],
            [-1, 0, 1],
            format="csr",
        ) / h2
        return (sp.kron(sp.eye(nz), tx) + sp.kron(tz, sp.eye(nx))).tocsr()

    @cached_property
    def damping(self) -> Array:
        nz, nx = self.shape
        width = max(1, min(self.damping_width, nz // 2, nx // 2))
        z = np.minimum(np.arange(nz), np.arange(nz)[::-1])
        x = np.minimum(np.arange(nx), np.arange(nx)[::-1])
        dist = np.minimum(z[:, None], x[None, :]).astype(float)
        ramp = np.clip((width - dist) / width, 0.0, 1.0) ** 2
        return (self.damping_strength * ramp).reshape(-1)

    def validate_model(self, m: Array) -> Array:
        arr = np.asarray(m, dtype=float).reshape(-1)
        if arr.size != self.n:
            raise ValueError(f"model has {arr.size} entries, expected {self.n}")
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
            raise ValueError("squared slowness must be finite and strictly positive")
        return arr

    def operator(self, m: Array, frequency_hz: float) -> sp.csr_matrix:
        m = self.validate_model(m)
        omega = 2.0 * np.pi * float(frequency_hz)
        diag = -(omega**2) * m + 1j * omega * self.damping
        return (self.stiffness + sp.diags(diag, 0, format="csr")).tocsr()

    def source(self, source_index: int, amplitude: complex = 1.0) -> Array:
        q = np.zeros(self.n, dtype=np.complex128)
        q[int(source_index)] = amplitude / (self.spacing**2)
        return q

    def solve_state(self, m: Array, q: Array, frequency_hz: float) -> Array:
        return np.asarray(spla.spsolve(self.operator(m, frequency_hz), q), dtype=np.complex128)

    def solve_tangent(self, m: Array, u: Array, v: Array, frequency_hz: float) -> Array:
        m = self.validate_model(m)
        v = np.asarray(v, dtype=float).reshape(-1)
        if v.size != self.n:
            raise ValueError("direction has wrong size")
        omega = 2.0 * np.pi * float(frequency_hz)
        rhs = (omega**2) * v * np.asarray(u)
        return np.asarray(spla.spsolve(self.operator(m, frequency_hz), rhs), dtype=np.complex128)

    def restrict(self, u: Array, receiver_indices: Iterable[int]) -> Array:
        idx = np.asarray(tuple(receiver_indices), dtype=int)
        return np.asarray(u)[idx]

    def inject_receivers(self, r: Array, receiver_indices: Iterable[int]) -> Array:
        idx = np.asarray(tuple(receiver_indices), dtype=int)
        rhs = np.zeros(self.n, dtype=np.complex128)
        # ``restrict`` may contain repeated receiver indices on very small
        # validation grids.  The true adjoint of repeated sampling must
        # accumulate those contributions rather than overwrite them.
        np.add.at(rhs, idx, np.asarray(r, dtype=np.complex128))
        return rhs

    def objective_and_gradient(
        self,
        m: Array,
        q: Array,
        observed: Array,
        receiver_indices: Iterable[int],
        frequency_hz: float,
    ) -> tuple[float, Array, Array, Array]:
        m = self.validate_model(m)
        u = self.solve_state(m, q, frequency_hz)
        pred = self.restrict(u, receiver_indices)
        residual = pred - np.asarray(observed)
        objective = 0.5 * float(np.vdot(residual, residual).real)
        rhs_adj = self.inject_receivers(residual, receiver_indices)
        a = self.operator(m, frequency_hz)
        lam = np.asarray(spla.spsolve(a.conjugate().transpose(), rhs_adj), dtype=np.complex128)
        omega = 2.0 * np.pi * float(frequency_hz)
        gradient = (omega**2) * np.real(np.conjugate(lam) * u)
        return objective, gradient, u, residual

    def directional_derivative(
        self,
        m: Array,
        q: Array,
        observed: Array,
        receiver_indices: Iterable[int],
        frequency_hz: float,
        direction: Array,
    ) -> float:
        u = self.solve_state(m, q, frequency_hz)
        du = self.solve_tangent(m, u, direction, frequency_hz)
        r = self.restrict(u, receiver_indices) - np.asarray(observed)
        return float(np.real(np.vdot(r, self.restrict(du, receiver_indices))))

    def data_jvp(
        self,
        m: Array,
        q: Array,
        receiver_indices: Iterable[int],
        frequency_hz: float,
        direction: Array,
    ) -> Array:
        """Apply the parameter-to-data Jacobian to a real model direction."""

        u = self.solve_state(m, q, frequency_hz)
        du = self.solve_tangent(m, u, direction, frequency_hz)
        return self.restrict(du, receiver_indices)

    def data_vjp(
        self,
        m: Array,
        q: Array,
        receiver_indices: Iterable[int],
        frequency_hz: float,
        data_dual: Array,
    ) -> Array:
        """Apply the real adjoint of the parameter-to-data Jacobian.

        For complex receiver data ``y``, this returns the real model-space
        gradient ``g`` satisfying

            dot(g, v) = Re <y, J v>

        for every real model direction ``v``.
        """

        m = self.validate_model(m)
        u = self.solve_state(m, q, frequency_hz)
        rhs_adj = self.inject_receivers(data_dual, receiver_indices)
        a = self.operator(m, frequency_hz)
        lam = np.asarray(
            spla.spsolve(a.conjugate().transpose(), rhs_adj), dtype=np.complex128
        )
        omega = 2.0 * np.pi * float(frequency_hz)
        return (omega**2) * np.real(np.conjugate(lam) * u)

    def smallest_singular_value(self, m: Array, frequency_hz: float) -> float:
        """Return the numerical 2-norm stability constant β = σ_min(A).

        This dense SVD is intentionally exact-ish for the small validation grids
        used by this repository.  Production runs should replace it with a
        certified lower-bound estimator rather than silently treating an
        iterative estimate as rigorous.
        """

        a = self.operator(m, frequency_hz).toarray()
        s = np.linalg.svd(a, compute_uv=False)
        beta = float(np.min(s))
        if not np.isfinite(beta) or beta <= 0:
            raise RuntimeError("failed to obtain positive stability constant")
        return beta
