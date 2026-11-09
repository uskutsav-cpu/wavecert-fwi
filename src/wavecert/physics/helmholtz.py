from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from wavecert.physics.certified_stability import (
    AnalyticalStabilityBound,
    dirichlet_stability_lower_bound,
)

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
    ) -> SurveyGeometry:
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
    factorization_cache_size: int = 8
    _factorizations: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)
    _cost: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _cache_signature: tuple | None = field(default=None, init=False, repr=False)

    _COST_KEYS = (
        "factorizations", "forward_rhs", "tangent_rhs", "adjoint_rhs",
        "stability_rhs", "dense_inverse_rhs", "svd_evaluations", "cache_hits",
        "analytic_bound_evaluations",
    )

    def cost_snapshot(self) -> dict[str, int]:
        """Cumulative executed work, including solves used by certificates.

        A matrix solve with k columns counts as k right-hand sides, even when
        SuperLU executes it in one batched call. Repaired gradient blocks are
        a different metric: they exclude stability solves and line searches.
        These counts describe the discrete reference backend, not FLOPs.
        """
        return {key: int(self._cost.get(key, 0)) for key in self._COST_KEYS}

    def reset_cost_counters(self, *, clear_cache: bool = False) -> None:
        """Reset accounting; optionally start with a cold factorization cache."""
        self._cost.clear()
        if clear_cache:
            self.clear_solver_cache()

    def clear_solver_cache(self) -> None:
        """Discard numerical factors without changing accumulated cost counts."""
        self._factorizations.clear()

    def _ensure_cache_signature(self) -> None:
        signature = (tuple(self.shape), self.spacing, self.damping_width, self.damping_strength)
        if signature != self._cache_signature:
            self.clear_solver_cache()
            self.__dict__.pop("stiffness", None)
            self.__dict__.pop("damping", None)
            self._cache_signature = signature
        if (isinstance(self.factorization_cache_size, (bool, np.bool_))
                or not isinstance(self.factorization_cache_size, (int, np.integer))
                or self.factorization_cache_size < 0):
            raise ValueError("factorization_cache_size must be a non-negative integer")
        while len(self._factorizations) > self.factorization_cache_size:
            self._factorizations.popitem(last=False)

    def _factorization(self, m: Array, frequency_hz: float):
        self._ensure_cache_signature()
        model = self.validate_model(m)
        # Store actual immutable bytes instead of array identity. In-place
        # mutation and views therefore cannot silently reuse stale factors.
        key = (float(frequency_hz), model.tobytes())
        if key in self._factorizations:
            self._cost["cache_hits"] = self._cost.get("cache_hits", 0) + 1
            self._factorizations.move_to_end(key)
            return self._factorizations[key]
        factor = spla.splu(self.operator(model, frequency_hz).tocsc())
        self._cost["factorizations"] = self._cost.get("factorizations", 0) + 1
        if self.factorization_cache_size:
            self._factorizations[key] = factor
            while len(self._factorizations) > self.factorization_cache_size:
                self._factorizations.popitem(last=False)
        return factor

    def _solve(
        self, m: Array, rhs: Array, frequency_hz: float, *, category: str, adjoint: bool = False,
    ) -> Array:
        values = np.asarray(rhs, dtype=np.complex128)
        if values.ndim not in (1, 2) or values.shape[0] != self.n:
            raise ValueError("right-hand side must have shape (n,) or (n, k)")
        if not np.all(np.isfinite(values)):
            raise ValueError("right-hand side must be finite")
        factor = self._factorization(m, frequency_hz)
        result = factor.solve(values, trans="H" if adjoint else "N")
        n_rhs = 1 if values.ndim == 1 else values.shape[1]
        self._cost[category] = self._cost.get(category, 0) + n_rhs
        return np.asarray(result, dtype=np.complex128)

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
        self._ensure_cache_signature()
        m = self.validate_model(m)
        if not np.isfinite(frequency_hz) or frequency_hz <= 0:
            raise ValueError("frequency_hz must be finite and positive")
        omega = 2.0 * np.pi * float(frequency_hz)
        diag = -(omega**2) * m + 1j * omega * self.damping
        return (self.stiffness + sp.diags(diag, 0, format="csr")).tocsr()

    def source(self, source_index: int, amplitude: complex = 1.0) -> Array:
        q = np.zeros(self.n, dtype=np.complex128)
        q[int(source_index)] = amplitude / (self.spacing**2)
        return q

    def solve_state(self, m: Array, q: Array, frequency_hz: float) -> Array:
        return self._solve(m, q, frequency_hz, category="forward_rhs")

    def solve_tangent(self, m: Array, u: Array, v: Array, frequency_hz: float) -> Array:
        m = self.validate_model(m)
        v = np.asarray(v, dtype=float).reshape(-1)
        if v.size != self.n:
            raise ValueError("direction has wrong size")
        omega = 2.0 * np.pi * float(frequency_hz)
        rhs = (omega**2) * v * np.asarray(u)
        return self._solve(m, rhs, frequency_hz, category="tangent_rhs")

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

    def objective(
        self, m: Array, q: Array, observed: Array,
        receiver_indices: Iterable[int], frequency_hz: float,
    ) -> float:
        """Exact data-misfit value with one forward solve and no adjoint."""
        state = self.solve_state(m, q, frequency_hz)
        residual = self.restrict(state, receiver_indices) - np.asarray(observed)
        return 0.5 * float(np.vdot(residual, residual).real)

    def analytical_stability_bound(self, m: Array, frequency_hz: float) -> AnalyticalStabilityBound:
        """Cheap discrete lower bound; zero explicitly means unavailable.

        This is specific to this class's five-point Dirichlet stiffness;
        alternate physics backends need their own stability derivation.
        """
        self._ensure_cache_signature()
        result = dirichlet_stability_lower_bound(
            shape=self.shape, spacing=self.spacing, squared_slowness=self.validate_model(m),
            damping=self.damping, frequency_hz=frequency_hz,
        )
        key = "analytic_bound_evaluations"
        self._cost[key] = self._cost.get(key, 0) + 1
        return result

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
        lam = self._solve(m, rhs_adj, frequency_hz, category="adjoint_rhs", adjoint=True)
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
        lam = self._solve(m, rhs_adj, frequency_hz, category="adjoint_rhs", adjoint=True)
        omega = 2.0 * np.pi * float(frequency_hz)
        return (omega**2) * np.real(np.conjugate(lam) * u)


    def inverse_operator_dense(self, m: Array, frequency_hz: float) -> Array:
        """Dense inverse of the discrete Helmholtz operator for reference certificates.

        This is intentionally a *small-problem validation* utility. Production
        WaveCert runs must replace it with scalable certified stability bounds.
        """

        return self._solve(
            m, np.eye(self.n, dtype=np.complex128), frequency_hz, category="dense_inverse_rhs"
        )

    @staticmethod
    def receiver_sampling_norm(receiver_indices: Iterable[int]) -> float:
        """Exact 2-norm of point sampling, including repeated receiver indices."""

        idx = np.asarray(tuple(receiver_indices), dtype=int)
        if idx.size == 0:
            return 0.0
        _, counts = np.unique(idx, return_counts=True)
        return float(np.sqrt(np.max(counts)))

    def receiver_resolvent_matrix(
        self,
        m: Array,
        frequency_hz: float,
        receiver_indices: Iterable[int],
        *,
        inverse_operator: Array | None = None,
    ) -> Array:
        r"""Return the dense *receiver* resolvent ``P A^{-1}``.

        When a full inverse is not supplied, this is formed efficiently from a
        sparse factorization of ``A^H`` with one right-hand side per receiver.
        It therefore avoids constructing the full dense inverse.
        """

        idx = np.asarray(tuple(receiver_indices), dtype=int)
        if inverse_operator is not None:
            return np.asarray(inverse_operator)[idx, :]
        if idx.size == 0:
            return np.zeros((0, self.n), dtype=np.complex128)
        rhs = np.zeros((self.n, idx.size), dtype=np.complex128)
        for j, i in enumerate(idx):
            rhs[i, j] += 1.0
        x = self._solve(m, rhs, frequency_hz, category="stability_rhs", adjoint=True)
        return np.asarray(x.conjugate().transpose(), dtype=np.complex128)

    def receiver_resolvent_norm(
        self,
        m: Array,
        frequency_hz: float,
        receiver_indices: Iterable[int],
        *,
        inverse_operator: Array | None = None,
        receiver_resolvent_matrix: Array | None = None,
    ) -> float:
        r"""Return ``||P A^{-1}||_2`` for the discrete receiver map."""

        rows = (
            np.asarray(receiver_resolvent_matrix)
            if receiver_resolvent_matrix is not None
            else self.receiver_resolvent_matrix(
                m,
                frequency_hz,
                receiver_indices,
                inverse_operator=inverse_operator,
            )
        )
        return self._spectral_norm(rows)

    def directional_receiver_tangent_resolvent_norm(
        self,
        m: Array,
        frequency_hz: float,
        receiver_indices: Iterable[int],
        direction: Array,
        *,
        inverse_operator: Array | None = None,
        receiver_resolvent_matrix: Array | None = None,
    ) -> float:
        r"""Return ``||P A^{-1} diag(v) A^{-1}||_2`` efficiently.

        Only a receiver-sized dense map is formed.  The second ``A^{-1}`` is
        applied by solving the adjoint system with one RHS per receiver.
        """

        return self._spectral_norm(self.directional_receiver_tangent_resolvent_matrix(
            m, frequency_hz, receiver_indices, direction,
            inverse_operator=inverse_operator,
            receiver_resolvent_matrix=receiver_resolvent_matrix,
        ))

    def directional_receiver_tangent_resolvent_matrix(
        self,
        m: Array,
        frequency_hz: float,
        receiver_indices: Iterable[int],
        direction: Array,
        *,
        inverse_operator: Array | None = None,
        receiver_resolvent_matrix: Array | None = None,
    ) -> Array:
        r"""Return ``P A^{-1} diag(v) A^{-1}`` before taking its norm.

        The same receiver-sized map used by the reference certificate also
        determines exact discrete receiver tangents for arbitrary sources.
        Exposing it avoids repeating its solves when comparing that control.
        Ordinary floating-point sparse solves do not establish a formal
        roundoff-enclosed certificate.
        """
        v = np.asarray(direction, dtype=float).reshape(-1)
        if v.size != self.n:
            raise ValueError("direction has wrong size")
        if not np.all(np.isfinite(v)):
            raise ValueError("direction must be finite")
        if inverse_operator is not None:
            inv_a = np.asarray(inverse_operator)
            idx = np.asarray(tuple(receiver_indices), dtype=int)
            return inv_a[idx, :] @ (v[:, None] * inv_a)

        b = (
            np.asarray(receiver_resolvent_matrix)
            if receiver_resolvent_matrix is not None
            else self.receiver_resolvent_matrix(m, frequency_hz, receiver_indices)
        )
        if b.size == 0:
            return np.zeros((0, self.n), dtype=np.complex128)
        # C = P A^{-1} diag(v) A^{-1}.  Compute C^H by solving
        # A^H X = (P A^{-1} diag(v))^H.
        m_left = b * v[None, :]
        x = self._solve(
            m, m_left.conjugate().transpose(), frequency_hz,
            category="stability_rhs", adjoint=True,
        )
        return x.conjugate().transpose()

    def _spectral_norm(self, values: Array) -> float:
        if not values.size:
            return 0.0
        self._cost["svd_evaluations"] = self._cost.get("svd_evaluations", 0) + 1
        return float(np.linalg.svd(values, compute_uv=False)[0])

    def smallest_singular_value(self, m: Array, frequency_hz: float) -> float:
        """Return the numerical 2-norm stability constant β = σ_min(A).

        This dense SVD is intentionally exact-ish for the small validation grids
        used by this repository.  Production runs should replace it with a
        certified lower-bound estimator rather than silently treating an
        iterative estimate as rigorous.
        """

        a = self.operator(m, frequency_hz).toarray()
        self._cost["svd_evaluations"] = self._cost.get("svd_evaluations", 0) + 1
        s = np.linalg.svd(a, compute_uv=False)
        beta = float(np.min(s))
        if not np.isfinite(beta) or beta <= 0:
            raise RuntimeError("failed to obtain positive stability constant")
        return beta
