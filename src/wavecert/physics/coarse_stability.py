"""DCT-assembled low-Laplacian stability diagnostics for Helmholtz2D.

The block inequalities are exact-arithmetic results for the real Dirichlet
five-point stiffness and a complex diagonal potential only. Evaluation uses
ordinary floating point, including a *heuristic* PSD allowance: this is not a
verified roundoff enclosure. No fine-grid factorization, inverse map, dense
fine matrix, or online n-by-r basis/Gram assembly is performed.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections import OrderedDict
from collections.abc import Iterable
from copy import deepcopy

import numpy as np
import scipy.fft as fft
import scipy.linalg as la

from wavecert.physics.helmholtz import Helmholtz2D


def _readonly(values: np.ndarray) -> np.ndarray:
    out = np.array(values, copy=True)
    out.setflags(write=False)
    return out


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _fold(indices: np.ndarray, intervals: int) -> np.ndarray:
    indices = indices % (2 * intervals)
    return np.where(indices > intervals, 2 * intervals - indices, indices)


class FastCoarseStability:
    """Snapshot one reference grid and assemble rank-r blocks from DCT moments.

    Cache keys contain the canonical model bytes, never array identities.
    Construction costs two real DCT-I transforms for d and d². On a model
    cache miss, exactly two real DCT-I transforms assemble m and m² moments.
    Counts describe executed operations/RHS columns, not FLOPs or timings.
    """

    _COUNT_KEYS = (
        "offline_dct_I_real_transforms", "online_dct_I_real_transforms",
        "model_cache_hits", "coarse_multiplication_matrices",
        "small_svd_evaluations", "small_hermitian_eigenproblems",
        "coarse_lu_factorizations", "coarse_solve_calls", "coarse_solve_rhs",
        "coarse_inverse_rhs", "coarse_rhs", "coarse_oriented_rhs",
        "coarse_correction_rhs", "coarse_receiver_rhs",
        "coarse_projection_calls", "dst_I_real_transforms",
        "coarse_lift_calls", "idst_I_real_transforms",
        "full_operator_applications", "receiver_geometry_evaluations",
        "receiver_cache_hits", "fine_grid_factorizations", "fine_grid_rhs",
        "fine_grid_dense_svd",
    )

    def __init__(self, physics: Helmholtz2D, rank: int, model_cache_size: int = 4):
        if type(physics) is not Helmholtz2D:
            raise TypeError("only the exact Helmholtz2D reference class is supported")
        if len(physics.shape) != 2:
            raise ValueError("shape must have two dimensions")
        shape = tuple(_integer(v, "shape entry", 1) for v in physics.shape)
        if not np.isfinite(physics.spacing) or physics.spacing <= 0:
            raise ValueError("spacing must be finite and positive")
        width = _integer(physics.damping_width, "damping_width")
        if not np.isfinite(physics.damping_strength) or physics.damping_strength < 0:
            raise ValueError("damping_strength must be finite and nonnegative")
        self._config = dict(shape=list(shape), spacing=float(physics.spacing),
                            damping_width=width, damping_strength=float(physics.damping_strength))
        self._shape = shape
        self._n = int(np.prod(shape))
        self._rank = _integer(rank, "rank", 1)
        if self._rank >= self._n:
            raise ValueError("rank must be less than the number of grid points")
        self._cache_size = _integer(model_cache_size, "model_cache_size")
        self._models: OrderedDict = OrderedDict()
        self._cost: dict[str, int] = {}
        nz, nx = shape
        h = self._config["spacing"]
        pairs = [(4 / h**2 * (np.sin(np.pi * k / (2 * (nz + 1)))**2
                              + np.sin(np.pi * p / (2 * (nx + 1)))**2), k, p)
                 for k in range(1, nz + 1) for p in range(1, nx + 1)]
        pairs.sort()  # deterministic (eigenvalue, z-mode, x-mode), including ties
        self._eigenvalues = _readonly(np.array([v[0] for v in pairs]))
        self._modes = _readonly(np.array([v[1:] for v in pairs[:self.rank]], dtype=int))
        k, p = self._modes.T
        self._lookup = (_fold(abs(k[:, None] - k[None, :]), nz + 1),
                        _fold(k[:, None] + k[None, :], nz + 1),
                        _fold(abs(p[:, None] - p[None, :]), nx + 1),
                        _fold(p[:, None] + p[None, :], nx + 1))
        snapshot = Helmholtz2D(shape=shape, spacing=h, damping_width=width,
                               damping_strength=self._config["damping_strength"])
        self._damping = _readonly(snapshot.damping)
        self._md = _readonly(self._multiplication(self._damping, offline=True))
        self._md2 = _readonly(self._multiplication(self._damping**2, offline=True))

    @property
    def physics_config(self) -> dict:
        return deepcopy(self._config)

    @property
    def n(self) -> int:
        return self._n

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def mode_indices(self) -> np.ndarray:
        return _readonly(self._modes)

    @property
    def laplacian_eigenvalues(self) -> np.ndarray:
        return _readonly(self._eigenvalues)

    @property
    def operator_signature(self) -> dict:
        """Content identity of the fixed basis/grid; no model or outcome data."""
        config = json.dumps(self._config, sort_keys=True, separators=(",", ":")).encode()
        return dict(
            method="real-low-Dirichlet-Laplacian-DCT-I-DST-I-v1",
            physics_config_sha256=hashlib.sha256(config).hexdigest(),
            selected_modes_int64_le_sha256=hashlib.sha256(
                self._modes.astype("<i8").tobytes(order="C")).hexdigest(),
            ordered_eigenvalues_float64_le_sha256=hashlib.sha256(
                self._eigenvalues.astype("<f8").tobytes(order="C")).hexdigest(),
            damping_float64_le_sha256=hashlib.sha256(
                self._damping.astype("<f8").tobytes(order="C")).hexdigest(),
            rank=self.rank, n=self.n,
            first_omitted_laplacian_eigenvalue=float(self._eigenvalues[self.rank]),
            formally_verified_roundoff_enclosure=False,
        )

    def _count(self, key: str, count: int = 1) -> None:
        self._cost[key] = self._cost.get(key, 0) + count

    def cost_snapshot(self) -> dict[str, int]:
        return {key: int(self._cost.get(key, 0)) for key in self._COUNT_KEYS}

    def reset_cost_counters(self, *, clear_cache: bool = False) -> None:
        """Reset executed counts (constructor's offline work is then excluded)."""
        self._cost.clear()
        if clear_cache:
            self._models.clear()

    def _multiplication(self, field: np.ndarray, *, offline: bool = False) -> np.ndarray:
        transformed = fft.dctn(np.pad(np.asarray(field).reshape(self._shape), ((1, 1), (1, 1))),
                               type=1, norm=None, orthogonalize=False, workers=1)
        self._count("offline_dct_I_real_transforms" if offline else "online_dct_I_real_transforms")
        self._count("coarse_multiplication_matrices")
        zd, zs, xd, xs = self._lookup
        nz, nx = self._shape
        return (transformed[zd, xd] - transformed[zs, xd]
                - transformed[zd, xs] + transformed[zs, xs]) / (4 * (nz + 1) * (nx + 1))

    def evaluate(self, m: np.ndarray, frequency_hz: float) -> CoarseStabilityResult:
        if np.iscomplexobj(m):
            raise ValueError("squared slowness must be real; complex inputs are not supported")
        model = np.asarray(m, dtype=float).reshape(-1)
        if model.size != self.n or not np.all(np.isfinite(model)) or np.any(model <= 0):
            raise ValueError("squared slowness must match the grid and be finite and positive")
        if not np.isfinite(frequency_hz) or frequency_hz <= 0:
            raise ValueError("frequency_hz must be finite and positive")
        result = CoarseStabilityResult(self, model, float(frequency_hz))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            omega = result.omega
            gamma = float(self._eigenvalues[self.rank] - omega**2 * np.max(model))
        result.gamma = gamma
        if not np.isfinite(gamma):
            return result._unavailable("nonfinite coarse arithmetic")
        if gamma <= 0:
            return result._unavailable("nonpositive complement coercivity gamma")
        key = model.tobytes()
        if key in self._models:
            self._count("model_cache_hits")
            self._models.move_to_end(key)
            mm, mm2 = self._models[key]
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                squared = model**2
            if not np.all(np.isfinite(squared)):
                return result._unavailable("nonfinite model second moment")
            mm, mm2 = self._multiplication(model), self._multiplication(squared)
            if self._cache_size:
                self._models[key] = (_readonly(mm), _readonly(mm2))
                while len(self._models) > self._cache_size:
                    self._models.popitem(last=False)
        with np.errstate(over="ignore", invalid="ignore"):
            vr = -omega**2 * mm + 1j * omega * self._md
            ar = np.diag(self._eigenvalues[:self.rank]) + vr
            moment = omega**4 * mm2 + omega**2 * self._md2
            raw = moment - vr.conjugate().T @ vr
        if not all(np.all(np.isfinite(v)) for v in (ar, moment, raw)):
            return result._unavailable("nonfinite coarse arithmetic")
        result._ar = _readonly(ar)
        try:
            self._count("small_svd_evaluations")
            singular = la.svdvals(ar, check_finite=False)
            threshold = np.finfo(float).eps * self.rank * max(float(singular[0]), 1.0)
            result._diagnostics["coarse_singular_value_threshold"] = threshold
            if singular[-1] <= threshold:
                return result._unavailable("singular or numerically unresolved reduced operator")
            result.a = float(1 / singular[-1])
            hermitian = (raw + raw.conjugate().T) / 2
            self._count("small_hermitian_eigenproblems")
            eigen = la.eigvalsh(hermitian, check_finite=False)
            scale = float(la.norm(moment, "fro") + la.norm(vr, "fro")**2)
            allowance = float(128 * np.finfo(float).eps * self.rank * max(scale, 1.0))
            if not np.isfinite(allowance):
                return result._unavailable("nonfinite Gram numerical allowance")
            result._diagnostics.update(raw_gram_minimum_eigenvalue=float(eigen[0]),
                                       gram_scale=scale, gram_psd_allowance=allowance,
                                       strong_indefiniteness_threshold=-10 * allowance)
            if eigen[0] < -10 * allowance:
                return result._unavailable("strongly indefinite computed coupling Gram")
            shift = max(0.0, -float(eigen[0])) + allowance
            guarded = hermitian + shift * np.eye(self.rank)
            result._diagnostics["gram_diagonal_shift"] = shift
            result._raw_gram = _readonly(hermitian)
            result._gram = _readonly(guarded)
            result.ell = float(np.sqrt(max(0.0, float(eigen[-1]) + shift)))
            with warnings.catch_warnings():
                warnings.simplefilter("error", la.LinAlgWarning)
                self._count("coarse_lu_factorizations")
                result._lu = la.lu_factor(ar, check_finite=False)
            inverse = result._solve(np.eye(self.rank), adjoint=False, category="coarse_inverse_rhs")
            # L^T (L^T)^H = conjugate(L^H L), not G itself.
            response = inverse @ guarded.conjugate() @ inverse.conjugate().T
            self._count("small_hermitian_eigenproblems")
            top = la.eigvalsh((response + response.conjugate().T) / 2, check_finite=False)[-1]
            result.k = float(np.sqrt(max(0.0, top)))
            result.tau = float(result.k * result.ell / gamma)
            if not np.isfinite(result.tau):
                return result._unavailable("nonfinite small-gain arithmetic")
            if result.tau >= 1:
                return result._unavailable("small-gain tau is not less than one")
            a, k, ell, tau = result.a, result.k, result.ell, result.tau
            matrix = np.array([[a, k / gamma],
                               [ell * a / gamma, 1 / gamma]]) / (1 - tau)
            # The bottom-right expression simplifies exactly to 1/[gamma(1-tau)].
            if not np.all(np.isfinite(matrix)):
                return result._unavailable("nonfinite component-bound matrix")
            self._count("small_svd_evaluations")
            result.beta_lower_bound = float(1 / la.svdvals(matrix, check_finite=False)[0])
            if not np.isfinite(result.beta_lower_bound) or result.beta_lower_bound <= 0:
                return result._unavailable("nonfinite component-bound matrix")
            result._matrix = _readonly(matrix)
            result.available, result.reason = True, "available (ordinary floating point)"
            return result
        except (la.LinAlgError, la.LinAlgWarning, ValueError, FloatingPointError) as exc:
            return result._unavailable(f"coarse linear algebra failed: {type(exc).__name__}")


class CoarseStabilityResult:
    """One immutable-input coarse block and ordinary-FP component diagnostics.

    Counters are shared with the creating engine. Projection/operator methods
    also work when unavailable; inverse-dependent methods raise ValueError.
    """

    def __init__(self, engine: FastCoarseStability, model: np.ndarray, frequency_hz: float):
        self._engine = engine
        self._model = _readonly(model)
        self._model_sha256 = hashlib.sha256(
            self._model.astype("<f8").tobytes(order="C")).hexdigest()
        self._frequency_hz = frequency_hz
        self._omega = 2 * np.pi * frequency_hz
        self.available = False
        self.reason = "not evaluated"
        self.gamma = self.a = self.ell = self.k = self.tau = float("nan")
        self.beta_lower_bound = 0.0
        self._ar = self._raw_gram = self._gram = self._matrix = None
        self._lu = None
        self._receivers: OrderedDict = OrderedDict()
        self._diagnostics = {"formally_verified_roundoff_enclosure": False,
                             "floating_point_scope": "ordinary floating point; heuristic PSD allowance",
                             "offline_damping_real_dct_I_transforms": 2}

    @property
    def physics_config(self) -> dict:
        return self._engine.physics_config

    @property
    def frequency_hz(self) -> float:
        return self._frequency_hz

    @property
    def omega(self) -> float:
        return self._omega

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    @property
    def n(self) -> int:
        return self._engine.n

    @property
    def rank(self) -> int:
        return self._engine.rank

    @property
    def M(self) -> np.ndarray:
        self._require_available()
        return _readonly(self._matrix)

    @property
    def reduced_operator(self) -> np.ndarray | None:
        return None if self._ar is None else _readonly(self._ar)

    @property
    def numerical_diagnostics(self) -> dict:
        return deepcopy(self._diagnostics)

    @property
    def operator_signature(self) -> dict:
        return self._engine.operator_signature

    @property
    def raw_coupling_gram(self) -> np.ndarray | None:
        return None if self._raw_gram is None else _readonly(self._raw_gram)

    @property
    def coupling_gram(self) -> np.ndarray | None:
        return None if self._gram is None else _readonly(self._gram)

    def _unavailable(self, reason: str) -> CoarseStabilityResult:
        self.available, self.reason, self.beta_lower_bound = False, reason, 0.0
        return self

    def _require_available(self) -> None:
        if not self.available:
            raise ValueError(f"coarse stability unavailable: {self.reason}")

    def cost_snapshot(self) -> dict[str, int]:
        return self._engine.cost_snapshot()

    def to_dict(self) -> dict:
        def finite(value):
            return float(value) if np.isfinite(value) else None
        return dict(available=self.available, reason=self.reason, rank=self.rank, n=self.n,
                    frequency_hz=self.frequency_hz, physics_config=self.physics_config,
                    operator_signature=self.operator_signature,
                    model_sha256=self.model_sha256,
                    model_sha256_encoding="C-order flat little-endian float64 squared slowness",
                    gamma=finite(self.gamma), a=finite(self.a), ell=finite(self.ell),
                    k=finite(self.k), tau=finite(self.tau),
                    beta_lower_bound=self.beta_lower_bound,
                    M=self._matrix.tolist() if self.available else None,
                    numerical_diagnostics=self.numerical_diagnostics)

    def _vector(self, values: np.ndarray, size: int, name: str) -> np.ndarray:
        arr = np.asarray(values)
        if arr.ndim != 1 or arr.size != size or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be a finite vector of length {size}")
        return arr

    def project(self, vector: np.ndarray) -> np.ndarray:
        values = self._vector(vector, self.n, "field").reshape(self._engine._shape)
        self._engine._count("coarse_projection_calls")
        self._engine._count("dst_I_real_transforms", 2 if np.iscomplexobj(values) else 1)
        transformed = fft.dstn(values, type=1, norm="ortho", workers=1)
        indices = self._engine._modes
        return transformed[indices[:, 0] - 1, indices[:, 1] - 1]

    def lift(self, coefficients: np.ndarray) -> np.ndarray:
        values = self._vector(coefficients, self.rank, "coefficients")
        self._engine._count("coarse_lift_calls")
        self._engine._count("idst_I_real_transforms", 2 if np.iscomplexobj(values) else 1)
        tensor = np.zeros(self._engine._shape, dtype=values.dtype)
        indices = self._engine._modes
        tensor[indices[:, 0] - 1, indices[:, 1] - 1] = values
        return fft.idstn(tensor, type=1, norm="ortho", workers=1).reshape(-1)

    def component_norms(self, vector: np.ndarray) -> np.ndarray:
        values = self._vector(vector, self.n, "field")
        coarse = self.project(values)
        complement = values - self.lift(coarse)
        return np.array([la.norm(coarse), la.norm(complement)])

    def _solve(self, rhs: np.ndarray, *, adjoint: bool, category: str) -> np.ndarray:
        values = np.asarray(rhs, dtype=np.complex128)
        if values.ndim not in (1, 2) or values.shape[0] != self.rank:
            raise ValueError("coarse rhs must have shape (rank,) or (rank, k)")
        if not np.all(np.isfinite(values)):
            raise ValueError("coarse rhs must be finite")
        if category not in {"coarse_inverse_rhs", "coarse_rhs", "coarse_oriented_rhs",
                            "coarse_correction_rhs", "coarse_receiver_rhs"}:
            raise ValueError("unknown coarse RHS accounting category")
        if self._lu is None:
            raise ValueError("no reduced factorization is available")
        columns = 1 if values.ndim == 1 else values.shape[1]
        answer = la.lu_solve(self._lu, values, trans=2 if adjoint else 0, check_finite=False)
        self._engine._count("coarse_solve_calls")
        self._engine._count("coarse_solve_rhs", columns)
        self._engine._count(category, columns)
        if not np.all(np.isfinite(answer)):
            raise ValueError("nonfinite reduced solve")
        return answer

    def solve_coarse(self, rhs: np.ndarray, *, adjoint: bool = False,
                     category: str = "coarse_rhs") -> np.ndarray:
        self._require_available()
        return self._solve(rhs, adjoint=adjoint, category=category)

    def apply_operator(self, field: np.ndarray, *, adjoint: bool = False) -> np.ndarray:
        values = self._vector(field, self.n, "field").reshape(self._engine._shape)
        with np.errstate(over="ignore", invalid="ignore"):
            out = 4 * values.astype(np.complex128)
            out[1:, :] -= values[:-1, :]
            out[:-1, :] -= values[1:, :]
            out[:, 1:] -= values[:, :-1]
            out[:, :-1] -= values[:, 1:]
            out /= self._engine._config["spacing"]**2
            potential = -self.omega**2 * self._model + 1j * self.omega * self._engine._damping
            if adjoint:
                potential = potential.conjugate()
            out += potential.reshape(values.shape) * values
        self._engine._count("full_operator_applications")
        if not np.all(np.isfinite(out)):
            raise ValueError("nonfinite stencil application")
        return out.reshape(-1)

    def correct(self, field: np.ndarray, rhs: np.ndarray, *, adjoint: bool = False
                ) -> tuple[np.ndarray, np.ndarray]:
        self._require_available()
        values = self._vector(field, self.n, "field")
        forcing = self._vector(rhs, self.n, "rhs")
        residual = self.apply_operator(values, adjoint=adjoint) - forcing
        delta = self.solve_coarse(self.project(residual), adjoint=adjoint,
                                  category="coarse_correction_rhs")
        corrected = values - self.lift(delta)
        return corrected, self.apply_operator(corrected, adjoint=adjoint) - forcing

    def oriented_error_bounds(self, residual: np.ndarray, *, adjoint: bool = False
                              ) -> np.ndarray:
        self._require_available()
        values = self._vector(residual, self.n, "residual")
        rb = self.project(values)
        rq = la.norm(values - self.lift(rb))
        response = la.norm(self.solve_coarse(rb, adjoint=adjoint, category="coarse_oriented_rhs"))
        eb = (response + self.k * rq / self.gamma) / (1 - self.tau)
        return np.array([eb, (rq + self.ell * eb) / self.gamma])

    def receiver_constants(self, receiver_indices: Iterable[int]) -> dict:
        self._require_available()
        raw_indices = tuple(receiver_indices)
        indices = tuple(_integer(v, "receiver index") for v in raw_indices)
        if any(v >= self.n for v in indices):
            raise ValueError("receiver index out of range")
        if indices in self._receivers:
            self._engine._count("receiver_cache_hits")
            self._receivers.move_to_end(indices)
            return deepcopy(self._receivers[indices])
        self._engine._count("receiver_geometry_evaluations")
        r = len(indices)
        sampling_norm = float(np.sqrt(np.max(np.unique(indices, return_counts=True)[1]))) if r else 0.0
        if r > self.rank:
            out = dict(pB=sampling_norm, pQ=sampling_norm, rho_adjoint=self.a * sampling_norm,
                       basis=None, mode="conservative sampling norm (receiver count exceeds rank)")
        elif not r:
            out = dict(pB=0.0, pQ=0.0, rho_adjoint=0.0, basis=np.empty((0, self.rank)),
                       mode="empty receiver geometry")
        else:
            nz, nx = self._engine._shape
            z, x = np.unravel_index(indices, (nz, nx))
            k, p = self._engine._modes.T
            pb = (2 / np.sqrt((nz + 1) * (nx + 1))
                  * np.sin(np.pi * (z[:, None] + 1) * k[None, :] / (nz + 1))
                  * np.sin(np.pi * (x[:, None] + 1) * p[None, :] / (nx + 1)))
            gram_b = pb @ pb.T
            gram_q = np.equal.outer(indices, indices).astype(float) - gram_b
            self._engine._count("small_hermitian_eigenproblems", 2)
            allowance = 64 * np.finfo(float).eps * max(r, 1) * max(sampling_norm**2, 1.0)
            p_b = np.sqrt(max(0.0, la.eigvalsh(gram_b)[-1]) + allowance)
            p_q = np.sqrt(max(0.0, la.eigvalsh((gram_q + gram_q.T) / 2)[-1]) + allowance)
            response = self.solve_coarse(pb.T, adjoint=True, category="coarse_receiver_rhs")
            self._engine._count("small_svd_evaluations")
            rho = float(la.svdvals(response)[0])
            out = dict(pB=float(p_b), pQ=float(p_q), rho_adjoint=rho, basis=_readonly(pb),
                       mode="analytic sine samples and small receiver Gram",
                       receiver_gram_psd_allowance=float(allowance))
        self._receivers[indices] = deepcopy(out)
        while len(self._receivers) > 4:
            self._receivers.popitem(last=False)
        return deepcopy(out)

    def oriented_adjoint_error_bounds(self, ra: np.ndarray, eta_receiver_state: float,
                                      receiver_indices: Iterable[int]) -> np.ndarray:
        self._require_available()
        if not np.isfinite(eta_receiver_state) or eta_receiver_state < 0:
            raise ValueError("receiver state error bound must be finite and nonnegative")
        values = self._vector(ra, self.n, "adjoint residual")
        rb = self.project(values)
        rq = la.norm(values - self.lift(rb))
        response = la.norm(self.solve_coarse(rb, adjoint=True, category="coarse_oriented_rhs"))
        constants = self.receiver_constants(receiver_indices)
        forcing_q = rq + constants["pQ"] * eta_receiver_state
        eb = (response + constants["rho_adjoint"] * eta_receiver_state
              + self.k * forcing_q / self.gamma) / (1 - self.tau)
        return np.array([eb, (forcing_q + self.ell * eb) / self.gamma])
