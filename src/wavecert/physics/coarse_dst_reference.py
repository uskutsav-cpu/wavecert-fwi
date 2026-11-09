"""Norm-free coarse/DST inverse and true-residual right-GMRES reference.

This is a classical linear solver for the finite five-point Dirichlet operator
A = K - omega**2*diag(m) + 1j*omega*diag(d).  It is neither a directional
certificate nor a verified floating-point enclosure.  Preparation deliberately
does not form second moments, coupling Grams, inverse norms, SVDs or eigenproblems.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
from scipy import fft
from scipy.linalg.lapack import zlartg


def _readonly(values):
    out = np.array(values, copy=True)
    out.setflags(write=False)
    return out


def _sha(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _integer(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _real(values, name):
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real (including its input dtype)")
    out = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    return out


def _matrix(values, n, name):
    out = np.asarray(values, dtype=np.complex128)
    if out.ndim not in (1, 2) or out.shape[0] != n or (out.ndim == 2 and out.shape[1] < 1):
        raise ValueError(f"{name} must have shape ({n},) or ({n}, k), k>=1")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite")
    return out, out.ndim == 1


def _fold(indices, intervals):
    modular = indices % (2*intervals)
    return np.minimum(modular, 2*intervals-modular)


def _finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


class PreparationUnavailable(RuntimeError):
    """The declared inverse could not be prepared; no exact solve was performed."""


class LinearSolveFailure(RuntimeError):
    """An ndarray-only consumer must not receive a nonconverged approximation."""

    def __init__(self, result):
        self.result = result
        super().__init__("right-GMRES failed true-residual convergence for one or more columns")


@dataclass(frozen=True)
class ColumnReport:
    converged: bool
    reason: str
    iterations: int
    restart_cycles: int
    rhs_norm: float | None
    true_residual_norm: float | None
    tolerance: float | None
    estimated_residual_norm: float | None

    def to_dict(self):
        return dict(self.__dict__)


@dataclass(frozen=True)
class GMRESResult:
    solution: np.ndarray
    columns: tuple[ColumnReport, ...]
    context_identity: dict
    rhs_sha256: str
    warm_start_sha256: str
    adjoint: bool
    rtol: float
    atol: float
    maxiter: int
    restart: int
    costs: dict

    @property
    def converged(self):
        return all(column.converged for column in self.columns)

    def to_dict(self):
        return {"status": "tolerance_converged" if self.converged else "nonconverged",
                    "converged": self.converged, "columns": [c.to_dict() for c in self.columns],
                    "context_identity": deepcopy(self.context_identity), "rhs_sha256": self.rhs_sha256,
                    "warm_start_sha256": self.warm_start_sha256, "adjoint": self.adjoint,
                    "rtol": self.rtol, "atol": self.atol, "maxiter": self.maxiter, "restart": self.restart,
                    "costs": dict(self.costs), "solution_shape": list(self.solution.shape),
                    "exact_arithmetic_claim": False, "formally_verified_roundoff_enclosure": False}


class CoarseDSTEngine:
    """Immutable operator/basis setup with bounded content-keyed moment/LU caches.

``damping`` may be any finite real diagonal field.  No PDE coercivity or
convergence theorem is assumed.  A zero/nonfinite high denominator or failed
coarse factorization fails preparation explicitly.  No certificate constants
are computed.  Contexts snapshot m and share only this engine's cost ledger.
"""

    COST_KEYS = (
        "basis_setups", "laplacian_eigenvalue_entries", "offline_dct_I_real_transforms",
        "online_dct_I_real_transforms", "moment_matrices", "model_moment_cache_hits",
        "context_cache_hits", "reduced_operator_assemblies", "coarse_lu_factorizations",
        "high_denominator_builds", "operator_dispatches", "operator_columns",
        "forward_operator_columns", "adjoint_operator_columns", "dst_I_dispatches",
        "dst_I_real_column_transforms", "idst_I_dispatches", "idst_I_real_column_transforms",
        "projection_columns", "lift_columns", "high_inverse_columns",
        "coarse_solve_dispatches", "coarse_forward_rhs", "coarse_adjoint_rhs",
        "preconditioner_dispatches", "preconditioner_columns", "linear_solve_columns",
        "arnoldi_iterations", "arnoldi_dot_products", "arnoldi_axpy_updates",
        "arnoldi_norm_evaluations", "givens_constructions", "givens_rotations",
        "projected_triangular_solves", "krylov_solution_update_columns",
        "true_residual_checks", "restart_cycles",
        "fine_factorizations", "fine_direct_rhs", "certificate_svd_evaluations",
        "certificate_eigenproblems", "certificate_inverse_rhs",
    )

    def __init__(self, shape, spacing, damping, rank, *, model_cache_size=4, context_cache_size=8):
        if len(shape) != 2:
            raise ValueError("shape must have two entries")
        self._shape = tuple(_integer(v, "shape entry", 1) for v in shape)
        self._n = int(np.prod(self._shape))
        self._rank = _integer(rank, "rank", 1)
        if self.rank >= self.n:
            raise ValueError("rank must be less than grid size")
        if isinstance(spacing, (bool, np.bool_)) or not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("spacing must be finite and positive")
        self._spacing = float(spacing)
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            inverse_h2 = np.float64(1)/np.float64(self._spacing)**2
        if not np.isfinite(inverse_h2) or inverse_h2 <= 0:
            raise ValueError("spacing does not define a finite nonzero discrete stiffness")
        d = _real(damping, "damping")
        if d.size != self.n:
            raise ValueError("damping size does not match grid")
        self._damping = _readonly(d.reshape(-1))
        self._model_cache_size = _integer(model_cache_size, "model_cache_size")
        self._context_cache_size = _integer(context_cache_size, "context_cache_size")
        self._models, self._contexts = OrderedDict(), OrderedDict()
        self._cost = {}
        nz, nx = self._shape
        z, x = np.arange(1, nz+1)[:, None], np.arange(1, nx+1)[None, :]
        eigen_grid = 4/self._spacing**2*(np.sin(np.pi*z/(2*(nz+1)))**2
                                        + np.sin(np.pi*x/(2*(nx+1)))**2)
        ordered = sorted((float(eigen_grid[k-1, p-1]), k, p)
                         for k in range(1, nz+1) for p in range(1, nx+1))
        self._eigen_grid = _readonly(eigen_grid)
        self._eigenvalues = _readonly([entry[0] for entry in ordered])
        self._modes = _readonly(np.array([entry[1:] for entry in ordered[:self.rank]], dtype=np.int64))
        k, p = self._modes.T
        self._lookup = (_fold(abs(k[:, None]-k[None, :]), nz+1),
                        _fold(k[:, None]+k[None, :], nz+1),
                        _fold(abs(p[:, None]-p[None, :]), nx+1),
                        _fold(p[:, None]+p[None, :], nx+1))
        mask = np.ones(self._shape, dtype=bool)
        mask[k-1, p-1] = False
        self._high_mask = _readonly(mask)
        self._count("basis_setups")
        self._count("laplacian_eigenvalue_entries", self.n)
        self._md = _readonly(self._moment(self._damping, offline=True))
        self._signature = {
            "operator": "five-point-Dirichlet-real-diagonal-potential-v1",
            "shape": list(self._shape), "spacing": self._spacing, "rank": self.rank, "n": self.n,
            "damping_float64_le_sha256": _sha(self._damping, "<f8"),
            "selected_modes_int64_le_sha256": _sha(self._modes, "<i8"),
            "ordered_eigenvalues_float64_le_sha256": _sha(self._eigenvalues, "<f8"),
            "norm_free_preparation": True, "formally_verified_roundoff_enclosure": False,
        }
        self._signature["operator_basis_sha256"] = _json_sha(self._signature)

    @classmethod
    def from_helmholtz(cls, physics, rank, **kwargs):
        # Reconstruct the declared config to avoid inheriting a stale cached damping.
        from wavecert.physics.helmholtz import Helmholtz2D
        if type(physics) is not Helmholtz2D:
            raise TypeError("use the explicit shape/spacing/damping constructor for another backend")
        snapshot = Helmholtz2D(shape=tuple(physics.shape), spacing=float(physics.spacing),
                               damping_width=physics.damping_width,
                               damping_strength=physics.damping_strength)
        return cls(snapshot.shape, snapshot.spacing, snapshot.damping, rank, **kwargs)

    @property
    def shape(self):
        return self._shape

    @property
    def n(self):
        return self._n

    @property
    def rank(self):
        return self._rank

    @property
    def signature(self):
        return deepcopy(self._signature)

    @property
    def mode_indices(self):
        return _readonly(self._modes)

    def _count(self, name, amount=1):
        self._cost[name] = self._cost.get(name, 0)+int(amount)

    def cost_snapshot(self):
        return {key: int(self._cost.get(key, 0)) for key in self.COST_KEYS}

    def _moment(self, values, *, offline=False):
        transformed = fft.dctn(np.pad(values.reshape(self._shape), ((1, 1), (1, 1))),
                               type=1, norm=None, orthogonalize=False, workers=1)
        self._count("offline_dct_I_real_transforms" if offline else "online_dct_I_real_transforms")
        self._count("moment_matrices")
        zd, zs, xd, xs = self._lookup
        nz, nx = self._shape
        moment = (transformed[zd, xd]-transformed[zs, xd]
                  - transformed[zd, xs]+transformed[zs, xs])/(4*(nz+1)*(nx+1))
        if not np.all(np.isfinite(moment)):
            raise PreparationUnavailable("nonfinite DCT Galerkin moment")
        return moment

    def prepare(self, model, frequency_hz):
        values = _real(model, "squared slowness").reshape(-1)
        if values.size != self.n or np.any(values <= 0):
            raise ValueError("squared slowness must be strictly positive and match grid")
        if (isinstance(frequency_hz, (bool, np.bool_)) or not np.isfinite(frequency_hz)
                or frequency_hz <= 0):
            raise ValueError("frequency_hz must be finite and positive")
        frozen = _readonly(values)
        key = (frozen.tobytes(), float(frequency_hz))
        if key in self._contexts:
            self._count("context_cache_hits")
            self._contexts.move_to_end(key)
            return self._contexts[key]
        model_key = key[0]
        if model_key in self._models:
            self._count("model_moment_cache_hits")
            self._models.move_to_end(model_key)
            mm = self._models[model_key]
        else:
            mm = _readonly(self._moment(frozen))
            if self._model_cache_size:
                self._models[model_key] = mm
                while len(self._models) > self._model_cache_size:
                    self._models.popitem(last=False)
        try:
            context = CoarseDSTContext(self, frozen, float(frequency_hz), mm)
        except (FloatingPointError, OverflowError) as exc:
            raise PreparationUnavailable("nonfinite preparation arithmetic") from exc
        if self._context_cache_size:
            self._contexts[key] = context
            while len(self._contexts) > self._context_cache_size:
                self._contexts.popitem(last=False)
        return context


class CoarseDSTContext:
    """One immutable m/frequency operator and shared forward/adjoint coarse LU."""

    def __init__(self, engine, model, frequency_hz, mm):
        self._engine, self._model = engine, model
        self._frequency_hz = frequency_hz
        self._omega = 2*np.pi*frequency_hz
        with np.errstate(over="ignore", invalid="ignore"):
            ar = (np.diag(engine._eigenvalues[:self.rank])-self.omega**2*mm
                  + 1j*self.omega*engine._md)
            mid_m = float(np.min(model))/2+float(np.max(model))/2
            mid_d = float(np.min(engine._damping))/2+float(np.max(engine._damping))/2
            denominator = engine._eigen_grid-self.omega**2*mid_m+1j*self.omega*mid_d
            potential = -self.omega**2*model+1j*self.omega*engine._damping
        engine._count("reduced_operator_assemblies")
        engine._count("high_denominator_builds")
        high = denominator[engine._high_mask]
        if (not np.all(np.isfinite(ar)) or not np.all(np.isfinite(potential))
                or not np.all(np.isfinite(high)) or np.any(high == 0)):
            raise PreparationUnavailable("nonfinite operator or zero/nonfinite high denominator")
        denominator[~engine._high_mask] = 1
        self._denominator, self._potential, self._ar = map(_readonly, (denominator, potential, ar))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", la.LinAlgWarning)
                engine._count("coarse_lu_factorizations")
                lu, piv = la.lu_factor(ar, check_finite=False)
            if not np.all(np.isfinite(lu)):
                raise PreparationUnavailable("nonfinite coarse LU")
            self._lu = (_readonly(lu), _readonly(piv))
        except (la.LinAlgWarning, la.LinAlgError, ValueError) as exc:
            raise PreparationUnavailable("coarse LU factorization failed") from exc
        identity = engine.signature
        identity.update(model_float64_le_sha256=_sha(model, "<f8"),
                        frequency_hz=frequency_hz, frequency_float64_le_sha256=_sha([frequency_hz], "<f8"),
                        m_mid=mid_m, damping_mid=mid_d,
                        high_minimum_real_denominator=float(np.min(high.real)),
                        high_minimum_absolute_denominator=float(np.min(abs(high))),
                        reduced_operator_complex128_le_sha256=_sha(ar, "<c16"))
        identity["context_sha256"] = _json_sha(identity)
        self._identity = identity

    @property
    def identity(self):
        return deepcopy(self._identity)

    @property
    def n(self):
        return self._engine.n

    @property
    def rank(self):
        return self._engine.rank

    @property
    def frequency_hz(self):
        return self._frequency_hz

    @property
    def omega(self):
        return self._omega

    @property
    def reduced_operator(self):
        return _readonly(self._ar)

    @property
    def model(self):
        return _readonly(self._model)

    def cost_snapshot(self):
        """Shared cumulative engine work; take before/after deltas for a call."""
        return self._engine.cost_snapshot()

    def _transform(self, grid, *, inverse=False):
        engine = self._engine
        columns = 1 if grid.ndim == 2 else grid.shape[2]
        prefix = "idst" if inverse else "dst"
        engine._count(prefix+"_I_dispatches")
        engine._count(prefix+"_I_real_column_transforms", columns*(2 if np.iscomplexobj(grid) else 1))
        operation = fft.idstn if inverse else fft.dstn
        with np.errstate(over="ignore", invalid="ignore"):
            answer = operation(grid, type=1, norm="ortho", axes=(0, 1), workers=1)
        if not np.all(np.isfinite(answer)):
            raise FloatingPointError("nonfinite DST transform")
        return answer

    def project(self, values):
        array, vector = _matrix(values, self.n, "field")
        count = 1 if vector else array.shape[1]
        self._engine._count("projection_columns", count)
        grid = array.reshape((*self._engine.shape, *(() if vector else (count,))))
        transformed = self._transform(grid)
        modes = self._engine._modes
        return transformed[modes[:, 0]-1, modes[:, 1]-1]

    def lift(self, values):
        array, vector = _matrix(values, self.rank, "coarse coefficients")
        count = 1 if vector else array.shape[1]
        self._engine._count("lift_columns", count)
        grid = np.zeros((*self._engine.shape, *(() if vector else (count,))), dtype=np.complex128)
        modes = self._engine._modes
        grid[modes[:, 0]-1, modes[:, 1]-1] = array
        return self._transform(grid, inverse=True).reshape((self.n,) if vector else (self.n, count))

    def apply_operator(self, values, *, adjoint=False):
        array, vector = _matrix(values, self.n, "field")
        count = 1 if vector else array.shape[1]
        self._engine._count("operator_dispatches")
        self._engine._count("operator_columns", count)
        self._engine._count("adjoint_operator_columns" if adjoint else "forward_operator_columns", count)
        grid = array.reshape((*self._engine.shape, *(() if vector else (count,))))
        with np.errstate(over="ignore", invalid="ignore"):
            out = 4*grid.copy()
            out[1:] -= grid[:-1]
            out[:-1] -= grid[1:]
            out[:, 1:] -= grid[:, :-1]
            out[:, :-1] -= grid[:, 1:]
            out = out.reshape(array.shape)/self._engine._spacing**2
            potential = self._potential.conjugate() if adjoint else self._potential
            out += potential*array if vector else potential[:, None]*array
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("nonfinite full operator application")
        return out

    def solve_coarse(self, values, *, adjoint=False):
        array, vector = _matrix(values, self.rank, "coarse RHS")
        self._engine._count("coarse_solve_dispatches")
        self._engine._count("coarse_adjoint_rhs" if adjoint else "coarse_forward_rhs",
                            1 if vector else array.shape[1])
        answer = la.lu_solve(self._lu, array, trans=2 if adjoint else 0, check_finite=False)
        if not np.all(np.isfinite(answer)):
            raise FloatingPointError("nonfinite coarse solve")
        return answer

    def high_inverse(self, values, *, adjoint=False):
        array, vector = _matrix(values, self.n, "high inverse RHS")
        count = 1 if vector else array.shape[1]
        self._engine._count("high_inverse_columns", count)
        grid = array.reshape((*self._engine.shape, *(() if vector else (count,))))
        coefficients = self._transform(grid)
        denominator = self._denominator.conjugate() if adjoint else self._denominator
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            coefficients /= denominator if vector else denominator[:, :, None]
        coefficients[~self._engine._high_mask] = 0
        answer = self._transform(coefficients, inverse=True).reshape(array.shape)
        if not np.all(np.isfinite(answer)):
            raise FloatingPointError("nonfinite high-space inverse")
        return answer

    def apply_preconditioner(self, values, *, adjoint=False):
        """H r + B Ar^-1 Bᵀ(r−AHr); adjoint uses its own upper-triangular action."""
        array, vector = _matrix(values, self.n, "preconditioner RHS")
        self._engine._count("preconditioner_dispatches")
        self._engine._count("preconditioner_columns", 1 if vector else array.shape[1])
        high = self.high_inverse(array, adjoint=adjoint)
        coarse_rhs = self.project(array-self.apply_operator(high, adjoint=adjoint))
        with np.errstate(over="ignore", invalid="ignore"):
            answer = high+self.lift(self.solve_coarse(coarse_rhs, adjoint=adjoint))
        if not np.all(np.isfinite(answer)):
            raise FloatingPointError("nonfinite preconditioner action")
        return answer

    def solve(self, rhs, *, adjoint=False, **options):
        result = self.solve_report(rhs, adjoint=adjoint, **options)
        if not result.converged:
            raise LinearSolveFailure(result)
        return np.array(result.solution, copy=True)

    def solve_report(self, rhs, *, adjoint=False, x0=None, rtol=1e-10, atol=1e-12,
                     maxiter=None, restart=None):
        """Per-column right-GMRES, default restart=min(n,50), maxiter=n.

        The bounded default avoids an implicit n-by-n Krylov allocation.
        Set restart=n explicitly for unrestarted GMRES when storage permits;
        its residual-minimization comparison does not extend across restarts.
        """
        array, vector = _matrix(rhs, self.n, "RHS")
        array = np.array(array, copy=True)
        warm = np.zeros_like(array) if x0 is None else _matrix(x0, self.n, "warm start")[0]
        if warm.shape != array.shape:
            raise ValueError("warm start must have the same shape as RHS; no broadcasting")
        warm = np.array(warm, copy=True)
        for name, value in (("rtol", rtol), ("atol", atol)):
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value < 0:
                raise ValueError(name+" must be finite and nonnegative")
        cap = self.n if maxiter is None else _integer(maxiter, "maxiter", 0)
        width = min(self.n, 50) if restart is None else _integer(restart, "restart", 1)
        width = min(width, self.n)
        before = self.cost_snapshot()
        matrix = array[:, None] if vector else array
        starts = warm[:, None] if vector else warm
        results, reports = [], []
        for column in range(matrix.shape[1]):
            answer, report = self._column_gmres(matrix[:, column], starts[:, column],
                                               adjoint, float(rtol), float(atol), cap, width)
            results.append(answer)
            reports.append(report)
        solution = np.column_stack(results)
        after = self.cost_snapshot()
        return GMRESResult(_readonly(solution[:, 0] if vector else solution), tuple(reports),
                           self.identity, _sha(array, "<c16"), _sha(warm, "<c16"), bool(adjoint),
                           float(rtol), float(atol), cap, width,
                           {key: after[key]-before[key] for key in after})

    def _column_gmres(self, rhs, initial, adjoint, rtol, atol, cap, width):
        engine = self._engine
        engine._count("linear_solve_columns")
        x = np.array(initial, copy=True)
        last_solution = x
        iterations = cycles = 0
        estimated = actual = rhs_norm = tolerance = None

        def norm(vector):
            engine._count("arnoldi_norm_evaluations")
            return float(la.norm(vector, check_finite=False))

        def residual(candidate):
            engine._count("true_residual_checks")
            value = rhs-self.apply_operator(candidate, adjoint=adjoint)
            return value, norm(value)

        def report(ok, reason):
            return ColumnReport(ok, reason, iterations, cycles, _finite(rhs_norm), _finite(actual),
                                _finite(tolerance), _finite(estimated))

        try:
            rhs_norm = norm(rhs)
            tolerance = max(atol, rtol*rhs_norm)
            if not np.isfinite(rhs_norm) or not np.isfinite(tolerance):
                return x, report(False, "nonfinite_rhs_norm_or_tolerance")
            r, actual = residual(x)
            if actual <= tolerance:
                return x, report(True, "initial_true_residual")
            while iterations < cap:
                beta = norm(r)
                if not np.isfinite(beta) or beta == 0:
                    return x, report(False, "nonfinite_or_zero_unaccepted_residual")
                steps = min(width, cap-iterations)
                basis = np.zeros((self.n, steps+1), dtype=np.complex128)
                directions = np.zeros((self.n, steps), dtype=np.complex128)
                hessenberg = np.zeros((steps+1, steps), dtype=np.complex128)
                cosines = np.zeros(steps)
                sines = np.zeros(steps, dtype=np.complex128)
                projected_rhs = np.zeros(steps+1, dtype=np.complex128)
                projected_rhs[0] = beta
                basis[:, 0] = r/beta
                cycles += 1
                engine._count("restart_cycles")
                for j in range(steps):
                    directions[:, j] = self.apply_preconditioner(basis[:, j], adjoint=adjoint)
                    work = self.apply_operator(directions[:, j], adjoint=adjoint)
                    original_work_norm = norm(work)
                    # Two-pass modified Gram-Schmidt; every executed dot/AXPY is paid.
                    for _ in range(2):
                        for i in range(j+1):
                            coefficient = np.vdot(basis[:, i], work)
                            engine._count("arnoldi_dot_products")
                            hessenberg[i, j] += coefficient
                            work -= coefficient*basis[:, i]
                            engine._count("arnoldi_axpy_updates")
                    tail = norm(work)
                    hessenberg[j+1, j] = tail
                    breakdown = tail <= 32*np.finfo(float).eps*max(original_work_norm, np.finfo(float).tiny)
                    if not breakdown:
                        basis[:, j+1] = work/tail
                    for i in range(j):
                        a, b = hessenberg[i, j], hessenberg[i+1, j]
                        hessenberg[i, j] = cosines[i]*a+sines[i]*b
                        hessenberg[i+1, j] = -sines[i].conjugate()*a+cosines[i]*b
                        engine._count("givens_rotations")
                    c, s, diagonal = zlartg(hessenberg[j, j], hessenberg[j+1, j])
                    engine._count("givens_constructions")
                    cosines[j], sines[j], hessenberg[j, j] = c, s, diagonal
                    hessenberg[j+1, j] = 0
                    a, b = projected_rhs[j], projected_rhs[j+1]
                    projected_rhs[j] = c*a+s*b
                    projected_rhs[j+1] = -s.conjugate()*a+c*b
                    engine._count("givens_rotations")
                    iterations += 1
                    engine._count("arnoldi_iterations")
                    estimated = float(abs(projected_rhs[j+1]))
                    if not np.isfinite(estimated) or not np.all(np.isfinite(hessenberg[:j+1, j])):
                        return last_solution, report(False, "nonfinite_arnoldi_arithmetic")
                    if estimated <= tolerance or breakdown or j == steps-1:
                        engine._count("projected_triangular_solves")
                        try:
                            coefficients = la.solve_triangular(hessenberg[:j+1, :j+1],
                                                               projected_rhs[:j+1], check_finite=False)
                        except la.LinAlgError:
                            return last_solution, report(False, "singular_projected_system")
                        engine._count("krylov_solution_update_columns", j+1)
                        with np.errstate(over="ignore", invalid="ignore"):
                            candidate = x+directions[:, :j+1]@coefficients
                        if not np.all(np.isfinite(candidate)):
                            return last_solution, report(False, "nonfinite_candidate")
                        candidate_r, actual = residual(candidate)
                        last_solution = candidate
                        if actual <= tolerance:
                            return candidate, report(True, "true_residual_tolerance")
                        if breakdown:
                            return candidate, report(False, "arnoldi_breakdown_before_true_convergence")
                        if j == steps-1:
                            x, r = candidate, candidate_r
            return x, report(False, "iteration_limit")
        except (FloatingPointError, la.LinAlgError, OverflowError, ValueError) as exc:
            return last_solution, report(False, "numerical_failure:"+type(exc).__name__)
