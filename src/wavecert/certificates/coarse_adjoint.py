"""Experimental POD physical-adjoint proposals with spectral coarse error bounds.

This is scoped to the current symmetric Dirichlet reference operator and a
fixed real POD basis. Sequentially corrected fields define a physical adjoint
proposal, not an automatic derivative of the composite corrected state map.
No optimizer, fine exact recovery, statistical calibration, or trajectory
guarantee is implemented here. Floating-point margins are not verified error
enclosures; the underlying inequalities are exact-arithmetic statements.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as la

from wavecert.physics.coarse_stability import FastCoarseStability
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock
from wavecert.surrogates.pod import PODGalerkinSurrogate

METHODS = ("uncorrected_generic_M", "uncorrected_oriented", "coarse_corrected_oriented")


def _readonly(values):
    array = np.array(values, copy=True)
    array.setflags(write=False)
    return array


def _delta(before, after):
    return {key: int(after.get(key, 0)-before.get(key, 0)) for key in sorted(set(before) | set(after))
            if key not in {"full_dimension", "reduced_dimension"}}


@dataclass(frozen=True)
class CoarseAdjointBlockCertificate:
    label: str
    source_index: int
    frequency_hz: float
    stability_available: bool
    reason: str | None
    correction_applied: bool
    correction_skipped: bool
    approximate_objective: float | None
    approximate_directional_derivative: float | None
    directional_error_bound: float | None
    primal_residual_components: tuple[float, float]
    adjoint_residual_components: tuple[float, float]
    eta_u_components: tuple[float, float] | None
    eta_lambda_components: tuple[float, float] | None
    eta_receiver_state: float | None
    bound_terms: tuple[float, float, float] | None
    state: np.ndarray = field(repr=False)
    adjoint: np.ndarray = field(repr=False)
    physical_gradient: np.ndarray | None = field(repr=False)

    @property
    def available(self) -> bool:
        return (self.stability_available and self.directional_error_bound is not None
                and self.certified_upper_derivative is not None)

    @property
    def certified_upper_derivative(self) -> float | None:
        if self.approximate_directional_derivative is None or self.directional_error_bound is None:
            return None
        upper = self.approximate_directional_derivative+self.directional_error_bound
        return float(upper) if np.isfinite(upper) else None

    def to_dict(self) -> dict:
        """Small strict-JSON-compatible diagnostics; full fields stay separate."""
        result = {name: getattr(self, name) for name in self.__dataclass_fields__
                  if name not in {"state", "adjoint", "physical_gradient"}}
        result.update(available=self.available, certified_upper_derivative=self.certified_upper_derivative)
        return result


@dataclass(frozen=True)
class CoarseAdjointSurveyResult:
    method: str
    model_sha256: str
    stability_contexts: dict
    available: bool
    certified_descent: bool
    exact_fallback_required: bool
    reason: str | None
    direction: np.ndarray
    approximate_gradient: np.ndarray | None
    approximate_objective: float | None
    approximate_directional_derivative: float | None
    directional_error_bound: float | None
    certified_upper_derivative: float | None
    block_certificates: tuple[CoarseAdjointBlockCertificate, ...]
    costs: dict

    def to_dict(self) -> dict:
        return {"method": self.method, "model_sha256": self.model_sha256,
            "stability_contexts": self.stability_contexts, "available": self.available,
            "certified_descent": self.certified_descent, "exact_fallback_required": self.exact_fallback_required,
            "reason": self.reason, "direction": self.direction.tolist(),
            "approximate_gradient": None if self.approximate_gradient is None else self.approximate_gradient.tolist(),
            "approximate_objective": self.approximate_objective,
            "approximate_directional_derivative": self.approximate_directional_derivative,
            "directional_error_bound": self.directional_error_bound,
            "certified_upper_derivative": self.certified_upper_derivative,
            "block_certificates": [block.to_dict() for block in self.block_certificates], "costs": self.costs}


class PODCoarseAdjointCertifier:
    """All-block physical proposal and deterministic discrete directional bound.

    An unavailable frequency retains its raw POD fields, explicitly skips any
    requested correction, and makes the complete aggregate uncertified. Its
    paid work and descriptive proposal remain reported. Even with available
    stability, an upper derivative >=0 or a zero/nonfinite proposal requires
    exact fallback. This research API never performs that exact recovery.
    """

    def __init__(self, physics: Helmholtz2D, surrogate: PODGalerkinSurrogate,
                 stability: FastCoarseStability):
        if type(physics) is not Helmholtz2D or type(surrogate) is not PODGalerkinSurrogate:
            raise TypeError("this experimental adapter requires the unchanged reference Helmholtz2D and POD classes")
        if type(stability) is not FastCoarseStability:
            raise TypeError("this adapter requires the reference FastCoarseStability engine")
        config = {"shape": list(physics.shape), "spacing": float(physics.spacing),
            "damping_width": int(physics.damping_width), "damping_strength": float(physics.damping_strength)}
        if surrogate.physics_config != config or stability.physics_config != config:
            raise ValueError("POD, stability and physical operator snapshots must match exactly")
        self.surrogate, self.stability = surrogate, stability
        self._physics = Helmholtz2D(**{**config, "shape": tuple(config["shape"])})
        self._config, self._cost = config, {}

    def _count(self, key, amount=1):
        self._cost[key] = self._cost.get(key, 0)+amount

    def cost_snapshot(self):
        return {"pod": self.surrogate.cost_snapshot(), "coarse_stability": self.stability.cost_snapshot(),
            "certificate": dict(self._cost), "reference_solver": self._physics.cost_snapshot()}

    def _validate(self, m, blocks, receiver_indices):
        if self.surrogate.physics_config != self._config or self.stability.physics_config != self._config:
            raise ValueError("operator snapshots changed after adapter construction")
        raw = np.asarray(m)
        if raw.shape != (self._physics.n,) or np.iscomplexobj(raw):
            raise ValueError("the common model must be a real vector matching the reference grid")
        model = self._physics.validate_model(raw)
        receivers = tuple(receiver_indices)
        if not receivers or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                                or not 0 <= i < self._physics.n for i in receivers):
            raise ValueError("receivers must be nonempty valid integer grid positions")
        blocks = tuple(blocks)
        if not blocks:
            raise ValueError("at least one source/frequency block is required")
        seen = set()
        for block in blocks:
            source, frequency = block.source_index, block.frequency_hz
            if (isinstance(source, (bool, np.bool_)) or not isinstance(source, (int, np.integer))
                    or not 0 <= source < self._physics.n):
                raise ValueError("source indices must be valid integers")
            if not np.isfinite(frequency) or frequency <= 0:
                raise ValueError("frequencies must be finite and positive")
            observations = np.asarray(block.observed)
            if observations.shape != (len(receivers),) or not np.all(np.isfinite(observations)):
                raise ValueError("each observed vector must be finite and match the common receivers")
            key = (int(source), float(frequency))
            if key in seen:
                raise ValueError("source/frequency blocks must be unique")
            seen.add(key)
        return model, blocks, tuple(int(i) for i in receivers)

    def _pod_fields(self, m, q, observed, receivers, frequency):
        """Explicit adapter to the unchanged POD implementation, with its counters."""
        _, omega2, factor = self.surrogate._prepare(m, frequency)
        state = self.surrogate._state(factor, q)
        residual = state[list(receivers)]-observed
        self._count("receiver_injections")
        injected = self._physics.inject_receivers(residual, receivers)
        self.surrogate._count("adjoint_rhs_projections")
        coefficient = self.surrogate._solve(factor, self.surrogate.basis.T @ injected,
            category="reduced_adjoint_rhs", adjoint=True)
        self.surrogate._count("fine_adjoint_lifts")
        return state, self.surrogate.basis @ coefficient, omega2

    def certify_survey(self, m, blocks: Iterable[SurveyBlock], receiver_indices,
                       *, method: str = "coarse_corrected_oriented") -> CoarseAdjointSurveyResult:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        model, blocks, receivers = self._validate(m, blocks, receiver_indices)
        before = self.cost_snapshot()
        stability = {frequency: self.stability.evaluate(model, frequency)
                     for frequency in sorted({float(block.frequency_hz) for block in blocks})}
        packs, total_gradient, objective = [], np.zeros(self._physics.n), 0.0
        finite_proposal = True
        for block in blocks:
            frequency = float(block.frequency_hz)
            reference = stability[frequency]
            q = self._physics.source(int(block.source_index))
            observed = np.asarray(block.observed, dtype=np.complex128)
            state, adjoint, omega2 = self._pod_fields(model, q, observed, receivers, frequency)
            corrected = method == "coarse_corrected_oriented" and reference.available
            if corrected:
                state, rp = reference.correct(state, q)
            else:
                rp = reference.apply_operator(state)-q
            residual = state[list(receivers)]-observed
            self._count("receiver_injections")
            adjoint_rhs = self._physics.inject_receivers(residual, receivers)
            if corrected:
                adjoint, ra = reference.correct(adjoint, adjoint_rhs, adjoint=True)
            else:
                ra = reference.apply_operator(adjoint, adjoint=True)-adjoint_rhs
            self._count("physical_gradient_contractions")
            with np.errstate(over="ignore", invalid="ignore"):
                gradient = omega2*np.real(np.conjugate(adjoint)*state)
                value = .5*float(np.vdot(residual, residual).real)
                total_gradient += gradient
            valid = np.all(np.isfinite(gradient)) and np.isfinite(value)
            finite_proposal = finite_proposal and valid
            objective += value
            packs.append({"block": block, "reference": reference, "state": state, "adjoint": adjoint,
                "rp": rp, "ra": ra, "omega2": omega2, "gradient": gradient if valid else None,
                "objective": value if np.isfinite(value) else None, "corrected": corrected})
        finite_proposal = bool(finite_proposal and np.all(np.isfinite(total_gradient)) and np.isfinite(objective))
        norm = float(la.norm(total_gradient)) if finite_proposal else None
        finite_proposal = finite_proposal and np.isfinite(norm)
        direction = -total_gradient/norm if finite_proposal and norm > 1e-14 else np.zeros(self._physics.n)
        all_available, total_bound, approximate_dd = finite_proposal, 0.0, 0.0
        numeric_failure = False
        complete_dd = finite_proposal
        certificates = []
        for pack in packs:
            block, reference = pack["block"], pack["reference"]
            rp_components, ra_components = reference.component_norms(pack["rp"]), reference.component_norms(pack["ra"])
            with np.errstate(over="ignore", invalid="ignore"):
                dd = float(np.dot(pack["gradient"], direction)) if finite_proposal else None
            if dd is not None and not np.isfinite(dd):
                dd, complete_dd, numeric_failure = None, False, True
            error_bound = eta_pu = eta_u = eta_lambda = terms = None
            reason = reference.reason if not reference.available else None
            available = bool(reference.available and finite_proposal and dd is not None)
            if available:
                constants = reference.receiver_constants(receivers)
                p_components = np.array([constants["pB"], constants["pQ"]])
                if method == "uncorrected_generic_M":
                    eta_u = reference.M @ rp_components
                    eta_pu = float(p_components @ eta_u)
                    eta_lambda = reference.M @ (ra_components+p_components*eta_pu)
                else:
                    eta_u = reference.oriented_error_bounds(pack["rp"])
                    eta_pu = float(p_components @ eta_u)
                    eta_lambda = reference.oriented_adjoint_error_bounds(pack["ra"], eta_pu, receivers)
                first = float(reference.component_norms(direction*pack["state"]) @ eta_lambda)
                second = float(reference.component_norms(direction*pack["adjoint"]) @ eta_u)
                cross = float(la.norm(direction, np.inf)*la.norm(eta_u)*la.norm(eta_lambda))
                terms = tuple(float(pack["omega2"]*term) for term in (first, second, cross))
                error_bound = float(sum(terms))
                if not np.isfinite(error_bound):
                    available, reason, error_bound = False, "nonfinite directional error bound", None
                    eta_u = eta_lambda = eta_pu = terms = None
                elif not np.isfinite(dd+error_bound):
                    available, reason, error_bound = False, "nonfinite block upper derivative", None
                    numeric_failure = True
            elif not finite_proposal:
                reason = "nonfinite approximate physical proposal"
            elif dd is None:
                reason = "nonfinite block directional derivative"
            all_available = all_available and available
            if dd is not None:
                approximate_dd += dd
            if error_bound is not None:
                total_bound += error_bound
            certificates.append(CoarseAdjointBlockCertificate(str(block.label), int(block.source_index), float(block.frequency_hz),
                bool(reference.available), reason, pack["corrected"],
                method == "coarse_corrected_oriented" and not reference.available,
                pack["objective"], dd, error_bound, tuple(float(v) for v in rp_components),
                tuple(float(v) for v in ra_components), None if eta_u is None else tuple(float(v) for v in eta_u),
                None if eta_lambda is None else tuple(float(v) for v in eta_lambda), eta_pu, terms,
                _readonly(pack["state"]), _readonly(pack["adjoint"]),
                None if pack["gradient"] is None else _readonly(pack["gradient"])))
        if not np.isfinite(total_bound) or not np.isfinite(approximate_dd):
            all_available, numeric_failure = False, True
        upper = approximate_dd+total_bound if all_available else None
        if upper is not None and not np.isfinite(upper):
            all_available, numeric_failure, upper = False, True, None
        certified = bool(all_available and upper < 0)
        if not finite_proposal:
            reason = "nonfinite approximate physical proposal; no stationarity claim"
        elif numeric_failure:
            reason = "nonfinite aggregate directional bound or derivative"
        elif not all_available:
            reason = "one or more block bounds unavailable"
        elif norm <= 1e-14:
            reason = "zero approximate proposal; no exact stationarity claim"
        elif not certified:
            reason = "nonnegative certified upper derivative"
        else:
            reason = None
        after = self.cost_snapshot()
        costs = {name: _delta(before[name], after[name]) for name in before}
        costs["dimensions"] = {"fine": self._physics.n, "pod": self.surrogate.reduced_dimension,
                               "spectral_coarse": self.stability.rank}
        contexts = {str(frequency): context.to_dict() for frequency, context in stability.items()}
        model_sha256 = next(iter(stability.values())).model_sha256
        return CoarseAdjointSurveyResult(method, model_sha256, contexts, bool(all_available), certified, not certified, reason,
            _readonly(direction), _readonly(total_gradient) if finite_proposal else None,
            float(objective) if np.isfinite(objective) else None,
            approximate_dd if complete_dd and np.isfinite(approximate_dd) else None,
            total_bound if all_available else None, upper, tuple(certificates), costs)
