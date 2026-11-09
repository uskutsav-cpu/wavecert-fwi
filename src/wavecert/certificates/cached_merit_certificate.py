"""Ordinary-FP coupled bounds for native paid-merit cached fields.

Classical physical-adjoint/duality identities, not a new theorem or a verified
roundoff enclosure.  The issuing oracle validates the immutable export first.
Paid U/Z residuals and affine data-residual defects are measured, never zeroed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import scipy.linalg as la

from wavecert.physics.coarse_batch import BatchedCoarseOperations


def _immutable(values):
    array = np.ascontiguousarray(values)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _hash(values, dtype="<f8"):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


def _matrix(values, shape, name):
    array = np.asarray(values, dtype=np.complex128)
    if array.shape != shape:
        raise ValueError(name+" has wrong shape")
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(name+" is nonfinite")
    return np.array(array, copy=True)


def _delta(before, after):
    return {key: after.get(key, 0)-before.get(key, 0) for key in set(before)|set(after)}


def _strict(value):
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _validate(expected_oracle, snapshot, contexts):
    # This mandatory registry/owner/type/byte validation is deliberately owned
    # by the oracle.  No public constructor for unissued snapshots is supplied.
    expected_oracle.validate_merit_fields(snapshot)
    frequencies = tuple(frame.frequency_hz for frame in snapshot.frequencies)
    if set(contexts) != set(frequencies) or len(frequencies) != len(set(frequencies)):
        raise ValueError("contexts must match every exported frequency exactly")
    n = int(np.prod(snapshot.shape))
    damping_hash = _hash(snapshot.damping)
    if snapshot.model_sha256 != _hash(snapshot.model):
        raise ValueError("export model hash mismatch")
    for frame in snapshot.frequencies:
        c = contexts[frame.frequency_hz]
        signature = getattr(c, "operator_signature", getattr(c, "identity", None))
        if callable(signature):
            signature = signature()
        config = getattr(c, "physics_config", None)
        if config is None:
            config = signature
        if (c.n != n or c.frequency_hz != frame.frequency_hz
                or tuple(config["shape"]) != tuple(snapshot.shape)
                or config["spacing"] != snapshot.spacing
                or signature["damping_float64_le_sha256"] != damping_hash):
            raise ValueError("context operator/frequency mismatch")
        model_hash = getattr(c, "model_sha256", None)
        if model_hash is None:
            model_hash = signature["model_float64_le_sha256"]
        if model_hash != snapshot.model_sha256:
            raise ValueError("context model mismatch")
        if frame.source_amplitude != 1/snapshot.spacing**2:
            raise ValueError("native source scale must be real1/spacing²")
        if frame.cache_kind not in {"forward_states", "receiver_map"}:
            raise ValueError("unsupported cached merit route")
        if frame.cache_kind == "receiver_map" and not frame.transpose_symmetric:
            raise ValueError("Z certificate requires verified transpose symmetry")


def _source_rhs(frame, n):
    rhs = np.zeros((n, len(frame.source_indices)), dtype=np.complex128)
    np.add.at(rhs, (np.asarray(frame.source_indices), np.arange(rhs.shape[1])), frame.source_amplitude)
    return rhs


def _inject(indices, values, n):
    rhs = np.zeros((n, values.shape[1]), dtype=np.complex128)
    np.add.at(rhs, np.asarray(indices), values)
    return rhs


def _missing_rhs(frame, n):
    if frame.cache_kind == "forward_states":
        return _inject(frame.receiver_indices, frame.residual_slots, n), True
    return _inject(frame.source_indices, frame.source_amplitude*frame.residual_unique.conjugate().T, n), False


@np.errstate(over="ignore", invalid="ignore", divide="ignore")
def fixed_sweep_missing_fields(expected_oracle, snapshot, contexts, *, sweeps=0, initial=None):
    """Norm-free fixed coarse/DST proposal, shared with uncertified controls.

Contexts are the norm-free linear contexts, not stability contexts.  Constructor
and prepare costs must be charged by their owner; this reports query deltas.
No convergence or descent claim is made for a fixed sweep count.
"""
    _validate(expected_oracle, snapshot, contexts)
    if isinstance(sweeps, (bool, np.bool_)) or not isinstance(sweeps, (int, np.integer)) or sweeps < 0:
        raise ValueError("sweeps must be a nonnegative integer")
    if initial is not None and set(initial) != set(contexts):
        raise ValueError("initial fields must match every frequency")
    engines = {id(c._engine): c._engine for c in contexts.values()}
    before = {key: value.cost_snapshot() for key, value in engines.items()}
    fields = {}
    for frame in snapshot.frequencies:
        c = contexts[frame.frequency_hz]
        rhs, adjoint = _missing_rhs(frame, c.n)
        if initial is None:
            field = c.lift(c.solve_coarse(c.project(rhs), adjoint=adjoint))
        else:
            field = _matrix(initial[frame.frequency_hz], rhs.shape, "initial missing field")
        for _ in range(int(sweeps)):
            field += c.apply_preconditioner(rhs-c.apply_operator(field, adjoint=adjoint), adjoint=adjoint)
        if not np.all(np.isfinite(field)):
            raise FloatingPointError("nonfinite fixed-sweep proposal")
        fields[frame.frequency_hz] = _immutable(field)
    costs = {}
    for key, engine in engines.items():
        for name, value in _delta(before[key], engine.cost_snapshot()).items():
            costs[name] = costs.get(name, 0)+value
    return fields, {"sweeps": int(sweeps), "initializer": "coarse" if initial is None else "provided",
                       "query_costs": costs, "setup_costs_included": False,
                       "field_sha256": {str(k): _hash(v, "<c16") for k, v in fields.items()}}


@dataclass(frozen=True)
class CachedMeritCertificate:
    receipt_key: str
    binding: dict
    direction: np.ndarray | None
    approximate_gradient: np.ndarray | None
    approximate_directional_derivative: float | None
    directional_error_bound: float | None
    certified_upper_derivative: float | None
    available: bool
    certified_descent: bool
    exact_fallback_required: bool
    evaluation_complete: bool
    reason: str
    frequency_records: tuple
    costs: dict

    def to_dict(self):
        return _strict({"receipt_key": self.receipt_key, "binding": self.binding,
            "approximate_directional_derivative": self.approximate_directional_derivative,
            "directional_error_bound": self.directional_error_bound,
            "certified_upper_derivative": self.certified_upper_derivative,
            "available": self.available, "certified_descent": self.certified_descent,
            "exact_fallback_required": self.exact_fallback_required,
            "evaluation_complete": self.evaluation_complete, "reason": self.reason,
            "approximate_gradient_sha256": None if self.approximate_gradient is None else _hash(self.approximate_gradient),
            "direction_sha256": None if self.direction is None else _hash(self.direction),
            "frequency_records": list(self.frequency_records), "costs": self.costs,
            "scope": "ordinary floating point; existing unverified stability allowance; pointwise derivative only",
            "formally_verified_roundoff_enclosure": False, "stationarity_claim": False})


@np.errstate(over="ignore", invalid="ignore", divide="ignore")
def evaluate_cached_merit(expected_oracle, snapshot, missing_fields, contexts, *, direction=None):
    """Bound one shared survey direction using issued paid U/C and missing fields.

Missing fields are Lambda(n,s) for forward-state merit or V(n,r_unique) for
receiver-map merit.  Their construction costs are external, never treated as
free.  This function reports only executed bound-query work; full stability
setup/current merit/proposal/line-search costs remain the caller's obligation.
"""
    _validate(expected_oracle, snapshot, contexts)
    if set(missing_fields) != set(contexts):
        raise ValueError("missing fields must match every frequency")
    n = len(snapshot.model)
    for frame in snapshot.frequencies:
        columns = len(frame.source_indices) if frame.cache_kind == "forward_states" else len(frame.unique_receiver_indices)
        if np.asarray(missing_fields[frame.frequency_hz]).shape != (n, columns):
            raise ValueError("missing field has wrong shape")
    if direction is not None:
        raw = np.asarray(direction)
        if raw.shape != (n,) or np.iscomplexobj(raw) or not np.all(np.isfinite(raw)):
            raise ValueError("direction must be a finite real model vector")
        direction = np.array(raw, dtype=float, copy=True)
    engines = {id(c._engine): c._engine for c in contexts.values()}
    before = {key: value.cost_snapshot() for key, value in engines.items()}
    dispatches, packs, records = {}, [], []
    local = {"paid_field_residual_columns": 0, "missing_field_residual_columns": 0,
                 "affine_data_defect_columns": 0, "injection_columns": 0,
                 "physical_gradient_contractions": 0, "directional_bound_columns": 0,
                 "diagnostic_frobenius_norm_calls": 0, "survey_vector_norm_calls": 0,
                 "directional_dot_products": 0}

    def result(gradient, chosen, center, bound, upper, available, complete, reason):
        work = {}
        for key, engine in engines.items():
            for name, value in _delta(before[key], engine.cost_snapshot()).items():
                work[name] = work.get(name, 0)+value
        certified = bool(available and upper is not None and upper < 0)
        binding = snapshot.to_dict() if hasattr(snapshot, "to_dict") else {
            "receipt_key": snapshot.receipt_key, "model_sha256": snapshot.model_sha256,
            "scope": "controlled synthetic mathematical issuer; not a native oracle export"}
        return CachedMeritCertificate(snapshot.receipt_key, binding,
            None if chosen is None else _immutable(chosen),
            None if gradient is None else _immutable(gradient), center, bound, upper,
            available, certified, not certified, complete, reason, tuple(records),
            {"stability_query": work, "dispatches": dict(dispatches), "local": dict(local),
                 "setup_costs_included": False, "proposal_costs_included": False, "merit_costs_included": False})

    try:
        gradient = np.zeros(n)
        for frame in snapshot.frequencies:
            c = contexts[frame.frequency_hz]
            ops = BatchedCoarseOperations(c, dispatches)
            s, r = len(frame.source_indices), len(frame.unique_receiver_indices)
            missing = _matrix(missing_fields[frame.frequency_hz], (n, s if frame.cache_kind == "forward_states" else r), "missing field")
            receivers = np.asarray(frame.receiver_indices)
            expand = np.asarray(frame.receiver_expand_indices)
            if frame.cache_kind == "forward_states":
                paid = _matrix(frame.cached_field, (n, s), "cached U")
                target = paid[receivers]-frame.observed_slots
                defect = frame.residual_slots-target
                first_residual = ops.apply_operator(paid)-_source_rhs(frame, n)
                rhs = _inject(frame.receiver_indices, frame.residual_slots, n)
                defect_rhs = _inject(frame.receiver_indices, defect, n)
                second_raw = ops.apply_operator(missing, adjoint=True)-rhs
                second_residual = second_raw+defect_rhs
                local["injection_columns"] += 2*s
                contribution = c.omega**2*np.real(np.sum(np.conjugate(missing)*paid, axis=1))
            else:
                paid = _matrix(frame.cached_field, (r, n), "cached C").T
                weights = np.bincount(expand, minlength=r).astype(float)
                data_sum = np.zeros((r, s), dtype=np.complex128)
                np.add.at(data_sum, expand, frame.observed_slots)
                target_unique = weights[:, None]*frame.source_amplitude*paid[np.asarray(frame.source_indices)].T-data_sum
                defect = frame.residual_unique-target_unique
                unit_receivers = np.zeros((n, r), dtype=np.complex128)
                unit_receivers[np.asarray(frame.unique_receiver_indices), np.arange(r)] = 1
                first_residual = ops.apply_operator(paid)-unit_receivers
                rhs = _inject(frame.source_indices, frame.source_amplitude*frame.residual_unique.conjugate().T, n)
                defect_rhs = _inject(frame.source_indices, frame.source_amplitude*defect.conjugate().T, n)
                second_raw = ops.apply_operator(missing)-rhs
                second_residual = second_raw+defect_rhs
                local["injection_columns"] += 2*r
                contribution = c.omega**2*np.real(np.sum(paid*missing, axis=1))
            local["paid_field_residual_columns"] += paid.shape[1]
            local["missing_field_residual_columns"] += missing.shape[1]
            local["affine_data_defect_columns"] += defect.shape[1]
            local["physical_gradient_contractions"] += paid.shape[1]
            if not all(np.all(np.isfinite(v)) for v in (contribution, first_residual, second_residual, defect)):
                raise FloatingPointError("nonfinite field/residual/gradient arithmetic")
            gradient += contribution
            first_b, first_components = ops.decompose(first_residual)
            second_b, second_components = ops.decompose(second_residual)
            raw_norms = ops.norm_columns(second_raw)
            affine_norm = float(la.norm(defect))
            local["diagnostic_frobenius_norm_calls"] += 1
            if not all(np.all(np.isfinite(v)) for v in (first_components, second_components,
                                                        raw_norms, affine_norm)):
                raise FloatingPointError("nonfinite residual diagnostics")
            record = {"frequency_hz": frame.frequency_hz, "cache_kind": frame.cache_kind,
                          "paid_field_residual_components": first_components.tolist(),
                          "missing_raw_residual_norms": raw_norms.tolist(),
                          "missing_effective_residual_components": second_components.tolist(),
                          "affine_data_defect_norm": affine_norm,
                          "stability_available": bool(c.available), "stability": c.to_dict(),
                          "approximate_directional_derivative": None, "directional_error_bound": None}
            records.append(record)
            packs.append((frame, c, ops, paid, missing, first_b, first_components, second_b, second_components, contribution, record))
        if not np.all(np.isfinite(gradient)):
            raise FloatingPointError("nonfinite survey gradient")
        norm = float(la.norm(gradient))
        local["survey_vector_norm_calls"] += 1
        if not np.isfinite(norm):
            raise FloatingPointError("nonfinite survey gradient norm")
        chosen = (-gradient/norm if norm > 0 else np.zeros(n)) if direction is None else direction
        center = float(np.dot(gradient, chosen))
        local["directional_dot_products"] += 1
        if not np.isfinite(center):
            raise FloatingPointError("nonfinite survey directional center")
        total = 0.
        available = all(c.available for _, c, *_ in packs)
        direction_inf_norm = float(la.norm(chosen, np.inf))
        local["survey_vector_norm_calls"] += 1
        for frame, c, ops, paid, missing, rb1, rc1, rb2, rc2, contribution, record in packs:
            record["approximate_directional_derivative"] = float(np.dot(contribution, chosen))
            local["directional_dot_products"] += 1
            if not np.isfinite(record["approximate_directional_derivative"]):
                raise FloatingPointError("nonfinite frequency directional center")
            if not c.available:
                continue
            eta1 = ops.oriented_error_bounds(rb1, rc1)
            if frame.cache_kind == "forward_states":
                constants = c.receiver_constants(frame.receiver_indices)
                eta_sample = np.array([constants["pB"], constants["pQ"]])@eta1
                eta2 = ops.oriented_adjoint_error_bounds(rb2, rc2, eta_sample, constants)
            else:
                constants = c.receiver_constants(frame.source_indices)
                weights = np.bincount(np.asarray(frame.receiver_expand_indices), minlength=paid.shape[1])
                eta_sample = np.array([constants["pB"], constants["pQ"]])@eta1
                eta_force = weights*frame.source_amplitude**2*eta_sample
                response = ops.norm_columns(ops.solve(rb2, category="coarse_oriented_rhs"))
                forcing = rc2[1]+constants["pQ"]*eta_force
                eb = (response+constants["rho_adjoint"]*eta_force+c.k*forcing/c.gamma)/(1-c.tau)
                eta2 = np.stack((eb, (forcing+c.ell*eb)/c.gamma))
                record["source_forcing_coefficient_bounds"] = eta_force.tolist()
                record["receiver_multiplicities"] = weights.tolist()
            first = np.sum(ops.component_norms(chosen[:, None]*paid)*eta2, axis=0)
            second = np.sum(ops.component_norms(chosen[:, None]*missing)*eta1, axis=0)
            cross = direction_inf_norm*ops.norm_columns(eta1)*ops.norm_columns(eta2)
            terms = c.omega**2*np.stack((first, second, cross))
            bound = float(np.sum(terms))
            local["directional_bound_columns"] += paid.shape[1]
            if not all(np.all(np.isfinite(v)) for v in (eta1, eta2, terms, bound)):
                raise FloatingPointError("nonfinite component/directional bound")
            record.update(paid_field_error_components=eta1.tolist(), missing_field_error_components=eta2.tolist(),
                          directional_bound_terms=terms.tolist(), directional_error_bound=bound)
            total += bound
        upper = center+total if available else None
        if not np.isfinite(total) or (upper is not None and not np.isfinite(upper)):
            raise FloatingPointError("nonfinite aggregate bound")
        if norm == 0 or not np.any(chosen):
            return result(gradient, chosen, center, total if available else None, upper, False, True,
                          "zero proposal/direction; no true-stationarity claim")
        return result(gradient, chosen, center, total if available else None, upper, available, True,
                      "available (ordinary floating point)" if available else "stability unavailable")
    except (FloatingPointError, OverflowError, la.LinAlgError, ValueError) as exc:
        return result(None, None, None, None, None, False, False, "numeric abstention:"+str(exc))
