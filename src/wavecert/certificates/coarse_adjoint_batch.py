"""Batched execution of the frozen coarse physical-adjoint certificate.

Only dispatch/grouping and repeated-residual reuse change. The scalar module,
its stability evaluation, proof, parameters and fallback rule are unchanged.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as la

from wavecert.certificates.coarse_adjoint import (
    METHODS,
    CoarseAdjointBlockCertificate,
    CoarseAdjointSurveyResult,
    PODCoarseAdjointCertifier,
    _delta,
    _readonly,
)
from wavecert.physics.coarse_batch import BatchedCoarseOperations
from wavecert.surrogates.pod_batch import REFERENCE_SOURCE_SHA256, _batched_pod_survey_fields


class BatchedPODCoarseAdjointCertifier(PODCoarseAdjointCertifier):
    """Compatible survey result with grouped matrix RHS and spatial FFT axes."""

    def __init__(self, physics, surrogate, stability):
        super().__init__(physics, surrogate, stability)
        self._batch_dispatches = {}

    @property
    def reference_source_sha256(self):
        return dict(REFERENCE_SOURCE_SHA256)

    def cost_snapshot(self):
        return {**super().cost_snapshot(), "batch_dispatches": dict(self._batch_dispatches)}

    def _dispatch(self, name):
        self._batch_dispatches[name] = self._batch_dispatches.get(name, 0)+1

    def certify_survey(self, m, blocks, receiver_indices, *, method="coarse_corrected_oriented"):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        model, blocks, receivers = self._validate(m, blocks, receiver_indices)
        before = self.cost_snapshot()
        stability = {frequency: self.stability.evaluate(model, frequency)
                     for frequency in sorted({float(block.frequency_hz) for block in blocks})}
        fields = _batched_pod_survey_fields(self.surrogate, model, blocks, receivers, compute_physical=False)
        for key, value in fields.batch_dispatches.items():
            self._batch_dispatches[key] = self._batch_dispatches.get(key, 0)+value
        self._count("receiver_injections", len(blocks))  # Raw POD adjoint RHS, paid inside the adapter.
        packs, total_gradient, objective = {}, np.zeros(self._physics.n), 0.0
        ordered_gradients, ordered_objectives = {}, {}
        finite_proposal = True
        for frequency, indices in fields.group_indices.items():
            reference = stability[frequency]
            ops = BatchedCoarseOperations(reference, self._batch_dispatches)
            state, adjoint = fields.block_states[:, indices], fields.block_adjoints[:, indices]
            columns = len(indices)
            q = np.zeros((self._physics.n, columns), dtype=np.complex128)
            q[[blocks[i].source_index for i in indices], np.arange(columns)] = 1/self._physics.spacing**2
            self._dispatch("source_rhs_assembly_dispatches")
            corrected = method == "coarse_corrected_oriented" and reference.available
            if corrected:
                state, rp = ops.correct(state, q)
            else:
                rp = ops.apply_operator(state)-q
            observed = np.stack([blocks[i].observed for i in indices], axis=1)
            residual = state[list(receivers), :]-observed
            adjoint_rhs = np.zeros_like(state, dtype=np.complex128)
            np.add.at(adjoint_rhs, np.asarray(receivers), residual)
            self._count("receiver_injections", columns)
            self._dispatch("receiver_injection_dispatches")
            if corrected:
                adjoint, ra = ops.correct(adjoint, adjoint_rhs, adjoint=True)
            else:
                ra = ops.apply_operator(adjoint, adjoint=True)-adjoint_rhs
            omega2 = (2*np.pi*frequency)**2
            self._count("physical_gradient_contractions", columns)
            self._dispatch("physical_gradient_dispatches")
            with np.errstate(over="ignore", invalid="ignore"):
                gradients = omega2*np.real(np.conjugate(adjoint)*state)
                values = .5*np.sum(np.conjugate(residual)*residual, axis=0).real
            valid = np.all(np.isfinite(gradients), axis=0) & np.isfinite(values)
            finite_proposal = finite_proposal and bool(np.all(valid))
            for column, index in enumerate(indices):
                ordered_gradients[index], ordered_objectives[index] = gradients[:, column], values[column]
            packs[frequency] = dict(indices=indices, reference=reference, ops=ops, state=state, adjoint=adjoint,
                rp=rp, ra=ra, omega2=omega2, gradients=gradients, values=values, valid=valid, corrected=corrected)
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(len(blocks)):
                total_gradient += ordered_gradients[i]
                objective += float(ordered_objectives[i])
        finite_proposal = bool(finite_proposal and np.all(np.isfinite(total_gradient)) and np.isfinite(objective))
        norm = float(la.norm(total_gradient)) if finite_proposal else None
        finite_proposal = finite_proposal and np.isfinite(norm)
        direction = -total_gradient/norm if finite_proposal and norm > 1e-14 else np.zeros(self._physics.n)
        certificates, numeric_failure, complete_dd = {}, False, finite_proposal
        all_available = finite_proposal
        for pack in packs.values():
            reference, ops = pack["reference"], pack["ops"]
            rpb, rp_components = ops.decompose(pack["rp"])
            rab, ra_components = ops.decompose(pack["ra"])
            with np.errstate(over="ignore", invalid="ignore"):
                dd_values = np.sum(pack["gradients"]*direction[:, None], axis=0) if finite_proposal else None
            bounds = eta_u = eta_lambda = eta_pu = terms = None
            if reference.available and finite_proposal:
                constants = reference.receiver_constants(receivers)
                p_components = np.array([constants["pB"], constants["pQ"]])
                with np.errstate(over="ignore", invalid="ignore"):
                    if method == "uncorrected_generic_M":
                        eta_u = reference.M @ rp_components
                        eta_pu = p_components @ eta_u
                        eta_lambda = reference.M @ (ra_components+p_components[:, None]*eta_pu[None, :])
                    else:
                        eta_u = ops.oriented_error_bounds(rpb, rp_components)
                        eta_pu = p_components @ eta_u
                        eta_lambda = ops.oriented_adjoint_error_bounds(rab, ra_components, eta_pu, constants)
                    first = np.sum(ops.component_norms(direction[:, None]*pack["state"])*eta_lambda, axis=0)
                    second = np.sum(ops.component_norms(direction[:, None]*pack["adjoint"])*eta_u, axis=0)
                    cross = la.norm(direction, np.inf)*ops.norm_columns(eta_u)*ops.norm_columns(eta_lambda)
                    terms = pack["omega2"]*np.stack((first, second, cross))
                    bounds = np.sum(terms, axis=0)
            for column, index in enumerate(pack["indices"]):
                block = blocks[index]
                dd = None if dd_values is None else float(dd_values[column])
                if dd is not None and not np.isfinite(dd):
                    dd, complete_dd, numeric_failure = None, False, True
                reason = reference.reason if not reference.available else None
                available = bool(reference.available and finite_proposal and dd is not None)
                error_bound = float(bounds[column]) if available else None
                if available and not np.isfinite(error_bound):
                    available, error_bound, reason = False, None, "nonfinite directional error bound"
                elif available and not np.isfinite(dd+error_bound):
                    available, error_bound, reason, numeric_failure = False, None, "nonfinite block upper derivative", True
                elif not finite_proposal:
                    reason = "nonfinite approximate physical proposal"
                elif dd is None:
                    reason = "nonfinite block directional derivative"
                all_available = all_available and available
                certificates[index] = CoarseAdjointBlockCertificate(str(block.label), int(block.source_index), float(block.frequency_hz),
                    bool(reference.available), reason, pack["corrected"],
                    method == "coarse_corrected_oriented" and not reference.available,
                    float(pack["values"][column]) if np.isfinite(pack["values"][column]) else None,
                    dd, error_bound, tuple(float(v) for v in rp_components[:, column]),
                    tuple(float(v) for v in ra_components[:, column]),
                    tuple(float(v) for v in eta_u[:, column]) if available else None,
                    tuple(float(v) for v in eta_lambda[:, column]) if available else None,
                    float(eta_pu[column]) if available else None,
                    tuple(float(v) for v in terms[:, column]) if available else None,
                    _readonly(pack["state"][:, column]), _readonly(pack["adjoint"][:, column]),
                    _readonly(pack["gradients"][:, column]) if pack["valid"][column] else None)
        ordered = tuple(certificates[i] for i in range(len(blocks)))
        approximate_dd, total_bound = 0.0, 0.0
        for block in ordered:
            if block.approximate_directional_derivative is not None:
                approximate_dd += block.approximate_directional_derivative
            if block.directional_error_bound is not None:
                total_bound += block.directional_error_bound
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
        return CoarseAdjointSurveyResult(method, next(iter(stability.values())).model_sha256,
            {str(frequency): context.to_dict() for frequency, context in stability.items()},
            bool(all_available), certified, not certified, reason, _readonly(direction),
            _readonly(total_gradient) if finite_proposal else None,
            float(objective) if np.isfinite(objective) else None,
            approximate_dd if complete_dd and np.isfinite(approximate_dd) else None,
            total_bound if all_available else None, upper, ordered, costs)
