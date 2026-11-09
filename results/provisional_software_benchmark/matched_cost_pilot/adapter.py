"""AI-authored provisional software benchmark adapters; no study outcomes.

All five controls use the SAME paid sparse-LU merit for finite-step acceptance.
The norm-free GMRES control is a tolerance-converged missing-field proposal,
not a different merit oracle. It pays the same coarse initializer and inverse
action as the fixed-sweep controls. No old production implementation is changed.
"""

from __future__ import annotations

import hashlib

import numpy as np
import scipy.linalg as la

from wavecert.certificates.cached_merit_certificate import (
    evaluate_cached_merit,
    fixed_sweep_missing_fields,
)
from wavecert.optimization.paid_descent import DirectionProposal, run_exact_lbfgsb, run_paid_descent
from wavecert.physics.coarse_dst_reference import CoarseDSTEngine, LinearSolveFailure
from wavecert.physics.coarse_stability import FastCoarseStability
from wavecert.physics.exact_merit import ExactMeritOracle
from wavecert.physics.helmholtz import Helmholtz2D

METHODS = (
    "exact_lu_projected", "exact_lu_lbfgsb", "same_M_gmres_projected",
    "fixed_sweeps_uncertified", "fixed_sweeps_certified",
)


def array_sha(values, dtype="<f8"):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


def missing_rhs(frame, n):
    """Same native scale, receiver-slot multiplicity and conjugation in both routes."""
    forward = frame.cache_kind == "receiver_map"
    values = (frame.source_amplitude*frame.residual_unique.conjugate().T
              if forward else frame.residual_slots)
    indices = frame.source_indices if forward else frame.receiver_indices
    rhs = np.zeros((n, values.shape[1]), dtype=np.complex128)
    np.add.at(rhs, np.asarray(indices), values)
    return rhs, not forward


@np.errstate(over="ignore", invalid="ignore", divide="ignore")
def physical_gradient(oracle, snapshot, fields, *, counters=None):
    """Uncertified contraction only; never computes a stability constant or bound."""
    oracle.validate_merit_fields(snapshot)
    if set(fields) != {frame.frequency_hz for frame in snapshot.frequencies}:
        raise ValueError("missing fields must match every issued frequency")
    gradient = np.zeros(snapshot.model.size)
    columns = 0
    for frame in snapshot.frequencies:
        missing = np.asarray(fields[frame.frequency_hz])
        if frame.cache_kind == "forward_states":
            paid = frame.cached_field
        else:
            if not frame.transpose_symmetric:
                raise ValueError("receiver-map proposal needs measured transpose symmetry")
            paid = frame.cached_field.T
        expected = paid.shape
        if missing.shape != expected or not np.all(np.isfinite(missing)):
            raise ValueError("missing field has wrong shape or is nonfinite")
        if counters is not None:
            counters["physical_gradient_columns"] += expected[1]
        contribution = (np.sum(np.conjugate(missing)*paid, axis=1) if frame.cache_kind == "forward_states"
                        else np.sum(paid*missing, axis=1))
        gradient += (2*np.pi*frame.frequency_hz)**2*np.real(contribution)
        columns += expected[1]
    if not np.all(np.isfinite(gradient)):
        raise FloatingPointError("nonfinite physical proposal gradient")
    if counters is not None:
        counters["gradient_norm_calls"] += 1
    norm = float(la.norm(gradient))
    if not np.isfinite(norm):
        raise FloatingPointError("nonfinite physical proposal norm")
    direction = -gradient/norm if norm > 0 else np.zeros_like(gradient)
    return gradient, direction, columns


class CachedFieldProposal:
    """Policy-owned setup, current-model contexts and all query work are counted."""

    def __init__(self, oracle, physics, *, mode, rank, sweeps, gmres):
        if mode not in METHODS[2:]:
            raise ValueError("unknown cached-field policy")
        self.oracle, self.physics, self.mode = oracle, physics, mode
        self.rank, self.sweeps, self.gmres = rank, sweeps, dict(gmres)
        self.failures = []
        self.normfree = CoarseDSTEngine(physics.shape, physics.spacing, physics.damping, rank,
            model_cache_size=4, context_cache_size=8)
        self.stability = FastCoarseStability(physics, rank, model_cache_size=4) if mode == METHODS[4] else None
        self._cost = dict(proposal_calls=0, physical_gradient_columns=0, gradient_norm_calls=0,
            gmres_report_columns=0, gmres_failed_columns=0, missing_rhs_injection_columns=0)
        self._certificate_local, self._certificate_dispatches = {}, {}

    def cost_snapshot(self):
        return dict(normfree=self.normfree.cost_snapshot(),
            stability={} if self.stability is None else self.stability.cost_snapshot(), adapter=dict(self._cost),
            certificate_local=dict(self._certificate_local), certificate_dispatches=dict(self._certificate_dispatches))

    def propose(self, model, receipt):
        self._cost["proposal_calls"] += 1
        snapshot = self.oracle.export_merit_fields(receipt)
        if not np.array_equal(model, snapshot.model):
            raise ValueError("proposal model differs from own paid current merit")
        contexts = {frame.frequency_hz: self.normfree.prepare(model, frame.frequency_hz)
                    for frame in snapshot.frequencies}
        # Fixed sweeps and GMRES start from exactly this coarse-Galerkin field.
        fields, sweep_receipt = fixed_sweep_missing_fields(self.oracle, snapshot, contexts,
            sweeps=0 if self.mode == METHODS[2] else self.sweeps)
        self._cost["missing_rhs_injection_columns"] += sum(field.shape[1] for field in fields.values())
        reports = []
        if self.mode == METHODS[2]:
            solved = {}
            for frame in snapshot.frequencies:
                rhs, adjoint = missing_rhs(frame, model.size)
                self._cost["missing_rhs_injection_columns"] += rhs.shape[1]
                report = contexts[frame.frequency_hz].solve_report(rhs, adjoint=adjoint,
                    x0=fields[frame.frequency_hz], **self.gmres)
                reports.append(report.to_dict())
                self._cost["gmres_report_columns"] += len(report.columns)
                self._cost["gmres_failed_columns"] += sum(not column.converged for column in report.columns)
                if not report.converged:
                    # Never return a nonconverged approximation as the accurate
                    # reference. The engine retains all costs and the failed run.
                    self.failures.append(dict(receipt_key=snapshot.receipt_key,
                        model_sha256=snapshot.model_sha256, completed_reports=reports))
                    raise LinearSolveFailure(report)
                solved[frame.frequency_hz] = report.solution
            fields = solved
        diagnostics = dict(policy=self.mode, rank=self.rank,
            fixed_sweeps=None if self.mode == METHODS[2] else self.sweeps,
            initializer="same coarse Galerkin missing field", fixed_sweep_receipt=sweep_receipt,
            field_sha256={str(f): array_sha(v, "<c16") for f, v in fields.items()}, gmres_reports=reports,
            same_paid_lu_merit=True, formally_verified_roundoff_enclosure=False)
        if self.stability is not None:
            stability = {frame.frequency_hz: self.stability.evaluate(model, frame.frequency_hz)
                         for frame in snapshot.frequencies}
            certificate = evaluate_cached_merit(self.oracle, snapshot, fields, stability)
            # The stability ledger already charges stability_query. Only these
            # two separate executed-unit ledgers are accumulated here; adding
            # stability_query again would double-count its transforms/solves.
            for target, key in ((self._certificate_local, "local"),
                                (self._certificate_dispatches, "dispatches")):
                for name, value in certificate.costs[key].items():
                    target[name] = target.get(name, 0)+int(value)
            diagnostics["certificate"] = certificate.to_dict()
            # Pass this exact vector as both proposal and certified direction;
            # only the common engine may positively rescale/project the step.
            return DirectionProposal(certificate.direction,
                fallback_required=certificate.exact_fallback_required,
                certified_descent=certificate.certified_descent,
                certificate_direction=certificate.direction,
                certificate_upper=certificate.certified_upper_derivative, diagnostics=diagnostics)
        gradient, direction, _ = physical_gradient(self.oracle, snapshot, fields, counters=self._cost)
        diagnostics.update(gradient_sha256=array_sha(gradient),
            gradient_l2_norm=float(la.norm(gradient)), certificate_attempted=False)
        self._cost["gradient_norm_calls"] += 1
        return DirectionProposal(direction, diagnostics=diagnostics)


def run_control(method, *, physics_config, model, blocks, receivers, options,
                lower_bound, upper_bound, rank, sweeps, gmres, lbfgsb, clock):
    """Cold per-policy factories are called INSIDE the shared engine's clock."""
    if method not in METHODS:
        raise ValueError("unknown matched-cost policy")
    objects = {}

    def oracle_factory():
        physics = Helmholtz2D(**{**physics_config, "shape": tuple(physics_config["shape"])})
        objects["physics"] = physics
        return ExactMeritOracle(physics, blocks, receivers)

    if method == METHODS[1]:
        result = run_exact_lbfgsb(oracle_factory=oracle_factory, m0=model, lower_bound=lower_bound,
            upper_bound=upper_bound, options=options, clock=clock, **lbfgsb)
    else:
        def proxy_factory(oracle):
            proposal = CachedFieldProposal(oracle, objects["physics"], mode=method,
                rank=rank, sweeps=sweeps, gmres=gmres)
            objects["proposal"] = proposal
            return proposal

        result = run_paid_descent(oracle_factory=oracle_factory,
            proxy_factory=None if method == METHODS[0] else proxy_factory,
            m0=model, lower_bound=lower_bound, upper_bound=upper_bound, options=options, clock=clock)
    result["proposal_solver_failures"] = [] if "proposal" not in objects else objects["proposal"].failures
    return result
