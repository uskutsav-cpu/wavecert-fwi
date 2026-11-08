"""Pure coarse/DST field repair with the existing physical-direction bound.

The repair and residual-gradient inequalities are established methods. This
adapter exposes a no-training proposal and full work accounting, not a new
convergence theorem or a speedup claim. Bounds remain ordinary floating point.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.fft as fft
import scipy.linalg as la

from wavecert.certificates.coarse_adjoint import (
    CoarseAdjointBlockCertificate,
    CoarseAdjointSurveyResult,
    _delta,
)
from wavecert.physics.coarse_batch import BatchedCoarseOperations
from wavecert.physics.coarse_stability import FastCoarseStability
from wavecert.physics.helmholtz import Helmholtz2D
from wavecert.surrogates.pod_batch import REFERENCE_SOURCE_SHA256, _verify_reference_sources

_verify_reference_sources()
_COARSE_BATCH_SHA = "6adf41b5b836a1331661a598c4e9b6e5e24ebd35ec659b7d86db95134e22a7be"
_reference_path = Path(__file__).resolve().parents[1] / "physics/coarse_batch.py"
if (
    Path(BatchedCoarseOperations.__init__.__code__.co_filename).resolve()
    != _reference_path.resolve()
    or hashlib.sha256(_reference_path.read_bytes()).hexdigest() != _COARSE_BATCH_SHA
):
    raise RuntimeError("pure deflated adapter requires the frozen batch coarse operations")


def _readonly(values):
    values = np.ascontiguousarray(values)
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


class DeflatedDirectionCertifier:
    """Fixed coupled Richardson sweeps, followed by a pointwise DD bound.

    An unavailable frequency uses the explicit zero-field seed and forces
    aggregate abstention. No fine solve or stationarity inference occurs.
    Finite-step acceptance and exact recovery belong to the optimizer.
    """

    def __init__(self, physics, *, rank=64, repair_sweeps=4):
        if type(physics) is not Helmholtz2D:
            raise TypeError("this implementation is bound to the reference Helmholtz2D")
        if (
            isinstance(repair_sweeps, (bool, np.bool_))
            or not isinstance(repair_sweeps, (int, np.integer))
            or not 0 <= repair_sweeps <= 16
        ):
            raise ValueError("repair_sweeps must be an integer in [0,16]")
        self._physics = Helmholtz2D(
            shape=tuple(physics.shape),
            spacing=physics.spacing,
            damping_width=physics.damping_width,
            damping_strength=physics.damping_strength,
        )
        self.stability = FastCoarseStability(self._physics, rank)
        self._config = self.stability.physics_config
        self._repair_sweeps = int(repair_sweeps)
        self._batch_dispatches, self._cost = {}, {}
        nz, nx = self._physics.shape
        h = self._physics.spacing
        z, x = np.arange(1, nz + 1)[:, None], np.arange(1, nx + 1)[None, :]
        self._eigen_grid = _readonly(
            4
            / h**2
            * (np.sin(np.pi * z / (2 * (nz + 1))) ** 2 + np.sin(np.pi * x / (2 * (nx + 1))) ** 2)
        )
        modes = self.stability.mode_indices
        mask = np.ones((nz, nx), dtype=bool)
        mask[modes[:, 0] - 1, modes[:, 1] - 1] = False
        self._high_mask = _readonly(mask)
        damping = self._physics.damping
        self._middle_d = (float(np.min(damping)) + float(np.max(damping))) / 2
        self._high_references = OrderedDict()

    @property
    def repair_sweeps(self):
        return self._repair_sweeps

    def _count(self, name, amount=1):
        self._cost[name] = self._cost.get(name, 0) + amount

    def _validate(self, m, blocks, receiver_indices):
        if self.stability.physics_config != self._config:
            raise ValueError("coarse operator snapshot changed")
        raw = np.asarray(m)
        if raw.shape != (self._physics.n,) or np.iscomplexobj(raw):
            raise ValueError("model must be a real flat vector matching the reference grid")
        model = self._physics.validate_model(raw)
        receivers = tuple(receiver_indices)
        if not receivers or any(
            isinstance(i, (bool, np.bool_))
            or not isinstance(i, (int, np.integer))
            or not 0 <= i < self._physics.n
            for i in receivers
        ):
            raise ValueError("receiver indices must be nonempty valid integer positions")
        snapshot, seen = [], set()
        for block in blocks:
            source, frequency = block.source_index, block.frequency_hz
            if (
                isinstance(source, (bool, np.bool_))
                or not isinstance(source, (int, np.integer))
                or not 0 <= source < self._physics.n
            ):
                raise ValueError("source must be a valid integer position")
            if (
                isinstance(frequency, (bool, np.bool_))
                or not isinstance(frequency, (int, float, np.integer, np.floating))
                or not np.isfinite(frequency)
                or frequency <= 0
                or frequency > np.sqrt(np.finfo(float).max) / (2 * np.pi)
            ):
                raise ValueError("frequency must be finite real and positive")
            observed = np.array(block.observed, dtype=np.complex128, copy=True)
            if observed.shape != (len(receivers),) or not np.all(np.isfinite(observed)):
                raise ValueError("observations must be finite and match ordered receivers")
            if la.norm(observed) > np.sqrt(np.finfo(float).max) / 4:
                raise ValueError("observations exceed the representable squared-objective domain")
            key = int(source), float(frequency)
            if key in seen:
                raise ValueError("source/frequency pairs must be unique")
            seen.add(key)
            snapshot.append(
                SimpleNamespace(
                    source_index=key[0],
                    frequency_hz=key[1],
                    observed=observed,
                    label=str(block.label),
                )
            )
        if not snapshot:
            raise ValueError("at least one block is required")
        return model.copy(), tuple(snapshot), tuple(int(i) for i in receivers)

    @property
    def reference_source_sha256(self):
        return {**REFERENCE_SOURCE_SHA256, "physics/coarse_batch.py": _COARSE_BATCH_SHA}

    def cost_snapshot(self):
        return {
            "coarse_stability": self.stability.cost_snapshot(),
            "certificate": dict(self._cost),
            "reference_solver": self._physics.cost_snapshot(),
            "batch_dispatches": dict(self._batch_dispatches),
        }

    def _dispatch(self, name):
        self._batch_dispatches[name] = self._batch_dispatches.get(name, 0) + 1

    def certify_survey(self, m, blocks, receiver_indices):
        method = f"pure_deflated_coarse{self.stability.rank}_sweeps{self.repair_sweeps}"
        model, blocks, receivers = self._validate(m, blocks, receiver_indices)
        before = self.cost_snapshot()
        stability = {
            frequency: self.stability.evaluate(model, frequency)
            for frequency in sorted({float(block.frequency_hz) for block in blocks})
        }
        fields = self._coarse_fields(model, blocks, receivers, stability)
        packs, total_gradient, objective = {}, np.zeros(self._physics.n), 0.0
        ordered_gradients, ordered_objectives = {}, {}
        finite_proposal = not fields.failures
        for frequency, indices in fields.group_indices.items():
            reference = stability[frequency]
            ops = BatchedCoarseOperations(reference, self._batch_dispatches)
            state, adjoint = fields.block_states[:, indices], fields.block_adjoints[:, indices]
            columns = len(indices)
            q = np.zeros((self._physics.n, columns), dtype=np.complex128)
            q[[blocks[i].source_index for i in indices], np.arange(columns)] = (
                1 / self._physics.spacing**2
            )
            self._dispatch("source_rhs_assembly_dispatches")
            corrected = reference.available and frequency not in fields.failures
            observed = np.stack([blocks[i].observed for i in indices], axis=1)
            residual = state[list(receivers), :] - observed
            residuals = getattr(fields, "residuals", {})
            if frequency in residuals:
                rp, ra = residuals[frequency]
            else:
                rp = ops.apply_operator(state) - q
                adjoint_rhs = self._inject(residual, receivers)
                ra = ops.apply_operator(adjoint, adjoint=True) - adjoint_rhs
            omega2 = (2 * np.pi * frequency) ** 2
            self._count("physical_gradient_contractions", columns)
            self._dispatch("physical_gradient_dispatches")
            with np.errstate(over="ignore", invalid="ignore"):
                gradients = omega2 * np.real(np.conjugate(adjoint) * state)
                values = 0.5 * np.sum(np.conjugate(residual) * residual, axis=0).real
            valid = np.all(np.isfinite(gradients), axis=0) & np.isfinite(values)
            finite_proposal = finite_proposal and bool(np.all(valid))
            for column, index in enumerate(indices):
                ordered_gradients[index], ordered_objectives[index] = (
                    gradients[:, column],
                    values[column],
                )
            packs[frequency] = dict(
                indices=indices,
                reference=reference,
                ops=ops,
                state=state,
                adjoint=adjoint,
                rp=rp,
                ra=ra,
                omega2=omega2,
                gradients=gradients,
                values=values,
                valid=valid,
                corrected=corrected,
            )
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(len(blocks)):
                total_gradient += ordered_gradients[i]
                objective += float(ordered_objectives[i])
        finite_proposal = bool(
            finite_proposal and np.all(np.isfinite(total_gradient)) and np.isfinite(objective)
        )
        norm = float(la.norm(total_gradient)) if finite_proposal else None
        finite_proposal = finite_proposal and np.isfinite(norm)
        direction = (
            -total_gradient / norm
            if finite_proposal and norm > 1e-14
            else np.zeros(self._physics.n)
        )
        certificates, numeric_failure, complete_dd = {}, False, finite_proposal
        all_available = finite_proposal
        for pack in packs.values():
            reference, ops = pack["reference"], pack["ops"]
            rpb, rp_components = ops.decompose(pack["rp"])
            rab, ra_components = ops.decompose(pack["ra"])
            with np.errstate(over="ignore", invalid="ignore"):
                dd_values = (
                    np.sum(pack["gradients"] * direction[:, None], axis=0)
                    if finite_proposal
                    else None
                )
            bounds = eta_u = eta_lambda = eta_pu = terms = None
            if reference.available and finite_proposal:
                constants = reference.receiver_constants(receivers)
                p_components = np.array([constants["pB"], constants["pQ"]])
                with np.errstate(over="ignore", invalid="ignore"):
                    eta_u = ops.oriented_error_bounds(rpb, rp_components)
                    eta_pu = p_components @ eta_u
                    eta_lambda = ops.oriented_adjoint_error_bounds(
                        rab, ra_components, eta_pu, constants
                    )
                    first = np.sum(
                        ops.component_norms(direction[:, None] * pack["state"]) * eta_lambda, axis=0
                    )
                    second = np.sum(
                        ops.component_norms(direction[:, None] * pack["adjoint"]) * eta_u, axis=0
                    )
                    cross = (
                        la.norm(direction, np.inf)
                        * ops.norm_columns(eta_u)
                        * ops.norm_columns(eta_lambda)
                    )
                    terms = pack["omega2"] * np.stack((first, second, cross))
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
                    available, error_bound, reason = (
                        False,
                        None,
                        "nonfinite directional error bound",
                    )
                elif available and not np.isfinite(dd + error_bound):
                    available, error_bound, reason, numeric_failure = (
                        False,
                        None,
                        "nonfinite block upper derivative",
                        True,
                    )
                elif not finite_proposal:
                    reason = "nonfinite approximate physical proposal"
                elif dd is None:
                    reason = "nonfinite block directional derivative"
                all_available = all_available and available
                certificates[index] = CoarseAdjointBlockCertificate(
                    str(block.label),
                    int(block.source_index),
                    float(block.frequency_hz),
                    bool(reference.available),
                    reason,
                    pack["corrected"],
                    not pack["corrected"],
                    float(pack["values"][column]) if np.isfinite(pack["values"][column]) else None,
                    dd,
                    error_bound,
                    tuple(float(v) if np.isfinite(v) else None for v in rp_components[:, column]),
                    tuple(float(v) if np.isfinite(v) else None for v in ra_components[:, column]),
                    tuple(float(v) for v in eta_u[:, column]) if available else None,
                    tuple(float(v) for v in eta_lambda[:, column]) if available else None,
                    float(eta_pu[column]) if available else None,
                    tuple(float(v) for v in terms[:, column]) if available else None,
                    _readonly(pack["state"][:, column]),
                    _readonly(pack["adjoint"][:, column]),
                    _readonly(pack["gradients"][:, column]) if pack["valid"][column] else None,
                )
        ordered = tuple(certificates[i] for i in range(len(blocks)))
        approximate_dd, total_bound = 0.0, 0.0
        for block in ordered:
            if block.approximate_directional_derivative is not None:
                approximate_dd += block.approximate_directional_derivative
            if block.directional_error_bound is not None:
                total_bound += block.directional_error_bound
        if not np.isfinite(total_bound) or not np.isfinite(approximate_dd):
            all_available, numeric_failure = False, True
        upper = approximate_dd + total_bound if all_available else None
        if upper is not None and not np.isfinite(upper):
            all_available, numeric_failure, upper = False, True, None
        certified = bool(all_available and upper < 0)
        if fields.failures:
            reason = "field generation failed; no valid physical proposal or stationarity claim"
        elif not finite_proposal:
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
        costs["field_generation_failures"] = {str(k): v for k, v in fields.failures.items()}
        costs["dimensions"] = {
            "fine": self._physics.n,
            "spectral_coarse": self.stability.rank,
            "repair_sweeps": self.repair_sweeps,
        }
        return CoarseAdjointSurveyResult(
            method,
            next(iter(stability.values())).model_sha256,
            {str(frequency): context.to_dict() for frequency, context in stability.items()},
            bool(all_available),
            certified,
            not certified,
            reason,
            _readonly(direction),
            _readonly(total_gradient) if finite_proposal else None,
            float(objective) if np.isfinite(objective) else None,
            approximate_dd if complete_dd and np.isfinite(approximate_dd) else None,
            total_bound if all_available else None,
            upper,
            ordered,
            costs,
        )

    def _high_inverse(self, context, rhs, model, *, adjoint=False):
        nz, nx = self._physics.shape
        key = context.model_sha256, context.frequency_hz
        if key in self._high_references:
            denominator = self._high_references[key]
            self._high_references.move_to_end(key)
            self._count("high_reference_cache_hits")
        else:
            middle_m = (float(np.min(model)) + float(np.max(model))) / 2
            denominator = (
                self._eigen_grid - context.omega**2 * middle_m + 1j * context.omega * self._middle_d
            )
            if np.min(denominator.real[self._high_mask]) < context.gamma - 1e-8 * max(
                1.0, context.gamma
            ):
                raise ValueError("high-complement reference violates the positive real bound")
            denominator[~self._high_mask] = 1.0
            denominator = _readonly(denominator)
            self._high_references[key] = denominator
            self._count("high_reference_assemblies")
            if len(self._high_references) > 8:
                self._high_references.popitem(last=False)
                self._count("high_reference_evictions")
        if adjoint:
            denominator = denominator.conjugate()
        columns = rhs.shape[1]
        transformed = fft.dstn(
            rhs.reshape(nz, nx, columns), type=1, axes=(0, 1), norm="ortho", workers=1
        )
        transformed /= denominator[:, :, None]
        transformed[~self._high_mask] = 0.0
        field = fft.idstn(transformed, type=1, axes=(0, 1), norm="ortho", workers=1).reshape(
            self._physics.n, columns
        )
        self._dispatch("high_dst_I_dispatches")
        self._dispatch("high_idst_I_dispatches")
        self._count("high_dst_I_real_transforms", 2 * columns)
        self._count("high_idst_I_real_transforms", 2 * columns)
        self._count("high_inverse_rhs_columns", columns)
        if not np.all(np.isfinite(field)):
            raise ValueError("nonfinite high-complement inverse")
        return field

    def _precondition(self, context, rhs, model, *, adjoint=False):
        ops = BatchedCoarseOperations(context, self._batch_dispatches)
        high = self._high_inverse(context, rhs, model, adjoint=adjoint)
        # Only the corrected field is needed. Avoid an unused second stencil.
        coarse_rhs = ops.project(rhs - ops.apply_operator(high, adjoint=adjoint))
        low = ops.solve(coarse_rhs, adjoint=adjoint, category="coarse_correction_rhs")
        self._count("preconditioner_rhs_columns", rhs.shape[1])
        self._dispatch("preconditioner_dispatches")
        return high + ops.lift(low)

    def _inject(self, residual, receivers):
        rhs = np.zeros((self._physics.n, residual.shape[1]), dtype=np.complex128)
        self._count("receiver_injections", residual.shape[1])
        self._dispatch("receiver_injection_dispatches")
        with np.errstate(over="ignore", invalid="ignore"):
            np.add.at(rhs, np.asarray(receivers), residual)
        if not np.all(np.isfinite(rhs)):
            raise FloatingPointError("nonfinite receiver injection")
        return rhs

    def _coarse_fields(self, model, blocks, receivers, contexts):
        groups = {
            f: tuple(i for i, b in enumerate(blocks) if b.frequency_hz == f)
            for f in sorted(contexts)
        }
        states = np.zeros((self._physics.n, len(blocks)), dtype=np.complex128)
        adjoints = np.zeros_like(states)
        failures, residuals = {}, {}
        for frequency, indices in groups.items():
            context = contexts[frequency]
            if not context.available:
                self._count("unavailable_zero_seed_blocks", len(indices))
                continue
            ops = BatchedCoarseOperations(context, self._batch_dispatches)
            q = np.zeros((self._physics.n, len(indices)), dtype=np.complex128)
            q[[blocks[i].source_index for i in indices], np.arange(len(indices))] = (
                1 / self._physics.spacing**2
            )
            self._dispatch("source_rhs_assembly_dispatches")
            observed = np.stack([blocks[i].observed for i in indices], axis=1)
            try:
                state = ops.lift(ops.solve(ops.project(q)))
                rhs = self._inject(state[list(receivers), :] - observed, receivers)
                adjoint = ops.lift(ops.solve(ops.project(rhs), adjoint=True))
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    for _ in range(self.repair_sweeps):
                        self._count("paired_repair_block_sweep_attempts", len(indices))
                        state += self._precondition(context, q - ops.apply_operator(state), model)
                        rhs = self._inject(state[list(receivers), :] - observed, receivers)
                        adjoint += self._precondition(
                            context,
                            rhs - ops.apply_operator(adjoint, adjoint=True),
                            model,
                            adjoint=True,
                        )
                        self._count("paired_repair_block_sweeps", len(indices))
                    rp = ops.apply_operator(state) - q
                    rhs = self._inject(state[list(receivers), :] - observed, receivers)
                    ra = ops.apply_operator(adjoint, adjoint=True) - rhs
                    if not np.all(np.isfinite(rp)) or not np.all(np.isfinite(ra)):
                        raise FloatingPointError("nonfinite final field residuals")
                    residuals[frequency] = rp, ra
            except (ValueError, FloatingPointError, la.LinAlgError) as error:
                failures[frequency] = f"{type(error).__name__}: {error}"
                self._count("failed_zero_seed_blocks", len(indices))
                continue
            states[:, indices], adjoints[:, indices] = state, adjoint
        return SimpleNamespace(
            group_indices=groups,
            block_states=states,
            block_adjoints=adjoints,
            failures=failures,
            residuals=residuals,
        )
