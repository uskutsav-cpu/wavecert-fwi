"""Exact survey merit/gradient receipts with paid forward and LU reuse.

Scoped to the frozen point-source Helmholtz2D reference operator. This is
ordinary floating-point linear algebra, not a roundoff enclosure or a new
gradient identity. A cached accepted merit is genuine paid online work.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Protocol

import numpy as np

from wavecert.physics.helmholtz import Helmholtz2D

EXACT_REFERENCE_SOURCE_SHA256 = "caf51b58da24ed6d3633bda4eaaab804714f5c7932d9d5f343644763bc158131"


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _json_sha(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _immutable(values, dtype):
    values = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(values.tobytes(), dtype=dtype).reshape(values.shape)


def _array_record(values, dtype=None):
    array = np.asarray(values)
    canonical = np.ascontiguousarray(array, dtype=dtype)
    return dict(shape=list(canonical.shape), dtype=str(canonical.dtype), sha256=_sha(canonical.tobytes()))


def _configuration(physics):
    if (type(physics) is not Helmholtz2D or len(physics.shape) != 2
            or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) or i < 1 for i in physics.shape)
            or isinstance(physics.damping_width, (bool, np.bool_))
            or not isinstance(physics.damping_width, (int, np.integer)) or physics.damping_width < 0
            or not np.isfinite(physics.spacing) or physics.spacing <= 0
            or not np.isfinite(physics.damping_strength) or physics.damping_strength < 0):
        raise ValueError("invalid or changed reference configuration")
    if (getattr(physics.operator, "__func__", None) is not Helmholtz2D.operator
            or getattr(physics.source, "__func__", None) is not Helmholtz2D.source):
        raise ValueError("reference operator/source method changed outside the frozen scope")
    return dict(shape=list(physics.shape), spacing=float(physics.spacing),
                damping_width=int(physics.damping_width), damping_strength=float(physics.damping_strength))


def _operator_record(physics):
    stiffness = physics.stiffness.tocsr()
    damping = np.asarray(physics.damping)
    if (stiffness.shape != (physics.n, physics.n) or np.iscomplexobj(stiffness.data)
            or not np.all(np.isfinite(stiffness.data)) or damping.shape != (physics.n,)
            or np.iscomplexobj(damping) or not np.all(np.isfinite(damping))):
        raise ValueError("reference operator arrays changed outside the finite real scope")
    return dict(configuration=_configuration(physics),
        stiffness_shape=list(stiffness.shape), stiffness_indptr=_array_record(stiffness.indptr, "<i8"),
        stiffness_indices=_array_record(stiffness.indices, "<i8"), stiffness_data=_array_record(stiffness.data, "<f8"),
        damping=_array_record(damping, "<f8"), source_sha256=EXACT_REFERENCE_SOURCE_SHA256)


def _verify_reference_source(reference_package_root):
    here = Path(__file__).resolve()
    packaged = here.parent.name == "physics" and here.parents[1].name == "wavecert"
    if packaged:
        intended_root = here.parents[1]
        if reference_package_root is not None and Path(reference_package_root).resolve() != intended_root:
            raise RuntimeError("a packaged exact oracle cannot bind a foreign reference package")
    elif reference_package_root is None:
        raise RuntimeError("unpackaged staging requires an explicit reference_package_root")
    else:
        intended_root = Path(reference_package_root).resolve()
    module = importlib.import_module("wavecert.physics.helmholtz")
    intended = intended_root / "physics" / "helmholtz.py"
    actual = Path(module.__file__).resolve()
    if (actual != intended or module.Helmholtz2D is not Helmholtz2D
            or Path(Helmholtz2D.operator.__code__.co_filename).resolve() != intended):
        raise RuntimeError("exact oracle rejects a foreign cached reference implementation")
    if _sha(actual.read_bytes()) != EXACT_REFERENCE_SOURCE_SHA256:
        raise RuntimeError("exact oracle requires the frozen Helmholtz reference source")
    return intended_root


def _index(value, n, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not 0 <= value < n):
        raise ValueError(f"{name} must be an integer grid position")
    return int(value)


def _survey_record(blocks, receivers, n):
    receivers = tuple(_index(i, n, "receiver") for i in receivers)
    if not receivers or not blocks:
        raise ValueError("the survey needs nonempty blocks and receivers")
    records, seen = [], set()
    for block in blocks:
        source = _index(block.source_index, n, "source")
        frequency = float(block.frequency_hz)
        omega = 2*np.pi*frequency
        if (isinstance(block.frequency_hz, (bool, np.bool_)) or not np.isfinite(frequency)
                or frequency <= 0 or not np.isfinite(omega) or omega > np.sqrt(np.finfo(float).max)):
            raise ValueError("frequency must be finite, positive and numerically representable")
        data = np.asarray(block.observed)
        if data.shape != (len(receivers),) or not np.all(np.isfinite(data)):
            raise ValueError("observations must be finite and match the ordered receiver slots")
        key = source, frequency
        if key in seen:
            raise ValueError("duplicate source/frequency blocks are not supported")
        seen.add(key)
        records.append(dict(source=source, frequency_hz=frequency, label=str(block.label),
                            original_data=_array_record(data), canonical_data=_array_record(data, "<c16")))
    geometry = dict(ordered_receivers=list(receivers), ordered_blocks=[
        {k: r[k] for k in ("source", "frequency_hz", "label")} for r in records])
    return dict(geometry=geometry, observed_data=records)


class LinearSolveContext(Protocol):
    """One immutable model/frequency context; iterative adapters must fail closed.

    A future tolerance-converged GMRES adapter can implement this interface
    without changing merit/gradient algebra. Its setup/iteration cost receipts
    and operator-identity validation must be wired in before enabling it.
    """

    def solve(self, rhs: np.ndarray, *, adjoint: bool = False) -> np.ndarray: ...


@dataclass(frozen=True)
class _SparseLUContext:
    factor: object = dc_field(repr=False)

    def solve(self, rhs, *, adjoint=False):
        return self.factor.solve(rhs, trans="H" if adjoint else "N")


@dataclass(frozen=True)
class _FrequencyReceipt:
    frequency: float
    sources: tuple[int, ...]
    factor: LinearSolveContext = dc_field(repr=False)
    route: str
    field: np.ndarray = dc_field(repr=False)
    residual: np.ndarray = dc_field(repr=False)
    _symmetric: bool | None = dc_field(default=None, repr=False, compare=False)


@dataclass(frozen=True, eq=False)
class CachedFrequencySnapshot:
    """Readonly paid fields, with receiver-slot order and multiplicity intact.

    ``receiver_map`` means C=P_unique A^-1, not its adjoint. Only after the
    measured transpose-symmetry check may a consumer use Z=C.T. PDE residuals
    of the ordinary-floating-point fields have NOT been computed by export.
    """

    frequency_hz: float
    source_indices: tuple[int, ...]
    block_indices: tuple[int, ...]
    receiver_indices: tuple[int, ...]
    unique_receiver_indices: tuple[int, ...]
    receiver_expand_indices: np.ndarray = dc_field(repr=False)
    observed_slots: np.ndarray = dc_field(repr=False)
    cache_kind: str
    cached_field: np.ndarray = dc_field(repr=False)
    residual_slots: np.ndarray = dc_field(repr=False)
    residual_unique: np.ndarray = dc_field(repr=False)
    source_amplitude: float
    transpose_symmetric: bool

    def to_dict(self):
        return dict(frequency_hz=self.frequency_hz, source_indices=list(self.source_indices),
            block_indices=list(self.block_indices), receiver_indices=list(self.receiver_indices),
            unique_receiver_indices=list(self.unique_receiver_indices), cache_kind=self.cache_kind,
            source_amplitude=self.source_amplitude, transpose_symmetric=self.transpose_symmetric,
            **{name: _array_record(getattr(self, name)) for name in (
                "receiver_expand_indices", "observed_slots", "cached_field", "residual_slots", "residual_unique")})


@dataclass(frozen=True, eq=False)
class CachedMeritSnapshot:
    """An issued, immutable snapshot; validation requires its own policy oracle.

    This carries no LU handle and does not keep an expired receipt alive. It
    remains valid only while its issued receipt is alive and its oracle's
    reference configuration, ordered survey and data remain unchanged.
    """

    receipt_key: str
    model_sha256: str
    operator_sha256: str
    geometry_sha256: str
    data_sha256: str
    model: np.ndarray = dc_field(repr=False)
    shape: tuple[int, int]
    spacing: float
    damping_width: int
    damping_strength: float
    damping: np.ndarray = dc_field(repr=False)
    frequencies: tuple[CachedFrequencySnapshot, ...]
    accuracy_scope: str
    _issuer: object = dc_field(repr=False, compare=False)

    @property
    def physics_config(self):
        return dict(shape=list(self.shape), spacing=self.spacing, damping_width=self.damping_width,
                    damping_strength=self.damping_strength)

    def validate(self, expected_oracle):
        """Require an explicit expected owner; never infer one from an export."""
        expected_oracle.validate_merit_fields(self)

    def to_dict(self):
        return dict(receipt_key=self.receipt_key, model_sha256=self.model_sha256,
            operator_sha256=self.operator_sha256, geometry_sha256=self.geometry_sha256,
            data_sha256=self.data_sha256, model=_array_record(self.model),
            physics_config=self.physics_config, damping=_array_record(self.damping),
            accuracy_scope=self.accuracy_scope, frequencies=[frame.to_dict() for frame in self.frequencies])


@dataclass(frozen=True, eq=False)
class ExactMeritReceipt:
    objective: float
    key: str
    model_sha256: str
    operator_sha256: str
    geometry_sha256: str
    data_sha256: str
    model: np.ndarray = dc_field(repr=False)
    _records: tuple[_FrequencyReceipt, ...] = dc_field(repr=False)
    _owner: object = dc_field(repr=False, compare=False)
    _gradient: np.ndarray | None = dc_field(default=None, repr=False, compare=False)
    _gradient_routes: dict | None = dc_field(default=None, repr=False, compare=False)

    def to_dict(self):
        return dict(objective=self.objective, key=self.key, model_sha256=self.model_sha256,
            operator_sha256=self.operator_sha256, geometry_sha256=self.geometry_sha256,
            data_sha256=self.data_sha256, gradient_cached=self._gradient is not None,
            frequency_merit_routes={str(r.frequency): r.route for r in self._records},
            frequency_field_shapes={str(r.frequency): list(r.field.shape) for r in self._records})


@dataclass(frozen=True)
class ExactGradientEvaluation:
    objective: float
    gradient: np.ndarray
    receipt: ExactMeritReceipt
    additional_rhs: int
    frequency_strategies: dict


class ExactMeritOracle:
    """Own one cold solver and at most current + latest-trial receipt internally.

    Each merit uses min(s,r) RHS at a frequency, where r counts unique receiver
    positions. Cached forward fields need min(s,r-c) further RHS for a full
    gradient, while cached receiver maps need min(r,s-c); c is collocation count.
    These reciprocity routes are used only after actual A.T==A verification.
    Non-symmetric alternatives omit c reuse. All repeated receiver slots remain
    in the objective and their residuals are summed for adjoint injection.

    The owned Helmholtz LU cache is disabled. Receipts pin their actually paid
    factors, so rejected trial factorizations cannot evict the accepted point's
    factors. A caller retaining additional receipts also retains that memory.
    No work/receipt sharing is permitted between independent policy oracles.
    """

    def __init__(self, physics, blocks, receiver_indices, *, reference_package_root=None):
        self._reference_root = _verify_reference_source(reference_package_root)
        if type(physics) is not Helmholtz2D:
            raise TypeError("the exact merit oracle supports the frozen Helmholtz2D class only")
        self._template, self._config = physics, _configuration(physics)
        self._p = Helmholtz2D(**{**self._config, "shape": tuple(physics.shape)}, factorization_cache_size=0)
        physics._ensure_cache_signature()
        self._p._ensure_cache_signature()
        self._operator = _operator_record(physics)
        if self._operator != _operator_record(self._p):
            raise ValueError("a customized/mutated operator is outside the reference oracle's scope")
        self._operator_base_sha = _json_sha(self._operator)
        self._block_reference = blocks if isinstance(blocks, Sequence) else None
        self._receiver_reference = receiver_indices if isinstance(receiver_indices, (Sequence, np.ndarray)) else None
        self._original_blocks, self._original_receivers = tuple(blocks), tuple(receiver_indices)
        self._survey = _survey_record(self._original_blocks, self._original_receivers, self._p.n)
        self._survey_sha = _json_sha(self._survey)
        self._geometry_sha256 = _json_sha(self._survey["geometry"])
        self._data_sha256 = _json_sha(self._survey["observed_data"])
        self._receivers = tuple(int(i) for i in self._original_receivers)
        self._unique = tuple(dict.fromkeys(self._receivers))
        self._expand = np.asarray([self._unique.index(i) for i in self._receivers], dtype=int)
        self._groups = {}
        for i, block in enumerate(self._original_blocks):
            self._groups.setdefault(float(block.frequency_hz), []).append(i)
        self._data = tuple(_immutable(np.asarray(block.observed), "<c16") for block in self._original_blocks)
        self._token, self._current, self._trial, self._cost = object(), None, None, {}
        self._issued_receipts = weakref.WeakSet()
        self._issued_exports = weakref.WeakKeyDictionary()

    @property
    def n(self):
        return self._p.n

    @property
    def geometry_sha256(self):
        return self._geometry_sha256

    @property
    def data_sha256(self):
        return self._data_sha256

    @property
    def contract(self):
        return dict(operator_base_sha256=self._operator_base_sha,
            operator_base_record=json.loads(json.dumps(self._operator)), geometry_sha256=self.geometry_sha256,
            data_sha256=self.data_sha256, survey_sha256=self._survey_sha,
            reference_source_sha256=EXACT_REFERENCE_SOURCE_SHA256,
            reference_source_path=str(self._reference_root / "physics" / "helmholtz.py"),
            operator_hash_encoding="frozen source/config/stiffness-CSR/damping plus canonical model and frequencies",
            model_hash_encoding="flat C-order little-endian float64 squared slowness",
            cache_scope="owned current and latest trial fields/LU factors; no cross-policy sharing")

    def _count(self, key, amount=1):
        self._cost[key] = self._cost.get(key, 0)+int(amount)

    def cost_snapshot(self):
        return dict(physics=self._p.cost_snapshot(), oracle=dict(self._cost))

    def _assert_contract(self):
        self._count("contract_checks")
        _verify_reference_source(self._reference_root)
        if _operator_record(self._template) != self._operator or _operator_record(self._p) != self._operator:
            raise ValueError("physical operator/configuration changed after oracle construction")
        blocks = tuple(self._block_reference) if self._block_reference is not None else self._original_blocks
        receivers = tuple(self._receiver_reference) if self._receiver_reference is not None else self._original_receivers
        if _json_sha(_survey_record(blocks, receivers, self.n)) != self._survey_sha:
            raise ValueError("ordered geometry or observed data changed after oracle construction")

    def _model(self, m):
        raw = np.asarray(m)
        if raw.shape != (self.n,) or np.iscomplexobj(raw):
            raise ValueError("model must be a finite positive real flat vector")
        return _immutable(self._p.validate_model(raw), "<f8")

    def _key(self, model):
        model_sha = _sha(model.tobytes())
        operator_sha = _json_sha(dict(base=self._operator_base_sha, model=model_sha,
                                     frequencies=sorted(self._groups)))
        key = _json_sha(dict(operator=operator_sha, geometry=self.geometry_sha256, data=self.data_sha256))
        return key, model_sha, operator_sha

    def _solve(self, model, frequency, rhs, *, category, adjoint=False, factor=None):
        rhs = np.asarray(rhs, dtype=np.complex128)
        if rhs.ndim != 2 or rhs.shape[0] != self.n or rhs.shape[1] < 1 or not np.all(np.isfinite(rhs)):
            raise ValueError("matrix RHS must be finite and have a positive column count")
        if factor is None:
            self._count("factorization_attempts")
            self._count("operator_assemblies")
            try:
                factor = _SparseLUContext(self._p._factorization(model, frequency))
            except Exception:
                self._count("failed_factorizations")
                raise
        else:
            self._count("pinned_factor_reuses")
        self._count("linear_solve_dispatches")
        self._count("adjoint_solve_dispatches" if adjoint else "forward_solve_dispatches")
        self._p._cost[category] = self._p._cost.get(category, 0)+rhs.shape[1]
        result = factor.solve(rhs, adjoint=adjoint)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("exact reference solve returned a nonfinite field")
        return result, factor

    def evaluate(self, m):
        """Pay for fine merit, or reuse exactly this oracle's immutable receipt."""
        self._assert_contract()
        model = self._model(m)
        key, model_sha, operator_sha = self._key(model)
        for receipt in (self._current, self._trial):
            if receipt is not None and receipt.key == key:
                self._count("objective_cache_hits")
                return receipt
        self._count("objective_evaluations")
        records, objective = [], 0.0
        amplitude = 1/self._p.spacing**2
        for frequency, indices in sorted(self._groups.items()):
            sources = tuple(int(self._original_blocks[i].source_index) for i in indices)
            observed = np.stack([self._data[i] for i in indices], axis=1)
            if len(sources) <= len(self._unique):
                rhs = np.zeros((self.n, len(sources)), dtype=np.complex128)
                rhs[np.asarray(sources), np.arange(len(sources))] = amplitude
                state, factor = self._solve(model, frequency, rhs, category="forward_rhs")
                residual, route, values = state[list(self._receivers)]-observed, "forward_states", state
            else:
                rhs = np.zeros((self.n, len(self._unique)), dtype=np.complex128)
                rhs[np.asarray(self._unique), np.arange(len(self._unique))] = 1.
                response, factor = self._solve(model, frequency, rhs, category="adjoint_rhs", adjoint=True)
                matrix = response.conjugate().T
                residual = (amplitude*matrix[:, list(sources)])[self._expand]-observed
                route, values = "receiver_map", matrix
            self._count("objective_contractions")
            value = .5*float(np.vdot(residual, residual).real)
            objective += value
            records.append(_FrequencyReceipt(frequency, sources, factor, route,
                _immutable(values, "<c16"), _immutable(residual, "<c16")))
        if not np.isfinite(objective):
            raise FloatingPointError("exact objective overflowed")
        receipt = ExactMeritReceipt(float(objective), key, model_sha, operator_sha, self.geometry_sha256,
            self.data_sha256, model, tuple(records), self._token)
        self._issued_receipts.add(receipt)
        self._trial = receipt
        return receipt

    def _check_receipt(self, receipt):
        self._assert_contract()
        if (not isinstance(receipt, ExactMeritReceipt) or receipt._owner is not self._token
                or receipt not in self._issued_receipts):
            raise ValueError("foreign exact-merit receipt; online work cannot be shared between policies")
        key, model_sha, operator_sha = self._key(self._model(receipt.model))
        if (key != receipt.key or model_sha != receipt.model_sha256 or operator_sha != receipt.operator_sha256
                or receipt.geometry_sha256 != self.geometry_sha256 or receipt.data_sha256 != self.data_sha256):
            raise ValueError("exact-merit receipt fingerprint mismatch")

    def _symmetry(self, receipt, record):
        if record._symmetric is None:
            operator = self._p.operator(receipt.model, record.frequency)
            self._count("operator_assemblies")
            self._count("transpose_symmetry_checks")
            difference = operator-operator.T
            object.__setattr__(record, "_symmetric", not difference.nnz or not np.any(difference.data != 0))
        else:
            self._count("transpose_symmetry_cache_hits")
        return bool(record._symmetric)

    def _export_digest(self, snapshot):
        # Hash the actual exposed bytes on every validation, not stored hashes.
        arrays = [snapshot.model, snapshot.damping]
        for frame in snapshot.frequencies:
            arrays.extend(getattr(frame, name) for name in ("receiver_expand_indices", "observed_slots",
                "cached_field", "residual_slots", "residual_unique"))
        self._count("export_fingerprint_bytes", sum(np.asarray(array).nbytes for array in arrays))
        self._count("export_fingerprint_evaluations")
        return _json_sha(snapshot.to_dict())

    def export_merit_fields(self, receipt):
        """Expose only this oracle's paid fields, without solving another RHS.

        Data-misfit residuals are exported, including SUM aggregation of repeated
        receiver slots. Consumers must measure PDE residuals if their bound
        needs them; sparse LU fields are ordinary FP, not exact enclosures.
        Actual transpose-symmetry checks and copied/hashed array bytes are paid
        and counted. No factorization or cached-gradient handle is exported.
        """
        self._check_receipt(receipt)
        frames = []
        copied_bytes = 0
        for record in receipt._records:
            indices = tuple(self._groups[record.frequency])
            observed = _immutable(np.stack([self._data[i] for i in indices], axis=1), "<c16")
            residual = np.zeros((len(self._unique), len(record.sources)), dtype=np.complex128)
            np.add.at(residual, self._expand, record.residual)
            residual = _immutable(residual, "<c16")
            expand = _immutable(self._expand, "<i8")
            copied_bytes += observed.nbytes+residual.nbytes+expand.nbytes
            frames.append(CachedFrequencySnapshot(record.frequency, record.sources, indices,
                self._receivers, self._unique, expand, observed, record.route, record.field,
                record.residual, residual, 1/self._p.spacing**2, self._symmetry(receipt, record)))
        damping = _immutable(self._p.damping, "<f8")
        copied_bytes += damping.nbytes
        snapshot = CachedMeritSnapshot(receipt.key, receipt.model_sha256, receipt.operator_sha256,
            receipt.geometry_sha256, receipt.data_sha256, receipt.model, tuple(self._p.shape),
            self._p.spacing, self._p.damping_width, self._p.damping_strength, damping, tuple(frames),
            "native sparse LU; ordinary floating point; PDE residuals not analytically zero; not a verified roundoff enclosure",
            self)
        digest = self._export_digest(snapshot)
        self._issued_exports[snapshot] = (weakref.ref(receipt), digest)
        self._count("merit_field_exports")
        self._count("export_copied_array_bytes", copied_bytes)
        return snapshot

    def validate_merit_fields(self, snapshot):
        """Fail closed on foreign, fabricated, expired or altered exported fields."""
        self._count("merit_field_validation_attempts")
        if (type(snapshot) is not CachedMeritSnapshot or snapshot._issuer is not self
                or snapshot not in self._issued_exports):
            raise ValueError("foreign or unissued cached-merit snapshot")
        receipt_ref, digest = self._issued_exports[snapshot]
        receipt = receipt_ref()
        if receipt is None:
            raise ValueError("expired cached-merit receipt; fields cannot borrow old online work")
        self._check_receipt(receipt)
        if self._export_digest(snapshot) != digest:
            raise ValueError("cached-merit snapshot fingerprint mismatch")
        self._count("merit_field_validations")

    def accept(self, receipt):
        """Pin already paid accepted fields/factors; this performs no PDE solve."""
        self._check_receipt(receipt)
        self._current, self._trial = receipt, None
        self._count("accepted_receipts")

    def gradient(self, receipt):
        """Compute the strongest implemented incremental full-gradient route."""
        self._check_receipt(receipt)
        if receipt._gradient is not None:
            self._count("gradient_cache_hits")
            return ExactGradientEvaluation(receipt.objective, receipt._gradient, receipt, 0,
                                           dict(receipt._gradient_routes))
        self._count("gradient_evaluations")
        gradient, strategies, rhs_count = np.zeros(self.n), {}, 0
        amplitude = 1/self._p.spacing**2
        for record in receipt._records:
            frequency, sources = record.frequency, record.sources
            residual = np.zeros((len(self._unique), len(sources)), dtype=np.complex128)
            np.add.at(residual, self._expand, record.residual)
            symmetric = self._symmetry(receipt, record)
            source_positions = {source: i for i, source in enumerate(sources)}
            receiver_positions = {receiver: i for i, receiver in enumerate(self._unique)}
            common = len(set(sources) & set(self._unique)) if symmetric else 0
            if record.route == "forward_states":
                state = record.field
                if len(sources) <= len(self._unique)-common:
                    rhs = np.zeros_like(state)
                    np.add.at(rhs, np.asarray(self._receivers), record.residual)
                    adjoint, _ = self._solve(receipt.model, frequency, rhs, category="adjoint_rhs", adjoint=True, factor=record.factor)
                    route, work = "forward_cache/source_adjoints", len(sources)
                else:
                    matrix = np.empty((len(self._unique), self.n), dtype=np.complex128)
                    missing = []
                    for i, receiver in enumerate(self._unique):
                        if symmetric and receiver in source_positions:
                            matrix[i] = state[:, source_positions[receiver]]/amplitude
                        else:
                            missing.append(i)
                    if missing:
                        rhs = np.zeros((self.n, len(missing)), dtype=np.complex128)
                        rhs[np.asarray(self._unique)[missing], np.arange(len(missing))] = 1.
                        response, _ = self._solve(receipt.model, frequency, rhs, category="adjoint_rhs", adjoint=True, factor=record.factor)
                        matrix[missing] = response.conjugate().T
                    adjoint = matrix.conjugate().T @ residual
                    route, work = "forward_cache/receiver_reuse", len(missing)
                contribution = np.real(np.sum(np.conjugate(adjoint)*state, axis=1))
            else:
                matrix = record.field
                missing = [i for i, source in enumerate(sources) if not symmetric or source not in receiver_positions]
                if len(missing) <= len(self._unique):
                    state = np.empty((self.n, len(sources)), dtype=np.complex128)
                    for i, source in enumerate(sources):
                        if symmetric and source in receiver_positions:
                            state[:, i] = amplitude*matrix[receiver_positions[source]]
                    if missing:
                        rhs = np.zeros((self.n, len(missing)), dtype=np.complex128)
                        rhs[np.asarray(sources)[missing], np.arange(len(missing))] = amplitude
                        response, _ = self._solve(receipt.model, frequency, rhs, category="forward_rhs", factor=record.factor)
                        state[:, missing] = response
                    adjoint = matrix.conjugate().T @ residual
                    contribution = np.real(np.sum(np.conjugate(adjoint)*state, axis=1))
                    route, work = "receiver_cache/source_states", len(missing)
                else:
                    rhs = np.zeros((self.n, len(self._unique)), dtype=np.complex128)
                    np.add.at(rhs, np.asarray(sources), amplitude*residual.conjugate().T)
                    coded, _ = self._solve(receipt.model, frequency, rhs, category="forward_rhs", factor=record.factor)
                    contribution = np.real(np.sum(matrix.T*coded, axis=1))
                    route, work = "receiver_cache/residual_coded", len(self._unique)
            gradient += (2*np.pi*frequency)**2*contribution
            strategies[str(frequency)], rhs_count = route, rhs_count+work
            self._count("gradient_contractions")
        if not np.all(np.isfinite(gradient)):
            raise FloatingPointError("exact gradient overflowed")
        object.__setattr__(receipt, "_gradient", _immutable(gradient, "<f8"))
        object.__setattr__(receipt, "_gradient_routes", dict(strategies))
        return ExactGradientEvaluation(receipt.objective, receipt._gradient, receipt, rhs_count, strategies)
