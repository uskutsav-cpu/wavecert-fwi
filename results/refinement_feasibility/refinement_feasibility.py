"""Fixed development-only refinement/repair feasibility; no timing inference.

Interpolates existing coarse validation velocities and the frozen train-fit POD
space. This is not native high-resolution geology or independent confirmation.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.fft as fft
import scipy.linalg as la
import scipy.ndimage as ndi
from scipy.interpolate import RegularGridInterpolator

from wavecert.physics.coarse_batch import BatchedCoarseOperations
from wavecert.physics.coarse_stability import FastCoarseStability
from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock
from wavecert.provenance import environment_record, sha256_file, source_digest
from wavecert.surrogates.pod import PODGalerkinSurrogate
from wavecert.surrogates.pod_batch import batched_pod_survey_fields

ROOT = Path(__file__).resolve().parent / "wavecert-fwi"
WORK = Path(__file__).resolve().parent
PROTOCOL = WORK / "refinement_feasibility_protocol.json"
OUTPUT = WORK / "refinement_feasibility"
GRIDS = (24, 49, 99)
SOURCES = (50, 54, 58, 61, 65, 69)
RECEIVERS = tuple(range(50, 70))
FREQUENCIES = (2.5, 3.0, 3.5, 4.0)
METHODS = (
    ("raw_pod", None, 0, "pod"),
    ("pod_coarse96_k0", 96, 0, "pod"),
    ("pod_coarse96_k2", 96, 2, "pod"),
    ("pod_coarse64_k2", 64, 2, "pod"),
    ("pure_coarse96_k4", 96, 4, "pure"),
    ("pure_coarse64_k4", 64, 4, "pure"),
)
INPUTS = (
    "data/reference/subsurfacegen_parent_split/validation.npz",
    "data/reference/subsurfacegen_parent_split/validation_metadata.csv",
    "data/reference/subsurfacegen_parent_split/split_manifest.csv",
    "results/pod_development/bases/state_tangent_rank64.npz",
)


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def transfer_velocity(values, n):
    if n == 24:
        return np.array(values, dtype=float, copy=True)
    coordinates = np.arange(26, dtype=float)
    points = np.arange(1, n + 1) / (n + 1) * 25
    zz, xx = np.meshgrid(points, points, indexing="ij")
    interpolator = RegularGridInterpolator(
        (coordinates, coordinates), np.pad(values, 1, mode="edge")
    )
    return interpolator(np.stack((zz, xx), axis=-1))


def transfer_basis(basis, n):
    if n == 24:
        return basis.copy()
    coordinates = np.arange(26, dtype=float)
    points = np.arange(1, n + 1) / (n + 1) * 25
    zz, xx = np.meshgrid(points, points, indexing="ij")
    values = np.pad(basis.reshape(24, 24, 64), ((1, 1), (1, 1), (0, 0)))
    lifted = RegularGridInterpolator((coordinates, coordinates), values)(
        np.stack((zz, xx), axis=-1)
    ).reshape(n * n, 64)
    q, _ = la.qr(lifted, mode="economic")
    assert la.norm(q.T @ q - np.eye(64), 2) < 1e-10
    return q


def map_index(index, n):
    factor = (n + 1) // 25
    assert (n + 1) % 25 == 0
    z, x = divmod(index, 24)
    return (factor * (z + 1) - 1) * n + factor * (x + 1) - 1


def load_inputs():
    with (ROOT / INPUTS[1]).open(newline="") as stream:
        metadata = list(csv.DictReader(stream))
    identities = []
    for index, row in enumerate(metadata):
        if not any(row["slice_id"] == item["slice_id"] for item in identities):
            identities.append(
                dict(array_index=index, parent_id=row["model_id"], slice_id=row["slice_id"])
            )
    assert len(identities) == 8
    return identities, {name: sha256_file(ROOT / name) for name in INPUTS}


def prepare():
    identities, hashes = load_inputs()
    for n in GRIDS:
        assert all(
            np.isclose((map_index(i, n) // n + 1) * (1.25 / (n + 1)), (i // 24 + 1) * 0.05)
            and np.isclose((map_index(i, n) % n + 1) * (1.25 / (n + 1)), (i % 24 + 1) * 0.05)
            for i in SOURCES + RECEIVERS
        )
    protocol = dict(
        created_before_outcomes_utc=datetime.now(timezone.utc).isoformat(),
        grids=list(GRIDS),
        methods=[dict(label=a, rank=b, sweeps=c, initialization=d) for a, b, c, d in METHODS],
        identities=identities,
        inputs_sha256=hashes,
        script_sha256=sha256_file(Path(__file__)),
        provenance=environment_record(ROOT),
        scope="PRIVATE existing-validation discovery; interpolated coarse velocity/basis, not native high-resolution geology/new independent parents/confirmation. No timing or acceleration inference.",
        geometry=dict(
            interior_domain_extent_km=1.25,
            sources=list(SOURCES),
            receivers=list(RECEIVERS),
            frequencies=list(FREQUENCIES),
            mapping="Preserve exact physical source/receiver coordinates using integer refinement factors1/2/4.",
            damping="Same Helmholtz implementation, damping_width=3*factor, nominal sponge thickness. Grid sampling of its boundary profile shifts with h; not an exact pointwise-damping refinement theorem.",
        ),
        initialization="Interpolate raw velocity with edge extension, smooth each grid at physical sigma.08km using reflect, then squared-slowness inverse. Interpolate POD columns with zero Dirichlet boundary and orthonormalize; no new snapshots/learning.",
        repair="Classical fixed high-complement DST reference inverse, low Galerkin correction, coupled updated-state adjoint residual; midpoints of current m/damping. Unchanged oriented state/adjoint three-term DD inequality.",
        expected=dict(aggregate_rows=144, block_rows=3456),
        statistical_scope="All8 reused validation slices. Existing model IDs are not independent ancestors (SEAM common ancestor known); no IID/site/untouched inference.",
        counters="Untimed reused contexts/fields include proof diagnostics; not independent cold benchmark costs. All fine references separated offline.",
        retention="All6 methods and grids, failed/unavailable cases retained; no adaptive best checkpoint or dropped case.",
    )
    write_new(PROTOCOL, protocol)
    print(json.dumps(dict(status="pinned before any new outcomes", sha256=sha256_file(PROTOCOL))))


class Repair:
    def __init__(self, context, model, damping, dispatches):
        self.context, self.ops = context, BatchedCoarseOperations(context, dispatches)
        n, _ = context.physics_config["shape"]
        h = context.physics_config["spacing"]
        z, x = np.arange(1, n + 1)[:, None], np.arange(1, n + 1)[None, :]
        eigen = (
            4
            / h**2
            * (np.sin(np.pi * z / (2 * (n + 1))) ** 2 + np.sin(np.pi * x / (2 * (n + 1))) ** 2)
        )
        self.high = np.ones((n, n), dtype=bool)
        modes = context._engine.mode_indices
        self.high[modes[:, 0] - 1, modes[:, 1] - 1] = False
        self.denominator = (
            eigen
            - context.omega**2 * (np.min(model) + np.max(model)) / 2
            + 1j * context.omega * (np.min(damping) + np.max(damping)) / 2
        )
        assert np.min(self.denominator.real[self.high]) >= context.gamma - 1e-8
        self.denominator[~self.high] = 1
        self.n, self.dispatches = n, dispatches

    def apply(self, rhs, adjoint=False):
        columns = rhs.shape[1]
        coefficients = fft.dstn(
            rhs.reshape(self.n, self.n, columns), type=1, axes=(0, 1), norm="ortho", workers=1
        )
        denominator = self.denominator.conjugate() if adjoint else self.denominator
        coefficients /= denominator[:, :, None]
        coefficients[~self.high] = 0
        field = fft.idstn(coefficients, type=1, axes=(0, 1), norm="ortho", workers=1).reshape(
            self.n * self.n, columns
        )
        self.dispatches["high_DST_dispatches"] = self.dispatches.get("high_DST_dispatches", 0) + 2
        self.dispatches["high_DST_real_equivalent_transforms"] = (
            self.dispatches.get("high_DST_real_equivalent_transforms", 0) + 4 * columns
        )
        return self.ops.correct(field, rhs, adjoint=adjoint)[0]


def inject(residual, receivers, n):
    output = np.zeros((n, residual.shape[1]), dtype=np.complex128)
    np.add.at(output, np.asarray(receivers), residual)
    return output


def assess(fields, contexts, blocks, receivers, physics, exact, identity):
    gradient = np.zeros(physics.n)
    block_gradients = []
    for i, block in enumerate(blocks):
        g = (2 * np.pi * block.frequency_hz) ** 2 * np.real(
            fields[1][:, i].conjugate() * fields[0][:, i]
        )
        block_gradients.append(g)
        gradient += g
    norm = float(la.norm(gradient))
    assert np.isfinite(norm) and np.all(np.isfinite(gradient))
    direction = -gradient / norm if norm > 1e-14 else np.zeros_like(gradient)
    exact_gradient = np.sum([p[1] for p in exact], axis=0)
    exact_norm = float(la.norm(exact_gradient))
    rows, total_bound, available = [], 0.0, contexts is not None
    for i, block in enumerate(blocks):
        center = float(block_gradients[i] @ direction)
        truth_dd = float(exact[i][1] @ direction)
        row = dict(
            **identity,
            block_index=i,
            source=block.source_index,
            frequency=block.frequency_hz,
            center=center,
            exact_dd=truth_dd,
            error=abs(truth_dd - center),
            bound=None,
            available=False,
            covered=None,
            state_relative_l2=float(la.norm(fields[0][:, i] - exact[i][2]) / la.norm(exact[i][2])),
        )
        context = None if contexts is None else contexts[block.frequency_hz]
        if context is not None and context.available:
            q = physics.source(block.source_index)
            u, adjoint = fields[0][:, i], fields[1][:, i]
            residual = u[list(receivers)] - block.observed
            rp, ra = (
                context.apply_operator(u) - q,
                context.apply_operator(adjoint, adjoint=True)
                - physics.inject_receivers(residual, receivers),
            )
            eta_u = context.oriented_error_bounds(rp)
            constants = context.receiver_constants(receivers)
            eta_p = float(np.array([constants["pB"], constants["pQ"]]) @ eta_u)
            eta_a = context.oriented_adjoint_error_bounds(ra, eta_p, receivers)
            terms = context.omega**2 * np.array(
                [
                    context.component_norms(direction * u) @ eta_a,
                    context.component_norms(direction * adjoint) @ eta_u,
                    la.norm(direction, np.inf) * la.norm(eta_u) * la.norm(eta_a),
                ]
            )
            bound = float(np.sum(terms))
            actual_u, actual_a = (
                context.component_norms(exact[i][2] - u),
                context.component_norms(exact[i][3] - adjoint),
            )
            row.update(
                bound=bound,
                available=True,
                covered=bool(row["error"] <= bound + 1e-9 * max(1.0, bound)),
                state_components_covered=bool(
                    np.all(actual_u <= eta_u + 1e-9 * np.maximum(1.0, eta_u))
                ),
                adjoint_components_covered=bool(
                    np.all(actual_a <= eta_a + 1e-9 * np.maximum(1.0, eta_a))
                ),
                terms=terms.tolist(),
                eta_u=eta_u.tolist(),
                eta_a=eta_a.tolist(),
            )
            total_bound += bound
        else:
            available = False
        rows.append(row)
    center, exact_dd = float(gradient @ direction), float(exact_gradient @ direction)
    upper = center + total_bound if available else None
    certified = bool(available and upper < 0)
    record = dict(
        **identity,
        complete=True,
        available=available,
        certified=certified,
        true_descent=exact_dd < 0,
        false_certification=bool(certified and exact_dd >= 0),
        bound=total_bound if available else None,
        upper=upper,
        center=center,
        exact_dd=exact_dd,
        directional_error=abs(exact_dd - center),
        aggregate_covered=None
        if not available
        else abs(exact_dd - center) <= total_bound + 1e-9 * max(1.0, total_bound),
        all_block_bounds_cover=None
        if not available
        else all(
            row["covered"] and row["state_components_covered"] and row["adjoint_components_covered"]
            for row in rows
        ),
        gradient_cosine=float(
            np.clip(exact_gradient @ gradient / max(norm * exact_norm, 1e-300), -1.0, 1.0)
        ),
        gradient_relative_l2=float(la.norm(gradient - exact_gradient) / max(exact_norm, 1e-300)),
        bound_margin_ratio=None if not available else total_bound / max(-center, 1e-300),
    )
    return record, rows, gradient, exact_gradient


def execute():
    assert not OUTPUT.exists()
    protocol = json.loads(PROTOCOL.read_text())
    identities, hashes = load_inputs()
    assert identities == protocol["identities"] and hashes == protocol["inputs_sha256"]
    assert source_digest(ROOT) == protocol["provenance"]["source_sha256"]
    assert sha256_file(Path(__file__)) == protocol["script_sha256"]
    OUTPUT.mkdir()
    with np.load(ROOT / INPUTS[0], allow_pickle=False) as source:
        raw_models = source["models"].copy()
    with np.load(ROOT / INPUTS[3], allow_pickle=False) as archive:
        original_basis = archive["basis"].copy()
    records, block_rows, receipts = [], [], []
    for n in GRIDS:
        h, factor = 1.25 / (n + 1), (n + 1) // 25
        basis = transfer_basis(original_basis, n)
        vectors, vector_cases, vector_labels, exact_vectors = [], [], [], []
        for case_index, selection in enumerate(identities):
            raw_v = 1 / np.sqrt(raw_models[selection["array_index"]].astype(float))
            velocity = transfer_velocity(raw_v, n)
            model = (
                1 / ndi.gaussian_filter(velocity, sigma=0.08 / h, mode="reflect") ** 2
            ).reshape(-1)
            config = dict(shape=(n, n), spacing=h, damping_width=3 * factor, damping_strength=2.0)
            physics, truth = Helmholtz2D(**config), Helmholtz2D(**config)
            receivers = tuple(map_index(i, n) for i in RECEIVERS)
            blocks, exact = [], []
            for source in SOURCES:
                q = physics.source(map_index(source, n))
                for f in FREQUENCIES:
                    observed = truth.solve_state((1 / velocity**2).reshape(-1), q, f)[
                        list(receivers)
                    ]
                    block = SurveyBlock(map_index(source, n), f, observed, f"{source}@{f}")
                    value, g, u, residual = physics.objective_and_gradient(
                        model, q, observed, receivers, f
                    )
                    adjoint = physics._solve(
                        model,
                        physics.inject_receivers(residual, receivers),
                        f,
                        category="adjoint_rhs",
                        adjoint=True,
                    )
                    exact.append((value, g, u, adjoint))
                    blocks.append(block)
            reference_cost = dict(truth=truth.cost_snapshot(), verification=physics.cost_snapshot())
            pod = PODGalerkinSurrogate(physics, basis)
            raw = batched_pod_survey_fields(pod, model, blocks, receivers)
            engines = {rank: FastCoarseStability(physics, rank) for rank in (64, 96)}
            contexts_by_rank = {
                rank: {f: engine.evaluate(model, f) for f in FREQUENCIES}
                for rank, engine in engines.items()
            }
            for label, rank, sweeps, initialization in METHODS:
                identity = dict(
                    grid=n,
                    case_index=case_index,
                    parent_id=selection["parent_id"],
                    slice_id=selection["slice_id"],
                    method=label,
                )
                contexts = None if rank is None else contexts_by_rank[rank]
                fields = [raw.block_states.copy(), raw.block_adjoints.copy()]
                dispatches = {}
                try:
                    if rank is not None:
                        for f in FREQUENCIES:
                            context = contexts[f]
                            if not context.available:
                                continue
                            indices = [
                                i for i, block in enumerate(blocks) if block.frequency_hz == f
                            ]
                            q = np.stack(
                                [physics.source(blocks[i].source_index) for i in indices], axis=1
                            )
                            ops = BatchedCoarseOperations(context, dispatches)
                            if initialization == "pure":
                                u = ops.lift(ops.solve(ops.project(q)))
                                adjoint_rhs = inject(
                                    u[list(receivers)]
                                    - np.stack([blocks[i].observed for i in indices], axis=1),
                                    receivers,
                                    physics.n,
                                )
                                adjoint = ops.lift(
                                    ops.solve(ops.project(adjoint_rhs), adjoint=True)
                                )
                            else:
                                u = ops.correct(fields[0][:, indices], q)[0]
                                adjoint_rhs = inject(
                                    u[list(receivers)]
                                    - np.stack([blocks[i].observed for i in indices], axis=1),
                                    receivers,
                                    physics.n,
                                )
                                adjoint = ops.correct(
                                    fields[1][:, indices], adjoint_rhs, adjoint=True
                                )[0]
                            repair = Repair(context, model, physics.damping, dispatches)
                            observed = np.stack([blocks[i].observed for i in indices], axis=1)
                            for _ in range(sweeps):
                                u += repair.apply(q - ops.apply_operator(u))
                                adjoint_rhs = inject(
                                    u[list(receivers)] - observed, receivers, physics.n
                                )
                                adjoint += repair.apply(
                                    adjoint_rhs - ops.apply_operator(adjoint, adjoint=True),
                                    adjoint=True,
                                )
                            fields[0][:, indices], fields[1][:, indices] = u, adjoint
                    record, rows, gradient, exact_gradient = assess(
                        fields, contexts, blocks, receivers, physics, exact, identity
                    )
                    records.append(record)
                    block_rows.extend(rows)
                    vectors.append(gradient)
                    vector_cases.append(case_index)
                    vector_labels.append(label)
                    if label == "raw_pod":
                        exact_vectors.append(exact_gradient)
                except (ValueError, AssertionError, FloatingPointError) as error:
                    records.append(
                        dict(
                            **identity,
                            complete=False,
                            reason=f"{type(error).__name__}: {error}",
                            available=False,
                            certified=False,
                        )
                    )
            receipts.append(
                dict(
                    grid=n,
                    case_index=case_index,
                    reference_cost=reference_cost,
                    pod=pod.cost_snapshot(),
                    coarse={str(rank): engine.cost_snapshot() for rank, engine in engines.items()},
                    contexts={
                        str(rank): {str(f): context.to_dict() for f, context in contexts.items()}
                        for rank, contexts in contexts_by_rank.items()
                    },
                )
            )
            (OUTPUT / "progress.json").write_text(
                json.dumps(dict(grid=n, completed_case=case_index, aggregate_rows=len(records)))
            )
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            proposals=np.asarray(vectors),
            case_index=np.asarray(vector_cases),
            labels=np.asarray(vector_labels),
            exact=np.asarray(exact_vectors),
        )
        (OUTPUT / f"vectors_{n}.npz").write_bytes(buffer.getvalue())
    unchanged = (
        source_digest(ROOT) == protocol["provenance"]["source_sha256"]
        and load_inputs()[1] == hashes
        and sha256_file(Path(__file__)) == protocol["script_sha256"]
    )
    complete = (
        len(records) == 144 and all(row["complete"] for row in records) and len(block_rows) == 3456
    )
    write_new(
        OUTPUT / "summary.json",
        dict(
            status="complete" if complete and unchanged else "incomplete-or-stale",
            source_inputs_unchanged=unchanged,
            protocol_sha256=sha256_file(PROTOCOL),
            records=records,
            blocks=block_rows,
            receipts=receipts,
            provenance_before=protocol["provenance"],
            provenance_after=environment_record(ROOT),
            scope=protocol["scope"],
        ),
    )
    print(
        json.dumps(
            dict(
                status="complete" if complete and unchanged else "incomplete-or-stale",
                aggregates=len(records),
                blocks=len(block_rows),
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    assert args.prepare != args.execute
    prepare() if args.prepare else execute()
