"""Private frozen-POD state/adjoint with independent low-mode stability feasibility."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.fft as fft
import scipy.linalg as la
import scipy.ndimage as ndi

from wavecert.physics.helmholtz import Helmholtz2D, SurveyGeometry
from wavecert.provenance import environment_record, sha256_file, source_digest


ROOT = Path("/Users/swethasunilkumar/Documents/Codex/2026-11-07/wavecert-fwi/work/wavecert-fwi")
STABILITY = Path(__file__).with_name("coarse_dct_feasibility.json")
ORIGINAL = Path(__file__).with_name("coarse_bound_feasibility.json")
OUTPUT = Path(__file__).with_suffix(".json")
CREATED = datetime.now(timezone.utc).isoformat()
stability, original = json.loads(STABILITY.read_text()), json.loads(ORIGINAL.read_text())
provenance = environment_record(ROOT)
shape = (24, 24)
p = Helmholtz2D(shape=shape, spacing=0.05, damping_width=3, damping_strength=2.0)
geometry = SurveyGeometry.from_grid(shape, n_sources=6, n_receivers=20, source_depth=2, receiver_depth=2, margin=2)
receivers = tuple(int(v) for v in geometry.receiver_indices)
POD_PATH = Path(__file__).with_name("wavecert-pod-controls") / "results/pod_development/bases/state_tangent_rank64.npz"
with np.load(POD_PATH, allow_pickle=False) as data:
    pod_basis = data["basis"]
    pod_physics = json.loads(str(data["physics_config"]))
assert pod_basis.shape == (576, 64) and np.isrealobj(pod_basis)
assert la.norm(pod_basis.T @ pod_basis-np.eye(64), 2) < 1e-10
assert pod_physics == {"shape": [24, 24], "spacing": 0.05, "damping_width": 3, "damping_strength": 2.0}
source = int(geometry.source_indices[1])
assert source == 54
q = p.source(source)
ppt = np.eye(len(receivers))  # Geometry has unique point receivers.
eigenmodes = []
for k in range(1, 25):
    for x in range(1, 25):
        eigenvalue = 4/0.05**2*(np.sin(np.pi*k/50)**2+np.sin(np.pi*x/50)**2)
        eigenmodes.append((eigenvalue, k, x))
eigenmodes.sort()


def basis(rank):
    grid = np.arange(1, 25)
    return np.stack([np.outer(np.sqrt(2/25)*np.sin(np.pi*grid*k/25),
                             np.sqrt(2/25)*np.sin(np.pi*grid*x/25)).reshape(-1)
                     for _, k, x in eigenmodes[:rank]], axis=1)


def reduced_multiplication(field, rank):
    transformed = fft.dctn(np.pad(field, 1), type=1, norm=None, orthogonalize=False, workers=1)
    modes = np.asarray([[k, x] for _, k, x in eigenmodes[:rank]])
    k, x = modes[:, 0], modes[:, 1]

    def fold(v):
        v = v % 50
        return np.where(v > 25, 50-v, v)

    kd, ks = fold(np.abs(k[:, None]-k[None, :])), fold(k[:, None]+k[None, :])
    xd, xs = fold(np.abs(x[:, None]-x[None, :])), fold(x[:, None]+x[None, :])
    return (transformed[kd, xd]-transformed[ks, xd]-transformed[kd, xs]+transformed[ks, xs])/2500


def parts(b, value):
    coarse = b.T @ value
    return np.asarray([la.norm(coarse), la.norm(value-b @ coarse)], dtype=float)


rows, input_files = [], {}
for selection in original["selected_models"]:
    data_path = ROOT/f"data/reference/subsurfacegen_parent_split/{selection['role']}.npz"
    input_files[str(data_path)] = sha256_file(data_path)
    with np.load(data_path, allow_pickle=False) as data:
        true_m = data["models"][selection["array_index"]].astype(float)
    initial_m = 1/ndi.gaussian_filter(1/np.sqrt(true_m), sigma=1.6, mode="reflect")**2
    m = initial_m.reshape(-1)
    for frequency in (3.0, 4.0):
        omega2 = (2*np.pi*frequency)**2
        # All fine solves below are offline verification labels, not the
        # candidate certificate. Share them across the three tested ranks.
        truth_a = p.operator(true_m.reshape(-1), frequency).toarray()
        observed = la.solve(truth_a, q)[list(receivers)]
        fine_a = p.operator(m, frequency).toarray()
        fine_lu = la.lu_factor(fine_a)
        fine_u = la.lu_solve(fine_lu, q)
        fine_residual = fine_u[list(receivers)]-observed
        fine_lambda = la.lu_solve(fine_lu, p.inject_receivers(fine_residual, receivers), trans=2)
        exact_gradient = omega2*np.real(np.conjugate(fine_lambda)*fine_u)
        exact_objective = 0.5*float(np.vdot(fine_residual, fine_residual).real)
        for rank in (32, 64, 96):
            b = basis(rank)
            # Approximation basis is the frozen real POD basis; b is ONLY
            # the independent spectral stability/projection basis.
            ar = pod_basis.T @ fine_a @ pod_basis
            reduced_lu = la.lu_factor(ar)
            uhat = pod_basis @ la.lu_solve(reduced_lu, pod_basis.T @ q)
            approximate_residual = uhat[list(receivers)]-observed
            lambda_hat = pod_basis @ la.lu_solve(reduced_lu,
                pod_basis.T @ p.inject_receivers(approximate_residual, receivers), trans=2)
            approximate_gradient = omega2*np.real(np.conjugate(lambda_hat)*uhat)
            norm = float(la.norm(approximate_gradient))
            direction = np.zeros(p.n) if norm <= 1e-14 else -approximate_gradient/norm
            approximate_dd = float(np.dot(approximate_gradient, direction))
            assert abs(approximate_dd-omega2*np.real(np.vdot(lambda_hat, direction*uhat))) < 1e-10
            exact_dd = float(np.dot(exact_gradient, direction))
            fine_tangent = la.lu_solve(fine_lu, omega2*direction*fine_u)
            tangent_dd = float(np.vdot(fine_residual, fine_tangent[list(receivers)]).real)
            assert abs(exact_dd-tangent_dd) <= 1e-10*max(1, abs(exact_dd))
            rp = fine_a @ uhat-q
            ra = fine_a.conjugate().T @ lambda_hat-p.inject_receivers(approximate_residual, receivers)
            rp_parts, ra_parts = parts(b, rp), parts(b, ra)
            pu = b[list(receivers), :]
            receiver_gram = pu @ pu.T
            pb = float(np.sqrt(max(0, la.eigvalsh(receiver_gram)[-1])))
            complement_gram = ppt-receiver_gram
            pq = float(np.sqrt(max(0, la.eigvalsh((complement_gram+complement_gram.T)/2)[-1])))
            prior = next(c for c in stability["train_validation_cases"] if c["model_id"] == selection["model_id"]
                and c["role"] == selection["role"] and c["frequency_hz"] == frequency and c["rank"] == rank)
            bound = prior["guarded_fast_bound"]
            row = {**selection, "frequency_hz": frequency, "rank": rank, "source_index": source,
                "approximation": "frozen_state_tangent_POD", "pod_rank": 64, "stability_available": bound["available"], "stability_reason": bound.get("reason"),
                "stability_tau": bound["tau"], "stability_beta_lower": bound["beta_lower"],
                "exact_objective": exact_objective,
                "approximate_objective": 0.5*float(np.vdot(approximate_residual, approximate_residual).real),
                "exact_gradient_norm": float(la.norm(exact_gradient)), "approximate_gradient_norm": norm,
                "gradient_cosine": float(np.dot(exact_gradient, approximate_gradient)/max(la.norm(exact_gradient)*norm, 1e-15)),
                "exact_directional_derivative": exact_dd, "independent_tangent_directional_derivative": tangent_dd,
                "approximate_directional_derivative": approximate_dd, "actual_directional_error": abs(exact_dd-approximate_dd),
                "approximate_state_relative_error": float(la.norm(fine_u-uhat)/la.norm(fine_u)),
                "approximate_receiver_relative_error": float(la.norm(fine_u[list(receivers)]-uhat[list(receivers)])/la.norm(fine_u[list(receivers)])),
                "primal_residual_components": rp_parts.tolist(), "approximate_adjoint_residual_components": ra_parts.tolist(),
                "pB": pb, "pQ": pq, "neural_or_jvp_used": False, "true_descent": exact_dd < 0,
                "directional_error_bound": None, "certified_upper_derivative": None,
                "certified_descent": False, "covers_directional_error": None, "effectivity": None}
            if bound["available"]:
                matrix = np.asarray(bound["M"])
                eta_u = matrix @ rp_parts
                p_components = np.asarray([pb, pq])
                eta_pu = float(np.dot(p_components, eta_u))
                eta_lambda = matrix @ (ra_parts+p_components*eta_pu)
                actual_u_components = parts(b, fine_u-uhat)
                actual_lambda_components = parts(b, fine_lambda-lambda_hat)
                assert np.all(eta_u+1e-9*np.maximum(1, actual_u_components) >= actual_u_components)
                assert np.all(eta_lambda+1e-9*np.maximum(1, actual_lambda_components) >= actual_lambda_components)
                assert eta_pu+1e-9 >= la.norm((fine_u-uhat)[list(receivers)])
                state_multiplier_parts = parts(b, direction*uhat)
                adjoint_multiplier_parts = parts(b, direction*lambda_hat)
                first = float(np.dot(state_multiplier_parts, eta_lambda))
                second = float(np.dot(adjoint_multiplier_parts, eta_u))
                cross = float(la.norm(direction, np.inf)*la.norm(eta_u)*la.norm(eta_lambda))
                eta_direction = omega2*(first+second+cross)
                covers = abs(exact_dd-approximate_dd) <= eta_direction+1e-12
                assert covers
                row.update(eta_u_components=eta_u.tolist(), eta_lambda_components=eta_lambda.tolist(),
                    eta_receiver_state=eta_pu, actual_u_error_components=actual_u_components.tolist(),
                    actual_lambda_error_components=actual_lambda_components.tolist(),
                    multiplier_state_components=state_multiplier_parts.tolist(), multiplier_adjoint_components=adjoint_multiplier_parts.tolist(),
                    directional_bound_terms={"state_approx_times_adjoint_error": omega2*first,
                        "adjoint_approx_times_state_error": omega2*second, "cross_error": omega2*cross},
                    directional_error_bound=eta_direction, certified_upper_derivative=approximate_dd+eta_direction,
                    certified_descent=approximate_dd+eta_direction < 0, covers_directional_error=covers,
                    effectivity=eta_direction/max(abs(exact_dd-approximate_dd), 1e-15),
                    bound_to_descent_margin=eta_direction/max(-approximate_dd, 1e-15))
            rows.append(row)

summary = {}
for rank in (32, 64, 96):
    group = [row for row in rows if row["rank"] == rank]
    available = [row for row in group if row["stability_available"]]
    summary[str(rank)] = {"cases": len(group), "stability_available": len(available),
        "true_descent": sum(row["true_descent"] for row in group),
        "certified_descent": sum(row["certified_descent"] for row in group),
        "false_certified_descent": sum(row["certified_descent"] and not row["true_descent"] for row in group),
        "all_available_covered": all(row["covers_directional_error"] for row in available),
        "median_gradient_cosine": float(np.median([row["gradient_cosine"] for row in group])),
        "median_receiver_relative_error": float(np.median([row["approximate_receiver_relative_error"] for row in group])),
        "median_effectivity_available": float(np.median([row["effectivity"] for row in available])) if available else None,
        "median_bound_to_descent_margin_available": float(np.median([row["bound_to_descent_margin"] for row in available])) if available else None}
report = {"schema_version": 1, "created_before_computation_utc": CREATED,
    "scope": "Private fixed four train/validation model frozen POD Galerkin state/adjoint with independent spectral-stability feasibility. No held-out data, NN/JVP, public source, optimizer, or timings.",
    "observed_data": "Fine reference state at the original unsmoothed model, evaluated with the same fixed source/receivers/operator",
    "evaluation_model": "Gaussian-smoothed original velocity, sigma1.6 reflect, then squared-slowness transform",
    "shape": list(shape), "spacing": p.spacing, "source_index": source,
    "receiver_indices": list(receivers), "ranks": [32, 64, 96], "frequencies_hz": [3.0, 4.0],
    "direction": "normalized negative physical approximate adjoint-form gradient; zero if its norm <=1e-14",
    "summary_by_rank": summary, "cases": rows,
    "offline_verification_costs_per_model_frequency": {"fine_factorizations": 2,
        "fine_rhs": 6, "detail": "1 truth observation +1 current state +1 exact adjoint +3 exact directional tangents; all offline verification, shared across ranks"},
    "candidate_cost_scope": "Only reduced state/adjoint solves, field reconstruction, sparse residual applications, component projections, small receiver Gram/norm operations and the previously validated fast stability bound. No fine solve/map or neural JVP is part of the candidate certificate.",
    "limitations": ["Ordinary floating-point stability with an explicit unverified engineering allowance, not a formal roundoff enclosure.",
        "Correlated development feasibility on four fixed train/validation models only; unavailable cases and failed acceptance retained.",
        "Approximate derivative is the physical state/adjoint formula for a fixed real POD Galerkin basis; no finite-step guarantee, trajectory guarantee, or acceleration claim."],
    "pod_basis_file": str(POD_PATH), "pod_basis_file_sha256": sha256_file(POD_PATH), "pod_physics_config": pod_physics,
    "stability_evidence_sha256": sha256_file(STABILITY), "original_selection_sha256": sha256_file(ORIGINAL),
    "input_sha256": input_files, "scratch_script_sha256": sha256_file(__file__),
    "provenance": provenance, "main_source_unchanged": source_digest(ROOT) == provenance["source_sha256"]}
OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
print(json.dumps({"output": str(OUTPUT), "summary": summary, "main_source_unchanged": report["main_source_unchanged"]}, indent=2))
