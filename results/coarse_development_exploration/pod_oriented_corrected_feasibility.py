"""Private final orientation/coarse-correction check on the fixed development cases."""

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
POD = Path(__file__).with_name("wavecert-pod-controls") / "results/pod_development/bases/state_tangent_rank64.npz"
STABILITY = Path(__file__).with_name("coarse_dct_feasibility.json")
ORIGINAL = Path(__file__).with_name("coarse_bound_feasibility.json")
PREVIOUS = Path(__file__).with_name("pod_coarse_certificate_feasibility.json")
OUTPUT = Path(__file__).with_suffix(".json")
CREATED = datetime.now(timezone.utc).isoformat()
previous = json.loads(PREVIOUS.read_text())
assert sha256_file(POD) == previous["pod_basis_file_sha256"]
with np.load(POD, allow_pickle=False) as data:
    pod = data["basis"]
assert pod.shape == (576, 64) and np.isrealobj(pod)
assert la.norm(pod.T @ pod-np.eye(64), 2) < 1e-10
stability, original = json.loads(STABILITY.read_text()), json.loads(ORIGINAL.read_text())
provenance = environment_record(ROOT)
p = Helmholtz2D(shape=(24, 24), spacing=0.05, damping_width=3, damping_strength=2.0)
geometry = SurveyGeometry.from_grid(p.shape, n_sources=6, n_receivers=20, source_depth=2, receiver_depth=2, margin=2)
receivers = tuple(int(i) for i in geometry.receiver_indices)
source = int(geometry.source_indices[1])
assert source == 54
q = p.source(source)
ordered = sorted((4/0.05**2*(np.sin(np.pi*k/50)**2+np.sin(np.pi*x/50)**2), k, x)
    for k in range(1, 25) for x in range(1, 25))
METHODS = ("uncorrected_generic_M", "uncorrected_oriented", "coarse_corrected_oriented")


def basis(rank):
    grid = np.arange(1, 25)
    return np.stack([np.outer(np.sqrt(2/25)*np.sin(np.pi*grid*k/25),
        np.sqrt(2/25)*np.sin(np.pi*grid*x/25)).reshape(-1) for _, k, x in ordered[:rank]], axis=1)


def multiply(field, rank):
    transformed = fft.dctn(np.pad(field, 1), type=1, norm=None, orthogonalize=False, workers=1)
    modes = np.asarray([[k, x] for _, k, x in ordered[:rank]])
    k, x = modes[:, 0], modes[:, 1]

    def fold(v):
        v %= 50
        return np.where(v > 25, 50-v, v)

    kd, ks = fold(np.abs(k[:, None]-k[None, :])), fold(k[:, None]+k[None, :])
    xd, xs = fold(np.abs(x[:, None]-x[None, :])), fold(x[:, None]+x[None, :])
    return (transformed[kd, xd]-transformed[ks, xd]-transformed[kd, xs]+transformed[ks, xs])/2500


def components(b, value):
    coarse = b.T @ value
    return coarse, float(la.norm(value-b @ coarse))


def norms(b, value):
    coarse, complement = components(b, value)
    return np.array([la.norm(coarse), complement])


rows, input_hashes = [], {}
for selection in original["selected_models"]:
    data_path = ROOT/f"data/reference/subsurfacegen_parent_split/{selection['role']}.npz"
    input_hashes[str(data_path)] = sha256_file(data_path)
    with np.load(data_path, allow_pickle=False) as data:
        true_m = data["models"][selection["array_index"]].astype(float)
    initial_m = 1/ndi.gaussian_filter(1/np.sqrt(true_m), sigma=1.6, mode="reflect")**2
    m = initial_m.reshape(-1)
    for frequency in (3.0, 4.0):
        omega, omega2 = 2*np.pi*frequency, (2*np.pi*frequency)**2
        observed = la.solve(p.operator(true_m.reshape(-1), frequency).toarray(), q)[list(receivers)]
        fine_a = p.operator(m, frequency).toarray()
        fine_lu = la.lu_factor(fine_a)
        exact_u = la.lu_solve(fine_lu, q)
        exact_residual = exact_u[list(receivers)]-observed
        exact_lambda = la.lu_solve(fine_lu, p.inject_receivers(exact_residual, receivers), trans=2)
        exact_gradient = omega2*np.real(np.conjugate(exact_lambda)*exact_u)
        exact_objective = 0.5*float(np.vdot(exact_residual, exact_residual).real)
        pod_ar = pod.T @ fine_a @ pod
        pod_lu = la.lu_factor(pod_ar)
        pod_u = pod @ la.lu_solve(pod_lu, pod.T @ q)
        pod_residual = pod_u[list(receivers)]-observed
        pod_lambda = pod @ la.lu_solve(pod_lu, pod.T @ p.inject_receivers(pod_residual, receivers), trans=2)
        pod_rp = fine_a @ pod_u-q
        for rank in (32, 64, 96):
            b = basis(rank)
            ar0 = np.diag([mode[0] for mode in ordered[:rank]])-omega2*multiply(initial_m, rank)
            ar0 = ar0+1j*omega*multiply(p.damping.reshape(p.shape), rank)
            lu0 = la.lu_factor(ar0)
            corrected_u = pod_u-b @ la.lu_solve(lu0, b.T @ pod_rp)
            intermediate_ra = fine_a.conjugate().T @ pod_lambda-p.inject_receivers(corrected_u[list(receivers)]-observed, receivers)
            corrected_lambda = pod_lambda-b @ la.lu_solve(lu0, b.T @ intermediate_ra, trans=2)
            pb_matrix = b[list(receivers), :]
            receiver_gram = pb_matrix @ pb_matrix.T
            pb = float(np.sqrt(max(0, la.eigvalsh(receiver_gram)[-1])))
            pq_gram = np.eye(len(receivers))-receiver_gram
            pq = float(np.sqrt(max(0, la.eigvalsh((pq_gram+pq_gram.T)/2)[-1])))
            receiver_response = la.lu_solve(lu0, pb_matrix.T, trans=2)
            rho = float(la.svdvals(receiver_response)[0])
            prior = next(c for c in stability["train_validation_cases"] if c["model_id"] == selection["model_id"]
                and c["role"] == selection["role"] and c["frequency_hz"] == frequency and c["rank"] == rank)
            bound = prior["guarded_fast_bound"]
            gamma = prior["gamma"]
            for method in METHODS:
                uhat, lambda_hat = (corrected_u, corrected_lambda) if method == "coarse_corrected_oriented" else (pod_u, pod_lambda)
                approximate_residual = uhat[list(receivers)]-observed
                rp = fine_a @ uhat-q
                ra = fine_a.conjugate().T @ lambda_hat-p.inject_receivers(approximate_residual, receivers)
                rp_b, rp_q = components(b, rp)
                ra_b, ra_q = components(b, ra)
                state_response = la.lu_solve(lu0, rp_b)
                adjoint_response = la.lu_solve(lu0, ra_b, trans=2)
                gradient = omega2*np.real(np.conjugate(lambda_hat)*uhat)
                grad_norm = float(la.norm(gradient))
                direction = np.zeros(p.n) if grad_norm <= 1e-14 else -gradient/grad_norm
                dd_hat = float(np.dot(gradient, direction))
                dd_exact = float(np.dot(exact_gradient, direction))
                tangent = la.lu_solve(fine_lu, omega2*direction*exact_u)
                dd_tangent = float(np.vdot(exact_residual, tangent[list(receivers)]).real)
                assert abs(dd_exact-dd_tangent) <= 1e-10*max(1, abs(dd_exact))
                row = {**selection, "frequency_hz": frequency, "rank": rank, "pod_rank": 64,
                    "source_index": source, "method": method, "stability_available": bound["available"],
                    "stability_reason": bound.get("reason"), "tau": bound["tau"],
                    "exact_objective": exact_objective,
                    "approximate_objective": 0.5*float(np.vdot(approximate_residual, approximate_residual).real),
                    "exact_gradient_norm": float(la.norm(exact_gradient)), "approximate_gradient_norm": grad_norm,
                    "gradient_cosine": float(np.dot(exact_gradient, gradient)/max(la.norm(exact_gradient)*grad_norm, 1e-15)),
                    "approximate_receiver_relative_error": float(la.norm((exact_u-uhat)[list(receivers)])/la.norm(exact_u[list(receivers)])),
                    "exact_directional_derivative": dd_exact, "approximate_directional_derivative": dd_hat,
                    "actual_directional_error": abs(dd_exact-dd_hat), "true_descent": dd_exact < 0,
                    "primal_residual_norms": [float(la.norm(rp_b)), rp_q],
                    "adjoint_residual_norms": [float(la.norm(ra_b)), ra_q],
                    "primal_coarse_fraction": float(la.norm(rp_b)/max(la.norm(rp), 1e-15)),
                    "adjoint_coarse_fraction": float(la.norm(ra_b)/max(la.norm(ra), 1e-15)),
                    "primal_reduced_response_norm": float(la.norm(state_response)),
                    "adjoint_reduced_response_norm": float(la.norm(adjoint_response)),
                    "reduced_receiver_adjoint_response_norm": rho, "pB": pb, "pQ": pq,
                    "directional_error_bound": None, "certified_upper_derivative": None,
                    "certified_descent": False, "covers_directional_error": None, "effectivity": None,
                    "bound_to_descent_margin": None}
                if method == "coarse_corrected_oriented":
                    assert la.norm(rp_b) <= 1e-9*max(la.norm(q), 1)
                    assert la.norm(ra_b) <= 1e-9*max(la.norm(p.inject_receivers(approximate_residual, receivers)), 1)
                if bound["available"]:
                    if method == "uncorrected_generic_M":
                        matrix = np.asarray(bound["M"])
                        eta_u = matrix @ np.array([la.norm(rp_b), rp_q])
                        eta_pu = float(np.dot([pb, pq], eta_u))
                        eta_lambda = matrix @ (np.array([la.norm(ra_b), ra_q])+np.array([pb, pq])*eta_pu)
                    else:
                        k, ell, tau = bound["k"], bound["ell"], bound["tau"]
                        eu_b = (la.norm(state_response)+k*rp_q/gamma)/(1-tau)
                        eu_q = (rp_q+ell*eu_b)/gamma
                        eta_u = np.array([eu_b, eu_q], dtype=float)
                        eta_pu = pb*eu_b+pq*eu_q
                        el_b = (la.norm(adjoint_response)+rho*eta_pu+(k/gamma)*(ra_q+pq*eta_pu))/(1-tau)
                        el_q = (ra_q+pq*eta_pu+ell*el_b)/gamma
                        eta_lambda = np.array([el_b, el_q], dtype=float)
                    actual_eu = norms(b, exact_u-uhat)
                    actual_el = norms(b, exact_lambda-lambda_hat)
                    assert np.all(eta_u+1e-9*np.maximum(1, actual_eu) >= actual_eu)
                    assert np.all(eta_lambda+1e-9*np.maximum(1, actual_el) >= actual_el)
                    first = float(np.dot(norms(b, direction*uhat), eta_lambda))
                    second = float(np.dot(norms(b, direction*lambda_hat), eta_u))
                    cross = float(la.norm(direction, np.inf)*la.norm(eta_u)*la.norm(eta_lambda))
                    eta_dd = omega2*(first+second+cross)
                    covers = abs(dd_exact-dd_hat) <= eta_dd+1e-12
                    assert covers
                    row.update(eta_u_components=eta_u.tolist(), eta_lambda_components=eta_lambda.tolist(),
                        actual_u_error_components=actual_eu.tolist(), actual_lambda_error_components=actual_el.tolist(),
                        eta_receiver_state=eta_pu, directional_error_bound=eta_dd,
                        certified_upper_derivative=dd_hat+eta_dd, certified_descent=dd_hat+eta_dd < 0,
                        covers_directional_error=covers, effectivity=eta_dd/max(abs(dd_exact-dd_hat), 1e-15),
                        bound_to_descent_margin=eta_dd/max(-dd_hat, 1e-15),
                        directional_bound_terms={"first": omega2*first, "second": omega2*second, "cross": omega2*cross})
                rows.append(row)

summary = {}
for method in METHODS:
    summary[method] = {}
    for rank in (32, 64, 96):
        group = [r for r in rows if r["method"] == method and r["rank"] == rank]
        available = [r for r in group if r["stability_available"]]
        summary[method][str(rank)] = {"cases": len(group), "available": len(available),
            "true_descent": sum(r["true_descent"] for r in group),
            "certified_descent": sum(r["certified_descent"] for r in group),
            "false_certifications": sum(r["certified_descent"] and not r["true_descent"] for r in group),
            "all_available_covered": all(r["covers_directional_error"] for r in available),
            "median_gradient_cosine": float(np.median([r["gradient_cosine"] for r in group])),
            "median_receiver_relative_error": float(np.median([r["approximate_receiver_relative_error"] for r in group])),
            "median_effectivity_available": float(np.median([r["effectivity"] for r in available])),
            "median_bound_margin_ratio_available": float(np.median([r["bound_to_descent_margin"] for r in available])),
            "minimum_bound_margin_ratio_available": float(min(r["bound_to_descent_margin"] for r in available))}
report = {"schema_version": 1, "created_before_computation_utc": CREATED,
    "scope": "Last private orientation/coarse-correction feasibility on identical four train/validation models, 24 fixed cases per method. No held-out data, rank tuning, public method, optimizer, NN JVP or timings.",
    "methods": list(METHODS), "summary": summary, "cases": rows,
    "correction": "u'=u-B0 Ar0^-1 B0.T(Au-q); recompute adjoint residual using u', then lambda'=lambda-B0 Ar0^-H B0.T ra. Physical approximate derivative only; no ROM-autograd claim for this sequential correction.",
    "coverage_tolerance": 1e-12, "direction": "negative physical approximate gradient, L2-normalized; zero if norm<=1e-14",
    "cost_scope": {"fine_solves": "Offline reference/truth and independent tangent derivative checks only",
        "extra_online_orientation": "Reduced residual solves and rank-by-receiver adjoint response, no fine solves/maps",
        "extra_online_correction": "One reduced primal and one reduced adjoint correction plus reconstruction/residual applications",
        "timings": "not measured"},
    "limits": ["Ordinary floating point, not formally verified rounding enclosures.",
        "No extension to arbitrary backends or complex coarse bases.", "All unavailable cases and failed acceptances retained.",
        "No additional aggregate experiment is authorized by a failed first test; stop this bound route if both oriented and corrected variants remain unusable."],
    "pod_file_sha256": sha256_file(POD), "stability_evidence_sha256": sha256_file(STABILITY),
    "previous_uncorrected_evidence_sha256": sha256_file(PREVIOUS), "input_sha256": input_hashes,
    "scratch_script_sha256": sha256_file(__file__), "provenance": provenance,
    "main_source_unchanged": source_digest(ROOT) == provenance["source_sha256"]}
OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
print(json.dumps({"output": str(OUTPUT), "summary": summary, "main_source_unchanged": report["main_source_unchanged"]}, indent=2))
