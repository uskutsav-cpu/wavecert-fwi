"""Private train/validation-only Schur-complement feasibility scratchpad.

Outside both repositories deliberately. No public method, held-out data,
timings, POD fitting, or source modification is performed.
"""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.ndimage as ndi

from wavecert.physics.helmholtz import Helmholtz2D, SurveyGeometry
from wavecert.provenance import environment_record, sha256_file, source_digest


ROOT = Path("/Users/swethasunilkumar/Documents/Codex/2026-11-07/wavecert-fwi/work/wavecert-fwi")
OUTPUT = Path(__file__).with_suffix(".json")
CREATED = datetime.now(timezone.utc).isoformat()
VARIANTS = ("exact_ell2_exact_k2", "ell2_triangle_k", "ellF_triangle_k", "ell2_min_triangle_product_k", "gram_ell2_gram_k2", "ell2_frobenius_k")
RANKS = (32, 64, 96)
FREQUENCIES = (3.0, 4.0)


def low_modes(shape, spacing):
    nz, nx = shape
    iz, ix = np.arange(1, nz+1), np.arange(1, nx+1)
    modes = []
    for kz in range(1, nz+1):
        z = np.sqrt(2/(nz+1))*np.sin(np.pi*iz*kz/(nz+1))
        for kx in range(1, nx+1):
            x = np.sqrt(2/(nx+1))*np.sin(np.pi*ix*kx/(nx+1))
            eigenvalue = 4/spacing**2*(np.sin(np.pi*kz/(2*(nz+1)))**2+np.sin(np.pi*kx/(2*(nx+1)))**2)
            modes.append((eigenvalue, kz, kx, np.outer(z, x).reshape(-1)))
    modes.sort(key=lambda item: (item[0], item[1], item[2]))
    return np.asarray([m[0] for m in modes]), np.stack([m[3] for m in modes], axis=1)


def block_bound(a, ell, k, gamma):
    if gamma <= 0:
        return {"available": False, "reason": "nonpositive complement coercivity", "gamma": gamma, "tau": None, "beta_lower": None, "M": None}
    tau = k*ell/gamma
    if not np.isfinite(tau) or tau >= 1:
        return {"available": False, "reason": "Schur small-gain condition not established", "gamma": gamma, "tau": float(tau), "beta_lower": None, "M": None}
    matrix = np.array([[a/(1-tau), k/(gamma*(1-tau))],
        [ell*a/(gamma*(1-tau)), 1/gamma+ell*k/(gamma**2*(1-tau))]])
    return {"available": True, "gamma": gamma, "tau": float(tau), "beta_lower": float(1/la.svdvals(matrix)[0]), "M": matrix.tolist()}


provenance = environment_record(ROOT)
dataset = ROOT / "data/reference/subsurfacegen_parent_split"
selected, input_files = [], {}
for role in ("train", "validation"):
    payload_path, metadata_path = dataset/f"{role}.npz", dataset/f"{role}_metadata.csv"
    input_files[str(payload_path)] = sha256_file(payload_path)
    input_files[str(metadata_path)] = sha256_file(metadata_path)
    with metadata_path.open(newline="") as stream:
        metadata = list(csv.DictReader(stream))
    with np.load(payload_path, allow_pickle=False) as arrays:
        models = arrays["models"]
        seen_types = set()
        for index, record in enumerate(metadata):
            if record["model_type"] in seen_types:
                continue
            raw = models[index].astype(float)
            velocity = 1/np.sqrt(raw)
            smooth = ndi.gaussian_filter(velocity, sigma=1.6, mode="reflect")
            selected.append({"role": role, "array_index": index, "model_id": record["model_id"],
                "slice_id": record["slice_id"], "model_type": record["model_type"],
                "raw_m_sha256": hashlib.sha256(models[index].tobytes()).hexdigest(),
                "model": (1/smooth**2).reshape(-1)})
            seen_types.add(record["model_type"])
            if len(seen_types) == 2:
                break

p = Helmholtz2D(shape=(24, 24), spacing=0.05, damping_width=3, damping_strength=2.0)
eigenvalues, complete_basis = low_modes(p.shape, p.spacing)
geometry = SurveyGeometry.from_grid(p.shape, n_sources=6, n_receivers=20, source_depth=2, receiver_depth=2, margin=2)
q = p.source(int(geometry.source_indices[1]))
rng = np.random.default_rng(2026110719)
random_residual = rng.normal(size=p.n)+1j*rng.normal(size=p.n)
random_residual /= la.norm(random_residual)
rows = []
for selection in selected:
    m = selection["model"]
    identity = {key: value for key, value in selection.items() if key != "model"}
    for frequency in FREQUENCIES:
        omega = 2*np.pi*frequency
        # Fine SVD/LU exist only in this offline verification block.
        fine_a = p.operator(m, frequency).toarray()
        fine_sigma = float(la.svdvals(fine_a)[-1])
        fine_lu = la.lu_factor(fine_a)
        random_error = la.lu_solve(fine_lu, random_residual)
        for rank in RANKS:
            basis = complete_basis[:, :rank]
            ab = basis*eigenvalues[:rank][None, :] + (-omega**2*m+1j*omega*p.damping)[:, None]*basis
            ar = basis.T @ ab
            coupling = ab-basis @ ar
            gamma = float(eigenvalues[rank]-omega**2*np.max(m))
            ar_singular = la.svdvals(ar)
            if ar_singular[-1] <= 0 or not np.isfinite(ar_singular[-1]):
                rows.append({**identity, "frequency_hz": frequency, "rank": rank, "available": False, "reason": "singular reduced block"})
                continue
            a = float(1/ar_singular[-1])
            reduced_lu = la.lu_factor(ar)
            inverse = la.lu_solve(reduced_lu, np.eye(rank))
            ell2, ellf = float(la.svdvals(coupling)[0]), float(la.norm(coupling, "fro"))
            direct_product = inverse @ coupling.T
            exact_k = float(la.svdvals(direct_product)[0])
            triangle_k = float(la.norm(np.abs(inverse) @ la.norm(coupling, axis=0)))
            gram = coupling.conjugate().T @ coupling
            gram_ell = float(np.sqrt(max(0, la.eigvalsh((gram+gram.conjugate().T)/2)[-1])))
            response_gram = inverse @ gram.conjugate() @ inverse.conjugate().T
            response_gram = (response_gram+response_gram.conjugate().T)/2
            gram_k = float(np.sqrt(max(0, la.eigvalsh(response_gram)[-1])))
            frob_k = float(np.sqrt(max(0, np.real(np.trace(response_gram)))))
            assert abs(gram_k-exact_k) <= 1e-10*max(exact_k, 1)
            assert abs(gram_ell-ell2) <= 1e-10*max(ell2, 1)
            assert triangle_k+1e-10 >= exact_k
            choices = {"exact_ell2_exact_k2": (ell2, exact_k), "ell2_triangle_k": (ell2, triangle_k),
                "ellF_triangle_k": (ellf, triangle_k), "ell2_min_triangle_product_k": (ell2, min(triangle_k, a*ell2)),
                "gram_ell2_gram_k2": (gram_ell, gram_k), "ell2_frobenius_k": (ell2, frob_k)}
            galerkin = basis @ la.lu_solve(reduced_lu, basis.T @ q)
            galerkin_residual = fine_a @ galerkin-q
            galerkin_error = la.lu_solve(fine_lu, galerkin_residual)
            residual_cases = {}
            for label, residual, error in (("random_complex", random_residual, random_error),
                                          ("galerkin_source", galerkin_residual, galerkin_error)):
                rb = basis.T @ residual
                rq = residual-basis @ rb
                eb = basis.T @ error
                eq = error-basis @ eb
                residual_cases[label] = {"component_residual_norms": [float(la.norm(rb)), float(la.norm(rq))],
                    "component_error_norms": [float(la.norm(eb)), float(la.norm(eq))],
                    "residual_norm": float(la.norm(residual)), "actual_error_norm": float(la.norm(error)),
                    "coarse_residual_fraction": float(la.norm(rb)/max(la.norm(residual), 1e-15))}
            variants = {}
            for name in VARIANTS:
                ell, k = choices[name]
                bound = block_bound(a, ell, k, gamma)
                bound.update(ell=ell, k=k)
                if bound["available"]:
                    matrix = np.asarray(bound["M"])
                    bound["beta_to_exact_sigma_ratio"] = bound["beta_lower"]/fine_sigma
                    assert bound["beta_lower"] <= fine_sigma*(1+1e-9)
                    checks = {}
                    for label, case in residual_cases.items():
                        component_upper = matrix @ np.asarray(case["component_residual_norms"])
                        actual_components = np.asarray(case["component_error_norms"])
                        assert np.all(component_upper+1e-10*np.maximum(1, actual_components) >= actual_components)
                        specific_upper = float(la.norm(component_upper))
                        global_upper = case["residual_norm"]/bound["beta_lower"]
                        checks[label] = {"component_error_upper": component_upper.tolist(),
                            "residual_specific_error_upper": specific_upper,
                            "global_error_upper": global_upper,
                            "specific_effectivity": specific_upper/case["actual_error_norm"],
                            "global_effectivity": global_upper/case["actual_error_norm"],
                            "all_component_checks_pass": True}
                    bound["offline_residual_checks"] = checks
                variants[name] = bound
            rows.append({**identity, "frequency_hz": frequency, "rank": rank, "fine_dimension": p.n,
                "complement_dimension": p.n-rank, "smoothed_m_range": [float(np.min(m)), float(np.max(m))],
                "gamma": gamma, "lambda_next": float(eigenvalues[rank]), "reduced_min_sigma": float(ar_singular[-1]),
                "a": a, "exact_fine_sigma_min_offline_only": fine_sigma,
                "basis_orthogonality_error_2": float(la.svdvals(basis.T @ basis-np.eye(rank))[0]),
                "gram_k_relative_agreement": abs(gram_k-exact_k)/max(exact_k, 1e-15),
                "triangle_to_exact_k_ratio": triangle_k/max(exact_k, 1e-15),
                "residual_cases": residual_cases, "variants": variants})
    print(json.dumps({"completed_model": identity, "rows": len(rows)}), flush=True)

availability = {name: {str(rank): {"available": sum(row.get("variants", {}).get(name, {}).get("available", False) for row in rows if row["rank"] == rank),
    "total": sum(row["rank"] == rank for row in rows)} for rank in RANKS} for name in VARIANTS}
report = {"schema_version": 1, "created_before_computation_utc": CREATED,
    "scope": "Private engineering feasibility, train/validation only, no held-out data, timings, POD fit, or public implementation.",
    "selection_rule": "First two distinct metadata model_type entries in each train and validation role, chosen before bound evaluation.",
    "selected_models": [{k: v for k, v in s.items() if k != "model"} for s in selected],
    "shape": [24, 24], "spacing": 0.05, "ranks": list(RANKS), "frequencies_hz": list(FREQUENCIES),
    "velocity_smoothing": {"sigma": 1.6, "mode": "reflect"}, "availability": availability, "cases": rows,
    "cost_accounting": {
        "online_bound_candidate_per_model_frequency_rank": {"fine_factorizations": 0, "fine_solve_rhs": 0,
            "fine_dense_svd": 0, "reduced_factorizations": 1, "reduced_identity_rhs": "rank",
            "reduced_svd_for_a": 1, "gram_variant_reduced_hermitian_eigenvalue_problems": 2,
            "basis_operator_application_columns": "rank; diagonal stiffness eigenvalues plus pointwise model/damping multiplication",
            "ar_assembly_arithmetic": "O(n*rank^2)", "coupling_gram_arithmetic": "O(n*rank^2)",
            "reduced_dense_arithmetic": "O(rank^3)", "basis_and_coupling_storage": "O(n*rank)",
            "triangle_k_extra_arithmetic": "O(rank^2), after coupling column norms",
            "gram_k_full_r_by_n_inverse_product": False},
        "offline_only_per_selected_model_frequency": {"fine_dense_svd": 1, "fine_factorizations": 1,
            "fine_solve_rhs": 1+len(RANKS), "purpose": "Verify sigma bound and random/Galerkin residual error bounds"},
        "offline_only_per_rank": {"direct_rectangular_svd_norms": 2,
            "explicit_inverse_times_L_transpose": True, "Galerkin_demonstration_reduced_rhs": 1},
        "timings": "not measured; other training may be active"},
    "assumptions": ["Real orthonormal B contains all declared low Dirichlet Laplacian modes.",
        "Current complex-symmetric constant-spacing Dirichlet operator, model-independent diagonal damping.",
        "gamma>0, invertible Ar, and tau<1 are sufficient conditions; failure means unavailable, not singularity.",
        "Ordinary floating point with no formally verified rounding enclosure.",
        "Small correlated development samples establish feasibility only, not generalization or performance."],
    "primary_context": ["https://msp.org/pjm/1962/12-4/pjm-v12-n4-p08-p.pdf",
        "https://www.numdam.org/articles/10.1016/j.crma.2007.09.019/"],
    "provenance": provenance, "input_sha256": input_files,
    "scratch_script_sha256": sha256_file(__file__), "main_source_unchanged": source_digest(ROOT) == provenance["source_sha256"]}
OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
print(json.dumps({"output": str(OUTPUT), "availability": availability, "main_source_unchanged": report["main_source_unchanged"]}, indent=2))
