"""Private DCT-I spectral-coarse assembly checks, outside all repositories."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.fft as fft
import scipy.linalg as la
import scipy.ndimage as ndi

from wavecert.physics.helmholtz import Helmholtz2D
from wavecert.provenance import environment_record, sha256_file, source_digest


ROOT = Path("/Users/swethasunilkumar/Documents/Codex/2026-11-07/wavecert-fwi/work/wavecert-fwi")
PREVIOUS = Path(__file__).with_name("coarse_bound_feasibility.json")
OUTPUT = Path(__file__).with_suffix(".json")
CREATED = datetime.now(timezone.utc).isoformat()
previous = json.loads(PREVIOUS.read_text())
provenance = environment_record(ROOT)
rng = np.random.default_rng(2026110721)


def modes(shape, spacing):
    nz, nx = shape
    pairs = [(4/spacing**2*(np.sin(np.pi*k/(2*(nz+1)))**2+np.sin(np.pi*p/(2*(nx+1)))**2), k, p)
             for k in range(1, nz+1) for p in range(1, nx+1)]
    pairs.sort()
    return np.asarray([v[0] for v in pairs]), np.asarray([[v[1], v[2]] for v in pairs])


def basis(shape, selected_modes):
    nz, nx = shape
    iz, ix = np.arange(1, nz+1), np.arange(1, nx+1)
    return np.stack([(np.sqrt(2/(nz+1))*np.sin(np.pi*iz*k/(nz+1)))[:, None]
        *(np.sqrt(2/(nx+1))*np.sin(np.pi*ix*p/(nx+1)))[None, :]
        for k, p in selected_modes], axis=-1).reshape(nz*nx, len(selected_modes))


def fold(indices, intervals):
    indices = indices % (2*intervals)
    return np.where(indices > intervals, 2*intervals-indices, indices)


def transform(field):
    return fft.dctn(np.pad(np.asarray(field, dtype=float), ((1, 1), (1, 1))),
        type=1, norm=None, orthogonalize=False, workers=1)


def multiplication_from_transform(transformed, shape, selected_modes):
    nz, nx = shape
    k, p = selected_modes[:, 0], selected_modes[:, 1]
    zd = fold(np.abs(k[:, None]-k[None, :]), nz+1)
    zs = fold(k[:, None]+k[None, :], nz+1)
    xd = fold(np.abs(p[:, None]-p[None, :]), nx+1)
    xs = fold(p[:, None]+p[None, :], nx+1)
    return (transformed[zd, xd]-transformed[zs, xd]-transformed[zd, xs]+transformed[zs, xs])/(4*(nz+1)*(nx+1))


def multiplication(field, selected_modes):
    return multiplication_from_transform(transform(field), field.shape, selected_modes)


def beta_bound(ar, gram, gamma):
    if gamma <= 0:
        return {"available": False, "reason": "nonpositive gamma", "tau": None, "beta_lower": None}
    smallest = float(la.svdvals(ar)[-1])
    if smallest <= 0 or not np.isfinite(smallest):
        return {"available": False, "reason": "singular reduced block", "tau": None, "beta_lower": None}
    a = 1/smallest
    inverse = la.solve(ar, np.eye(len(ar)))
    ell = float(np.sqrt(max(0, la.eigvalsh((gram+gram.conjugate().T)/2)[-1])))
    response = inverse @ gram.conjugate() @ inverse.conjugate().T
    k = float(np.sqrt(max(0, la.eigvalsh((response+response.conjugate().T)/2)[-1])))
    tau = k*ell/gamma
    result = {"a": a, "ell": ell, "k": k, "tau": tau, "available": tau < 1, "beta_lower": None}
    if tau >= 1:
        result["reason"] = "Schur small-gain condition unavailable"
        return result
    matrix = np.asarray([[a/(1-tau), k/(gamma*(1-tau))],
        [ell*a/(gamma*(1-tau)), 1/gamma+ell*k/(gamma**2*(1-tau))]])
    result.update(beta_lower=float(1/la.svdvals(matrix)[0]), M=matrix.tolist())
    return result


def gram_psd_margin(gram, moment, vr):
    hermitian = (gram+gram.conjugate().T)/2
    eigenvalues = la.eigvalsh(hermitian)
    scale = float(la.norm(moment, 2)+la.norm(vr, 2)**2)
    # Explicit engineering allowance, NOT a verified error enclosure.
    allowance = float(128*np.finfo(float).eps*len(gram)*max(scale, 1))
    shift = max(0, -float(eigenvalues[0]))+allowance
    return hermitian+shift*np.eye(len(gram)), {
        "raw_minimum_eigenvalue": float(eigenvalues[0]), "scale": scale,
        "engineering_allowance": allowance, "diagonal_shift": shift,
        "formally_verified_roundoff_enclosure": False,
    }


identity_checks = []
for shape in ((5, 6), (6, 5), (7, 7), (24, 24)):
    eigenvalues, all_modes = modes(shape, 0.05)
    # Mix low modes with modes whose sums exceed Nyquist to exercise folding.
    chosen = np.vstack((all_modes[:8], [[shape[0], shape[1]], [shape[0]-1, shape[1]]]))
    b = basis(shape, chosen)
    for label, field in (("random_real", rng.normal(size=shape)), ("constant", np.full(shape, 1.7)),
                         ("corner_impulse", np.eye(shape[0], shape[1], dtype=float))):
        direct = b.T @ (field.reshape(-1, 1)*b)
        fast = multiplication(field, chosen)
        absolute = float(np.max(np.abs(fast-direct)))
        relative = float(la.norm(fast-direct)/max(la.norm(direct), 1e-15))
        assert absolute < 2e-12 and relative < 2e-12
        if label == "constant":
            assert np.max(np.abs(fast-1.7*np.eye(len(chosen)))) < 2e-12
        identity_checks.append({"shape": list(shape), "field": label, "selected_modes": chosen.tolist(),
            "maximum_absolute_error": absolute, "relative_frobenius_error": relative,
            "folding_exercised": bool(np.any(chosen[:, 0, None]+chosen[None, :, 0] > shape[0]+1)
                                      or np.any(chosen[:, 1, None]+chosen[None, :, 1] > shape[1]+1))})


shape, spacing = (24, 24), 0.05
eigenvalues, all_modes = modes(shape, spacing)
p = Helmholtz2D(shape=shape, spacing=spacing, damping_width=3, damping_strength=2.0)
damping = p.damping.reshape(shape)
dct_d, dct_d2 = transform(damping), transform(damping**2)
rows = []
for selection in previous["selected_models"]:
    with np.load(ROOT/f"data/reference/subsurfacegen_parent_split/{selection['role']}.npz", allow_pickle=False) as data:
        raw = data["models"][selection["array_index"]]
        assert hashlib.sha256(raw.tobytes()).hexdigest() == selection["raw_m_sha256"]
        smooth = ndi.gaussian_filter(1/np.sqrt(raw.astype(float)), sigma=1.6, mode="reflect")
        m = 1/smooth**2
    t_m, t_m2 = transform(m), transform(m**2)
    for frequency in (3.0, 4.0):
        omega = 2*np.pi*frequency
        for rank in (32, 64, 96):
            selected_modes = all_modes[:rank]
            b = basis(shape, selected_modes)  # Offline explicit verification only.
            mm = multiplication_from_transform(t_m, shape, selected_modes)
            mm2 = multiplication_from_transform(t_m2, shape, selected_modes)
            md = multiplication_from_transform(dct_d, shape, selected_modes)
            md2 = multiplication_from_transform(dct_d2, shape, selected_modes)
            vr = -omega**2*mm+1j*omega*md
            ar = np.diag(eigenvalues[:rank])+vr
            moment = omega**4*mm2+omega**2*md2
            gram = moment-vr.conjugate().T @ vr
            explicit_vb = (-omega**2*m.reshape(-1)+1j*omega*p.damping)[:, None]*b
            explicit_vr = b.T @ explicit_vb
            explicit_l = explicit_vb-b @ explicit_vr
            explicit_gram = explicit_l.conjugate().T @ explicit_l
            ar_reference = np.diag(eigenvalues[:rank])+explicit_vr
            gamma = float(eigenvalues[rank]-omega**2*np.max(m))
            guarded_gram, margin = gram_psd_margin(gram, moment, vr)
            direct_bound = beta_bound(ar_reference, explicit_gram, gamma)
            fast_bound = beta_bound(ar, gram, gamma)
            guarded_bound = beta_bound(ar, guarded_gram, gamma)
            previous_case = next(case for case in previous["cases"] if case["model_id"] == selection["model_id"]
                and case["role"] == selection["role"] and case["frequency_hz"] == frequency and case["rank"] == rank)
            sigma = previous_case["exact_fine_sigma_min_offline_only"]
            for label, bound in (("direct", direct_bound), ("fast", fast_bound), ("guarded_fast", guarded_bound)):
                if bound["available"]:
                    assert bound["beta_lower"] <= sigma*(1+1e-9)
                    bound["beta_to_offline_exact_sigma_ratio"] = bound["beta_lower"]/sigma
                    checks = {}
                    for residual_name, case in previous_case["residual_cases"].items():
                        component_upper = np.asarray(bound["M"]) @ np.asarray(case["component_residual_norms"])
                        actual_components = np.asarray(case["component_error_norms"])
                        assert np.all(component_upper+1e-9*np.maximum(1, actual_components) >= actual_components)
                        checks[residual_name] = {"component_checks_pass": True,
                            "specific_error_effectivity": float(la.norm(component_upper)/case["actual_error_norm"])}
                    bound["reused_offline_residual_checks"] = checks
            assert fast_bound["available"] == direct_bound["available"] == guarded_bound["available"]
            assert la.norm(ar-ar_reference)/la.norm(ar_reference) < 1e-12
            assert la.norm(gram-explicit_gram)/la.norm(explicit_gram) < 1e-10
            if fast_bound["available"]:
                assert abs(fast_bound["beta_lower"]-direct_bound["beta_lower"])/direct_bound["beta_lower"] < 1e-10
            rows.append({**selection, "frequency_hz": frequency, "rank": rank, "gamma": gamma,
                "cutoff_degenerate": bool(abs(eigenvalues[rank]-eigenvalues[rank-1]) <= 1e-12),
                "ar_relative_frobenius_error": float(la.norm(ar-ar_reference)/la.norm(ar_reference)),
                "gram_relative_frobenius_error": float(la.norm(gram-explicit_gram)/la.norm(explicit_gram)),
                "gram_maximum_absolute_error": float(np.max(np.abs(gram-explicit_gram))),
                "roundoff_margin": margin, "direct_bound": direct_bound,
                "fast_bound": fast_bound, "guarded_fast_bound": guarded_bound})


constant_checks = []
for label, m_constant, d_constant in (("constant_m_d", 0.22, 0.8),
                                     ("constant_near_resonance", float(eigenvalues[4]/(2*np.pi*3)**2), 0.001)):
    for rank in (32, 64, 96):
        chosen = all_modes[:rank]
        mm, mm2 = multiplication(np.full(shape, m_constant), chosen), multiplication(np.full(shape, m_constant**2), chosen)
        md, md2 = multiplication(np.full(shape, d_constant), chosen), multiplication(np.full(shape, d_constant**2), chosen)
        omega = 2*np.pi*3
        vr = -omega**2*mm+1j*omega*md
        moment = omega**4*mm2+omega**2*md2
        gram = moment-vr.conjugate().T @ vr
        guarded, margin = gram_psd_margin(gram, moment, vr)
        ar = np.diag(eigenvalues[:rank])+vr
        gamma = float(eigenvalues[rank]-omega**2*m_constant)
        bound = beta_bound(ar, guarded, gamma)
        true_sigma = float(np.min(np.hypot(eigenvalues-omega**2*m_constant, omega*d_constant)))
        if bound["available"]:
            assert bound["beta_lower"] <= true_sigma*(1+1e-8)
        constant_checks.append({"case": label, "rank": rank, "m": m_constant, "d": d_constant,
            "exact_coupling_gram": "zero", "computed_gram_norm_2": float(la.norm(gram, 2)),
            "roundoff_margin": margin, "guarded_bound": bound, "exact_normal_sigma_min": true_sigma})


degenerate_checks = []
cuts = [rank for rank in range(25, 100) if abs(eigenvalues[rank]-eigenvalues[rank-1]) <= 1e-12]
cuts = sorted({min(cuts, key=lambda rank: abs(rank-target)) for target in (32, 64)})
first = previous["selected_models"][0]
with np.load(ROOT/f"data/reference/subsurfacegen_parent_split/{first['role']}.npz", allow_pickle=False) as data:
    raw = data["models"][first["array_index"]].astype(float)
    first_m = 1/ndi.gaussian_filter(1/np.sqrt(raw), sigma=1.6, mode="reflect")**2
first_tm, first_tm2 = transform(first_m), transform(first_m**2)
q = p.source(54)
for frequency in (3.0, 4.0):
    omega = 2*np.pi*frequency
    fine_matrix = p.operator(first_m.reshape(-1), frequency).toarray()
    fine_lu = la.lu_factor(fine_matrix)  # Offline degenerate-cut verification only.
    sigma = next(c["exact_fine_sigma_min_offline_only"] for c in previous["cases"]
        if c["model_id"] == first["model_id"] and c["frequency_hz"] == frequency)
    for rank in cuts:
        for label in ("ordered", "swap_equal_eigenvalue_cut_mode"):
            indices = np.arange(rank)
            if label != "ordered":
                indices[-1] = rank
            chosen = all_modes[indices]
            b = basis(shape, chosen)
            mm = multiplication_from_transform(first_tm, shape, chosen)
            mm2 = multiplication_from_transform(first_tm2, shape, chosen)
            md = multiplication_from_transform(dct_d, shape, chosen)
            md2 = multiplication_from_transform(dct_d2, shape, chosen)
            vr = -omega**2*mm+1j*omega*md
            moment = omega**4*mm2+omega**2*md2
            gram = moment-vr.conjugate().T @ vr
            guarded, margin = gram_psd_margin(gram, moment, vr)
            ar = np.diag(eigenvalues[indices])+vr
            gamma = float(min(eigenvalues[rank-1], eigenvalues[rank])-omega**2*np.max(first_m))
            bound = beta_bound(ar, guarded, gamma)
            explicit_vb = (-omega**2*first_m.reshape(-1)+1j*omega*p.damping)[:, None]*b
            explicit_vr = b.T @ explicit_vb
            explicit_l = explicit_vb-b @ explicit_vr
            direct_gram = explicit_l.conjugate().T @ explicit_l
            assert la.norm(gram-direct_gram)/la.norm(direct_gram) < 1e-10
            residual = fine_matrix @ (b @ la.solve(ar, b.T @ q))-q
            error = la.lu_solve(fine_lu, residual)
            rb, eb = b.T @ residual, b.T @ error
            residual_parts = [float(la.norm(rb)), float(la.norm(residual-b @ rb))]
            error_parts = [float(la.norm(eb)), float(la.norm(error-b @ eb))]
            if bound["available"]:
                assert bound["beta_lower"] <= sigma*(1+1e-9)
                assert np.all(np.asarray(bound["M"]) @ residual_parts+1e-9*np.maximum(1, error_parts) >= error_parts)
            degenerate_checks.append({"frequency_hz": frequency, "rank": rank, "case": label,
                "cut_eigenvalues": [float(eigenvalues[rank-1]), float(eigenvalues[rank])],
                "selected_cut_mode": chosen[-1].tolist(), "gamma": gamma,
                "gram_relative_frobenius_error": float(la.norm(gram-direct_gram)/la.norm(direct_gram)),
                "guarded_bound": bound, "offline_exact_sigma_min": sigma,
                "residual_component_norms": residual_parts, "actual_error_component_norms": error_parts,
                "roundoff_margin": margin, "all_available_checks_pass": True})


report = {"schema_version": 1, "created_before_computation_utc": CREATED,
    "scope": "Private fast-assembly feasibility only; no held-out data, new POD fit, timings, public source, or optimizer.",
    "identity_checks": identity_checks, "train_validation_cases": rows, "constant_cancellation_checks": constant_checks,
    "degenerate_cut_checks": degenerate_checks,
    "availability": {str(rank): {"fast_available": sum(row["fast_bound"]["available"] for row in rows if row["rank"] == rank),
        "guarded_fast_available": sum(row["guarded_fast_bound"]["available"] for row in rows if row["rank"] == rank),
        "total": sum(row["rank"] == rank for row in rows)} for rank in (32, 64, 96)},
    "counts": {"online_dct_I_per_model": 2, "offline_damping_dct_I_once": 2,
        "online_fine_factorizations": 0, "online_fine_rhs": 0, "online_dense_fine_svd": 0,
        "online_explicit_fine_by_reduced_basis_or_gram": False,
        "online_work": "O(n_grid log n_grid + rank^3), including rank^2 coefficient lookups; reuse model DCTs across frequencies/ranks",
        "online_storage": "O(n_grid + rank^2), excluding offline mode metadata/damping transforms",
        "reduced_factorization": "one per model/frequency/rank", "reduced_identity_rhs": "rank",
        "reduced_svd": 1, "reduced_hermitian_eigenproblems": "G and response Gram; PSD guard uses additional G eigenvalue information",
        "verification_fine_svd_lu": "Train/validation sigma and residual labels reused from previous scratch evidence; additional degenerate-cut LU/RHS counts are listed separately. Explicit bases/matrices are offline checks, never the fast path.",
        "additional_offline_degenerate_cut_fine_factorizations": 2,
        "additional_offline_degenerate_cut_fine_rhs": len(degenerate_checks),
        "timings": "not measured"},
    "important_limits": ["Zero padding is only the exact DCT-I product-to-sum quadrature identity; original physical model/damping values are unchanged.",
        "Pure real tensor Dirichlet Laplacian eigenmodes only; adding arbitrary POD vectors breaks this DCT-only assembly without new cross terms.",
        "G is a difference of positive matrices and can lose numerical PSD under cancellation. The recorded diagonal allowance is an unverified engineering margin, not a proof of floating-point upper enclosure.",
        "Availability/accuracy here are correlated development evidence on the same four train/validation models, not held-out generalization or runtime evidence.",
        "The derived Schur condition still requires gamma>0, invertible Ar and tau<1; failure means unavailable."],
    "primary_context": ["https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html",
        "https://www.math.purdue.edu/~shen7/pub/SWX16.pdf"],
    "previous_offline_evidence_sha256": sha256_file(PREVIOUS), "scratch_script_sha256": sha256_file(__file__),
    "provenance": provenance, "main_source_unchanged": source_digest(ROOT) == provenance["source_sha256"]}
OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
print(json.dumps({"output": str(OUTPUT), "availability": report["availability"],
    "maximum_multiplication_relative_error": max(c["relative_frobenius_error"] for c in identity_checks),
    "maximum_ar_relative_error": max(c["ar_relative_frobenius_error"] for c in rows),
    "maximum_gram_relative_error": max(c["gram_relative_frobenius_error"] for c in rows),
    "main_source_unchanged": report["main_source_unchanged"]}, indent=2))
