"""Independent saved-array arithmetic/metadata audit. Never runs a PDE."""
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from wavecert.provenance import source_digest

REPO = Path(__file__).resolve().parents[1] / "wavecert-fwi"
RUN = Path(__file__).resolve().parents[1] / "devito_boundary_verification_run_01"
OUT = Path(__file__).with_name("devito_boundary_post_audit.json")

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def array_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()

def euclidean(values):
    # Separate stable norm implementation, not the runner or its NumPy norm.
    return math.hypot(*np.asarray(values).ravel().tolist())

def close(actual, expected):
    return math.isclose(float(actual), float(expected), rel_tol=5e-14, abs_tol=0.)

receipt = json.loads((RUN / "complete_receipt.json").read_text())
status = json.loads((RUN / "status.json").read_text())
protocol = json.loads((RUN / "protocol.json").read_text())
metrics = json.loads((RUN / "metrics.json").read_text())
assert {k: v for k, v in receipt.items() if k != "artifacts"} == status
assert receipt["status"] == "complete_software_verification"
assert len(receipt["artifacts"]) == 11
for filename, expected in receipt["artifacts"].items():
    assert digest(RUN / filename) == expected, filename
assert digest(RUN / "runner_source.py") == receipt["runner_sha256"] == "f8a522f34a3ce1be68293944dbef362ea86ae9731f28fd802eabb0434fb57cb1"
assert digest(RUN / "protocol.json") == receipt["protocol_sha256"] == "a83f2de93543b16af428d1fbe562f4d7db8d9df982b203a04bb0d445e93306b8"
assert receipt["source_before"] == receipt["source_after"]
assert source_digest(REPO) == receipt["source_before"]["production_source_sha256"] == "9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da"
for record in receipt["source_before"]["installed_example_modules"].values():
    assert digest(record["path"]) == record["sha256"]
for name, expected in receipt["source_before"]["native_module_sha256"].items():
    relative = "physics/devito_backend.py" if name == "backend" else "provenance.py"
    assert digest(REPO / "src/wavecert" / relative) == expected
assert [row["case_id"] for row in receipt["case_records"]] == [c["id"] for c in protocol["cases"]]
assert len(receipt["case_records"]) == 7
common = protocol["common"]
times = np.arange(0., 1101., 1.)
r = np.pi * .005 * (times - 200.)
wavelet = ((1 - 2. * r ** 2) * np.exp(-r ** 2))[:, None]
raw, case_audits = {}, []
for entry, case in zip(receipt["case_records"], protocol["cases"]):
    name = case["id"]
    assert entry["status"] == "complete" and all(entry["software_invariants"].values())
    assert len(entry["software_invariants"]) == 17
    assert entry["raw_artifact"] == name + "_raw.npz"
    assert digest(RUN / entry["raw_artifact"]) == entry["raw_sha256"]
    with np.load(RUN / entry["raw_artifact"], allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    assert len(arrays) == 11 and all(np.all(np.isfinite(a)) for a in arrays.values())
    assert all(a.dtype == np.float64 for a in arrays.values())
    for key in ["times_before_ms", "times_after_ms", "source_times_before_ms", "source_times_after_ms", "receiver_times_ms"]:
        assert np.array_equal(arrays[key], times), (name, key)
    assert np.array_equal(arrays["source_before"], wavelet)
    assert np.array_equal(arrays["source_after"], wavelet)
    assert arrays["receiver"].shape == (1101, 1)
    translation = np.array(case["translation_m"])
    source, receiver = np.array([300., 300.]) + translation, np.array([350., 300.]) + translation
    assert np.array_equal(arrays["source_coordinates_before_m"], source[None, :])
    assert np.array_equal(arrays["source_coordinates_after_m"], source[None, :])
    assert np.array_equal(arrays["receiver_coordinates_m"], receiver[None, :])
    rec = entry["record"]
    assert rec["physics_before"] == rec["physics_after"]
    phys, field = rec["physics_before"], rec["wavefield"]
    n = case["shape_x_z"][0] + 2 * case["nbl"]
    assert phys["grid_shape"] == [n, n] and phys["grid_spacing"] == [10., 10.]
    assert phys["grid_origin"] == [-10. * case["nbl"]] * 2
    assert phys["vp"]["sha256"] == array_digest(np.full((n, n), 2., dtype=np.float64))
    assert phys["derived_squared_slowness"]["sha256"] == array_digest(np.full((n, n), .25, dtype=np.float64))
    assert rec["dt_before_ms"] == rec["dt_after_ms"] == 1. <= phys["critical_dt_ms"]
    assert phys["critical_dt_ms"] == 2.773
    assert np.array_equal(rec["computational_source_index"], ((source + 10. * case["nbl"]) / 10.)[None, :])
    assert np.array_equal(rec["computational_receiver_index"], ((receiver + 10. * case["nbl"]) / 10.)[None, :])
    assert field["logical_shape"] == [3, n, n] and field["logical_bytes"] == 3 * n * n * 8
    assert field["halo_shape"] == [3, n + 8, n + 8] and field["halo_bytes"] == 3 * (n + 8) ** 2 * 8
    assert field["dtype"] == "float64" and not field["full_time_history_saved"]
    paths = []
    padded_paths = []
    for axis in (0, 1):
        for boundary in (0., float((case["shape_x_z"][axis] - 1) * 10.)):
            image = source.copy(); image[axis] = 2 * boundary - source[axis]
            paths.append(math.dist(receiver, image))
        for boundary in (-10. * case["nbl"], (case["shape_x_z"][axis] - 1) * 10. + 10. * case["nbl"]):
            image = source.copy(); image[axis] = 2 * boundary - source[axis]
            padded_paths.append(math.dist(receiver, image))
    assert all(close(a, b) for a, b in zip(paths, rec["geometry"]["nominal_physical_image_path_lengths_m"]))
    assert all(close(a, b) for a, b in zip(padded_paths, rec["geometry"]["nominal_padded_image_path_lengths_m"]))
    assert close(min(paths) / 2., rec["geometry"]["nearest_physical_image_travel_ms"])
    assert close(min(padded_paths) / 2., rec["geometry"]["nearest_padded_image_travel_ms"])
    raw[name] = arrays
    case_audits.append(dict(case_id=name, raw_sha256=entry["raw_sha256"], source_sha256=array_digest(arrays["source_before"]), all_saved_arrays_finite_float64=True, exact_time_wavelet_and_coordinates=True, constant_vp_and_m_fingerprints_reconstructed=True, recorded_grid_CFL_three_slice_metadata_verified=True, logical_bytes=field["logical_bytes"], halo_bytes=field["halo_bytes"]))
expected_against = set(raw) - {"reference_321"}
assert set(metrics["against_reference_321"]) == expected_against
assert set(metrics["matched_exterior_contrasts"]) == {"small_nbl20", "small_nbl40"}
comparisons = []
scalar_checks = 0
for group, group_metrics in metrics.items():
    for name, reported in group_metrics.items():
        reference_name = "reference_321" if group == "against_reference_321" else {"small_nbl20": "matched_exterior_20_undamped", "small_nbl40": "matched_exterior_40_undamped"}[name]
        a, ref = raw[name]["receiver"], raw[reference_name]["receiver"]
        early = times <= 250.
        difference = a - ref
        values = {}
        for key, num, den in [("early_relative", euclidean(difference[early]), euclidean(ref[early])), ("late_mismatch_over_full_reference", euclidean(difference[~early]), euclidean(ref)), ("full_relative", euclidean(difference), euclidean(ref))]:
            value = None if den == 0 else num / den
            assert close(num, reported[key]["numerator"]), (group, name, key, "numerator")
            assert close(den, reported[key]["denominator"]), (group, name, key, "denominator")
            assert reported[key]["denominator_zero"] == (den == 0)
            assert value is None and reported[key]["value"] is None or value is not None and close(value, reported[key]["value"])
            scalar_checks += 3
            values[key] = dict(numerator=num, denominator=den, value=value)
        assert reported["early_samples"] == 251 and reported["late_samples"] == 850
        assert reported["quality_threshold"] is None and not reported["physics_pass_claim"]
        comparisons.append(dict(group=group, case_id=name, reference_case_id=reference_name, recomputed=values, all_reported_numbers_agree=True))
assert receipt["quality_thresholds"] is None and not receipt["boundary_fidelity_pass_claim"] and not receipt["goal2_completion_claim"]
result = dict(status="PASS_saved_array_arithmetic_and_metadata_audit_only", recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), reviewer="separate AI /root/candidate_final_review; metadata/hash subreview /root/candidate_final_review/verification_helpers_review", audit_source_sha256=digest(__file__), complete_receipt_sha256=digest(RUN / "complete_receipt.json"), production_source_sha256=source_digest(REPO), artifact_hashes_verified=11, case_count=7, recorded_invariant_count=119, metric_comparisons=8, reported_metric_scalars_recomputed=scalar_checks, independent_norm_implementation="Python math.hypot; no runner metric helper imported", comparison_tolerance=dict(relative=5e-14, absolute=0.), case_audits=case_audits, comparisons=comparisons, maximum_recorded_logical_wavefield_bytes=max(c["logical_bytes"] for c in case_audits), maximum_recorded_halo_wavefield_bytes=max(c["halo_bytes"] for c in case_audits), source_wavelet_shared_exactly=True, times_and_actual_source_receiver_axes_shared_exactly=True, PDE_rerun=False, main_source_edits=False, student_final_research_design=False, scientific_goal2_complete=False, interpretation_limits=["These are conditional finite-record trace errors, not intrinsic reflection coefficients or energy-flux reflection ratios", "Same-code enlarged references agree numerically for this fixture, not independent implementation or continuum validation", "Damping field values and full wavefield buffers are not saved; their before/after fingerprints and recorded memory arithmetic are verified, not independently replayed", "Recorded three-slice buffer sizes are not peak process memory", "One homogeneous geometry/source band does not establish independent geology, certificate transfer, free-surface/PML validity, matched-cost inversion advantage or full goal completion", "Separate AI raw arithmetic checks are not human reproduction or student scientific interpretation"])
assert not OUT.exists()
OUT.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(dict(status=result["status"], scalar_checks=scalar_checks, output=str(OUT), sha256=digest(OUT))))
