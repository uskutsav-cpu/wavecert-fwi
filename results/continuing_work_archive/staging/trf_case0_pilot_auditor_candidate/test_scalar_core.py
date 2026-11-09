"""Stdlib manufactured parser/math/cost controls; no native or project arrays."""
import copy
import hashlib
import importlib.util
import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("held_scalar_core", HERE / "scalar_core.py")
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)


def npy(shape, descr, values, extra=b""):
    header = repr({"descr": descr, "fortran_order": False, "shape": shape}).encode()
    header += b" " * ((-10 - len(header) - 1) % 64) + b"\n"
    if descr == "<c16":
        body = b"".join(struct.pack("<dd", v.real, v.imag) for v in values)
    else:
        body = struct.pack("<" + str(len(values)) + {"<f8": "d", "<f4": "f", "<i8": "q"}[descr], *values)
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + body + extra


def archive(path, members, compression=zipfile.ZIP_DEFLATED):
    with zipfile.ZipFile(path, "w", compression=compression) as output:
        for name, raw in members:
            output.writestr(name, raw)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trf_payload():
    initial, selected = [.25] * 2401, [.24] * 2401
    ih, sh = C.sha(C.f64_bytes(initial)), C.sha(C.f64_bytes(selected))
    return {"policy_valid": True, "error": None, "status": "budget_exhausted", "truth_used_for_policy": False,
        "initial_model": initial, "selected_model": selected, "Phi0": 10., "common_Phi_ratio": .8,
        "selected": {"elapsed_seconds": 2., "model_sha256": sh, "raw_Phi": 8.}, "exact_engine": None,
        "TRF_result_not_used_for_selection": {"result_x_sha256_not_selected": "a" * 64},
        "TRF_scalar_events": [
            {"sequence": 0, "kind": "residual_completed", "elapsed_seconds": 1., "model_sha256": ih},
            {"sequence": 1, "kind": "initial", "elapsed_seconds": 1., "model_sha256": ih, "raw_Phi": 10., "within_budget": True},
            {"sequence": 2, "kind": "residual_completed", "elapsed_seconds": 1.9, "model_sha256": sh},
            {"sequence": 3, "kind": "accepted_callback", "elapsed_seconds": 2., "model_sha256": sh, "raw_Phi": 8., "within_budget": True},
            {"sequence": 4, "kind": "residual_completed", "elapsed_seconds": 3., "model_sha256": "a" * 64},
            {"sequence": 5, "kind": "accepted_callback", "elapsed_seconds": 7.6, "model_sha256": "a" * 64, "raw_Phi": 7., "within_budget": False}]}


def cost_payload(sets=1):
    frequencies = [.25, .5, 1., 2., 2.5, 3., 3.5, 4.]
    sources, receivers = [250, 258, 266, 272, 280, 288], list(range(250, 289, 2))
    rows = []
    for index in range(8 * sets):
        rows.append({"index": index, "unknown_work": False, "factory_attempted": True, "factory_completed": True,
            "map_dispatch_attempted": True, "map_completed": True, "cold_owner_verified": True,
            "actual_complete_batch_counts_verified": True, "final_snapshot_completed": True,
            "receivers": [v for v in receivers if v not in sources], "point_identity": ["a" * 64] * 5,
            "frequency_hz": frequencies[index % 8], "before_physics": C.expected_counts(),
            "last_known_physics_prefix": C.expected_counts(factorizations=1, stability_rhs=14), "elapsed_seconds": .01})
    native = {"physics": C.expected_counts(factorizations=2, forward_rhs=12, adjoint_rhs=6), "oracle": {"contract_checks": 1}}
    preparation = C.expected_counts(factorizations=8, forward_rhs=48)
    payload = {"preparation_costs_disjoint": preparation, "preparation_prefix": {
        "observation_dispatches_attempted": 8, "observation_dispatches_completed": 8,
        "unknown_work": False, "observation_call_inflight": False, "last_known_observation_counts": preparation},
        "unknown_work": False, "preparation_work_unknown": False, "native_cumulative_once": native,
        "normalization_prefix": None, "component_prefix": {"native": {"unknown_since_prefix": False,
            "last_known_cumulative": copy.deepcopy(native)}, "component_counts": {"jacobian_requests": 1000,
            "jacobian_operator_cache_hits": 999}, "map_builder": {"unknown_work_present": False, "cold_owner_batches": rows}}}
    return payload, {"native_final_cumulative_0": copy.deepcopy(native)}, frequencies, receivers, sources


class ScalarControls(unittest.TestCase):
    def test_npz_real_complex_integer_both_compressions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manufactured.npz"
            schema = {"real": ((2,), "<f8"), "complex": ((2,), "<c16"), "integer": ((2,), "<i8")}
            members = [("real.npy", npy((2,), "<f8", [.25, .5])), ("complex.npy", npy((2,), "<c16", [1 + 2j, -3 + 4j])),
                ("integer.npy", npy((2,), "<i8", [0, 1]))]
            for compression in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                result = C.read_npz(path, schema, archive(path, members, compression))
                self.assertEqual(result["complex"].values, (1 + 2j, -3 + 4j))
                self.assertEqual(result["real"].values, (.25, .5))
                self.assertEqual(result["integer"].values, (0, 1))

    def test_npy_nonfinite_shape_trailing_and_header_execution_rejected(self):
        for raw in (npy((1,), "<f8", [math.nan]), npy((1,), "<c16", [complex(1, math.inf)]),
                npy((1,), "<f8", [1.], b"extra"), npy((2,), "<f8", [1., 2.])):
            with self.assertRaises(ValueError):
                C.decode_npy(raw, (1,), "<f8")
        malicious = b"__import__('os').system('false')\n"
        with self.assertRaises(ValueError):
            C.decode_npy(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(malicious)) + malicious, (1,), "<f8")

    def test_npz_hash_cap_symlink_and_wrong_members_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.npz"
            digest = archive(path, [("x.npy", npy((1,), "<f8", [1.]))])
            for schema, expected, cap in (({"x": ((1,), "<f8")}, "0" * 64, 10000),
                    ({"other": ((1,), "<f8")}, digest, 10000), ({"x": ((1,), "<f8")}, digest, 1)):
                with self.assertRaises(ValueError):
                    C.read_npz(path, schema, expected, byte_cap=cap)
            link = Path(directory) / "alias.npz"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                C.read_npz(link, {"x": ((1,), "<f8")}, digest)

    def test_inflation_and_truncated_payload_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expanded.npz"
            digest = archive(path, [("x.npy", npy((10000,), "<f8", [1.] * 10000))])
            with self.assertRaises(ValueError):
                C.read_npz(path, {"x": ((1,), "<f8")}, digest)
        with self.assertRaises(ValueError):
            C.decode_npy(npy((2,), "<f8", [1.]), (2,), "<f8")

    def test_constant_truth_analytic(self):
        self.assertEqual(C.reconstruct_truth([.25] * 576), (.25,) * 2401)

    def test_affine_velocity_bilinear_interior_and_edge_clamp(self):
        raw = [1 / (2 + .1 * r + .2 * c) ** 2 for r in range(24) for c in range(24)]
        truth = C.reconstruct_truth(raw)
        for actual, expected in ((truth[0], .25), (truth[24 * 49 + 24], 1 / 5.45 ** 2), (truth[-1], 1 / 8.9 ** 2)):
            self.assertAlmostEqual(actual, expected, places=14)

    def test_velocity_error_analytic_and_domain(self):
        self.assertEqual(C.velocity_error([.25] * 8, [.25] * 8), 0.)
        self.assertAlmostEqual(C.velocity_error([.16] * 8, [.25] * 8), .25)
        for values in ([0.], [-1.], [math.nan]):
            with self.assertRaises(ValueError):
                C.velocity_error(values, [.25])

    def test_direct_box_initializer_margin(self):
        C.checked_model([1 / 36] * 2401)
        for values, interior in (([1 / 36] * 2401, True), ([1.1] * 2401, False)):
            with self.assertRaises(ValueError):
                C.checked_model(values, interior=interior)

    def test_rejected_result_x_and_late_do_not_select(self):
        payload = trf_payload()
        self.assertEqual(C.audit_selection(payload)["selected_elapsed_seconds"], 2.)
        for changed in (dict(payload["selected"], elapsed_seconds=7.6), dict(payload["selected"], model_sha256="a" * 64)):
            value = copy.deepcopy(payload)
            value["selected"] = changed
            with self.assertRaises(ValueError):
                C.audit_selection(value)

    def test_unknown_partial_and_excess_counter_prefix_rejected(self):
        payload, context, frequencies, receivers, sources = cost_payload()
        for change in ("unknown", "partial", "native_prefix"):
            value = copy.deepcopy(payload)
            if change == "unknown":
                value["component_prefix"]["map_builder"]["unknown_work_present"] = True
            elif change == "partial":
                value["component_prefix"]["map_builder"]["cold_owner_batches"].pop()
            else:
                value["component_prefix"]["native"]["last_known_cumulative"]["physics"]["factorizations"] = 999
            with self.assertRaises(ValueError):
                C.audit_costs(value, context, frequencies, receivers, sources)

    def test_map_consumption_once_and_revisited_point_not_request_count(self):
        for sets in (0, 1, 2):
            payload, context, frequencies, receivers, sources = cost_payload(sets)
            result = C.audit_costs(payload, context, frequencies, receivers, sources)
            self.assertEqual(result["complete_distinct_map_sets"], sets)
            self.assertEqual(result["map_disjoint"]["factorizations"], 8 * sets)
            self.assertEqual(result["map_disjoint"]["stability_rhs"], 112 * sets)
            self.assertEqual(result["consumed_total"]["factorizations"], 10 + 8 * sets)
            self.assertFalse(result["overlapping_snapshots_summed"])

    def test_default_actual_isolated_module_inventory(self):
        script = "import runpy,sys;runpy.run_path(sys.argv[1],run_name='__main__');assert not any(k.split('.')[0] in ('numpy','scipy','wavecert') for k in sys.modules)"
        result = subprocess.run([sys.executable, "-B", "-I", "-Werror", "-c", script, str(HERE / "scalar_core.py")],
            capture_output=True, text=True, timeout=10, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)["execution_authorized"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
