"""Stdlib manufactured audit controls; no native or numerical package imports."""
import argparse
import importlib.util
import io
import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_cached_j_scalar_audit_test", HERE/"audit_retained.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def npy(shape, dtype, values, *, fortran=False):
    header = repr(dict(descr=dtype, fortran_order=fortran, shape=tuple(shape))).encode("ascii")+b"\n"
    if dtype == "<c16":
        body = b"".join(struct.pack("<dd", complex(v).real, complex(v).imag) for v in values)
    else:
        body = b"".join(struct.pack("<d", v) for v in values)
    return b"\x93NUMPY\x01\x00"+struct.pack("<H", len(header))+header+body


def archive(path, members, compression=zipfile.ZIP_STORED):
    with zipfile.ZipFile(path, "w", compression=compression) as output:
        for name, value in members:
            output.writestr(name, value)


def cost_rows():
    rows = {"observation_generation": (8, 48, 0, 0, 0), "policy_template": (0, 0, 0, 0, 0),
        "public_jvp": (8, 48, 48, 0, 0), "public_vjp": (8, 48, 0, 48, 0),
        "complex_source_validation": (8, 48, 0, 0, 0), "full_receiver_map_validation": (8, 0, 0, 0, 160),
        "foreign_template": (0, 0, 0, 0, 0), "actual_native_policy": (16, 96, 0, 0, 0),
        "actual_foreign_native_validation": (0, 0, 0, 0, 0), "component_cold_map_batches": (8, 0, 0, 0, 112)}
    rows.update({f"fd_{step}_{sign}": (8, 48, 0, 0, 0) for step in (1e-4, 5e-5) for sign in (-1, 1)})
    return {name: dict(zip(A.PHYSICS_KEYS, values, strict=True)) for name, values in rows.items()}


def control_packet(root):
    lifecycle = dict(status="MANUFACTURED_LIFECYCLE_PASS", tests=14, failures=0, errors=0,
        source_files=A.SOURCE_PINS, control_source_sha256="2526a76b539af952a9a9707f6723493ad571723012ed48c8e3738b8432a80414",
        numerical_imports=0, native_owners_constructed=0, PDE_calls=0)
    lines = [f"test_control_{i} (manufactured) ... ok" for i in range(14)]
    (root/"lifecycle14.log").write_text("\n".join(lines)+"\n"+json.dumps(lifecycle)+"\n")
    summary = dict(lifecycle14=lifecycle, component44=dict(tests=44, failures=0, errors=0, skipped=0), total_mandatory_tests=58)
    (root/"mandatory58_controls.json").write_text(json.dumps(summary))
    xml = ET.Element("testsuites")
    suite = ET.SubElement(xml, "testsuite", tests="44", failures="0", errors="0", skipped="0")
    for i in range(44):
        ET.SubElement(suite, "testcase", classname="manufactured", name=f"case_{i}")
    ET.ElementTree(xml).write(root/"component44.xml")
    return xml


class AuditControls(unittest.TestCase):
    def test_default_subprocess_blocks_numerical_imports_and_ignores_future_path(self):
        wrapper = '''import runpy,sys
def block(event,args):
 if event=="import" and args[0].split(".")[0] in {"numpy","scipy","wavecert","cached_jacobian","public_map_builder"}:
  raise RuntimeError("numeric import forbidden")
sys.addaudithook(block)
sys.argv=[sys.argv[1],"--packet","FUTURE_UNREAD_PACKET"]
runpy.run_path(sys.argv[0],run_name="__main__")
'''
        with tempfile.TemporaryDirectory() as root:
            completed = subprocess.run([sys.executable, "-B", "-I", "-c", wrapper, str(HERE/"audit_retained.py")],
                cwd=root, text=True, capture_output=True, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            value = json.loads(completed.stdout)
            self.assertEqual((value["arrays"], value["mandatory_tests"], value["native_dispatches"]), (14, 58, 160))
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_float_and_complex_payload_decode_and_hash(self):
        for dtype, values in [("<f8", [1., -2.]), ("<c16", [1+2j, -3+.5j])]:
            value = A.decode_npy(npy((2,), dtype, values), (2,), dtype)
            self.assertEqual(value["values"], tuple(values))
            self.assertEqual(value["bytes"], 16 if dtype == "<f8" else 32)

    def test_npy_dtype_order_shape_nonfinite_and_truncation_rejected(self):
        for raw, shape, dtype in [(npy((2,), "<f8", [1., 2.], fortran=True), (2,), "<f8"),
                (npy((2,), "<f8", [1., 2.]), (3,), "<f8"),
                (npy((1,), "<f8", [math.nan]), (1,), "<f8"),
                (npy((1,), "<c16", [complex(1, math.inf)]), (1,), "<c16"),
                (npy((2,), "<f8", [1., 2.])[:-1], (2,), "<f8"),
                (npy((True,), "<f8", [1.]), (1,), "<f8")]:
            with self.subTest(shape=shape, dtype=dtype), self.assertRaises(ValueError):
                A.decode_npy(raw, shape, dtype)

    def test_npz_exact_members_and_real_complex_crc_read(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"toy.npz"
            archive(path, [("r.npy", npy((2,), "<f8", [1., 2.])), ("c.npy", npy((1,), "<c16", [3+4j]))])
            data = A.decode_npz(path, {"r": ((2,), "<f8"), "c": ((1,), "<c16")})
            self.assertEqual(data["c"]["values"], (3+4j,))

    def test_npz_compression_duplicate_or_extra_members_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"toy.npz"
            member = ("r.npy", npy((1,), "<f8", [1.]))
            cases = [([member], zipfile.ZIP_DEFLATED), ([member, member], zipfile.ZIP_STORED),
                ([member, ("extra.npy", member[1])], zipfile.ZIP_STORED)]
            for members, compression in cases:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Duplicate name:.*", category=UserWarning)
                    archive(path, members, compression)
                with self.assertRaises(ValueError):
                    A.decode_npz(path, {"r": ((1,), "<f8")})

    def test_corrupt_crc_or_truncated_archive_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"toy.npz"
            archive(path, [("r.npy", npy((2,), "<f8", [1., 2.]))])
            raw = bytearray(path.read_bytes())
            location = raw.find(struct.pack("<d", 2.))
            self.assertGreater(location, 0)
            raw[location] ^= 1
            path.write_bytes(raw)
            with self.assertRaises(zipfile.BadZipFile):
                A.decode_npz(path, {"r": ((2,), "<f8")})
            path.write_bytes(raw[:20])
            with self.assertRaises(zipfile.BadZipFile):
                A.decode_npz(path, {"r": ((2,), "<f8")})

    def test_scalar_real_adjoint_and_sign_error(self):
        p, z = (2., 3.), (4., -1., 2.)
        jp = (2*p[0]+p[1], -p[0]+3*p[1], 4*p[0]-2*p[1])
        star = (2*z[0]-z[1]+4*z[2], z[0]+3*z[1]-2*z[2])
        self.assertEqual(A.real_dot(z, jp), A.real_dot(p, star))
        self.assertNotEqual(A.real_dot(z, jp), A.real_dot(p, (-star[0], star[1])))

    def test_complex_covered_row_requires_plain_transpose_amplitude_division(self):
        row, alpha = (1+2j, -3+.5j, .25-4j), 2-.5j
        u = tuple(alpha*1600*v for v in row)
        correct = tuple(v/(alpha*1600) for v in u)
        wrong = tuple(v.conjugate()/(alpha*1600) for v in u)
        self.assertLess(A.relative(correct, row), 1e-14)
        self.assertGreater(A.relative(wrong, row), .1)

    def test_both_centered_steps_and_objective_scalar(self):
        residual, jp = (1., 2., -1.), (.3, -.2, .1)
        dd = A.real_dot(residual, jp)
        errors = []
        for step in (1e-4, 5e-5):
            minus = tuple(a-step*b for a, b in zip(residual, jp, strict=True))
            plus = tuple(a+step*b for a, b in zip(residual, jp, strict=True))
            errors.append(A.relative(A.centered_fd(minus, plus, step), jp))
            self.assertAlmostEqual(A.fd_objective(minus, plus, step), dd, places=9)
        self.assertEqual(len(errors), 2)
        self.assertTrue(all(e < 1e-9 for e in errors))
        self.assertGreater(A.relative(A.centered_fd(minus, tuple(v+.01 for v in plus), step), jp), 1e-4)

    def test_disjoint_costs_exact_expected_not_double_counted(self):
        expected = dict(factorizations=96, forward_rhs=480, tangent_rhs=48, adjoint_rhs=48, stability_rhs=272)
        rows, checks = cost_rows(), A.Checks()
        self.assertEqual(A.check_counts(rows, expected, checks), expected)
        self.assertEqual(checks.issues, [])
        rows["actual_native_policy"]["factorizations"] += 8
        A.check_counts(rows, expected, checks)
        self.assertTrue(checks.issues)

    def test_unknown_bool_or_extra_cost_owner_cannot_be_zero(self):
        expected = dict(factorizations=96, forward_rhs=480, tangent_rhs=48, adjoint_rhs=48, stability_rhs=272)
        for mutation in (lambda r: r["actual_native_policy"].update(forward_rhs=True),
                lambda r: r.update(actual_native_policy=None), lambda r: r.update(extra={})): 
            rows, checks = cost_rows(), A.Checks()
            mutation(rows)
            A.check_counts(rows, expected, checks)
            self.assertTrue(checks.issues)

    def test_public_dispatch_plan160_includes_two_expected_rejections(self):
        labels = A.expected_dispatch_labels()
        self.assertEqual(len(labels), 160)
        self.assertEqual(labels[-2:], ["expected_model_rejection", "foreign_owner_validation"])
        self.assertEqual(labels.count("public_jvp"), 48)
        self.assertEqual(labels.count("public_vjp"), 48)

    def test_actual_wait_unknown_signalled_or_bool_cannot_admit(self):
        row = dict(label="native", actual_wait_returncode=0, wait_observed=True, requested_new_session=True,
            launch_state="HANDLE_ASSIGNED", terminal_state_known=True, owned_process_group_alive_after_cleanup=False,
            deferred_parent_signals=[], actual_pid=123, observed_pgid=123, external_spawn_through_wait_wall_seconds=2.)
        checks = A.Checks()
        A.wait_check(row, "native", checks)
        self.assertEqual(checks.issues, [])
        for change in [dict(actual_wait_returncode=-15), dict(actual_wait_returncode=None), dict(actual_wait_returncode=False),
                dict(terminal_state_known=False), dict(observed_pgid=None), dict(launch_state="OUTCOME_UNKNOWN"), dict(deferred_parent_signals=["SIGINT"])]:
            checks = A.Checks()
            A.wait_check(dict(row, **change), "native", checks)
            self.assertTrue(checks.issues)

    def test_metadata_failure_prevents_any_npz_decode(self):
        with tempfile.TemporaryDirectory() as root:
            packet = Path(root)/"packet"
            packet.mkdir()
            args = argparse.Namespace(packet=packet, output=Path(root)/"fresh", expected_source_sha256=A.sha(HERE/"audit_retained.py"),
                expected_criteria_sha256=A.sha(HERE/"criteria.json"))
            def invalid(_args, _packet, _protocol, checks):
                checks.require(False, "manufactured_missing_actual58_or_terminal")
                return {}, {}
            with patch.object(A, "metadata_preflight", invalid), patch.object(A, "decode_npz", side_effect=AssertionError("must not decode")):
                self.assertEqual(A.evaluate(args), 2)
            result = json.loads((args.output/"report.json").read_bytes())
            self.assertFalse(result["verified_credit"])
            self.assertEqual(result["npz_arrays_decoded"], 0)

    def test_mandatory58_requires14_actual_names_and44_xml_cases_without_skips(self):
        with tempfile.TemporaryDirectory() as root:
            packet = Path(root)
            xml = control_packet(packet)
            checks = A.Checks()
            A.mandatory_controls(packet, checks)
            self.assertEqual(checks.issues, [])
            ET.SubElement(xml.find("testsuite/testcase"), "skipped")
            ET.ElementTree(xml).write(packet/"component44.xml")
            checks = A.Checks()
            A.mandatory_controls(packet, checks)
            self.assertTrue(checks.issues)

    def test_mandatory_controls_missing_actual_test_or_wrong_source_not_admitted(self):
        with tempfile.TemporaryDirectory() as root:
            packet = Path(root)
            control_packet(packet)
            text = (packet/"lifecycle14.log").read_text().replace("test_control_0 (manufactured) ... ok\n", "")
            (packet/"lifecycle14.log").write_text(text)
            checks = A.Checks()
            A.mandatory_controls(packet, checks)
            self.assertTrue(checks.issues)
            control_packet(packet)
            text = (packet/"lifecycle14.log").read_text().replace("2526a76b539af952a9a9707f6723493ad571723012ed48c8e3738b8432a80414", "0"*64)
            (packet/"lifecycle14.log").write_text(text)
            checks = A.Checks()
            A.mandatory_controls(packet, checks)
            self.assertTrue(checks.issues)

    def test_inventory_missing_omitted_and_path_escape_fail(self):
        with tempfile.TemporaryDirectory() as root:
            packet = Path(root)
            (packet/"one.log").write_text("manufactured")
            inventory = dict(files={"one.log": dict(bytes=12, sha256=A.sha(packet/"one.log"), included=True)},
                any_omitted=False, inventory_excludes_itself=True, copied_bytes=12, artifact_cap_bytes=A.MAX_PACKET)
            (packet/"artifact_inventory.json").write_text(json.dumps(inventory))
            checks = A.Checks()
            A.inventory_check(packet, inventory, checks)
            self.assertEqual(checks.issues, [])
            inventory["files"]["one.log"]["included"] = False
            A.inventory_check(packet, inventory, checks)
            self.assertTrue(checks.issues)
            inventory["files"]["../escape.log"] = inventory["files"].pop("one.log")
            with self.assertRaises(ValueError):
                A.inventory_check(packet, inventory, checks)

    def test_path_escape_and_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            with self.assertRaises(ValueError):
                A.safe_file(base, "../escape")
            path = base/"bad.json"
            path.write_text('{"bad":NaN}')
            with self.assertRaises(ValueError):
                A.read_json(path)

    def test_full_manufactured_scalar_shapes_packing_twoFD_and96rows(self):
        # Algebraic manufactured arrays only: these constant U values are NOT
        # claimed to solve any wave equation or to emulate the native worker.
        protocol = A.read_json(HERE/"native_protocol.json")
        phi0 = 480.
        m0 = (.25,)*2401
        p = tuple(.25*(1+.2*math.sin(2*math.pi*i/48)*math.cos(2*math.pi*j/48))/1.2 for i in range(49) for j in range(49))
        raw_z = tuple(math.sin(.017*k)+math.cos(.013*k) for k in range(1, 1921))
        z = tuple(v/A.norm(raw_z) for v in raw_z)
        f0 = (1/math.sqrt(phi0),)*960+(0.,)*960
        jp = f0
        star_scale = A.real_dot(z, jp)/A.real_dot(p, p)
        star = tuple(star_scale*v for v in p)
        alpha = tuple(complex(*v) for v in protocol["complex_source_multipliers"])
        payload = dict(m0=m0, observation_model=(1/2.05**2,)*2401, direction=p, dual=z,
            observations=(0j,)*960, initial_residual=f0, cached_jvp=jp, reference_jvp=jp,
            cached_vjp=star, reference_vjp=star, native_U=(1+0j,)*(8*2401*6),
            reference_C_covered=(1/1600+0j,)*(8*6*2401), complex_U=tuple(alpha[s] for _ in range(8*2401) for s in range(6)),
            fd_residual_vectors=tuple(v*(1+sign*step) for step in (1e-4, 5e-5) for sign in (-1, 1) for v in f0))
        arrays = {key: dict(values=value) for key, value in payload.items()}
        arrays["m0"]["data_sha256"] = "manufactured_exact_mass_hash"
        result = dict(native_initial_header=dict(objective=phi0, model_sha256="manufactured_exact_mass_hash"))
        checks = A.Checks()
        answer = A.numerical_audit(arrays, result, protocol, checks)
        self.assertEqual(checks.issues, [])
        self.assertEqual(len(answer["FD"]), 2)
        self.assertEqual(len(answer["covered_rows"]), 48)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuditControls))
    print(json.dumps(dict(status="STDLIB_MANUFACTURED_AUDIT_PASS" if result.wasSuccessful() else "FAIL",
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors), native_calls=0,
        source_sha256=A.sha(HERE/"audit_retained.py"), control_source_sha256=A.sha(Path(__file__)))))
    raise SystemExit(0 if result.wasSuccessful() else 2)
