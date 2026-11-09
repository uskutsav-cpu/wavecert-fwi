"""Reproduce the archived75 manufactured controls without project data or PDEs."""

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "results/fractional_velocity_prior_development/portable_controls"
AUDITOR = "fractional_velocity_prior48_post_audit"
PRODUCER = "fractional_velocity_prior48_driver_candidate"
MANIFEST_SHA256 = "3385d0b238586e99312226b545b817b3fae4f8ea90ca6b5129ddc401c5cb1f0b"
LAYOUT_SHA256 = "e5487f5b424cf5ffd57223ec8ca8db48129b8f9dc84927bd884d5f38cb86f919"
CONTROL_FILES = (
    AUDITOR + "/test_scalar_audit_manufactured.py",
    AUDITOR + "/test_trace_checks.py",
    AUDITOR + "/test_producer_schema_interop.py",
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_fixture(root):
    descriptors = {
        "PORTABLE_CONTROL_FILE_MANIFEST.json": MANIFEST_SHA256,
        "PORTABLE_CONTROL_LAYOUT.md": LAYOUT_SHA256,
    }
    for name, expected in descriptors.items():
        assert _sha(root / AUDITOR / name) == expected, name
    manifest = json.loads((root / AUDITOR / "PORTABLE_CONTROL_FILE_MANIFEST.json").read_text())
    assert manifest["literal_sibling_directories"] == [AUDITOR, PRODUCER]
    assert manifest["required_file_count"] == len(manifest["required_files"]) == 16
    assert manifest["controls_expected"] == 75
    assert manifest["project_input_arrays_needed"] is False
    assert manifest["actual_study_outputs_needed"] is False
    for name in (AUDITOR, PRODUCER):
        assert (root / name).is_dir() and not (root / name).is_symlink(), name
    files, retained_bytes = {}, 0
    for row in manifest["required_files"]:
        relative = Path(row["path"])
        assert len(relative.parts) == 2 and relative.parts[0] in (AUDITOR, PRODUCER)
        assert row["path"] not in files, "duplicate manifest path"
        path = root / relative
        assert not path.is_symlink() and path.resolve().is_relative_to(root.resolve())
        assert path.is_file() and path.stat().st_size == row["bytes"], row["path"]
        actual = _sha(path)
        assert actual == row["sha256"], row["path"]
        files[row["path"]] = actual
        retained_bytes += row["bytes"]
    assert retained_bytes == manifest["required_total_bytes"] == 296711
    assert all(name in files for name in CONTROL_FILES)
    return dict(manifest_sha256=MANIFEST_SHA256, layout_sha256=LAYOUT_SHA256,
                computational_files=files, computational_bytes=retained_bytes)


def _xml_counts(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    counts = {name: sum(int(s.attrib.get(name, "0")) for s in suites)
              for name in ("tests", "failures", "errors", "skipped")}
    cases = [case for suite in suites for case in suite.findall("testcase")]
    identities = [(case.attrib.get("classname"), case.attrib.get("name")) for case in cases]
    return dict(counts=counts, case_records=len(cases), unique_cases=len(set(identities)),
                identity_sha256=hashlib.sha256(json.dumps(sorted(identities)).encode()).hexdigest(),
                all_cases_passed=not any(case.find(tag) is not None for case in cases
                                         for tag in ("failure", "error", "skipped")))


def _run_once(command, env, log, cwd):
    """A dedicated process group keeps timed-out exporter children tracked."""
    with log.open("xb") as stream:
        with subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                              stderr=subprocess.STDOUT, start_new_session=True) as child:
            try:
                return dict(actual_exit_code=child.wait(timeout=180), timed_out=False)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return dict(actual_exit_code=child.wait(), timed_out=True)
            except BaseException:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
                raise


def test_archived_prior48_manufactured_controls(tmp_path, capfd, record_property):
    output = tmp_path / "prior48_archival75"
    output.mkdir()  # Existing parent for child XML, log and pytest-owned basetemp.
    log, xml = output / "controls.log", output / "controls.xml"
    report = dict(check="prior48_archival_manufactured_reproduction",
                  sentinel="WAVECERT_PRIOR48_ARCHIVAL75_V1", status="started",
                  python=sys.version.split()[0], platform=platform.platform(),
                  expected_controls=75, project_arrays_decoded=0, native_PDE_calls=0,
                  output=str(output), full_log=str(log), XML=str(xml),
                  scope="Exact archived manufactured controls and scalar producer-schema interoperability; no project recovery, new physics or independent geology.")
    try:
        report["packages"] = {name: importlib.metadata.version(name)
                              for name in ("numpy", "scipy", "pytest")}
        before = _verified_fixture(FIXTURE)
        report["fixture_before"] = before
        bundle = output / "bundle"
        bundle.mkdir()
        relative_files = [*before["computational_files"],
                          AUDITOR + "/PORTABLE_CONTROL_FILE_MANIFEST.json",
                          AUDITOR + "/PORTABLE_CONTROL_LAYOUT.md"]
        for relative in relative_files:
            target = bundle / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(FIXTURE / relative, target)
        assert _verified_fixture(bundle) == before
        report["isolated_bundle"] = str(bundle)
        report["only_16_computational_files_and_2_pinned_descriptors_copied"] = True
        env = dict(os.environ)
        for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONPATH", "PYTHONHOME"):
            env.pop(name, None)
        env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONHASHSEED="0")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "1"
        command = [sys.executable, "-B", "-I", "-W", "error", "-m", "pytest", "-q", "-x",
                   "-W", "error", "-p", "no:cacheprovider", "-c", os.devnull,
                   "--import-mode=prepend", "--noconftest", "--confcutdir", str(bundle),
                   "--basetemp", str(output / "control_tmp"), "--junitxml", str(xml),
                   *CONTROL_FILES]
        report["command"] = command
        report.update(_run_once(command, env, log, bundle))
        report["log_sha256"] = _sha(log)
        if xml.is_file():
            report["JUnit"] = _xml_counts(xml)
            report["XML_sha256"] = _sha(xml)
        after = _verified_fixture(FIXTURE)
        report["fixture_unchanged"] = before == after == _verified_fixture(bundle)
        assert report["actual_exit_code"] == 0 and report["timed_out"] is False
        assert report["fixture_unchanged"] is True
        junit = report["JUnit"]
        assert junit["counts"] == dict(tests=75, failures=0, errors=0, skipped=0)
        assert junit["case_records"] == junit["unique_cases"] == 75
        assert junit["all_cases_passed"] is True
        report["status"] = "pass_all75"
    except BaseException as exc:
        report.update(status="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}")
        if log.is_file():
            report["log_sha256"] = _sha(log)
            report["log_tail"] = log.read_text(errors="replace")[-10000:]
        if not isinstance(exc, Exception):
            raise
    finally:
        encoded = json.dumps(report, allow_nan=False, sort_keys=True)
        (output / "report.json").write_text(encoded + "\n")
        record_property("prior48_archival_manufactured_reproduction", encoded)
        with capfd.disabled():
            print(encoded, flush=True)
    assert report["status"] == "pass_all75", json.dumps(report, allow_nan=False)
