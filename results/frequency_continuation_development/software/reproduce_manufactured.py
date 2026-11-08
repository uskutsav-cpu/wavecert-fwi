#!/usr/bin/env python3
"""Reproduce five frozen manufactured suites; default is metadata only.

Archive inputs are immutable. A physical minimal checkout and exact staged
sources are copied into a new outer workspace; no corpus or study is executed.
"""
from __future__ import annotations

import argparse
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
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST_SHA256 = "0932ce13b0565133cab94c186022f62244b51e69fd1384f00c8ef0a9217c3120"
PRODUCTION_SHA256 = "9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    owned = False
    try:
        with temporary.open("x") as stream:
            owned = True
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if owned:
            temporary.unlink(missing_ok=True)


def production_snapshot(repo):
    paths = [repo / "pyproject.toml", *repo.glob("requirements*.txt"),
        *(repo / "src").rglob("*.py"),
        *(p for p in (repo / "configs").rglob("*") if p.is_file()),
        *(p for p in (repo / "scripts").rglob("*") if p.suffix in {".py", ".sh"}),
        *(repo / ".github/workflows").glob("*.yml")]
    hashes = {p.relative_to(repo).as_posix(): sha(p) for p in sorted(set(paths))}
    digest = hashlib.sha256()
    for name, value in hashes.items():
        digest.update(name.encode() + b"\0" + value.encode() + b"\0")
    return dict(files=len(hashes), sha256=digest.hexdigest(), file_sha256=hashes)


def manifest():
    path = HERE / "MANUFACTURED_REPRODUCTION_ASSETS.json"
    if sha(path) != MANIFEST_SHA256:
        raise ValueError("frozen reproduction asset manifest changed")
    value = json.loads(path.read_bytes())
    if value["expected_total"] != 108 or [s["expected_passed"] for s in value["suites"]] != [17, 24, 15, 15, 37]:
        raise ValueError("five actual-suite denominators must total108")
    if value["production"]["files"] != 161 or value["production"]["sha256"] != PRODUCTION_SHA256:
        raise ValueError("production9f/161 contract changed")
    return value


def archive_snapshot(value):
    result = {}
    for asset in value["archived_assets"]:
        path = (HERE / asset["source"]).resolve()
        if not path.is_relative_to(HERE) or sha(path) != asset["sha256"]:
            raise ValueError("frozen archive asset changed: " + asset["source"])
        result[asset["source"]] = asset["sha256"]
    return result


def copy_exact(source, destination, expected):
    if sha(source) != expected:
        raise ValueError("source hash mismatch before copy: " + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    if sha(destination) != expected:
        raise ValueError("copied bytes differ: " + str(destination))


def junit_counts(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    counts = dict(observed=len(cases), passed=0, failed=0, errors=0, skipped=0)
    identities = []
    for case in cases:
        identities.append((case.get("classname"), case.get("name")))
        if case.find("failure") is not None:
            counts["failed"] += 1
        elif case.find("error") is not None:
            counts["errors"] += 1
        elif case.find("skipped") is not None:
            counts["skipped"] += 1
        else:
            counts["passed"] += 1
    counts["unique_testcase_identities"] = len(set(identities)) == len(identities)
    return counts


def execute(repo, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    value = manifest()
    if (output.exists() or not output.parent.is_dir() or output.is_relative_to(repo)
            or output.is_relative_to(HERE)):
        raise ValueError("fresh outer output outside source checkout/archive required")
    output.mkdir()
    rows = [dict(**suite, status="unattempted", actual_exit_code=None) for suite in value["suites"]]
    status = dict(status="running", started_utc=utc(), expected_total=108, suites=rows,
        harness_sha256=sha(__file__), manifest_sha256=MANIFEST_SHA256,
        archive_root=str(HERE), source_checkout=str(repo), workspace=str(output),
        environment=dict(python=sys.version, platform=platform.platform()),
        scope="archival manufactured controls only; tiny synthetic API physics, no corpus/project96run/training",
        test_source_mutation=False, pytest_plugin_or_module_mutation=False)
    atomic_json(output / "status.json", status)
    process = active = None
    try:
        status["environment"]["packages"] = {
            n: importlib.metadata.version(n) for n in ("numpy", "scipy", "pytest")}
        original = production_snapshot(repo)
        if original != value["production"]:
            raise ValueError("specified checkout does not match frozen9f/161 file manifest")
        status["production_before"] = original
        status["archive_assets_before"] = archive_snapshot(value)
        fixture = output / "work/wavecert-fwi"
        fixture.mkdir(parents=True)
        for name, digest in original["file_sha256"].items():
            copy_exact(repo / name, fixture / name, digest)
        for name, digest in value["candidate_assets_sha256"].items():
            copy_exact(repo / name, fixture / name, digest)
        if production_snapshot(fixture) != original:
            raise ValueError("physical minimal checkout lost9f/161 identity")
        staged = output / "work/new-phase-staging"
        for asset in value["archived_assets"]:
            copy_exact(HERE / asset["source"], staged / asset["destination"], asset["sha256"])
        status["physical_fixture"] = dict(path=str(fixture), production=production_snapshot(fixture),
            copied_candidate_assets=value["candidate_assets_sha256"], corpus_files_copied=0,
            driver_protocol_context=value["driver_protocol_context"])
        atomic_json(output / "status.json", status)
        env = dict(os.environ)
        env.update(value["threads"])
        env.update(PYTHONPATH=str(fixture / "src"), PYTHONDONTWRITEBYTECODE="1",
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
            env.pop(name, None)
        status["controlled_child_environment"] = {k:env[k] for k in
            [*value["threads"], "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "PYTEST_DISABLE_PLUGIN_AUTOLOAD"]}
        (output / "pytest_tmp").mkdir()
        for active in rows:
            log = output / (active["name"] + ".log")
            xml = output / (active["name"] + ".xml")
            command = [sys.executable, "-B", "-m", "pytest",
                str(staged / active["directory"] / active["test_file"]),
                "-q", "-W", "error", "-p", "no:cacheprovider", "-o", "addopts=",
                "--rootdir", str(output), "--confcutdir", str(output),
                "--basetemp", str(output / "pytest_tmp" / active["name"]),
                "--junitxml", str(xml)]
            active.update(status="attempted", started_utc=utc(), command=command,
                log=str(log), junit=str(xml))
            atomic_json(output / "status.json", status)
            with log.open("xb") as stream:
                process = subprocess.Popen(command, cwd=output, env=env,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True)
                active["PID"] = process.pid
                atomic_json(output / "status.json", status)
                active["actual_exit_code"] = process.wait()
            active.update(finished_utc=utc(), log_sha256=sha(log))
            if xml.is_file():
                active.update(junit_sha256=sha(xml), **junit_counts(xml))
            good = (active["actual_exit_code"] == 0 and active.get("passed") == active["expected_passed"]
                and active.get("observed") == active["expected_passed"]
                and active.get("failed") == active.get("errors") == active.get("skipped") == 0
                and active.get("unique_testcase_identities") is True)
            active["status"] = "passed" if good else "failed_or_count_mismatch"
            process = None
            atomic_json(output / "status.json", status)
        status["production_after"] = production_snapshot(repo)
        status["archive_assets_after"] = archive_snapshot(value)
        status["harness_after_sha256"] = sha(__file__)
        status["candidate_assets_after"] = {n:sha(repo / n) for n in value["candidate_assets_sha256"]}
        if (status["production_after"] != original or production_snapshot(fixture) != original
                or status["archive_assets_after"] != status["archive_assets_before"]
                or status["harness_after_sha256"] != status["harness_sha256"]
                or sha(HERE / "MANUFACTURED_REPRODUCTION_ASSETS.json") != MANIFEST_SHA256
                or status["candidate_assets_after"] != value["candidate_assets_sha256"]
                or {n:sha(fixture / n) for n in value["candidate_assets_sha256"]}
                != value["candidate_assets_sha256"]):
            raise ValueError("source/archive/fixture identity changed during controls")
        for asset in value["archived_assets"]:
            if sha(staged / asset["destination"]) != asset["sha256"]:
                raise ValueError("copied staged source changed during controls")
        status.update(actual_passed=sum(r.get("passed", 0) for r in rows),
            actual_observed=sum(r.get("observed", 0) for r in rows), finished_utc=utc())
        status["status"] = "pass_all108" if all(r["status"] == "passed" for r in rows) else "failed_controls_retained"
        atomic_json(output / "status.json", status)
        atomic_json(output / "receipt.json", status)
        return status
    except BaseException as exc:
        if process is not None:
            try:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    code = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    code = process.wait()
                active["actual_exit_code"] = code
            except BaseException as cleanup_exc:
                active["cleanup_uncertainty"] = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
        if active is not None and active["status"] == "attempted":
            active["status"] = "interrupted_or_wrapper_failed"
        status.update(status="interrupted_or_wrapper_failed", error=f"{type(exc).__name__}: {exc}", finished_utc=utc())
        atomic_json(output / "status.json", status)
        atomic_json(output / "failure_receipt.json", status)
        raise


def compact_report(receipt):
    return dict(check="frequency_continuation_archival_manufactured_reproduction",
        status=receipt["status"], expected_total=108,
        actual_passed=receipt.get("actual_passed"), actual_observed=receipt.get("actual_observed"),
        production_sha256=receipt.get("production_after", {}).get("sha256"),
        production_files=receipt.get("production_after", {}).get("files"),
        environment=receipt["environment"],
        suites=[{k:r.get(k) for k in ("name", "expected_passed", "passed", "observed", "failed",
            "errors", "skipped", "actual_exit_code", "status", "PID")} for r in receipt["suites"]],
        workspace=receipt["workspace"], scope=receipt["scope"], scientific_claim=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-manufactured-controls", action="store_true")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.execute_manufactured_controls:
        print(json.dumps(dict(status="metadata_only_no_controls_or_corpus", expected_total=108,
            suite_counts=[17, 24, 15, 15, 37], manifest_sha256=MANIFEST_SHA256)))
        return 0
    if args.repo is None or args.output is None:
        parser.error("explicit checkout and fresh outer output required")
    receipt = execute(args.repo, args.output)
    print(json.dumps(compact_report(receipt), allow_nan=False, sort_keys=True), flush=True)
    return 0 if receipt["status"] == "pass_all108" else 2


if __name__ == "__main__":
    raise SystemExit(main())
