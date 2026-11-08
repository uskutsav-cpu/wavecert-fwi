"""Portable rerun of exact archived manufactured controls, with no corpus."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "results/frequency_continuation_development/software"
HARNESS_SHA256 = "4715f767f50821798f35020ab0e70d94d8934ce9293c9e6cffbc64a5dc636b5e"
MANIFEST_SHA256 = "0932ce13b0565133cab94c186022f62244b51e69fd1384f00c8ef0a9217c3120"


def test_archived_frequency_manufactured_controls(tmp_path, capfd, record_property):
    source = ARCHIVE / "reproduce_manufactured.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == HARNESS_SHA256
    assert hashlib.sha256((ARCHIVE / "MANUFACTURED_REPRODUCTION_ASSETS.json").read_bytes()).hexdigest() == MANIFEST_SHA256
    output = tmp_path / "archival108"
    log = tmp_path / "harness_cli.log"
    command = [sys.executable, "-B", str(source), "--execute-manufactured-controls",
        "--repo", str(REPO), "--output", str(output)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    with log.open("xb") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, check=False)
    text = log.read_text()
    try:
        report = json.loads(text.splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        report = dict(status="harness_exception", actual_cli_exit_code=result.returncode,
            full_log=str(log), failure_receipt=str(output / "failure_receipt.json"))
    encoded = json.dumps(report, allow_nan=False, sort_keys=True)
    record_property("frequency_manufactured_reproduction", encoded)
    with capfd.disabled():
        print(encoded, flush=True)
    assert result.returncode == 0, text[-10000:]
    receipt = json.loads((output / "receipt.json").read_text())
    failures = []
    for row in receipt["suites"]:
        if row["status"] != "passed":
            failures.append(dict(suite=row["name"], full_log=row.get("log"),
                tail=Path(row["log"]).read_text()[-5000:] if row.get("log") else None))
    assert receipt["status"] == "pass_all108", json.dumps(failures, allow_nan=False)
    assert receipt["actual_passed"] == receipt["actual_observed"] == 108
    assert [r["passed"] for r in receipt["suites"]] == [17, 24, 15, 15, 37]
    assert receipt["production_before"] == receipt["production_after"]
    assert receipt["production_after"]["files"] == 161
