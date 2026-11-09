"""Run the unchanged probe controls in a fresh process with portable frozen metadata."""
from __future__ import annotations
import hashlib,importlib.util,json,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
EXPECTED={"protocol.json":"27e5a331d6bd9eff0dd6b4fa0ca1bc666c6014ed70ffb3da01821c62f83fdcbc","RUNTIME_SOURCE_PINS.json":"bfbc2e31cd4ff7ae2a0f0342dc3633c5e4dc0189daea0d06eaeb12de1040350f"}
for name,value in EXPECTED.items():
    if digest(HERE/"frozen_v2/worker"/name)!=value:raise ValueError("exact V2 metadata fixture required")
spec=importlib.util.spec_from_file_location("portable_manufactured_probe_controls",HERE/"test_import_identity_probe.py")
test=importlib.util.module_from_spec(spec);spec.loader.exec_module(test)
test.V2=HERE/"frozen_v2"
suite=unittest.defaultTestLoader.loadTestsFromTestCase(test.ImportProbeControls)
result=unittest.TextTestRunner(verbosity=2).run(suite)
print(json.dumps({"sentinel":"CHECKPOINT23_IMPORT_PROBE_V2_MANUFACTURED","tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skips":len(result.skipped),"runtime_import_attempts":test.BLOCKED_IMPORT_ATTEMPTS,"PDE_calls":0,"package_actions":0,"actual_import_probe":False,"test_source_sha256":digest(HERE/"test_import_identity_probe.py"),"probe_source_sha256":digest(HERE/"import_identity_probe.py"),"fixture_source_sha256":EXPECTED},sort_keys=True))
raise SystemExit(0 if result.wasSuccessful() and result.testsRun==25 and not test.BLOCKED_IMPORT_ATTEMPTS else 1)
