"""Generated attribution notice checks only; no acquisition or velocity input."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bp2004_v2_notice_only", root / "collect_bp2004.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def forbidden_network(*args, **kwargs):
    raise AssertionError("No network is permitted in the notice validation")


module.urllib.request.urlopen = forbidden_network
protocol, protocol_raw = module.load_protocol(root / "protocol.json")
notice = module.modification_readme(protocol)
checks = {
    "correct_title": "The 2004 BP velocity benchmark" in notice,
    "incorrect_title_absent": "Complex subsalt imaging challenges" not in notice,
    "original_doi": "doi:10.3997/2214-4609-pdb.1.B035" in notice,
    "BP_and_Billette_acknowledgment": "Data courtesy of BP. Acknowledge BP and Frederic Billette." in notice,
    "original_documentation_preserved": "original_BP2004_READMES.pdf" in notice,
    "original_terms_no_new_license": "These files inherit its terms; no new data license is asserted." in notice,
    "one_composite_ancestor_notice": "ONE BP2004 composite synthetic numerical ancestor" in notice,
    "no_independent_sites": "not independent parents/sites" in notice,
    "protocol_single_ancestor": protocol["independent_ancestor_count"] == 1 and not protocol["crops_are_independent_parents"],
    "ancestor_identity": "Ancestor: " + protocol["ancestor_id"] in notice,
    "unchanged_protocol_sha256": hashlib.sha256(protocol_raw).hexdigest() == "a84ce64dcac7f93f035007daf313cb2ab7bde713800d10d9d37d812f12c836f4",
    "unchanged_original_document_sha256": hashlib.sha256((root / "original_BP2004_READMES.pdf").read_bytes()).hexdigest() == "e29b003fc2161c44d2b9f04448195e58ebe6f476a195b31c3be77440adb60e24",
}
(root / "validation_01" / "README_Modification.generated.md").write_text(notice)
report = {
    "scope": "Generated notice and original terms/one-ancestor metadata only; no fetch, velocity input or PDE.",
    "python": sys.version,
    "numpy": module.np.__version__,
    "checks": checks,
    "all_checks_passed": all(checks.values()),
}
print(json.dumps(report, indent=2, sort_keys=True))
if not all(checks.values()):
    raise SystemExit(1)
