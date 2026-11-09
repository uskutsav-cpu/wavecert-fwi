"""Read/hash/JSON-only review; never import or execute a candidate module."""
import difflib
import hashlib
import json
from datetime import datetime, timezone
from email.parser import BytesParser
from pathlib import Path


BASE = Path(__file__).resolve().parent.parent
OLD = BASE / "checkpoint_pilot23_controller_candidate"
NEW = BASE / "checkpoint_pilot23_runtime_contract_v2_candidate"
ENV = BASE / "checkpoint_pilot23_environment_preparation_candidate"
WORKER = BASE / "devito_checkpoint_pilot_candidate"
OUT = Path(__file__).resolve().parent / "BOUND_REVIEW.json"
seen = {}
issues = []
checks = 0


def require(ok, label):
    global checks
    checks += 1
    if not ok:
        issues.append(label)


def raw(path):
    require(path.is_file() and not path.is_symlink(), "regular_file:" + str(path))
    value = path.read_bytes()
    seen[str(path)] = hashlib.sha256(value).hexdigest()
    return value


def obj(path):
    return json.loads(raw(path))


def sha(path):
    return hashlib.sha256(raw(path)).hexdigest()


def inventory(directory, exclusions):
    manifest = obj(directory / "STAGED_FILE_MANIFEST.json")
    for name, entry in manifest["files"].items():
        value = raw(directory / name)
        require(hashlib.sha256(value).hexdigest() == entry["sha256"], "manifest_sha:" + str(directory / name))
        require(len(value) == entry["bytes"], "manifest_bytes:" + str(directory / name))
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    require(actual == set(manifest["files"]) | set(exclusions), "exact_inventory:" + str(directory))
    require(len(manifest["files"]) == manifest["file_count"], "inventory_count:" + str(directory))
    return manifest["file_count"]


require(sha(NEW / "FINAL_HANDOFF.json") == "518a7cf93386769b779d5083a2d5ed1e1d38c95cd6de7fb55eb937a2b5bb8860", "root_handoff_pin")
handoff = obj(NEW / "FINAL_HANDOFF.json")
for name, expected in handoff["evidence_sha256"].items():
    require(sha(NEW / name) == expected, "v2_handoff_evidence:" + name)
require(sha(OLD / "STAGED_FILE_MANIFEST.json") == "1a9f08cb5d7cef93440b428eb342f30208e64b171005ce155751061170209769", "v1_controller_manifest_pin")
old_handoff = obj(OLD / "FINAL_REVIEW_HANDOFF.json")
for name, expected in old_handoff["evidence_sha256"].items():
    require(sha(OLD / name) == expected, "v1_handoff_evidence:" + name)
require(sha(ENV / "FINAL_HANDOFF.json") == "600c32eb83dc62939334e7bbdf2f89023ebca6bbcb7faaf62a1444ac872ad946", "v1_environment_handoff_pin")
require(sha(ENV / "STAGED_FILE_MANIFEST.json") == obj(ENV / "FINAL_HANDOFF.json")["manifest_sha256"], "v1_environment_manifest_pin")
worker_handoff = obj(WORKER / "FINAL_REVIEW_HANDOFF.json")
require(sha(WORKER / "STAGED_FILE_MANIFEST.json") == worker_handoff["evidence_sha256"]["STAGED_FILE_MANIFEST.json"], "v1_original_worker_manifest_pin")
inventory_counts = {
    "v1_controller": inventory(OLD, ["STAGED_FILE_MANIFEST.json", "FINAL_REVIEW_HANDOFF.json"]),
    "v1_environment": inventory(ENV, ["STAGED_FILE_MANIFEST.json", "FINAL_HANDOFF.json"]),
    "v2_candidate": inventory(NEW, ["STAGED_FILE_MANIFEST.json", "FINAL_HANDOFF.json"]),
    "v1_original_worker": inventory(WORKER, ["STAGED_FILE_MANIFEST.json", "FINAL_REVIEW_HANDOFF.json"]),
}
old_bundle, new_bundle = obj(OLD / "BUNDLE_MANIFEST.json"), obj(NEW / "BUNDLE_MANIFEST.json")
require(len(new_bundle["files"]) == new_bundle["runtime_file_count"] == 10, "exact10_runtime_bundle")
for name, expected in new_bundle["files"].items():
    require(sha(NEW / name) == expected, "v2_bundle:" + name)
require(sha(NEW / "BUNDLE_MANIFEST.json") == handoff["v2_bundle_manifest_sha256"], "v2_bundle_handoff_binding")
old_protocol, new_protocol = obj(OLD / "worker/protocol.json"), obj(NEW / "worker/protocol.json")
old_p_sha, new_p_sha = sha(OLD / "worker/protocol.json"), sha(NEW / "worker/protocol.json")
canonical = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
changed = json.loads(json.dumps(old_protocol))
changed["runtime"]["numpy"] = "2.4.3"
require(old_protocol["runtime"]["numpy"] == "2.4.6" and changed == new_protocol, "only_protocol_leaf_runtime_numpy")
old_plan, new_plan = obj(OLD / "FROZEN_PLAN23.json"), obj(NEW / "FROZEN_PLAN23.json")
require(len(new_plan["slots"]) == new_plan["slot_count"] == 23 and old_plan["slots"] == new_plan["slots"], "same23_slot_objects")
replacements = {
    "worker/protocol.json": [(b'"numpy": "2.4.6"', b'"numpy": "2.4.3"')],
    "ENVIRONMENT_RECEIPT_TEMPLATE.json": [(b'"numpy": "2.4.6"', b'"numpy": "2.4.3"')],
    "pilot23_common.py": [(old_p_sha.encode(), new_p_sha.encode())],
    "comparison.py": [(old_p_sha.encode(), new_p_sha.encode()), (canonical(old_protocol).encode(), canonical(new_protocol).encode())],
    "FROZEN_PLAN23.json": [(old_p_sha.encode(), new_p_sha.encode())],
    "HELD_REGISTRATION.json": [(sha(OLD / "BUNDLE_MANIFEST.json").encode(), sha(NEW / "BUNDLE_MANIFEST.json").encode()), (sha(OLD / "FROZEN_PLAN23.json").encode(), sha(NEW / "FROZEN_PLAN23.json").encode()), (old_p_sha.encode(), new_p_sha.encode())],
    "BUNDLE_MANIFEST.json": [(old_bundle["files"][name].encode(), new_bundle["files"][name].encode()) for name in new_bundle["files"] if old_bundle["files"][name] != new_bundle["files"][name]],
}
all_names = set(old_bundle["files"]) | {"BUNDLE_MANIFEST.json", "ENVIRONMENT_RECEIPT_TEMPLATE.json", "HELD_REGISTRATION.json"}
changed_names = sorted(name for name in all_names if raw(OLD / name) != raw(NEW / name))
require(changed_names == sorted(replacements), "exact7_changed_runtime_pointer_files")
patch = ""
for name in sorted(all_names):
    before, after = raw(OLD / name), raw(NEW / name)
    candidate = before
    for source, target in replacements.get(name, []):
        require(source in candidate, "pointer_substitution_present:" + name)
        candidate = candidate.replace(source, target)
    require(candidate == after, "exact_allowed_substitutions:" + name)
    patch += "".join(difflib.unified_diff(before.decode().splitlines(True), after.decode().splitlines(True), fromfile="V1/" + name, tofile="V2/" + name))
require(patch.encode() == raw(NEW / "V1_TO_V2_POINTER_DIFF.patch"), "published_patch_exact_actual_diff")
require(raw(OLD / "test_controller_controls.py") == raw(NEW / "test_controller_controls.py"), "metadata_controls_source_unchanged")
reg, template = obj(NEW / "HELD_REGISTRATION.json"), obj(NEW / "ENVIRONMENT_RECEIPT_TEMPLATE.json")
require(reg["execution_authorized"] is False and reg["retry"] is False and reg["resume"] is False, "registration_false_no_retry_resume")
require(reg["bundle_manifest_sha256"] == sha(NEW / "BUNDLE_MANIFEST.json") and reg["plan_sha256"] == sha(NEW / "FROZEN_PLAN23.json"), "held_registration_binds_new_bundle_plan")
require(template["ephemeral"] is False and template["native_use_review_clear"] is False and template["modules"] is None and new_protocol["execution_authorized"] is False, "template_and_protocol_non_authority")
require(handoff["execution_authorized"] is False and handoff["environment_approved"] is False and handoff["root_GO_created"] is False, "handoff_non_authority")

receipt = obj(NEW / "PRIMARY_METADATA_RECEIPT.json")
metadata = {}
for name in ("devito", "numpy", "scipy"):
    path = NEW / "metadata" / (name + "-wheel.METADATA")
    meta = BytesParser().parsebytes(raw(path))
    metadata[name] = {"version": meta["Version"], "requires_python": meta["Requires-Python"], "requires_dist": meta.get_all("Requires-Dist", [])}
    row = next(row for row in receipt["rows"] if row.get("name") == name)
    require(sha(path) == row["sha256"], "retained_sidecar_sha:" + name)
    if "advertised_sha256" in row:
        require(row["sha256"] == row["advertised_sha256"], "advertised_sidecar_sha:" + name)
require("numpy<=2.4.3,>=1.26" in metadata["devito"]["requires_dist"], "devito_direct_numpy_bound")
require("numpy<2.7,>=1.26.4" in metadata["scipy"]["requires_dist"], "scipy_direct_numpy_bound")
require(metadata["numpy"]["version"] == "2.4.3" and metadata["numpy"]["requires_dist"] == [], "numpy_selected_version_no_direct_requirements")
require(metadata["devito"]["requires_python"] == "<3.15,>=3.10" and metadata["numpy"]["requires_python"] == metadata["scipy"]["requires_python"] == ">=3.11", "python_direct_bounds")
release_paths = {"numpy": NEW / "metadata/numpy-2.4.3-release.json", "scipy": ENV / "primary_metadata/scipy-1.17.1.json"}
for selected in receipt["selected_wheel_metadata_only"]:
    release = obj(release_paths[selected["name"]])
    wheel = next(row for row in release["urls"] if row["filename"] == selected["filename"])
    require(wheel["url"] == selected["archive_url"] and wheel["digests"]["sha256"] == selected["declared_archive_sha256"] and wheel["size"] == selected["declared_archive_bytes"], "retained_release_wheel_identity:" + selected["name"])
    require(wheel["core-metadata"]["sha256"] == selected["core_metadata_sha256"], "retained_release_pep658_advertisement:" + selected["name"])
    require("cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64" in wheel["filename"] and selected["archive_bytes_acquired"] == 0, "selected_wheel_declared_tag_no_archive:" + selected["name"])
pyrevolve = obj(ENV / "primary_metadata/pyrevolve-2.2.8.json")["info"]
require(pyrevolve["requires_python"] == "<3.15,>=3.10" and "numpy" in pyrevolve["requires_dist"], "pyrevolve_direct_bounds")
require(b'i_required = ["contexttimer", "numpy"]' in raw(ENV / "primary_metadata/pyrevolve-pinned-setup.py"), "pinned_pyrevolve_numpy_requirement")
compatibility = obj(NEW / "DIRECT_METADATA_COMPATIBILITY.json")
require(all(compatibility[key] is False for key in ("resolved_build_closure", "resolved_full_transitive_runtime_closure", "actual_binary_compatibility_proved", "environment_approved", "native_authority")), "compatibility_scope_flags_false")
controls = obj(NEW / "METADATA_ADMISSION_CONTROLS_V2.json")
require(controls["tests_run"] == 4 and controls["failures"] == controls["errors"] == controls["skips"] == 0 and controls["blocked_attempts"] == [], "retained_author_four_controls_receipt")
require(b"Ran 4 tests" in raw(NEW / "METADATA_ADMISSION_CONTROLS_V2.log") and controls["test_source_sha256"] == sha(NEW / "test_controller_controls.py"), "retained_author_four_controls_log_source")
for path, before in list(seen.items()):
    require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == before, "review_readonly_after_sha:" + path)
result = {
    "status": "CLEAR_SOURCE_METADATA_SCOPE_ONLY_NO_EXECUTION_AUTHORITY" if not issues else "BLOCKED",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "checks": checks, "issues": issues, "review_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "reviewed_handoff_sha256": sha(NEW / "FINAL_HANDOFF.json"),
    "v2_bundle_manifest_sha256": sha(NEW / "BUNDLE_MANIFEST.json"),
    "v2_protocol_sha256": new_p_sha,
    "v1_inventory_file_counts": inventory_counts,
    "exact_changed_files": changed_names,
    "unchanged": ["native_worker319", "accounting1b", "controller_ef0", "worker_entry045", "policy4e", "13_selected_source_pins_bfbc", "23_slot_objects", "all_math_FD_criteria_protocol_fields"],
    "direct_bound_conclusion": "2.4.3 satisfies Devito >=1.26,<=2.4.3 and SciPy >=1.26.4,<2.7; Python3.11.18 lies in all retained direct Python ranges. Retained release objects advertise the exact sidecar hashes and wheel identities. No fresh network retrieval by reviewer.",
    "controls": {"author_recorded_passes": 4, "reviewer_tests_rerun": 0, "scope": "Four unchanged metadata admission controls, not full51/native/runtime validation."},
    "review_actions": {"candidate_imports": 0, "runtime_package_imports": 0, "tests": 0, "installs": 0, "downloads": 0, "PDE": 0, "jobs": 0, "candidate_edits": 0, "GO": 0},
    "limits": ["Metadata/direct-range compatibility only; full transitive runtime/build lock, PyRevolve compiled scheduler, actual17origins/13selected-source matches, binary/system linkage and native numerical agreement remain unproved.", "Registration/protocol remain false; template unapproved. This review does not authorize environment preparation or native execution.", "V1 preservation means equality to its retained frozen manifests/handoff pins, including original histories; no claim about filesystem metadata or unmanifested remote sources."],
    "evidence_sha256": {str(Path(path).relative_to(BASE)): value for path, value in sorted(seen.items())},
}
with OUT.open("x") as stream:
    json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
print(json.dumps({key: result[key] for key in ("status", "checks", "issues", "v1_inventory_file_counts", "exact_changed_files")}))
print("receipt_sha256=" + hashlib.sha256(OUT.read_bytes()).hexdigest())
raise SystemExit(bool(issues))
