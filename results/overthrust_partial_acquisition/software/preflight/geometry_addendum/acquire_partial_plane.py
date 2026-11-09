"""Acquire a fixed raw anonymous plane, only after an exact root source freeze.

Default invocation checks local metadata only. No velocity conversion, physics,
value-dependent selection, retry, resume or whole-file fallback is implemented.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PROPOSAL = "FULL207_MIDDLE400_PARTIAL_ACQUISITION_PROPOSAL.json"
PROPOSAL_SHA256 = "f2cea059ad3c116ad08131d7b874eab193f30929297f3068544bbd5fa64a64e6"
REVIEW = "INDEPENDENT_GEOMETRY_REVIEW.json"
REVIEW_SHA256 = "d0499c4b387ec8955e3aecb910840eac6b9c81ff33e2775b45c5ef7949fb2ad2"
RECORD_URL = "https://zenodo.org/api/records/4252588"
FILE_URL = "https://zenodo.org/api/records/4252588/files/overthrust_3D_true_model.h5/content"
FILE_SIZE = 531253092
BODY_CAP = 8388608
NOTICE = "Partial selection of stored dataset m[:,400,:] from SEG/EAGE 3-D Overthrust Models, DOI 10.5281/zenodo.4252588, file UUID 436595b8-ed4b-4e1d-8991-e63608dcc169. CC BY 4.0: https://creativecommons.org/licenses/by/4.0/. Publisher attribution/license declaration and exact HDF5 copyright metadata are included. All 207 rows retained; no conversion, transpose or smoothing. Physical units and named axes remain unverified. A subset does not verify the published whole-file checksum.\n"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def validate_plan(plan: dict) -> None:
    expected_selection = {"dataset": "m", "stored_dtype": "<f4", "stored_shape": [207, 801, 801], "index": ["0:207", 400, "0:801"], "result_shape": [207, 801], "index_basis": "anonymous stored dimensions, C contiguous", "water_rows_removed": 0, "arithmetic_conversion": None, "axis_transpose": None}
    if plan["selection"] != expected_selection or plan["hard_response_body_cap_bytes"] != BODY_CAP:
        raise ValueError("selection or body cap changed")
    expected = []
    for row in range(207):
        start = 2192 + ((row * 801 + 400) * 801) * 4
        expected.append({"index0": row, "first_byte": start, "last_byte": start + 3203, "body_bytes": 3204})
    if plan["intervals"] != expected or plan["planned_plane_response_body_bytes"] != 663228:
        raise ValueError("exact full207 range plan changed")
    identity = plan["file"]
    if (identity["id"], identity["key"], identity["size"], identity["checksum"], identity["links"]["self"]) != ("436595b8-ed4b-4e1d-8991-e63608dcc169", "overthrust_3D_true_model.h5", FILE_SIZE, "md5:ae4082cdcebbc7714eedd382d7925804", FILE_URL):
        raise ValueError("file identity changed")


def verify_bundle() -> tuple[dict, dict, list[tuple[int, int, Path]]]:
    proposal_raw, review_raw = (HERE / PROPOSAL).read_bytes(), (HERE / REVIEW).read_bytes()
    if sha(proposal_raw) != PROPOSAL_SHA256 or sha(review_raw) != REVIEW_SHA256:
        raise ValueError("reviewed proposal or metadata review changed")
    plan, review = json.loads(proposal_raw), json.loads(review_raw)
    validate_plan(plan)
    for relative, expected in review["source_assets"].items():
        if sha((BASE / relative).read_bytes()) != expected:
            raise ValueError("reviewed metadata asset changed: " + relative)
    metadata = []
    for folder, name in ((BASE, "REQUEST_LEDGER.json"), (HERE, "REQUEST_LEDGER_GEOMETRY.json")):
        ledger = json.loads((folder / name).read_bytes())
        for entry in ledger["requests"]:
            if entry.get("requested_range") is not None:
                start, end = entry["requested_range"]
                if not (0 <= start <= end < 2192 or 531247820 <= start <= end < FILE_SIZE):
                    raise ValueError("metadata interval enters model storage")
                metadata.append((start, end, folder / entry["local_name"]))
    if len(metadata) != 25:
        raise ValueError("metadata range count changed")
    return plan, review, metadata


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class BoundedTransport:
    def __init__(self, output: Path, opener=None, cap=BODY_CAP):
        self.output = output
        self.opener = opener or urllib.request.build_opener(NoRedirect()).open
        self.cap = cap
        self.ledger = {"status": "running", "body_cap_bytes": cap, "body_bytes_read": 0, "requests": [], "complete_rows": [], "whole_file_checksum_verified": False, "ETag_available_in_reviewed_metadata": False}
        self.save()

    def save(self) -> None:
        save_json(self.output / "REQUEST_LEDGER.json", self.ledger)

    def fetch(self, url: str, label: str, *, interval=None, maximum=65536) -> bytes:
        if url not in (RECORD_URL, FILE_URL) or interval is None and url != RECORD_URL:
            raise ValueError("unregistered endpoint or whole-file request")
        if interval is not None:
            lo, hi = interval
            if url != FILE_URL or not (0 <= lo <= hi < FILE_SIZE) or hi - lo + 1 != maximum:
                raise ValueError("invalid exact bounded range")
        if maximum > self.cap - self.ledger["body_bytes_read"]:
            raise ValueError("remaining aggregate body cap cannot admit request")
        entry = {"label": label, "url": url, "requested_range": interval, "maximum_body_bytes": maximum, "body_bytes_read": 0, "status": "attempted", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        self.ledger["requests"].append(entry)
        self.save()
        headers = {"Accept-Encoding": "identity", "User-Agent": "WaveCert-bounded-partial-acquisition/1"}
        if interval is not None:
            headers["Range"] = f"bytes={lo}-{hi}"
        else:
            headers["Accept"] = "application/json"
        request = urllib.request.Request(url, headers=headers)
        started = time.monotonic()
        raw = bytearray()
        try:
            try:
                response = self.opener(request, timeout=20)
            except urllib.error.HTTPError as exc:
                entry.update(status="HTTP_error_no_body_read", HTTP_status=exc.code)
                exc.close()
                raise ValueError("HTTP error; no response body read") from None
            with response:
                entry.update(HTTP_status=response.status, final_url=response.geturl(), response_headers={key: response.headers.get(key) for key in ("Content-Length", "Content-Range", "Content-Encoding", "ETag", "Last-Modified")})
                expected_status = 200 if interval is None else 206
                if response.status != expected_status or response.geturl() != url:
                    raise ValueError("status or final endpoint differs before body read")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("encoded response refused before body read")
                if interval is not None and response.headers.get("Content-Range") != f"bytes {lo}-{hi}/{FILE_SIZE}":
                    raise ValueError("exact Content-Range differs before body read")
                declared = response.headers.get("Content-Length")
                if declared is None or not declared.isdecimal() or int(declared) > maximum:
                    raise ValueError("missing or excessive Content-Length")
                wanted = int(declared)
                if wanted <= 0 or interval is not None and wanted != maximum:
                    raise ValueError("exact response length differs before body read")
                while len(raw) < wanted:
                    if time.monotonic() - started > 30:
                        raise TimeoutError("per-response30-second deadline")
                    block = response.read(min(8192, wanted-len(raw)))
                    if not block:
                        raise ValueError("incomplete response body")
                    raw.extend(block)
                    entry["body_bytes_read"] += len(block)
                    self.ledger["body_bytes_read"] += len(block)
                    self.save()
                    if len(raw) > wanted or self.ledger["body_bytes_read"] > self.cap:
                        raise ValueError("transport exceeded admitted byte count")
            entry.update(status="complete", sha256=sha(raw), elapsed_seconds=time.monotonic()-started)
            self.save()
            return bytes(raw)
        except BaseException as exc:
            if raw:
                partial = self.output / f"failed_response_prefix_{len(self.ledger['requests'])-1:03d}.bin"
                partial.write_bytes(raw)
                entry.update(preserved_failed_body_prefix=partial.name, preserved_prefix_sha256=sha(raw))
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic()-started)
            self.save()
            raise


def verify_record(raw: bytes, plan: dict) -> None:
    record = json.loads(raw)
    files = [x for x in record["files"] if x["key"] == plan["file"]["key"]]
    if record["id"] != 4252588 or record["revision"] != 2 or len(files) != 1 or files[0] != plan["file"] or record["metadata"]["license"]["id"] != "cc-by-4.0":
        raise ValueError("primary record/file/license identity changed")


def npy_header() -> bytes:
    content = b"{'descr': '<f4', 'fortran_order': False, 'shape': (207, 801), }"
    padding = (16 - ((10 + len(content) + 1) % 16)) % 16
    header = content + b" " * padding + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header


def acquire(plan: dict, metadata: list, output: Path, *, opener=None, provenance=()) -> dict:
    validate_plan(plan)
    output.mkdir(parents=True, exist_ok=False)
    (output / "rows").mkdir()
    (output / "metadata").mkdir()
    (output / "provenance").mkdir()
    transport = BoundedTransport(output, opener=opener)
    try:
        for source, relative in provenance:
            target = output / "provenance" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        (output / "provenance" / "PARTIAL_SELECTION_NOTICE.txt").write_text(NOTICE)
        all_rows = bytearray()
        for phase in ("before", "after"):
            record = transport.fetch(RECORD_URL, phase + "_record")
            (output / "metadata" / (phase + "_record.json")).write_bytes(record)
            verify_record(record, plan)
            for index, (lo, hi, source) in enumerate(metadata):
                raw = transport.fetch(FILE_URL, f"{phase}_metadata_{index}", interval=(lo, hi), maximum=hi-lo+1)
                (output / "metadata" / f"{phase}_{index}_{lo}_{hi}.bin").write_bytes(raw)
                if raw != source.read_bytes():
                    raise ValueError("metadata identity changed")
            if phase == "before":
                for row in plan["intervals"]:
                    index, lo, hi = row["index0"], row["first_byte"], row["last_byte"]
                    raw = transport.fetch(FILE_URL, f"model_row_{index:03d}", interval=(lo, hi), maximum=3204)
                    path = output / "rows" / f"row_{index:03d}.bin"
                    path.write_bytes(raw)
                    all_rows.extend(raw)
                    transport.ledger["complete_rows"].append({"index0": index, "sha256": sha(raw), "path": str(path.relative_to(output))})
                    transport.save()
        if len(all_rows) != 663228 or len(transport.ledger["complete_rows"]) != 207:
            raise ValueError("incomplete fixed plane")
        (output / "anonymous_m_axis1_400.npy").write_bytes(npy_header() + all_rows)
        transport.ledger.update(status="complete", ordered_raw_plane_sha256=sha(all_rows), model_response_body_bytes=663228, model_rows_complete=207, arithmetic_conversion=False, axis_transpose=False, water_rows_removed=0, model_physical_units_verified=False, named_axis_order_verified=False)
        transport.save()
        receipt = {"status": "COMPLETE_PARTIAL_RAW_ANONYMOUS_PLANE_ONLY", "proposal_sha256": PROPOSAL_SHA256, "driver_sha256": sha(Path(__file__).read_bytes()), "actual_terminal_exit": None, "request_ledger_sha256": sha((output / "REQUEST_LEDGER.json").read_bytes()), "whole_file_checksum_verified": False, "ETag_limitation": "No ETag supplied by reviewed endpoints; exact repeated metadata and versioned record identity do not establish whole-file byte immutability.", "physics_acceptance": None, "independent_site_claim": False, "untouched_confirmation_claim": False, "artifacts": {str(path.relative_to(output)): sha(path.read_bytes()) for path in sorted(output.rglob("*")) if path.is_file()}}
        save_json(output / "complete_receipt.json", receipt)
        return receipt
    except BaseException as exc:
        transport.ledger.update(status="failed", valid_dataset_credit=False, error=f"{type(exc).__name__}: {exc}")
        transport.save()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root-freeze", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan, review, metadata = verify_bundle()
    if not args.execute:
        print(json.dumps({"status": "LOCAL_METADATA_ONLY_NO_NETWORK", "planned_rows": 207, "planned_model_body_bytes": 663228, "proposal_sha256": PROPOSAL_SHA256}))
        return
    if args.root_freeze is None or args.output is None:
        parser.error("--execute requires --root-freeze and --output")
    freeze = json.loads(args.root_freeze.read_bytes())
    if freeze.get("authorization") != "GO_BOUNDED_PARTIAL_ACQUISITION" or freeze.get("driver_sha256") != sha(Path(__file__).read_bytes()) or freeze.get("proposal_sha256") != PROPOSAL_SHA256 or freeze.get("review_sha256") != REVIEW_SHA256:
        raise ValueError("root source-review freeze does not bind this exact candidate")
    provenance = [(BASE / relative, relative) for relative in review["source_assets"]]
    provenance.extend((source, source.name) for source in (HERE / PROPOSAL, HERE / REVIEW, Path(__file__), args.root_freeze))
    receipt = acquire(plan, metadata, args.output, provenance=provenance)
    print(json.dumps({"status": receipt["status"], "ledger_sha256": receipt["request_ledger_sha256"], "whole_file_checksum_verified": False}))


if __name__ == "__main__":
    main()
