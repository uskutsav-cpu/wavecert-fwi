"""Manufactured transport only: no remote requests or project PDE calls."""
from __future__ import annotations

import ast
import copy
import importlib.util
import io
import json
import struct
import tempfile
from functools import partial
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("bounded_plane", HERE / "acquire_partial_plane.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class Response(io.BytesIO):
    def __init__(self, raw, url, status=206, headers=None):
        super().__init__(raw)
        self.status = status
        self.url = url
        self.headers = headers or {"Content-Length": str(len(raw))}
        self.read_count = 0

    def geturl(self):
        return self.url

    def read(self, size=-1):
        self.read_count += 1
        return super().read(size)


def must_fail(action):
    try:
        action()
    except (ValueError, FileExistsError):
        return
    raise AssertionError("negative control unexpectedly succeeded")


def main():
    controls = []
    plan, _review, metadata = M.verify_bundle()
    controls.append("local exact metadata/proposal bindings")
    for change in ("wrong row", "drop row", "convert", "file identity", "cap"):
        bad = copy.deepcopy(plan)
        if change == "wrong row":
            bad["intervals"][0]["first_byte"] += 4
        elif change == "drop row":
            bad["intervals"].pop()
        elif change == "convert":
            bad["selection"]["arithmetic_conversion"] = "sqrt"
        elif change == "file identity":
            bad["file"]["id"] = "different"
        else:
            bad["hard_response_body_cap_bytes"] += 1
        must_fail(partial(M.validate_plan, bad))
        controls.append("reject " + change)
    raw_record = (M.BASE / "zenodo_4252588_record.json").read_bytes()
    M.verify_record(raw_record, plan)
    for key in ("UUID", "revision", "checksum", "license"):
        record = json.loads(raw_record)
        if key == "UUID":
            record["files"][0]["id"] = "changed"
        elif key == "revision":
            record["revision"] += 1
        elif key == "checksum":
            record["files"][0]["checksum"] = "changed"
        else:
            record["metadata"]["license"]["id"] = "changed"
        must_fail(partial(M.verify_record, json.dumps(record).encode(), plan))
        controls.append("reject record " + key)
    original_open = M.urllib.request.build_opener
    with (
        patch.object(M.urllib.request, "build_opener", side_effect=AssertionError("real network forbidden")),
        tempfile.TemporaryDirectory(prefix="wavecert-plane-controls-") as temporary,
    ):
        root = Path(temporary)
        lo, hi = 1283792, 1286995
        good_headers = {"Content-Length": "3204", "Content-Range": f"bytes {lo}-{hi}/{M.FILE_SIZE}"}
        for label in ("HTTP200", "wrong total", "wrong interval", "encoding", "wrong length", "missing length", "redirect"):
            folder = root / label
            folder.mkdir()
            headers = good_headers.copy()
            status, url = 206, M.FILE_URL
            if label == "HTTP200":
                status = 200
            elif label == "wrong total":
                headers["Content-Range"] = f"bytes {lo}-{hi}/531253093"
            elif label == "wrong interval":
                headers["Content-Range"] = f"bytes {lo+1}-{hi}/{M.FILE_SIZE}"
            elif label == "encoding":
                headers["Content-Encoding"] = "gzip"
            elif label == "wrong length":
                headers["Content-Length"] = "3203"
            elif label == "missing length":
                headers.pop("Content-Length")
            else:
                url += "?different"
            response = Response(b"x"*3204, url, status=status, headers=headers)
            transport = M.BoundedTransport(folder, opener=lambda *_a, _response=response, **_k: _response)
            must_fail(partial(transport.fetch, M.FILE_URL, label, interval=(lo, hi), maximum=3204))
            assert response.read_count == 0 and transport.ledger["body_bytes_read"] == 0
            assert len(transport.ledger["requests"]) == 1
            controls.append("reject before body read: " + label)
        folder = root / "cap"
        folder.mkdir()
        transport = M.BoundedTransport(folder, opener=lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not send")), cap=3203)
        must_fail(lambda: transport.fetch(M.FILE_URL, "cap", interval=(lo, hi), maximum=3204))
        assert not transport.ledger["requests"]
        controls.append("reject exhausted cap before request")
        must_fail(lambda: transport.fetch(M.FILE_URL, "wholefile"))
        must_fail(lambda: transport.fetch("https://other.example/", "host"))
        controls.append("no whole-file fallback or unregistered host")
        folder = root / "short"
        folder.mkdir()
        response = Response(b"x"*100, M.FILE_URL, headers=good_headers)
        transport = M.BoundedTransport(folder, opener=lambda *_a, **_k: response)
        must_fail(lambda: transport.fetch(M.FILE_URL, "short", interval=(lo, hi), maximum=3204))
        assert transport.ledger["body_bytes_read"] == 100
        entry = transport.ledger["requests"][0]
        assert (folder / entry["preserved_failed_body_prefix"]).read_bytes() == b"x"*100
        controls.append("short body actual count and exact failed prefix preserved")

        mapping = {(a, b): p.read_bytes() for a, b, p in metadata}
        row_ranges = {(x["first_byte"], x["last_byte"]): x["index0"] for x in plan["intervals"]}
        row_raw = lambda index: struct.pack("<f", index + 1.25) * 801
        calls = []

        def opener(request, **_kwargs):
            calls.append(request.full_url)
            assert request.get_header("Accept-encoding") == "identity"
            if request.full_url == M.RECORD_URL:
                return Response(raw_record, M.RECORD_URL, status=200)
            interval = tuple(map(int, request.get_header("Range")[6:].split("-")))
            raw = mapping[interval] if interval in mapping else row_raw(row_ranges[interval])
            return Response(raw, M.FILE_URL, headers={"Content-Length": str(len(raw)), "Content-Range": f"bytes {interval[0]}-{interval[1]}/{M.FILE_SIZE}"})

        output = root / "complete"
        receipt = M.acquire(plan, metadata, output, opener=opener)
        ledger = json.loads((output / "REQUEST_LEDGER.json").read_bytes())
        assert len(calls) == 259 and ledger["model_rows_complete"] == 207
        assert ledger["body_bytes_read"] == 663228 + 2*len(raw_record) + 2*sum(b-a+1 for a, b, _p in metadata)
        assert ledger["body_bytes_read"] < M.BODY_CAP
        expected = b"".join(row_raw(i) for i in range(207))
        for i in range(207):
            assert (output / "rows" / f"row_{i:03d}.bin").read_bytes() == row_raw(i)
        assert (output / "anonymous_m_axis1_400.npy").read_bytes() == M.npy_header() + expected
        npy = (output / "anonymous_m_axis1_400.npy").read_bytes()
        assert npy[:8] == b"\x93NUMPY\x01\x00"
        header_size = struct.unpack_from("<H", npy, 8)[0]
        header = ast.literal_eval(npy[10:10+header_size].decode("ascii"))
        assert header == {"descr": "<f4", "fortran_order": False, "shape": (207, 801)}
        assert npy[10+header_size:] == expected
        assert receipt["whole_file_checksum_verified"] is False and receipt["physics_acceptance"] is None
        assert not ledger["arithmetic_conversion"] and ledger["water_rows_removed"] == 0
        controls.extend(["all207 synthetic rows and ordered bytes exact", "259 requests and aggregate actual body count", "no conversion or omitted rows", "no whole-file or physics claim"])
        controls.append("independently parsed NPY header and payload identity")
        must_fail(lambda: M.acquire(plan, metadata, output, opener=opener))
        controls.append("fresh output only; no resume")
        for relative, expected_sha in receipt["artifacts"].items():
            assert M.sha((output / relative).read_bytes()) == expected_sha
        controls.append("complete receipt artifact hashes exact")
        calls.clear()

        def short_row_opener(request, **kwargs):
            interval = request.get_header("Range")
            bad = plan["intervals"][1]
            if interval == f"bytes={bad['first_byte']}-{bad['last_byte']}":
                calls.append(request.full_url)
                return Response(b"z"*100, M.FILE_URL, headers={"Content-Length": "3204", "Content-Range": f"bytes {bad['first_byte']}-{bad['last_byte']}/{M.FILE_SIZE}"})
            return opener(request, **kwargs)

        failed = root / "failed_prefix"
        must_fail(lambda: M.acquire(plan, metadata, failed, opener=short_row_opener))
        ledger = json.loads((failed / "REQUEST_LEDGER.json").read_bytes())
        assert ledger["status"] == "failed" and ledger["valid_dataset_credit"] is False
        assert len(ledger["complete_rows"]) == 1 and len(calls) == 28
        assert (failed / "rows" / "row_000.bin").read_bytes() == row_raw(0)
        assert ledger["requests"][-1]["label"] == "model_row_001"
        assert ledger["requests"][-1]["body_bytes_read"] == 100
        assert not (failed / "complete_receipt.json").exists()
        assert not (failed / "anonymous_m_axis1_400.npy").exists()
        controls.append("failed row preserves exact completed prefix; no retry, plane, or complete credit")
        calls.clear()

        def changed_metadata_opener(request, **kwargs):
            response = opener(request, **kwargs)
            if request.get_header("Range") == "bytes=0-7":
                response = Response(b"changed!", M.FILE_URL, headers={"Content-Length": "8", "Content-Range": f"bytes 0-7/{M.FILE_SIZE}"})
            return response

        changed = root / "metadata_drift"
        must_fail(lambda: M.acquire(plan, metadata, changed, opener=changed_metadata_opener))
        ledger = json.loads((changed / "REQUEST_LEDGER.json").read_bytes())
        assert ledger["status"] == "failed" and len(calls) == 2 and not ledger["complete_rows"]
        controls.append("metadata drift aborts before any model row request")
    assert M.urllib.request.build_opener is original_open
    result = {"status": "MANUFACTURED_TRANSPORT_CONTROLS_PASS", "controls_count": len(controls), "controls": controls, "driver_sha256": M.sha((HERE / "acquire_partial_plane.py").read_bytes()), "controls_source_sha256": M.sha(Path(__file__).read_bytes()), "actual_network_requests": 0, "actual_model_values_acquired": 0, "project_PDE_calls": 0, "temporary_synthetic_bytes_removed": True}
    (HERE / "MANUFACTURED_TRANSPORT_CONTROLS.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ("status", "controls_count", "driver_sha256", "actual_network_requests")}))


if __name__ == "__main__":
    main()
