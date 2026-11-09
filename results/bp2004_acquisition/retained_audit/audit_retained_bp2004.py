"""Read-only retained-artifact audit; no collector imports, network or PDEs."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import platform
import re
import struct
import sys
import traceback
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SOURCE_SHA = "7a499a425ca495c8cffb2714af260e88e9ad929e9c6f85b0d9d7c0ef9624e28d"
PROTOCOL_SHA = "a84ce64dcac7f93f035007daf313cb2ab7bde713800d10d9d37d812f12c836f4"
PDF_SHA = "e29b003fc2161c44d2b9f04448195e58ebe6f476a195b31c3be77440adb60e24"
NOTICE_SHA = "897e0baa2441bdd52645769004c4e6f413789ec4eeb0eb8b60568cf39e7bf6f7"
COMPRESSED_SHA = "01b3d65c6318491b1412465c904614a6c35bd73ea0ffc6a8d866917181749b9c"
UNCOMPRESSED_SHA = "19f729b535d1629e0176443024d22344edd2083273c423a361e99d4cd53e62e1"
ANCESTOR = "BP2004_Billette_BrandsbergDahl_composite_exact_velocity"
WINDOWS = [
    ("upper_left", [382, 983], [574, 1175], [0.25, 0.20]),
    ("middle_center", [859, 2601], [1051, 2793], [0.50, 0.50]),
    ("lower_right", [1337, 4219], [1529, 4411], [0.75, 0.80]),
]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def inventory(directory):
    return {p.name: {"bytes": p.stat().st_size, "sha256": digest(p.read_bytes())}
            for p in sorted(directory.iterdir()) if p.is_file()}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.run, args.frozen = args.run.resolve(), args.frozen.resolve()
    args.report = args.report.resolve()
    if args.report.exists() or args.report.is_relative_to(args.run):
        raise SystemExit("Report must be a new path outside the immutable run")
    own_source_before = digest(Path(__file__).read_bytes())
    before = inventory(args.run)
    frozen_before = inventory(args.frozen)
    checks, arrays, decoded_headers = [], {}, {}

    def check(name, passed, actual=None, expected=None):
        row = {"id": name, "passed": bool(passed)}
        if actual is not None:
            row["actual"] = actual
        if expected is not None:
            row["expected"] = expected
        checks.append(row)

    report = {
        "audit_started_utc": utc_now(),
        "role": "AI independent retained-file/header/stride audit; software/provenance only",
        "root_terminal_witness": {"exec_session": 55256, "exit_code": 0,
                                  "basis": "Root observed successful terminal execution before authorizing this audit"},
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "auditor_sha256_before": own_source_before,
        "run_files_before": before,
        "frozen_top_level_files_before": frozen_before,
        "checks": checks,
        "arrays": arrays,
        "headers": decoded_headers,
        "independence_limits": [
            "The collector is not imported or executed by this auditor; no network request is made.",
            "Full compressed/uncompressed payloads were discarded. Complete-stream SHA256/CRC/ISIZE and raw crop-to-original sample assignment are collector-supported recorded evidence, not independently recomputed or re-decoded here.",
            "Header serialization, saved-file hashes, NPY layout/finite positivity, fixed-index arithmetic and raw-to-stride equality are independently checked from retained bytes.",
            "Source enforces identity HTTP transfer and direct unclipped sample assignment; the response encoding header and omitted original samples are not separately retained for independent replay.",
            "One composite synthetic ancestor supplies all six arrays. No independent sites, original BP shots, physical solver, reconstruction, scientific result or student design is established."
        ],
    }
    try:
        receipt = json.loads((args.run / "receipt.json").read_bytes())
        protocol_raw = (args.run / "protocol.json").read_bytes()
        protocol = json.loads(protocol_raw)
        freeze_raw = args.freeze.read_bytes()
        freeze = json.loads(freeze_raw)
        final_cli = json.loads(args.log.read_text().splitlines()[-1])
        report["external_evidence"] = {
            "freeze_sha256": digest(freeze_raw), "execution_log_sha256": digest(args.log.read_bytes()),
            "terminal_cli": final_cli, "receipt_sha256": before["receipt.json"]["sha256"]}
        names = {name + "_" + suffix + "_velocity_mps.npy" for name, _, _, _ in WINDOWS
                 for suffix in ["raw", "stride2z"]}
        expected_files = names | {"receipt.json", "protocol.json", "collector_source.py",
                                  "original_BP2004_READMES.pdf", "documentation_manifest.json", "README_Modification.md"}
        check("exact_twelve_regular_files_no_extra_payload", set(before) == expected_files
              and all(p.is_file() and not p.is_symlink() for p in args.run.iterdir()),
              sorted(p.name for p in args.run.iterdir()), sorted(expected_files))
        check("complete_acquisition_only", receipt["status"] == final_cli["status"] == "complete_acquisition_only")
        check("freeze_output_identity", Path(freeze["output"]).resolve() == args.run)
        check("freeze_and_source_protocol_pins", freeze["exact_pins"]["collect_bp2004.py"] == SOURCE_SHA
              and freeze["exact_pins"]["protocol.json"] == PROTOCOL_SHA
              and receipt["collector_source_sha256"] == before["collector_source.py"]["sha256"] == SOURCE_SHA
              and receipt["protocol_sha256"] == digest(protocol_raw) == PROTOCOL_SHA)
        for filename, pin in freeze["exact_pins"].items():
            check("frozen_asset_" + filename, digest((args.frozen / filename).read_bytes()) == pin)
        check("snapshot_source_byte_exact", (args.run / "collector_source.py").read_bytes()
              == (args.frozen / "collect_bp2004.py").read_bytes())
        check("snapshot_protocol_byte_exact", protocol_raw == (args.frozen / "protocol.json").read_bytes())
        check("receipt_artifact_map_exact", set(receipt["artifact_sha256"]) == expected_files - {"receipt.json"})
        for filename in sorted(expected_files - {"receipt.json"}):
            check("artifact_sha256_" + filename, before[filename]["sha256"] == receipt["artifact_sha256"].get(filename))
        start = datetime.fromisoformat(receipt["acquisition_started_utc"])
        completed = datetime.fromisoformat(receipt["completed_utc"])
        check("registered_then_frozen_then_acquired", datetime.fromisoformat(protocol["registered_utc"])
              <= datetime.fromisoformat(freeze["created_utc"]) <= start <= completed)
        http, segy = receipt["http"], receipt["segy"]
        source = protocol["source"]
        check("http_identity_and_size", http["status"] == 200 and http["final_url"] == source["url"]
              and http["content_length"] == http["compressed_bytes"] == source["expected_compressed_bytes"] == 19533256
              and http["etag"] == source["expected_etag"] and http["etag_is_not_sha256"] is True)
        check("recorded_full_compressed_sha_matches_root_and_log", http["compressed_sha256"]
              == final_cli["compressed_sha256"] == COMPRESSED_SHA)
        check("recorded_full_uncompressed_sha_matches_root_and_log", segy["uncompressed_sha256"]
              == final_cli["uncompressed_sha256"] == UNCOMPRESSED_SHA)
        check("recorded_one_complete_gzip_stream", segy["uncompressed_bytes"] == segy["gzip_isize"] == 42537780
              and segy["gzip_members"] == 1 and segy["full_gzip_crc_and_isize_verified"] is True
              and re.fullmatch(r"[0-9a-f]{8}", segy["gzip_crc32_hex"]) is not None)
        report["collector_supported_stream_record"] = {k: segy[k] for k in ["uncompressed_bytes", "uncompressed_sha256", "gzip_crc32_hex", "gzip_isize", "gzip_members", "full_gzip_crc_and_isize_verified"]}
        report["collector_supported_stream_record"].update(http=http, independently_recomputed_from_whole_payload=False)
        check("recorded_source_layout", segy["shape_z_x"] == [1911, 5395] and segy["byte_order"] == "big"
              and segy["sample_format_code"] in [1, 5] and segy["binary_samples_per_trace"] == 1911
              and segy["extended_text_headers"] == 0 and segy["trace_stride_bytes"] == 7884
              and segy["first_trace_header_offset"] == 3600 and segy["first_sample_offset"] == 3840
              and set(segy["trace_sample_counts_raw"]) <= {0, 1911})
        check("recorded_sample_validation_and_no_full_matrix", segy["all_samples_validated_finite_positive"] is True
              and segy["header_coordinates_used_for_reordering"] is False
              and segy["full_decoded_velocity_array_allocated"] is False)
        check("no_payload_or_scientific_outcome_claim", receipt["raw_payload_written_to_disk"] is False
              and receipt["plots_or_model_outcomes_computed"] is False and receipt["velocity_units"] == "m/s")

        headers = segy["headers"]
        text = base64.b64decode(headers["text_header_base64"], validate=True)
        binary = base64.b64decode(headers["binary_header_base64"], validate=True)
        packed = base64.b64decode(headers["trace_headers_base64"], validate=True)
        decoder = zlib.decompressobj()
        traces = decoder.decompress(packed, 1294801)
        check("bounded_trace_header_zlib_complete", decoder.eof and not decoder.unused_data
              and not decoder.unconsumed_tail and len(traces) == 1294800)
        check("text_header_lossless_serialization", len(text) == 3200 and digest(text) == headers["text_header_sha256"])
        check("binary_header_lossless_serialization", len(binary) == 400 and digest(binary) == headers["binary_header_sha256"])
        check("trace_header_lossless_serialization", len(traces) == headers["trace_headers_bytes"] == 1294800
              and digest(traces) == headers["trace_headers_sha256"]
              and headers["trace_headers_encoding"] == "zlib+base64, concatenated240-byte headers in original file order")
        decoded_headers.update(text_bytes=len(text), text_sha256=digest(text), binary_bytes=len(binary),
                               binary_sha256=digest(binary), trace_bytes=len(traces), trace_sha256=digest(traces))
        binary_fields = {
            "binary_sample_interval_raw": struct.unpack_from(">H", binary, 16)[0],
            "binary_samples_per_trace": struct.unpack_from(">H", binary, 20)[0],
            "sample_format_code": struct.unpack_from(">H", binary, 24)[0],
            "segy_revision_raw": struct.unpack_from(">H", binary, 300)[0],
            "fixed_length_trace_flag_raw": struct.unpack_from(">H", binary, 302)[0],
            "extended_text_headers": struct.unpack_from(">h", binary, 304)[0],
        }
        for name, value in binary_fields.items():
            check("reparsed_binary_" + name, value == segy[name], value, segy[name])
        trace_sets = {"trace_sample_counts_raw": set(), "trace_sample_intervals_raw": set(),
                      "coordinate_scalars_raw": set(), "coordinate_units_raw": set()}
        first = last = None
        for index in range(5395):
            offset = 240 * index
            for name, code, field in [("trace_sample_counts_raw", ">H", 114), ("trace_sample_intervals_raw", ">H", 116),
                                      ("coordinate_scalars_raw", ">h", 70), ("coordinate_units_raw", ">H", 88)]:
                trace_sets[name].add(struct.unpack_from(code, traces, offset + field)[0])
            fields = {name: struct.unpack_from(">i", traces, offset + field)[0]
                      for name, field in [("trace_sequence_line", 0), ("trace_sequence_file", 4),
                                          ("source_x", 72), ("source_y", 76), ("group_x", 80), ("group_y", 84)]}
            if index == 0:
                first = fields
            last = fields
        for name, values in trace_sets.items():
            check("reparsed_all_5395_headers_" + name, sorted(values) == segy[name], sorted(values), segy[name])
        check("reparsed_first_last_header_fields", first == segy["first_trace_fields"] and last == segy["last_trace_fields"])
        decoded_headers.update(binary_fields=binary_fields, trace_field_sets={k: sorted(v) for k, v in trace_sets.items()}, first_fields=first, last_fields=last)

        check("protocol_fixed_source_and_ancestry", protocol["ancestor_id"] == receipt["ancestor_id"] == ANCESTOR
              and protocol["independent_ancestor_count"] == receipt["independent_ancestor_count"] == 1
              and protocol["crops_are_independent_parents"] is False)
        check("exact_crop_and_protocol_order", [c["id"] for c in receipt["crops"]]
              == [w["id"] for w in protocol["windows"]] == [w[0] for w in WINDOWS])
        combined_array_bytes = 0
        for expected, window, crop in zip(WINDOWS, protocol["windows"], receipt["crops"], strict=True):
            name, origin, stop, fractions = expected
            derived = [math.floor(f * (n - 1) - (192 - 1) / 2) for f, n in zip(fractions, [1911, 5395], strict=True)]
            check(name + "_fixed_parent_window", window["start_z_x"] == crop["start_z_x"] == origin == derived
                  and window["stop_exclusive_z_x"] == crop["stop_exclusive_z_x"] == stop
                  and window["requested_center_fraction_z_x"] == crop["requested_center_fraction_z_x"] == fractions
                  and all(crop[k] == v for k, v in window.items())
                  and crop["raw_shape_z_x"] == [192, 192] and crop["decimated_shape_z_x"] == [96, 192]
                  and crop["stride_z_x"] == [2, 1] and crop["ancestor_id"] == ANCESTOR)
            physical_origin = [origin[0] * 6.25, origin[1] * 12.5]
            check(name + "_physical_coordinates", crop["raw_origin_z_x_m"] == crop["decimated_origin_z_x_m"] == physical_origin
                  and crop["raw_spacing_z_x_m"] == [6.25, 12.5] and crop["decimated_spacing_z_x_m"] == [12.5, 12.5]
                  and crop["raw_pixel_midpoint_z_x"] == [i + 191 / 2 for i in origin] and crop["velocity_units"] == "m/s")
            check(name + "_two_named_outputs", [v["filename"] for v in crop["outputs"]]
                  == [name + "_raw_velocity_mps.npy", name + "_stride2z_velocity_mps.npy"])
            loaded = []
            for suffix, shape, output in zip(["raw", "stride2z"], [[192, 192], [96, 192]], crop["outputs"], strict=True):
                filename = name + "_" + suffix + "_velocity_mps.npy"
                raw_file = (args.run / filename).read_bytes()
                stream = io.BytesIO(raw_file)
                array = np.load(stream, allow_pickle=False)
                canonical = np.ascontiguousarray(array, dtype="<f4").tobytes()
                row = {"shape_z_x": list(array.shape), "dtype_str": array.dtype.str,
                       "c_contiguous": bool(array.flags.c_contiguous), "data_bytes": int(array.nbytes),
                       "file_bytes": len(raw_file), "file_sha256": digest(raw_file),
                       "array_C_le_float32_sha256": digest(canonical),
                       "finite_positive": bool(np.isfinite(array).all() and (array > 0).all())}
                arrays[filename] = row
                check(filename + "_npy_shape_dtype_no_trailing", list(array.shape) == shape
                      and array.dtype.str == "<f4" and array.flags.c_contiguous and stream.tell() == len(raw_file))
                check(filename + "_receipt_identity", row["file_bytes"] == output["file_bytes"]
                      and row["file_sha256"] == output["file_sha256"]
                      and row["array_C_le_float32_sha256"] == output["array_C_le_float32_sha256"]
                      and output["shape_z_x"] == shape and output["dtype"] == "float32"
                      and output["encoding"] == "NPY little-endian C order")
                check(filename + "_all_finite_positive", row["finite_positive"])
                combined_array_bytes += array.nbytes
                loaded.append(array)
            check(name + "_exact_stride_no_interpolation", np.array_equal(loaded[1], loaded[0][::2, ::1])
                  and loaded[1].tobytes(order="C") == loaded[0][::2, ::1].tobytes(order="C"))
        check("combined_six_array_data_bytes", combined_array_bytes == 663552, int(combined_array_bytes), 663552)

        manifest_raw = (args.run / "documentation_manifest.json").read_bytes()
        doc = json.loads(manifest_raw)
        check("original_document_and_manifest_byte_exact", before["original_BP2004_READMES.pdf"]["sha256"] == PDF_SHA
              and before["original_BP2004_READMES.pdf"]["bytes"] == doc["bytes"] == 24615
              and doc["sha256"] == PDF_SHA and doc["source_url"] == protocol["documentation"]["url"]
              and (args.run / "original_BP2004_READMES.pdf").read_bytes() == (args.frozen / "original_BP2004_READMES.pdf").read_bytes()
              and manifest_raw == (args.frozen / "documentation_manifest.json").read_bytes() and receipt["documentation"] == doc)
        notice_raw = (args.run / "README_Modification.md").read_bytes()
        notice = notice_raw.decode()
        required_phrases = ["Data courtesy of BP. Acknowledge BP and Frederic Billette.",
                            "Billette and Brandsberg-Dahl (2005), The 2004 BP velocity benchmark",
                            "67th EAGE Conference, B035, doi:10.3997/2214-4609-pdb.1.B035.",
                            "original_BP2004_READMES.pdf", "These files inherit its terms; no new data license is asserted.",
                            "raw[::2,::1]", "Velocity stays in m/s.",
                            "No interpolation, clipping, normalization or value-driven window selection occurs.",
                            "ONE BP2004 composite synthetic numerical ancestor", "not independent parents/sites",
                            "a reproduction of the original shot gathers or an evaluated algorithm result.", "Ancestor: " + ANCESTOR]
        check("correct_citation_terms_and_one_ancestor_notice", all(v in notice for v in required_phrases)
              and "Complex subsalt imaging challenges" not in notice)
        check("notice_byte_exact_to_validated_v2_output", digest(notice_raw) == NOTICE_SHA
              and notice_raw == (args.frozen / "validation_01" / "README_Modification.generated.md").read_bytes())
    except BaseException as exc:
        report["audit_exception"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        check("audit_completed_without_exception", False)
    finally:
        after = inventory(args.run)
        frozen_after = inventory(args.frozen)
        own_source_after = digest(Path(__file__).read_bytes())
        check("run_files_unchanged_during_audit", after == before)
        check("frozen_top_level_files_unchanged_during_audit", frozen_after == frozen_before)
        check("auditor_source_unchanged", own_source_after == own_source_before)
        report.update(run_files_after=after, frozen_top_level_files_after=frozen_after,
                      auditor_sha256_after=own_source_after, audit_completed_utc=utc_now(),
                      runtime={"python": sys.version, "numpy": np.__version__, "platform": platform.platform()})
        failed = [row for row in checks if not row["passed"]]
        report.update(status="PASS_RETAINED_ARTIFACT_AUDIT_WITH_EXPLICIT_STREAM_LIMITS" if not failed else "FAIL_RETAINED_ARTIFACT_AUDIT",
                      check_count=len(checks), failed_check_count=len(failed), first_mismatch=failed[0] if failed else None)
        with args.report.open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": report["status"], "checks": len(checks), "failed": len(failed),
                          "report": str(args.report), "report_sha256": digest(args.report.read_bytes())}))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
