"""Prepare two frozen OpenFWI velocity-only stress batches with bounded ranges.

The author archive is 681 MB. This script fetches its ZIP directory and two
compressed members only; a server that ignores Range is rejected before its
body is read. No seismic data, model training, or evaluation is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import struct
import tempfile
import urllib.request
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np

from wavecert.provenance import environment_record, sha256_file

MAX_UNCOMPRESSED_BYTES = 10_000_000
MAX_COMPRESSED_BYTES = 2_000_000
MAX_METADATA_BYTES = 1_000_000
ZIP_TAIL_BYTES = 65_536


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_relative_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or ".." in path.parts:
        raise ValueError("path must be a contained relative POSIX path")
    return path


def _contained_path(root: Path, name: str) -> Path:
    candidate = (root / _safe_relative_name(name)).resolve()
    if not candidate.is_relative_to(root.resolve()) or candidate == root.resolve():
        raise ValueError("path escapes the configured root")
    return candidate


def _atomic_write(root: Path, name: str, data: bytes) -> None:
    destination = _contained_path(root, name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".openfwi-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def read_range(url: str, start: int, end: int, total: int, *, opener=urllib.request.urlopen) -> bytes:
    """Require an exact bounded 206 response; never consume a full-archive 200."""
    if not 0 <= start <= end < total:
        raise ValueError("invalid byte range")
    request = urllib.request.Request(url, headers={
        "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
    })
    expected_length = end - start + 1
    with opener(request, timeout=45) as response:
        if response.status != 206:
            raise ValueError("server must return HTTP 206; full-response fallback is forbidden")
        expected_range = f"bytes {start}-{end}/{total}"
        if response.headers.get("Content-Range") != expected_range:
            raise ValueError("Content-Range does not match requested bytes and pinned archive size")
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise ValueError("encoded range responses are unsupported")
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != expected_length:
            raise ValueError("Content-Length does not match requested range")
        data = response.read(expected_length + 1)
        if len(data) != expected_length:
            raise ValueError("range body is truncated or exceeds its declared bound")
    return data


def parse_central_directory(tail: bytes, archive_bytes: int) -> tuple[dict, dict]:
    """Parse only a conventional single-disk ZIP whose entire directory is in tail."""
    end_offset = tail.rfind(b"PK\x05\x06")
    if end_offset < 0 or end_offset + 22 > len(tail):
        raise ValueError("ZIP end-of-central-directory record is absent or truncated")
    end = struct.unpack_from("<4s4H2LH", tail, end_offset)
    _, disk, directory_disk, entries_on_disk, entries, directory_bytes, directory_offset, comment = end
    if disk or directory_disk or entries != entries_on_disk or entries == 65535:
        raise ValueError("multi-disk and ZIP64 archives are unsupported")
    if end_offset + 22 + comment != len(tail):
        raise ValueError("ZIP end record does not end at pinned archive boundary")
    tail_start = archive_bytes - len(tail)
    cursor = directory_offset - tail_start
    directory_end = cursor + directory_bytes
    if cursor < 0 or directory_end != end_offset:
        raise ValueError("complete ZIP central directory must fit in the bounded tail")
    records = {}
    for _ in range(entries):
        if cursor + 46 > directory_end or tail[cursor:cursor + 4] != b"PK\x01\x02":
            raise ValueError("invalid ZIP central-directory entry")
        entry = struct.unpack_from("<4s6H3L5H2L", tail, cursor)
        name_bytes, extra_bytes, comment_bytes = entry[10:13]
        next_cursor = cursor + 46 + name_bytes + extra_bytes + comment_bytes
        if next_cursor > directory_end:
            raise ValueError("truncated ZIP central-directory entry")
        name = tail[cursor + 46:cursor + 46 + name_bytes].decode(
            "utf-8" if entry[3] & 0x800 else "cp437"
        )
        _safe_relative_name(name)
        if name in records:
            raise ValueError("duplicate ZIP member names are unsupported")
        records[name] = {
            "name": name, "flags": entry[3], "compression_method": entry[4],
            "crc32": f"{entry[7]:08x}", "compressed_bytes": entry[8],
            "uncompressed_bytes": entry[9], "local_header_offset": entry[16],
            "external_attributes": entry[15], "disk_start": entry[13],
        }
        cursor = next_cursor
    if cursor != directory_end:
        raise ValueError("ZIP central-directory size or entry count is inconsistent")
    return records, {
        "entries": entries, "directory_bytes": directory_bytes,
        "directory_offset": directory_offset, "tail_bytes_downloaded": len(tail),
        "tail_sha256": _sha256(tail), "archive_bytes": archive_bytes,
    }


def bounded_inflate(compressed: bytes, expected_bytes: int, expected_crc32: str) -> bytes:
    """Raw-deflate extraction with a hard output cap, exact size and CRC check."""
    if not 0 < expected_bytes <= MAX_UNCOMPRESSED_BYTES:
        raise ValueError("member uncompressed size exceeds the hard cap")
    if len(compressed) > MAX_COMPRESSED_BYTES:
        raise ValueError("member compressed size exceeds the hard cap")
    inflater = zlib.decompressobj(wbits=-15)
    data = inflater.decompress(compressed, expected_bytes + 1)
    if len(data) > expected_bytes or inflater.unconsumed_tail:
        raise ValueError("decompressed member exceeds its pinned size")
    if not inflater.eof or inflater.unused_data or len(data) != expected_bytes:
        raise ValueError("compressed member is truncated, trailing, or has incorrect size")
    if f"{zlib.crc32(data) & 0xFFFFFFFF:08x}" != expected_crc32:
        raise ValueError("member CRC32 does not match central-directory identity")
    return data


def extract_member(url: str, total: int, entry: dict, family: dict, *, opener=urllib.request.urlopen):
    """Verify pinned central/local records before fetching a bounded payload."""
    pins = {
        "compressed_bytes": family["expected_compressed_bytes"],
        "uncompressed_bytes": family["expected_uncompressed_bytes"],
        "local_header_offset": family["expected_local_header_offset"],
        "crc32": family["expected_crc32"],
    }
    if entry["name"] != family["member"] or any(entry[key] != value for key, value in pins.items()):
        raise ValueError("ZIP member identity differs from frozen protocol")
    file_type = (entry["external_attributes"] >> 16) & 0o170000
    if entry["flags"] != 0 or entry["compression_method"] != 8 or entry["disk_start"] or file_type not in {0, 0o100000}:
        raise ValueError("only unencrypted regular raw-deflate members without data descriptors are supported")
    if not 0 < entry["compressed_bytes"] <= MAX_COMPRESSED_BYTES or not 0 < entry["uncompressed_bytes"] <= MAX_UNCOMPRESSED_BYTES:
        raise ValueError("member sizes exceed hard download/decompression caps")
    offset = entry["local_header_offset"]
    header = read_range(url, offset, offset + 29, total, opener=opener)
    local = struct.unpack("<4s5H3L2H", header)
    if local[0] != b"PK\x03\x04" or local[2] != entry["flags"] or local[3] != 8:
        raise ValueError("local ZIP header signature, flags or method differ")
    if f"{local[6]:08x}" != entry["crc32"] or local[7] != entry["compressed_bytes"] or local[8] != entry["uncompressed_bytes"]:
        raise ValueError("local ZIP CRC or sizes differ from central directory")
    name_bytes, extra_bytes = local[9:11]
    if name_bytes < 1 or name_bytes + extra_bytes > 65535:
        raise ValueError("local ZIP filename/extra metadata is invalid or oversized")
    name_and_extra = read_range(url, offset + 30, offset + 29 + name_bytes + extra_bytes, total, opener=opener)
    if name_and_extra[:name_bytes].decode("cp437") != entry["name"]:
        raise ValueError("local ZIP filename differs from central directory")
    payload_offset = offset + 30 + name_bytes + extra_bytes
    compressed = read_range(url, payload_offset, payload_offset + entry["compressed_bytes"] - 1, total, opener=opener)
    raw = bounded_inflate(compressed, entry["uncompressed_bytes"], entry["crc32"])
    return raw, {
        **entry, "local_header_sha256": _sha256(header + name_and_extra),
        "local_name_bytes": name_bytes, "local_extra_bytes": extra_bytes,
        "payload_offset": payload_offset, "compressed_sha256": _sha256(compressed),
        "crc32_verified": True,
        "downloaded_bytes": 30 + len(name_and_extra) + len(compressed),
    }


def _duplicate_groups(hashes):
    groups = defaultdict(list)
    for index, value in enumerate(hashes):
        groups[value].append(index)
    return [indices for indices in groups.values() if len(indices) > 1]


def resolve_calibration_snapshot(root: Path, checkpoint: dict) -> tuple[bytes, str]:
    """Resolve the exact protocol-pinned calibration bytes without recalibration.

    A later reproduction may overwrite the original results summary. Prefer
    that original path only when its bytes match; otherwise require the bundled
    historical snapshot to match the same frozen hash. Neither candidate can
    substitute a newly computed scale or bypass the protocol identity check.
    """
    root = root.resolve()
    label = checkpoint["label"]
    expected = checkpoint["calibration_summary_sha256"]
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", label):
        raise ValueError("checkpoint label must be safe")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("calibration summary must have a frozen SHA256")
    candidates = [
        checkpoint["calibration_summary_path"],
        f"data/reference/openfwi_source_calibration/{label}.json",
    ]
    for relative in candidates:
        candidate = _contained_path(root, relative)
        if candidate.is_file():
            data = candidate.read_bytes()
            if _sha256(data) == expected:
                return data, candidate.relative_to(root).as_posix()
    raise ValueError(
        "neither original calibration summary nor bundled snapshot matches the frozen protocol"
    )


def prepare(protocol_path: Path, output: Path, *, opener=urllib.request.urlopen) -> dict:
    root = Path(__file__).resolve().parents[1]
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol["schema_version"] != 1 or len(protocol["families"]) != 2:
        raise ValueError("this frozen preparation protocol requires schema1 and exactly two families")
    archive = protocol["archive"]
    if archive["content_url"] != "https://zenodo.org/api/records/7272136/files/WAVE_FNO.zip/content" or archive["expected_bytes"] != 681048652:
        raise ValueError("archive URL and size must match the pinned author release")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new or empty; existing preparation is preserved")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    selections = {}
    seen_codes = set()
    for family in protocol["families"]:
        name, code = family["name"], family["family_code"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name in selections or isinstance(code, bool) or not isinstance(code, int) or code < 0 or code in seen_codes:
            raise ValueError("family names/codes must be safe and unique")
        if family["expected_n_models"] != 500 or family["selected_n_models"] != 100:
            raise ValueError("frozen cohort requires 100 of 500 original models per family")
        _safe_relative_name(family["member"])
        rng = np.random.default_rng(np.random.SeedSequence([protocol["selection"]["seed"], code]))
        selections[name] = np.sort(rng.choice(500, size=100, replace=False)).tolist()
        seen_codes.add(code)
    calibration_snapshots = {}
    for checkpoint in protocol["checkpoints"]:
        label = checkpoint["label"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", label) or label in calibration_snapshots:
            raise ValueError("checkpoint labels must be safe and unique")
        if sha256_file(_contained_path(root, checkpoint["path"])) != checkpoint["sha256"]:
            raise ValueError("checkpoint differs from frozen protocol")
        calibration, resolved_path = resolve_calibration_snapshot(root, checkpoint)
        calibration_snapshots[label] = {"bytes": calibration, "resolved_snapshot_path": resolved_path}

    # Publish the exact protocol, selections and freeze timestamp before ANY
    # velocity payload request or inspection. Never resample after inspection.
    protocol_record = {
        "protocol_sha256": _sha256(protocol_bytes),
        "recorded_utc_before_payload": datetime.now(timezone.utc).isoformat(),
        "selected_indices_before_payload": selections,
        "scope": "Assistant-development engineering record, including any explicit layout-only amendment; not student preregistration",
    }
    _atomic_write(output, "protocol.json", protocol_bytes)
    _atomic_write(output, "protocol_record.json", _json_bytes(protocol_record))
    _atomic_write(output, "status.json", _json_bytes({"status": "running", **protocol_record}))
    frozen_calibrations = []
    for checkpoint in protocol["checkpoints"]:
        relative = f"calibration/{checkpoint['label']}.json"
        snapshot = calibration_snapshots[checkpoint["label"]]
        data = snapshot["bytes"]
        _atomic_write(output, relative, data)
        frozen_calibrations.append({
            "label": checkpoint["label"], "path": relative,
            "source_path": checkpoint["calibration_summary_path"], "sha256": _sha256(data),
            "resolved_snapshot_path": snapshot["resolved_snapshot_path"],
        })
    provenance = environment_record(root)
    request = urllib.request.Request(archive["metadata_url"], headers={"Accept-Encoding": "identity"})
    with opener(request, timeout=45) as response:
        if response.status != 200:
            raise ValueError("Zenodo record metadata request failed")
        metadata_bytes = response.read(MAX_METADATA_BYTES + 1)
    if len(metadata_bytes) > MAX_METADATA_BYTES:
        raise ValueError("Zenodo record metadata exceeds hard cap")
    metadata = json.loads(metadata_bytes)
    if metadata["id"] != archive["record_id"] or metadata.get("doi") != archive["doi"]:
        raise ValueError("Zenodo record identity differs from frozen protocol")
    released = [file for file in metadata["files"] if file["key"] == archive["file_name"]]
    if len(released) != 1 or released[0]["size"] != archive["expected_bytes"] or released[0]["checksum"] != f"md5:{archive['advertised_md5']}":
        raise ValueError("Zenodo release file metadata differs from frozen protocol")
    _atomic_write(output, "zenodo_record.json", metadata_bytes)
    total = archive["expected_bytes"]
    tail = read_range(archive["content_url"], total - ZIP_TAIL_BYTES, total - 1, total, opener=opener)
    entries, directory = parse_central_directory(tail, total)
    families = []
    all_model_groups = defaultdict(list)
    for family in protocol["families"]:
        entry = entries.get(family["member"])
        if entry is None:
            raise ValueError("frozen ZIP member is absent")
        raw, zip_record = extract_member(archive["content_url"], total, entry, family, opener=opener)
        pinned_raw_hash = protocol.get("layout_amendment", {}).get("verified_raw_sha256", {}).get(family["name"])
        if pinned_raw_hash is not None and _sha256(raw) != pinned_raw_hash:
            raise ValueError("raw velocity file differs from header-verified layout-amendment identity")
        batch = np.load(io.BytesIO(raw), allow_pickle=False)
        if list(batch.shape) != family["expected_shape"] or str(batch.dtype) != family["expected_dtype"]:
            raise ValueError(f"velocity batch dimensions/dtype differ from frozen protocol: {batch.shape}, {batch.dtype}")
        if not np.all(np.isfinite(batch)) or np.any(batch <= 0):
            raise ValueError("source velocities must be finite and positive; no clipping is permitted")
        hashes = [_sha256(np.ascontiguousarray(model).tobytes()) for model in batch]
        selected = selections[family["name"]]
        selected_hashes = [hashes[index] for index in selected]
        model_ids = [f"{family['name']}:{family['member']}:{index}" for index in selected]
        for index, value in enumerate(hashes):
            all_model_groups[value].append(f"{family['name']}:{family['member']}:{index}")
        relative = f"{family['name']}/velocity.npy"
        _atomic_write(output, relative, raw)
        families.append({
            "name": family["name"], "family_code": family["family_code"],
            "member": family["member"], "path": relative, "sha256": _sha256(raw),
            "shape": list(batch.shape), "dtype": str(batch.dtype), "raw_bytes": len(raw),
            "selected_indices": selected, "selected_model_ids": model_ids,
            "selected_model_sha256": selected_hashes,
            "model_hash_representation": f"SHA256 of original {batch.dtype} row shape{batch.shape[1:]}, C-contiguous bytes",
            "duplicate_hash_check": {
                "original_batch_unique_models": len(set(hashes)),
                "original_batch_duplicate_index_groups": _duplicate_groups(hashes),
                "selected_unique_models": len(set(selected_hashes)),
                "selected_duplicate_positions": _duplicate_groups(selected_hashes),
                "handling": protocol["selection"]["duplicate_handling"],
            },
            "zip_member": zip_record,
        })
    manifest = {
        "schema_version": 1, "status": "complete", **protocol_record,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "archive": {**archive, "full_archive_md5_verified": False},
        "zenodo_metadata_sha256": _sha256(metadata_bytes),
        "central_directory": directory, "families": families,
        "frozen_calibration_summaries": frozen_calibrations,
        "cross_family_or_batch_duplicate_model_groups": [
            group for group in all_model_groups.values() if len(group) > 1
        ],
        "total_archive_range_bytes_downloaded": len(tail) + sum(
            family["zip_member"]["downloaded_bytes"] for family in families
        ),
        "attribution": protocol["attribution"],
        "scope": "Velocity-only author-provided OpenFWI subset; no training or evaluation performed; not official benchmark/field data or verified IID site sampling",
        "provenance": provenance,
    }
    _atomic_write(output, "preparation_manifest.json", _json_bytes(manifest))
    _atomic_write(output, "status.json", _json_bytes({"status": "complete", **protocol_record}))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/openfwi_external_stress.json"))
    parser.add_argument("--output", type=Path, default=Path("data/external/openfwi_stress"))
    args = parser.parse_args()
    output_was_empty = not args.output.exists() or (args.output.is_dir() and not any(args.output.iterdir()))
    try:
        result = prepare(args.protocol, args.output)
    except BaseException as exc:
        if output_was_empty and (args.output / "protocol_record.json").is_file() and not (args.output / "preparation_manifest.json").exists():
            _atomic_write(args.output.resolve(), "status.json", _json_bytes({
                "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
            }))
        raise
    print(json.dumps({
        "status": result["status"], "archive_range_bytes": result["total_archive_range_bytes_downloaded"],
        "families": [{"name": item["name"], "selected_models": len(item["selected_indices"]),
                      "sha256": item["sha256"], "unique_models": item["duplicate_hash_check"]["selected_unique_models"]}
                     for item in result["families"]],
        "full_archive_md5_verified": False,
    }, indent=2))


if __name__ == "__main__":
    main()
