from __future__ import annotations

import importlib.util
import io
import struct
import zipfile
import zlib
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "openfwi_preparation", Path(__file__).resolve().parents[1] / "scripts/prepare_openfwi_stress.py",
)
prep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prep)


class Response:
    def __init__(self, body, *, status=206, headers=None):
        self.body, self.status, self.headers = body, status, headers or {}
        self.read_sizes = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self, size):
        self.read_sizes.append(size)
        return self.body[:size]


def test_range_ignoring_server_is_rejected_before_body_read():
    response = Response(b"large archive body must not be consumed", status=200)
    with pytest.raises(ValueError, match="full-response fallback is forbidden"):
        prep.read_range("https://example.test/archive", 3, 7, 100, opener=lambda *_a, **_k: response)
    assert response.read_sizes == []


@pytest.mark.parametrize("body,headers,message", [
    (b"abcde", {"Content-Range": "bytes 3-7/101"}, "Content-Range"),
    (b"abcde", {"Content-Range": "bytes 2-6/100"}, "Content-Range"),
    (b"abcde", {"Content-Range": "bytes 3-7/100", "Content-Length": "6"}, "Content-Length"),
    (b"abcd", {"Content-Range": "bytes 3-7/100"}, "truncated or exceeds"),
    (b"abcdef", {"Content-Range": "bytes 3-7/100"}, "truncated or exceeds"),
    (b"abcde", {"Content-Range": "bytes 3-7/100", "Content-Encoding": "gzip"}, "encoded range"),
])
def test_range_requires_exact_identity_and_bounded_body(body, headers, message):
    response = Response(body, headers=headers)
    with pytest.raises(ValueError, match=message):
        prep.read_range("https://example.test/archive", 3, 7, 100, opener=lambda *_a, **_k: response)
    assert all(size <= 6 for size in response.read_sizes)


def _archive(member="DATA/family/velocity.npy", raw=b"a valid bounded payload" * 30):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, raw)
    data = stream.getvalue()
    entries, _ = prep.parse_central_directory(data, len(data))
    entry = entries[member]
    family = {
        "member": member, "expected_compressed_bytes": entry["compressed_bytes"],
        "expected_uncompressed_bytes": entry["uncompressed_bytes"],
        "expected_local_header_offset": entry["local_header_offset"], "expected_crc32": entry["crc32"],
    }
    return data, entry, family, raw


def _range_opener(data):
    def opener(request, **_):
        first, last = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        return Response(data[first:last + 1], headers={
            "Content-Range": f"bytes {first}-{last}/{len(data)}", "Content-Length": str(last - first + 1),
        })
    return opener


def test_member_extraction_verifies_records_and_payload():
    data, entry, family, raw = _archive()
    extracted, identity = prep.extract_member(
        "https://example.test/archive", len(data), entry, family, opener=_range_opener(data),
    )
    assert extracted == raw
    assert identity["crc32_verified"] is True
    assert identity["payload_offset"] + identity["compressed_bytes"] <= len(data)


def test_wrong_crc_is_rejected_after_bounded_decompression():
    compressed = zlib.compress(b"the actual bytes")[2:-4]
    with pytest.raises(ValueError, match="CRC32"):
        prep.bounded_inflate(compressed, len(b"the actual bytes"), "00000000")


def test_decompression_size_cap_and_truncation():
    compressed = zlib.compress(b"x" * 10000)[2:-4]
    with pytest.raises(ValueError, match="exceeds its pinned size"):
        prep.bounded_inflate(compressed, 100, "00000000")
    with pytest.raises(ValueError, match="hard cap"):
        prep.bounded_inflate(compressed, prep.MAX_UNCOMPRESSED_BYTES + 1, "00000000")
    with pytest.raises(ValueError, match="truncated, trailing"):
        prep.bounded_inflate(compressed[:-2], 10000, "00000000")


def test_local_filename_and_method_must_match_directory():
    data, entry, family, _ = _archive()
    changed = bytearray(data)
    changed[30] ^= 1
    with pytest.raises(ValueError, match="filename differs"):
        prep.extract_member("https://example.test/archive", len(data), entry, family, opener=_range_opener(bytes(changed)))
    changed = bytearray(data)
    struct.pack_into("<H", changed, 8, 0)
    with pytest.raises(ValueError, match="signature, flags or method"):
        prep.extract_member("https://example.test/archive", len(data), entry, family, opener=_range_opener(bytes(changed)))


def test_member_metadata_cap_is_checked_before_any_request():
    data, entry, family, _ = _archive()
    entry["compressed_bytes"] = family["expected_compressed_bytes"] = prep.MAX_COMPRESSED_BYTES + 1
    with pytest.raises(ValueError, match="hard download/decompression caps"):
        prep.extract_member("https://example.test/archive", len(data), entry, family,
                            opener=lambda *_a, **_k: pytest.fail("must reject before network"))


@pytest.mark.parametrize("name", ["../outside", "/outside", "a/../../outside", "a\\outside"])
def test_output_paths_and_archive_members_reject_traversal(tmp_path, name):
    with pytest.raises(ValueError, match="contained relative"):
        prep._atomic_write(tmp_path, name, b"must not be written")
    data, _, _, _ = _archive()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, b"must not be extracted")
    data = stream.getvalue()
    with pytest.raises(ValueError, match="contained relative"):
        prep.parse_central_directory(data, len(data))


def test_output_symlink_cannot_escape_root_and_atomic_write_leaves_no_partial(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        prep._atomic_write(root, "link/payload.npy", b"no")
    prep._atomic_write(root, "family/payload.npy", b"verified payload")
    assert (root / "family/payload.npy").read_bytes() == b"verified payload"
    assert list((root / "family").iterdir()) == [root / "family/payload.npy"]


def _calibration_fixture(root):
    frozen = b'{"conformal_scale": 0.25, "frozen_development_summary": true}\n'
    checkpoint = {
        "label": "forward_only",
        "calibration_summary_path": "results/grouped_wavecert/summary.json",
        "calibration_summary_sha256": prep._sha256(frozen),
    }
    original = root / checkpoint["calibration_summary_path"]
    bundled = root / "data/reference/openfwi_source_calibration/forward_only.json"
    original.parent.mkdir(parents=True)
    bundled.parent.mkdir(parents=True)
    return frozen, checkpoint, original, bundled


@pytest.mark.parametrize("original_exists", [True, False])
def test_stale_or_missing_calibration_uses_exact_bundled_snapshot(tmp_path, original_exists):
    frozen, checkpoint, original, bundled = _calibration_fixture(tmp_path)
    if original_exists:
        original.write_bytes(b'{"conformal_scale": 0.9, "later_reproduction": true}\n')
    bundled.write_bytes(frozen)
    data, resolved = prep.resolve_calibration_snapshot(tmp_path, checkpoint)
    assert data == frozen
    assert resolved == "data/reference/openfwi_source_calibration/forward_only.json"


def test_calibration_resolution_rejects_both_corrupt_candidates(tmp_path):
    _, checkpoint, original, bundled = _calibration_fixture(tmp_path)
    original.write_bytes(b'{"conformal_scale": 0.9}\n')
    bundled.write_bytes(b'{"conformal_scale": 0.1}\n')
    with pytest.raises(ValueError, match="neither original calibration summary nor bundled snapshot"):
        prep.resolve_calibration_snapshot(tmp_path, checkpoint)


def test_matching_original_calibration_is_preferred(tmp_path):
    frozen, checkpoint, original, bundled = _calibration_fixture(tmp_path)
    original.write_bytes(frozen)
    bundled.write_bytes(b"corrupt fallback must not replace matching original")
    data, resolved = prep.resolve_calibration_snapshot(tmp_path, checkpoint)
    assert data == frozen
    assert resolved == checkpoint["calibration_summary_path"]
