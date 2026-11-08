"""Synthetic bytes/mock HTTP only. No BP velocity/network/PDE input is used."""
import base64
import gzip
import io
import json
import struct
import zlib
from pathlib import Path

import collect_bp2004 as module
import numpy as np
import pytest
from collect_bp2004 import (
    AcquisitionError,
    BoundedGzipStream,
    array_bytes,
    contained_path,
    decode_samples,
    fetch_bounded,
    load_protocol,
    parse_velocity_gzip,
    resolve_windows,
    sha256,
)


def window():
    return {"id": "fixed", "start_z_x": [1, 2], "stop_exclusive_z_x": [7, 6],
                "raw_shape_z_x": [6, 4], "stride_z_x": [2, 1], "decimated_shape_z_x": [3, 4]}


def synthetic(ns=8, ntr=10, code=5, trace_ns=None, extended=0):
    text = b"C 1 SYNTHETIC ONLY".ljust(3200, b" ")
    binary = bytearray(400)
    struct.pack_into(">H", binary, 20, ns)
    struct.pack_into(">H", binary, 24, code)
    struct.pack_into(">h", binary, 304, extended)
    parts, headers = [text, bytes(binary)], []
    values = np.arange(ns, dtype=np.float32)[:, None] + 10 * np.arange(ntr, dtype=np.float32)[None, :] + 1
    if code == 1:
        values[:] = 1000
    for x in range(ntr):
        header = bytearray(240)
        struct.pack_into(">i", header, 0, x + 1)
        struct.pack_into(">i", header, 4, x + 1)
        struct.pack_into(">H", header, 114, ns if trace_ns is None else trace_ns)
        struct.pack_into(">H", header, 116, 6250)
        struct.pack_into(">i", header, 72, x * 125)
        headers.append(bytes(header))
        samples = values[:, x].astype(">f4").tobytes() if code != 1 else struct.pack(">I", 0x433e8000) * ns
        parts.extend((bytes(header), samples))
    raw = b"".join(parts)
    return gzip.compress(raw, mtime=0), raw, values, b"".join(headers)


@pytest.mark.parametrize("code", [1, 5])
@pytest.mark.parametrize("trace_ns", [None, 0])
def test_full_integrity_headers_and_fixed_parent_crops_without_interpolation(code, trace_ns):
    payload, raw, values, headers = synthetic(code=code, trace_ns=trace_ns)
    crops, receipt = parse_velocity_gzip(payload, samples_per_trace=8, traces=10, windows=[window()])
    np.testing.assert_array_equal(crops["fixed"], values[1:7, 2:6])
    np.testing.assert_array_equal(crops["fixed"][::2], values[1:7:2, 2:6])
    assert receipt["uncompressed_sha256"] == sha256(raw)
    assert receipt["gzip_crc32_hex"] == f"{zlib.crc32(raw):08x}"
    assert receipt["full_gzip_crc_and_isize_verified"]
    assert receipt["uncompressed_bytes"] == len(raw)
    assert zlib.decompress(base64.b64decode(receipt["headers"]["trace_headers_base64"])) == headers
    assert receipt["headers"]["trace_headers_sha256"] == sha256(headers)
    assert base64.b64decode(receipt["headers"]["text_header_base64"]) == raw[:3200]
    assert base64.b64decode(receipt["headers"]["binary_header_base64"]) == raw[3200:3600]
    assert not receipt["full_decoded_velocity_array_allocated"]
    encoded = array_bytes(crops["fixed"])
    np.testing.assert_array_equal(np.load(io.BytesIO(encoded), allow_pickle=False), crops["fixed"])
    json.dumps(receipt, allow_nan=False)


def test_frozen_three_windows_and_native_stride_geometry():
    protocol, raw = load_protocol(Path(__file__).with_name("protocol.json"))
    assert len(raw) < 20000
    assert protocol["independent_ancestor_count"] == 1
    assert [w["start_z_x"] for w in protocol["windows"]] == [[382, 983], [859, 2601], [1337, 4219]]
    assert protocol["layout"]["expected_uncompressed_bytes"] == 3600 + 5395 * (240 + 1911 * 4)
    assert protocol["layout"]["native_spacing_z_x_m"][0] * 2 == 12.5
    changed = [window()]
    changed[0]["start_z_x"] = [7, 2]
    with pytest.raises(AcquisitionError, match="escapes"):
        resolve_windows((8, 10), changed)


@pytest.mark.parametrize("kind", ["crc", "isize", "truncated", "extra_member", "extra_compressed_bytes"])
def test_gzip_integrity_failures_are_not_published(kind):
    payload, _, _, _ = synthetic()
    if kind == "crc":
        b = bytearray(payload)
        b[-8] ^= 1
        payload = bytes(b)
    elif kind == "isize":
        b = bytearray(payload)
        b[-4] ^= 1
        payload = bytes(b)
    elif kind == "truncated":
        payload = payload[:-3]
    elif kind == "extra_member":
        payload += gzip.compress(b"", mtime=0)
    else:
        payload += b"\x00"
    with pytest.raises(AcquisitionError, match="gzip"):
        parse_velocity_gzip(payload, samples_per_trace=8, traces=10, windows=[window()])


def test_decompression_hard_cap_and_fixed_trace_count():
    reader = BoundedGzipStream(gzip.compress(b"A" * 1000000, mtime=0), maximum_output=1000)
    with pytest.raises(AcquisitionError, match="cap"):
        reader.read_exact(1001)
    payload, _, _, _ = synthetic()
    for count in (9, 11):
        with pytest.raises(AcquisitionError):
            parse_velocity_gzip(payload, samples_per_trace=8, traces=count, windows=[window()])


@pytest.mark.parametrize("kind", ["binary_ns", "trace_ns", "sample_format", "extended", "trailing_segy", "zero", "nan"])
def test_strict_layout_and_value_failures(kind):
    payload, raw, _, _ = synthetic(trace_ns=7 if kind == "trace_ns" else None,
                                    extended=1 if kind == "extended" else 0)
    raw = bytearray(raw)
    if kind == "binary_ns":
        struct.pack_into(">H", raw, 3220, 9)
    elif kind == "sample_format":
        struct.pack_into(">H", raw, 3224, 6)
    elif kind == "trailing_segy":
        raw += b"x"
    elif kind in ("zero", "nan"):
        struct.pack_into(">f", raw, 3840, 0. if kind == "zero" else np.nan)
    payload = gzip.compress(raw, mtime=0)
    with pytest.raises(AcquisitionError):
        parse_velocity_gzip(payload, samples_per_trace=8, traces=10, windows=[window()])


def test_known_ibm_exponents_and_unrepresentable_or_nonpositive_values():
    words = [0x41100000, 0x40800000, 0x42100000, 0x433e8000]
    decoded = decode_samples(b"".join(struct.pack(">I", v) for v in words), 1)
    np.testing.assert_array_equal(decoded, [1., .5, 16., 1000.])
    for invalid in (0x00000000, 0xc1100000, 0x7fffffff, 0x00100000):
        with pytest.raises(AcquisitionError, match="finite positive"):
            decode_samples(struct.pack(">I", invalid), 1)


class Response(io.BytesIO):
    def __init__(self, payload, *, status=200, headers=None, url="https://official.invalid/model"):
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Length": str(len(payload)), "ETag": '"frozen"', **(headers or {})}
        self.url = url

    def geturl(self):
        return self.url


@pytest.mark.parametrize("kind", ["status", "length", "etag", "encoding", "redirect", "oversize", "truncated"])
def test_bounded_http_failures_use_mock_responses_only(kind):
    payload = b"abcd"
    kwargs = {}
    if kind == "status":
        kwargs["status"] = 206
    elif kind == "length":
        kwargs["headers"] = {"Content-Length": "5"}
    elif kind == "etag":
        kwargs["headers"] = {"ETag": '"changed"'}
    elif kind == "encoding":
        kwargs["headers"] = {"Content-Encoding": "gzip"}
    elif kind == "redirect":
        kwargs["url"] = "https://different.invalid/model"
    elif kind == "oversize":
        payload = b"abcde"
        kwargs["headers"] = {"Content-Length": "4"}
    else:
        payload = b"abc"
        kwargs["headers"] = {"Content-Length": "4"}
    with pytest.raises(AcquisitionError):
        fetch_bounded("https://official.invalid/model", expected_bytes=4, maximum_bytes=4,
                      expected_etag="frozen", opener=lambda *a, **k: Response(payload, **kwargs))


def test_http_exact_body_and_sha_are_recorded():
    data, receipt = fetch_bounded("https://official.invalid/model", expected_bytes=4, maximum_bytes=4,
        expected_etag="frozen", opener=lambda *a, **k: Response(b"abcd"))
    assert data == b"abcd" and receipt["compressed_sha256"] == sha256(b"abcd")
    assert receipt["etag_is_not_sha256"]


@pytest.mark.parametrize("name", ["../escape.npy", "/escape.npy", "nested/escape.npy", ".."])
def test_output_paths_are_contained(tmp_path, name):
    with pytest.raises(AcquisitionError):
        contained_path(tmp_path, name)


def test_default_cli_never_performs_network(monkeypatch, capsys):
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: pytest.fail("unexpected network"))
    assert module.main([]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "protocol_only_no_network"


def test_unapproved_source_hash_refuses_before_network(tmp_path):
    with pytest.raises(AcquisitionError, match="SHA256 mismatch"):
        module.collect(Path(__file__).with_name("protocol.json"), tmp_path / "new", "wrong", "wrong",
                       opener=lambda *a, **k: pytest.fail("network before approval hashes"))
    assert not (tmp_path / "new").exists()


def synthetic_collection(tmp_path, monkeypatch):
    """Exercise publication with tiny invented geometry, never the BP payload."""
    protocol, _ = load_protocol(Path(__file__).with_name("protocol.json"))
    payload, raw, _, _ = synthetic()
    protocol["source"].update(expected_compressed_bytes=len(payload), maximum_compressed_bytes=len(payload),
                              expected_etag="frozen", url="https://official.invalid/model")
    protocol["layout"].update(samples_per_trace=8, traces=10)
    protocol["windows"] = [window()]
    protocol_raw = json.dumps(protocol, sort_keys=True).encode()
    path = tmp_path / "protocol.json"
    path.write_bytes(protocol_raw)
    for name in ("documentation_manifest.json", "original_BP2004_READMES.pdf"):
        (tmp_path / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    monkeypatch.setattr(module, "load_protocol", lambda _: (protocol, protocol_raw))
    expected_source = sha256(Path(module.__file__).read_bytes())
    return path, protocol_raw, expected_source, payload, raw


def test_synthetic_collection_publishes_only_crops_and_complete_provenance(tmp_path, monkeypatch):
    path, protocol_raw, source_sha, payload, raw = synthetic_collection(tmp_path, monkeypatch)
    output = tmp_path / "collected"
    receipt = module.collect(path, output, sha256(protocol_raw), source_sha,
                             opener=lambda *a, **k: Response(payload))
    assert receipt["segy"]["uncompressed_sha256"] == sha256(raw)
    assert receipt["http"]["compressed_sha256"] == sha256(payload)
    assert receipt["independent_ancestor_count"] == 1
    assert not receipt["raw_payload_written_to_disk"]
    assert not receipt["plots_or_model_outcomes_computed"]
    assert len(list(output.glob("*.npy"))) == 2
    assert not list(output.glob("*.gz")) and not list(output.glob("*.segy"))
    assert (output / "README_Modification.md").is_file()
    for filename, digest in receipt["artifact_sha256"].items():
        assert sha256((output / filename).read_bytes()) == digest
    assert json.loads((output / "receipt.json").read_text()) == receipt
    with pytest.raises(AcquisitionError, match="already exists"):
        module.collect(path, output, sha256(protocol_raw), source_sha,
                       opener=lambda *a, **k: pytest.fail("must refuse existing evidence before network"))


def test_publication_failure_removes_only_own_stage_and_exposes_no_partial_output(tmp_path, monkeypatch):
    path, protocol_raw, source_sha, payload, _ = synthetic_collection(tmp_path, monkeypatch)
    untouched = tmp_path / "unrelated.txt"
    untouched.write_text("retain")
    output = tmp_path / "collected"
    monkeypatch.setattr(module.os, "rename", lambda *a: (_ for _ in ()).throw(OSError("synthetic I/O failure")))
    with pytest.raises(OSError, match="I/O failure"):
        module.collect(path, output, sha256(protocol_raw), source_sha,
                       opener=lambda *a, **k: Response(payload))
    assert not output.exists() and not list(tmp_path.glob(".collected-*"))
    assert untouched.read_text() == "retain"
