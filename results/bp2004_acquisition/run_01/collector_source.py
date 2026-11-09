"""Bounded BP2004 velocity acquisition. Import/default CLI never fetches velocity.

The six fixed crop arrays inherit ONE synthetic ancestor. This is a file-format
and provenance collector, with no PDE, model evaluation, plotting or tuning.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import struct
import sys
import tempfile
import urllib.request
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DOCUMENT_SHA256 = "e29b003fc2161c44d2b9f04448195e58ebe6f476a195b31c3be77440adb60e24"


class AcquisitionError(ValueError):
    """A bounded input/layout/integrity check failed; no output is published."""


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AcquisitionError(name + " must be a positive integer")
    return value


def resolve_windows(shape, windows):
    """Validate fixed coordinates. No velocity values enter this calculation."""
    nz, nx = shape
    resolved, seen = [], set()
    for window in windows:
        name = window["id"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in seen:
            raise AcquisitionError("unsafe or duplicate crop id")
        seen.add(name)
        size = tuple(_positive_integer(v, "window size") for v in window["raw_shape_z_x"])
        start = tuple(window["start_z_x"])
        stride = tuple(_positive_integer(v, "stride") for v in window["stride_z_x"])
        if len(size) != 2 or len(start) != 2 or len(stride) != 2:
            raise AcquisitionError("window coordinates must be two dimensional")
        if any(isinstance(v, bool) or not isinstance(v, int) for v in start):
            raise AcquisitionError("window starts must be integer indices")
        stop = tuple(a + b for a, b in zip(start, size, strict=True))
        if any(a < 0 or b > c for a, b, c in zip(start, stop, (nz, nx), strict=True)):
            raise AcquisitionError("crop escapes original model")
        if tuple(window["stop_exclusive_z_x"]) != stop:
            raise AcquisitionError("inconsistent crop stop")
        decimated = tuple((n + s - 1) // s for n, s in zip(size, stride, strict=True))
        if tuple(window["decimated_shape_z_x"]) != decimated:
            raise AcquisitionError("inconsistent decimated shape")
        if "requested_center_fraction_z_x" in window:
            fractions = window["requested_center_fraction_z_x"]
            if len(fractions) != 2 or not all(math.isfinite(v) and 0 <= v <= 1 for v in fractions):
                raise AcquisitionError("invalid center fraction")
            expected = tuple(math.floor(f * (n - 1) - (s - 1) / 2)
                             for f, n, s in zip(fractions, (nz, nx), size, strict=True))
            if start != expected:
                raise AcquisitionError("crop differs from frozen center rule")
        resolved.append({**window, "start_z_x": list(start), "raw_shape_z_x": list(size),
                         "stride_z_x": list(stride)})
    if not resolved:
        raise AcquisitionError("at least one window is required")
    return resolved


def load_protocol(path):
    raw = Path(path).read_bytes()
    protocol = json.loads(raw)
    layout = protocol["layout"]
    if (protocol["schema_version"] != 1 or protocol["independent_ancestor_count"] != 1
            or protocol["crops_are_independent_parents"] is not False
            or layout["samples_per_trace"] != 1911 or layout["traces"] != 5395
            or layout["byte_order"] != "big" or layout["accepted_sample_format_codes"] != [1, 5]
            or layout["native_spacing_z_x_m"] != [6.25, 12.5]):
        raise AcquisitionError("not the registered BP2004 source/layout protocol")
    expected = 3600 + layout["traces"] * (240 + 4 * layout["samples_per_trace"])
    if expected != layout["expected_uncompressed_bytes"] or expected != layout["maximum_uncompressed_bytes"]:
        raise AcquisitionError("inconsistent uncompressed byte budget")
    windows = resolve_windows((1911, 5395), protocol["windows"])
    expected_centers = [[.25, .20], [.50, .50], [.75, .80]]
    if ([w["requested_center_fraction_z_x"] for w in windows] != expected_centers
            or any(w["raw_shape_z_x"] != [192, 192] or w["stride_z_x"] != [2, 1] for w in windows)):
        raise AcquisitionError("registered three-window design changed")
    return protocol, raw


def fetch_bounded(url, *, expected_bytes, maximum_bytes, expected_etag, opener=None):
    """Full bounded HTTP200 only; never spill the compressed payload to disk."""
    opener = urllib.request.urlopen if opener is None else opener
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    with opener(request, timeout=30) as response:
        if response.status != 200 or response.geturl() != url:
            raise AcquisitionError("unexpected HTTP status or redirected source URL")
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise AcquisitionError("unexpected HTTP content encoding")
        try:
            length = int(response.headers["Content-Length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AcquisitionError("missing/invalid Content-Length") from exc
        if length != expected_bytes or length > maximum_bytes:
            raise AcquisitionError("compressed Content-Length differs from frozen budget")
        etag = response.headers.get("ETag", "").strip('"')
        if etag != expected_etag:
            raise AcquisitionError("source ETag changed; protocol review required")
        data, count = bytearray(), 0
        while True:
            chunk = response.read(min(65536, maximum_bytes - count + 1))
            if not chunk:
                break
            count += len(chunk)
            if count > maximum_bytes or count > expected_bytes:
                raise AcquisitionError("compressed body exceeds frozen byte budget")
            data.extend(chunk)
        if count != expected_bytes:
            raise AcquisitionError("truncated compressed HTTP body")
        return data, {"status": response.status, "final_url": response.geturl(),
                      "content_length": length, "etag": etag,
                      "last_modified": response.headers.get("Last-Modified"),
                      "compressed_bytes": count, "compressed_sha256": sha256(data),
                      "etag_is_not_sha256": True}


class BoundedGzipStream:
    """One gzip member, bounded streaming output, full CRC/ISIZE verification."""

    def __init__(self, compressed, maximum_output, chunk_size=65536):
        self.compressed = compressed
        self.maximum_output = maximum_output
        self.chunk_size = chunk_size
        self.count = self.crc = 0
        self.digest = hashlib.sha256()
        self.finished = False
        self._buffer = bytearray()
        self._chunks = self._decompress()

    def _decompress(self):
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            for offset in range(0, len(self.compressed), self.chunk_size):
                pending = self.compressed[offset:offset + self.chunk_size]
                while pending:
                    data = decoder.decompress(pending, self.chunk_size)
                    pending = decoder.unconsumed_tail
                    if data:
                        self.count += len(data)
                        if self.count > self.maximum_output:
                            raise AcquisitionError("uncompressed byte cap exceeded")
                        self.digest.update(data)
                        self.crc = zlib.crc32(data, self.crc)
                        yield data
                    if decoder.eof:
                        if decoder.unused_data or offset + self.chunk_size < len(self.compressed):
                            raise AcquisitionError("trailing gzip member or bytes are forbidden")
                        break
                if decoder.eof:
                    break
            if not decoder.eof:
                raise AcquisitionError("truncated gzip member")
            if len(self.compressed) < 18:
                raise AcquisitionError("invalid gzip member")
            expected_crc, expected_size = struct.unpack("<II", self.compressed[-8:])
            if self.crc & 0xffffffff != expected_crc or self.count & 0xffffffff != expected_size:
                raise AcquisitionError("gzip CRC32 or ISIZE mismatch")
            self.finished = True
        except zlib.error as exc:
            raise AcquisitionError("gzip integrity/decompression failure") from exc

    def read(self, size):
        if size < 0:
            raise AcquisitionError("unbounded reads are forbidden")
        while len(self._buffer) < size:
            try:
                self._buffer.extend(next(self._chunks))
            except StopIteration:
                break
        answer = bytes(self._buffer[:size])
        del self._buffer[:size]
        return answer

    def read_exact(self, size):
        data = self.read(size)
        if len(data) != size:
            raise AcquisitionError("truncated SEG-Y layout")
        return data

    def finish(self):
        if self.read(1):
            raise AcquisitionError("unexpected trailing SEG-Y data")
        if not self.finished:
            raise AcquisitionError("gzip verification not completed")


def decode_samples(raw, code):
    if code == 5:
        result = np.frombuffer(raw, dtype=">f4").astype("<f4")
    elif code == 1:
        words = np.frombuffer(raw, dtype=">u4").astype(np.uint32)
        fraction = (words & 0xffffff).astype(np.float64) / 2**24
        exponent = ((words >> 24) & 0x7f).astype(np.int32) - 64
        sign = np.where(words & 0x80000000, -1., 1.)
        values = sign * np.ldexp(fraction, 4 * exponent)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            result = values.astype("<f4")
    else:
        raise AcquisitionError("unsupported SEG-Y sample format")
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise AcquisitionError("velocity is not finite positive float32; no clipping is allowed")
    return result


def parse_velocity_gzip(compressed, *, samples_per_trace, traces, windows):
    """Decode every sample for validity but retain only the predetermined crops."""
    ns = _positive_integer(samples_per_trace, "samples_per_trace")
    ntr = _positive_integer(traces, "traces")
    windows = resolve_windows((ns, ntr), windows)
    expected_bytes = 3600 + ntr * (240 + 4 * ns)
    reader = BoundedGzipStream(compressed, expected_bytes)
    text, binary = reader.read_exact(3200), reader.read_exact(400)
    binary_ns, sample_format = struct.unpack_from(">H", binary, 20)[0], struct.unpack_from(">H", binary, 24)[0]
    if binary_ns != ns:
        raise AcquisitionError("binary sample count differs from declared geometry")
    if sample_format not in (1, 5):
        raise AcquisitionError("unsupported SEG-Y sample format")
    extended = struct.unpack_from(">h", binary, 304)[0]
    if extended != 0:
        raise AcquisitionError("extended/unknown textual headers are not registered")
    crops = {w["id"]: np.empty(w["raw_shape_z_x"], dtype="<f4") for w in windows}
    headers = bytearray()
    trace_ns_values, interval_values, scalars, coordinate_units = set(), set(), set(), set()
    first_fields = last_fields = None
    for x in range(ntr):
        header = reader.read_exact(240)
        headers.extend(header)
        trace_ns = struct.unpack_from(">H", header, 114)[0]
        if trace_ns not in (0, ns):
            raise AcquisitionError("trace sample count differs from fixed geometry")
        trace_ns_values.add(trace_ns)
        interval_values.add(struct.unpack_from(">H", header, 116)[0])
        scalars.add(struct.unpack_from(">h", header, 70)[0])
        coordinate_units.add(struct.unpack_from(">H", header, 88)[0])
        fields = {"trace_sequence_line": struct.unpack_from(">i", header, 0)[0],
                  "trace_sequence_file": struct.unpack_from(">i", header, 4)[0],
                  "source_x": struct.unpack_from(">i", header, 72)[0],
                  "source_y": struct.unpack_from(">i", header, 76)[0],
                  "group_x": struct.unpack_from(">i", header, 80)[0],
                  "group_y": struct.unpack_from(">i", header, 84)[0]}
        if first_fields is None:
            first_fields = fields
        last_fields = fields
        values = decode_samples(reader.read_exact(4 * ns), sample_format)
        for window in windows:
            z0, x0 = window["start_z_x"]
            z1, x1 = window["stop_exclusive_z_x"]
            if x0 <= x < x1:
                crops[window["id"]][:, x - x0] = values[z0:z1]
    reader.finish()
    if reader.count != expected_bytes:
        raise AcquisitionError("unexpected uncompressed layout length")
    raw_headers = bytes(headers)
    metadata = {
        "uncompressed_bytes": reader.count, "uncompressed_sha256": reader.digest.hexdigest(),
        "gzip_crc32_hex": f"{reader.crc & 0xffffffff:08x}",
        "gzip_isize": reader.count & 0xffffffff, "full_gzip_crc_and_isize_verified": reader.finished,
        "gzip_members": 1, "sample_format_code": sample_format, "byte_order": "big",
        "shape_z_x": [ns, ntr], "trace_stride_bytes": 240 + 4 * ns,
        "first_trace_header_offset": 3600, "first_sample_offset": 3840,
        "extended_text_headers": extended,
        "binary_sample_interval_raw": struct.unpack_from(">H", binary, 16)[0],
        "binary_samples_per_trace": binary_ns,
        "segy_revision_raw": struct.unpack_from(">H", binary, 300)[0],
        "fixed_length_trace_flag_raw": struct.unpack_from(">H", binary, 302)[0],
        "trace_sample_counts_raw": sorted(trace_ns_values),
        "trace_sample_intervals_raw": sorted(interval_values), "coordinate_scalars_raw": sorted(scalars),
        "coordinate_units_raw": sorted(coordinate_units),
        "first_trace_fields": first_fields, "last_trace_fields": last_fields,
        "headers": {
            "text_header_base64": base64.b64encode(text).decode(), "text_header_sha256": sha256(text),
            "binary_header_base64": base64.b64encode(binary).decode(), "binary_header_sha256": sha256(binary),
            "trace_headers_encoding": "zlib+base64, concatenated240-byte headers in original file order",
            "trace_headers_base64": base64.b64encode(zlib.compress(raw_headers, 9)).decode(),
            "trace_headers_bytes": len(raw_headers), "trace_headers_sha256": sha256(raw_headers),
        },
        "all_samples_validated_finite_positive": True,
        "header_coordinates_used_for_reordering": False,
        "full_decoded_velocity_array_allocated": False,
    }
    return crops, metadata


def contained_path(directory, filename):
    root = Path(directory).resolve()
    name = Path(filename)
    if name.is_absolute() or len(name.parts) != 1 or name.name in ("", ".", ".."):
        raise AcquisitionError("output path must be a contained single filename")
    result = root / name
    if result.resolve().parent != root:
        raise AcquisitionError("output path escapes collector directory")
    return result


def array_bytes(array):
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(array, dtype="<f4"), allow_pickle=False)
    return stream.getvalue()


def modification_readme(protocol):
    return ("# BP2004 velocity crops: modification and scope notice\n\n"
            "Data courtesy of BP. Acknowledge BP and Frederic Billette. Reference: "
            "Billette and Brandsberg-Dahl (2005), The 2004 BP velocity benchmark, "
            "67th EAGE Conference, B035, doi:10.3997/2214-4609-pdb.1.B035.\n\n"
            "The complete original BP documentation and disclaimer are preserved in "
            "original_BP2004_READMES.pdf. These files inherit its terms; no new data license is asserted.\n\n"
            "Modifications: decode the registered big-endian SEG-Y samples; retain the three "
            "predeclared 192-by-192 windows as little-endian float32 NPY files; additionally retain "
            "raw[::2,::1] as 96-by-192 NPY files at 12.5 m isotropic spacing. Velocity stays in m/s. "
            "No interpolation, clipping, normalization or value-driven window selection occurs. "
            "Exact zero-based parent coordinates, hashes and all source headers are in receipt.json.\n\n"
            "All six arrays belong to ONE BP2004 composite synthetic numerical ancestor. "
            "They are named stress crops, not independent parents/sites, measured field data, "
            "a reproduction of the original shot gathers or an evaluated algorithm result.\n\n"
            f"Ancestor: {protocol['ancestor_id']}\n")


def collect(protocol_path, output, expected_protocol_sha256, expected_source_sha256, *, opener=None):
    protocol_path = Path(protocol_path)
    protocol, protocol_raw = load_protocol(protocol_path)
    source_raw = Path(__file__).read_bytes()
    if sha256(protocol_raw) != expected_protocol_sha256 or sha256(source_raw) != expected_source_sha256:
        raise AcquisitionError("approved protocol/source SHA256 mismatch before network access")
    doc_manifest_raw = (protocol_path.parent / "documentation_manifest.json").read_bytes()
    doc_manifest = json.loads(doc_manifest_raw)
    document = (protocol_path.parent / "original_BP2004_READMES.pdf").read_bytes()
    if (doc_manifest["source_url"] != protocol["documentation"]["url"]
            or sha256(document) != DOCUMENT_SHA256 or sha256(document) != doc_manifest["sha256"]
            or len(document) != doc_manifest["bytes"]
            or len(document) > protocol["documentation"]["maximum_bytes"]):
        raise AcquisitionError("original documentation integrity mismatch")
    output = Path(output).resolve()
    if output.exists():
        raise AcquisitionError("output already exists; no evidence is overwritten")
    started = utc_now()
    source = protocol["source"]
    compressed, http = fetch_bounded(source["url"], expected_bytes=source["expected_compressed_bytes"],
        maximum_bytes=source["maximum_compressed_bytes"], expected_etag=source["expected_etag"], opener=opener)
    layout = protocol["layout"]
    crops, metadata = parse_velocity_gzip(compressed, samples_per_trace=layout["samples_per_trace"],
                                         traces=layout["traces"], windows=protocol["windows"])
    del compressed
    artifacts, crop_records = {}, []
    dz, dx = layout["native_spacing_z_x_m"]
    for window in protocol["windows"]:
        raw = crops[window["id"]]
        stride = raw[::window["stride_z_x"][0], ::window["stride_z_x"][1]]
        record = dict(window)
        z0, x0 = window["start_z_x"]
        nz, nx = window["raw_shape_z_x"]
        record.update(ancestor_id=protocol["ancestor_id"],
                      raw_origin_z_x_m=[z0 * dz, x0 * dx], raw_spacing_z_x_m=[dz, dx],
                      raw_pixel_midpoint_z_x=[z0 + (nz - 1) / 2, x0 + (nx - 1) / 2],
                      decimated_origin_z_x_m=[z0 * dz, x0 * dx],
                      decimated_spacing_z_x_m=[dz * 2, dx], velocity_units="m/s", outputs=[])
        for suffix, values in (("raw", raw), ("stride2z", stride)):
            filename = window["id"] + "_" + suffix + "_velocity_mps.npy"
            encoded = array_bytes(values)
            artifacts[filename] = encoded
            record["outputs"].append({"filename": filename, "shape_z_x": list(values.shape),
                "dtype": "float32", "encoding": "NPY little-endian C order", "file_bytes": len(encoded),
                "file_sha256": sha256(encoded), "array_C_le_float32_sha256": sha256(np.ascontiguousarray(values, dtype="<f4").tobytes())})
        crop_records.append(record)
    artifacts.update({"protocol.json": protocol_raw, "collector_source.py": source_raw,
                      "original_BP2004_READMES.pdf": document,
                      "documentation_manifest.json": doc_manifest_raw,
                      "README_Modification.md": modification_readme(protocol).encode()})
    receipt = {"status": "complete_acquisition_only", "acquisition_started_utc": started, "completed_utc": utc_now(),
        "ancestor_id": protocol["ancestor_id"], "independent_ancestor_count": 1,
        "protocol_sha256": sha256(protocol_raw), "protocol_registered_utc": protocol["registered_utc"],
        "collector_source_sha256": sha256(source_raw), "documentation": doc_manifest, "http": http,
        "segy": metadata, "crops": crop_records, "raw_payload_written_to_disk": False,
        "plots_or_model_outcomes_computed": False, "velocity_units": "m/s",
        "runtime": {"python": sys.version, "numpy": np.__version__, "zlib": zlib.ZLIB_RUNTIME_VERSION, "platform": platform.platform()},
        "artifact_sha256": {name: sha256(value) for name, value in artifacts.items()}}
    artifacts["receipt.json"] = json_bytes(receipt)
    if protocol_path.read_bytes() != protocol_raw or Path(__file__).read_bytes() != source_raw:
        raise AcquisitionError("collector source/protocol changed during acquisition")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + output.name + "-", dir=output.parent))
    try:
        for filename, data in artifacts.items():
            contained_path(staging, filename).write_bytes(data)
        if output.exists():
            raise AcquisitionError("output appeared during acquisition; refusing overwrite")
        os.rename(staging, output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fetch", action="store_true", help="explicitly acquire velocity after root-approved freeze")
    parser.add_argument("--expected-protocol-sha256")
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args(argv)
    _, raw = load_protocol(args.protocol)
    if not args.fetch:
        print(json.dumps({"status": "protocol_only_no_network", "protocol_sha256": sha256(raw),
                          "source_sha256": sha256(Path(__file__).read_bytes())}))
        return 0
    if not all((args.output, args.expected_protocol_sha256, args.expected_source_sha256)):
        parser.error("--fetch requires --output and both approved SHA256 arguments")
    receipt = collect(args.protocol, args.output, args.expected_protocol_sha256, args.expected_source_sha256)
    print(json.dumps({"status": receipt["status"], "compressed_sha256": receipt["http"]["compressed_sha256"],
                      "uncompressed_sha256": receipt["segy"]["uncompressed_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
