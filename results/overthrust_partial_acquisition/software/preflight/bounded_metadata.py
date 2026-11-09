"""Quarantined metadata transport; no model reads. Not a dataset loader."""
import datetime
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAP = 262144


def fetch(label, url, *, kind="metadata_json", maximum=65536, byte_range=None):
    ledger_path = HERE / "REQUEST_LEDGER.json"
    ledger = json.loads(ledger_path.read_bytes()) if ledger_path.exists() else {
        "cap_bytes": CAP, "body_bytes_read": 0, "requests": []}
    if maximum > CAP-ledger["body_bytes_read"]:
        raise ValueError("aggregate metadata cap cannot cover request")
    entry = dict(label=label, url=url, kind=kind, requested_maximum=maximum,
        requested_range=byte_range, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        body_bytes_read=0, status="attempted")
    ledger["requests"].append(entry)

    def save():
        temp = ledger_path.with_suffix(".tmp")
        temp.write_text(json.dumps(ledger, indent=2)+"\n")
        temp.replace(ledger_path)

    save()
    headers = {"Accept-Encoding": "identity", "User-Agent": "WaveCert-metadata-preflight/1"}
    if kind == "metadata_json":
        headers["Accept"] = "application/json"
    if byte_range is not None:
        if kind != "hdf5_metadata_range":
            raise ValueError("range classification required")
        start, end = byte_range
        headers["Range"] = f"bytes={start}-{end}"
        if end < start or end-start+1 > maximum:
            raise ValueError("invalid bounded metadata range")
    request = urllib.request.Request(url, headers=headers,
        method="HEAD" if kind == "head" else "GET")
    data = bytearray()
    try:
        try:
            response = urllib.request.urlopen(request, timeout=20)
        except urllib.error.HTTPError as exc:
            entry.update(status="HTTP_error_no_body_read", HTTP_status=exc.code,
                final_url=exc.url, response_headers=dict(exc.headers))
            exc.close()
            save()
            return None
        with response:
            entry.update(HTTP_status=response.status, final_url=response.url,
                response_headers={k: response.headers.get(k) for k in ["Content-Length",
                    "Content-Type", "Content-Encoding", "Content-Range", "Accept-Ranges",
                    "ETag", "Last-Modified", "Content-Disposition"]})
            if kind == "head":
                entry["status"] = "HEAD_only_no_body_read"
                save()
                return b""
            if byte_range is not None:
                start, end = byte_range
                if response.status != 206:
                    entry["status"] = "REJECTED_non206_before_body_read"
                    save()
                    return None
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {start}-{end}/"):
                    raise ValueError("Content-Range differs from exact requested metadata interval")
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise ValueError("encoded metadata response refused")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > maximum:
                raise ValueError("declared body exceeds request cap")
            while len(data) < maximum:
                block = response.read(min(8192, maximum-len(data)))
                if not block:
                    break
                data.extend(block)
                entry["body_bytes_read"] += len(block)
                ledger["body_bytes_read"] += len(block)
                save()
            if byte_range is not None and len(data) != end-start+1:
                raise ValueError("partial metadata interval length differs")
            if byte_range is None and len(data) == maximum:
                raise ValueError("metadata response reached cap; completeness not established")
        target = HERE / label
        with target.open("xb") as stream:
            stream.write(data)
        entry.update(status="bounded_metadata_saved", local_name=label,
            sha256=hashlib.sha256(data).hexdigest())
        save()
        return bytes(data)
    except BaseException as exc:
        entry.update(status="failed_metadata_request", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
