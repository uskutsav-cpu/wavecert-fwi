# Saved partial Overthrust acquisition audit

This standard-library-only auditor reads saved acquisition artifacts and cached
metadata. It imports no producer and performs no network requests, wave solves,
unit conversion, axis transpose, smoothing, clipping or row removal.

The final audit passes 3,186 integrity checks with zero integrity failures. It
verifies the exact frozen 207-row integer range plan, all 259 ordered requests,
257 HTTP206 range records with exact Content-Range and body lengths, two HTTP200
record JSON responses, before/after/cached metadata identity, published record
and file identity claims, 207 raw row hashes, full saved-artifact hashes and exact
ordered NPY payload assembly. The response ledger totals 686,662 body bytes,
including 663,228 model-row bytes, below the fixed 8 MiB cap.

The stored NPY header describes a C-order `(207, 801)` array of `<f4` values.
Its payload is byte-identical to concatenating the ordered raw rows. Scalar
`struct.iter_unpack` checks all 165,807 stored values for finiteness and positivity.
No model values, extrema, converted velocities or physical interpretations are
reported by this audit.

There is one disclosure finding: both record JSON responses report ETag `"2"`,
whereas all HDF5 range responses have no ETag. The producer receipt's blanket
statement that no reviewed endpoint supplied an ETag is too broad. A record
revision ETag does not establish the identity of the entire HDF5 byte stream.
The raw receipt/ledger remain unchanged; this audit records the precise scope.

This is local byte/ledger consistency verification, not independent HTTP
authentication/refetching or whole-file checksum verification. The claimed
published MD5 was not tested against the entire file. Named axes, model physical
units, original CD1 velocity identity, independent geological sites, untouched
confirmation and physical acceptance remain unverified. The root's actual process
exit observation is separate; the producer appropriately leaves its own terminal
exit field null.

Two successful audit reports are retained. Between them the source changes one
blank line for lint only. Ruff and Python 3.10 grammar checks pass.
