# Bounded Overthrust HDF5 slice route

A selected plane is feasible at the transport/layout level without downloading
the 531 MB HDF5 file. This review read **15,504 response-body bytes** under a
262,144-byte metadata cap: two primary record JSON responses, one short README
and 712 HDF5 header/index bytes. No model values, raw chunks or selected slices
were read. No new dataset or physical-validation evidence is established.

The primary [versioned record 4252588](https://zenodo.org/records/4252588)
declares open CC BY 4.0 access. Its true-model file is
`overthrust_3D_true_model.h5`, file ID
`436595b8-ed4b-4e1d-8991-e63608dcc169`, size 531,253,092 bytes, with published
MD5 `ae4082cdcebbc7714eedd382d7925804`. Its exact content URL is
`https://zenodo.org/api/records/4252588/files/overthrust_3D_true_model.h5/content`.
The whole-file checksum was **not independently verified**. HEAD returned the
same size; all 13 requested HDF5 metadata intervals returned exact HTTP 206
Content-Range/length responses. No large-file HTTP 200 body was read. No ETag
was supplied by the inspected endpoints.

The [related CD1 record 23469292](https://zenodo.org/records/23469292) explicitly
links this HDF5 record as a variant. Its small README was acquired and matched
the record's published MD5; its actual SHA256 is
`690fe3549e26ded7e16c236d603bb8bf0e07ea2a32329ca84028141c18d38b8b`.
That README describes original SEG/EAGE big-endian float32 velocity on an
801 × 801 × 187 grid at 25 m, m/s, x fastest, and identifies the original
authors and CC BY 4.0 notice. Its CD1 identity/checksum statements remain
publisher claims; the 174,960,038-byte archive was not fetched.

HDF5 superblock, root links and dataset `m` object messages establish:

| Verified metadata | Value |
| --- | --- |
| Shape | `(207, 801, 801)` |
| Type | Little-endian IEEE float32 |
| Layout | Contiguous, uncompressed; no chunk index required |
| Raw model storage | Offset 2,192, length 531,245,628 bytes |
| Excluded model interval | `[2192, 531247820)`; no acquired range intersects it |

Installed h5py/HDF5 independently parsed the same cached header/index bytes
through a virtual read-only file. The reader rejects any model-storage read;
unknown non-model bytes were zero filler and are logged. Only shape, datatype,
layout, compression and storage offsets were queried—no dataset indexing or
values. This cross-check validates the named metadata, not the whole file.

An index-only candidate fixes the middle dimension at 400 and reads the entire
last dimension. Each depth-index row is one contiguous 3,204-byte interval.
All 207 first-axis rows would require 207 ranges and **663,228 bytes**. If a
later reviewed geometry interpretation removes the first 20 rows, 187 ranges
would require **599,148 bytes**. Either fits a separately registered 8 MiB body
cap with ample room for metadata, but entails many HTTP requests. The saved
candidate byte-range plan has not been executed.

The remaining interpretation is material. Pinned SWEEP source describes `m` as
squared slowness in s²/km² on `(z,x,y)` axes, with 20 added water rows, then
conversion by `1000/sqrt(m)`, an axis transpose and float32 rounding. It claims
that this recovers the CD1 velocity grid. Those unit/axis/water/equality claims
are not proved by the inspected object headers. Geometry datasets `n`, `d`,
`o`, the `copyright` dataset and all model values remain unread. Do not label
the prospective output an unchanged original float32 velocity slice yet.

Before a future acquisition, register the exact record/file identity, anonymous
index selection, source geometry interpretation, byte intervals, hard aggregate
cap, strict 206/Content-Range checks, stable size/identity checks and slice-byte
SHA256. Preserve the original attribution and terms and explicitly document any
water removal, unit conversion or axis change. This is a feasibility advisory,
not acquisition/decode authorization, a student final design, untouched
confirmation, an independent-site claim or accepted wave physics.
