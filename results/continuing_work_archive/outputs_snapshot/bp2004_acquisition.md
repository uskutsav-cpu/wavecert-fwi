# BP2004: one acquired synthetic ancestor, no method evaluation

The frozen v2 collector acquired exactly three predefined native 192×192 crops
and their `raw[::2,::1]` stride views. The native sampling is 6.25 m vertically
and 12.5 m horizontally; each stride view is 96×192 at isotropic 12.5 m spacing.
Velocity remains in **m/s**. No interpolation, clipping, normalization,
value-driven crop choice, PDE, plot or method-quality evaluation occurred.
These six arrays inherit **one BP2004 composite synthetic ancestor**, not six
independent models or three independent sites.

Actual execution completed with exit code 0. Its full compressed stream SHA256 is
`01b3d65c6318491b1412465c904614a6c35bd73ea0ffc6a8d866917181749b9c`;
uncompressed stream SHA256 is
`19f729b535d1629e0176443024d22344edd2083273c423a361e99d4cd53e62e1`.
These hashes were measured **after acquisition**; only the official URL,
19,533,256-byte size and ETag were known before it. An ETag is not a SHA256.
The full compressed payload and decoded original model were not written to disk.

The three centers were fixed before payload inspection at normalized `(z,x)`
coordinates `(.25,.20)`, `(.50,.50)` and `(.75,.80)`. Exact parent index windows,
origins, sample spacing, full original SEG-Y headers and all output hashes are in
[`run_01/receipt.json`](/Users/swethasunilkumar/Documents/Codex/2026-11-07/wavecert-fwi/work/wavecert-fwi/results/bp2004_acquisition/run_01/receipt.json). The collector streamed and checked
the complete single-member gzip, CRC/ISIZE, byte counts, layout and positivity
of every decoded sample while retaining only the registered windows.

An independent retained-artifact audit passed **84 checks, zero mismatches**:
all eleven artifact hashes, 5,395 retained trace headers (1,294,800 bytes), all six
NPY file/array hashes, shape/dtype/finite positivity, exact stride equality,
fixed coordinates, source snapshots, original PDF, citation and one-ancestor
provenance. See [`retained_audit/`](/Users/swethasunilkumar/Documents/Codex/2026-11-07/wavecert-fwi/work/wavecert-fwi/results/bp2004_acquisition/retained_audit). This audit did not fetch or
independently re-decode the discarded full payload. Whole-stream SHA/CRC and
original-parent crop decoding remain collector-supported execution evidence.
No independent full-payload replay is implied.

The v2 source differs from preserved v1 by exactly two citation-title corrections.
The original protocol, crop rule, URLs, numerical parser, original PDF and tests
are unchanged. V1 and adverse review are retained; its old 35-test receipt is not
presented as a v2 test result. Actual v2 validation passed 35 mock-HTTP/synthetic
tests with warnings as errors, metadata-only CLI and all twelve generated-notice
checks. Collector lint passed. The frozen test file retains its existing import
ordering style diagnostic; all failed lint invocations and corrected command
scope are recorded, with no concealed source change.

Reproduce acquisition only with the exact source/protocol and a fresh output:

```sh
python results/bp2004_acquisition/fixture_v2/collect_bp2004.py \
  --fetch --output /tmp/wavecert-bp2004-new \
  --expected-protocol-sha256 a84ce64dcac7f93f035007daf313cb2ab7bde713800d10d9d37d812f12c836f4 \
  --expected-source-sha256 7a499a425ca495c8cffb2714af260e88e9ad929e9c6f85b0d9d7c0ef9624e28d
```

Data courtesy of BP; acknowledge BP and Frederic Billette. Citation: Billette
and Brandsberg-Dahl (2005), *The 2004 BP velocity benchmark*, EAGE B035,
[doi:10.3997/2214-4609-pdb.1.B035](https://doi.org/10.3997/2214-4609-pdb.1.B035).
These are BP data under the original terms, **not the repository's MIT code
license**. Retain the complete
[original documentation/disclaimer](/Users/swethasunilkumar/Documents/Codex/2026-11-07/wavecert-fwi/work/wavecert-fwi/results/bp2004_acquisition/run_01/original_BP2004_READMES.pdf) and
[modification notice](/Users/swethasunilkumar/Documents/Codex/2026-11-07/wavecert-fwi/work/wavecert-fwi/results/bp2004_acquisition/run_01/README_Modification.md) with any redistribution.
Acquisition does not establish independent geological validation, physical
accuracy, a certificate transfer, student ownership or a scientific milestone.
