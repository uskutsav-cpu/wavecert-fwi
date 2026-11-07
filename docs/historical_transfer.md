# Historical test reuse and development transfer

This phase evaluates the three already frozen FNOs, the selected raw POD64
basis, and the fixed POD64/coarse96 physical-adjoint proposal on the existing
source-domain ID/OOD roles. These roles and old FNO outcomes were previously
inspected. This is **historical test reuse / development transfer**, never
pristine confirmation, a new benchmark split, or student preregistration.
There is no learning, rank search, model reselection or recalibration.

The prospective engineering protocol is
`configs/historical_transfer_protocol.json`, created at
2026-11-07 22:24:47.935556 UTC with SHA256
`b5ecc87c5d90e19a39040e4f57a2d88ac9da0d1b54991e574fdb0839ffaf2f1d`.
It was compiled from dataset/split manifests and metadata CSVs before any new
POD outcomes on these historical roles. Test NPZ arrays were not opened to
choose it. The protocol pins their existing manifest hashes for later checks.

## Complete fixed cohort

All eight ID parents have two cached slices each: **16 ID slices**. The one
OOD parent, `penobscot_000`, has **four cached slices**. An initial task assumed
two OOD slices; metadata corrected that assumption, and all four were approved
before new outcomes. Nothing is selected from their performance.

Every slice retains the original six sources and four frequencies
(2.5, 3, 3.5, 4 Hz), all24 blocks. The full phase therefore requires:

- 20 complete slices, eight ID parents and one OOD parent;
- five fixed models, 100 slice/model aggregate evaluations;
- 480 blocks per model, 2,400 block/model rows;
- 45 parent/model summaries, with ID and OOD kept separate.

The old manifest audits disjoint parents/slices among training, validation,
calibration, ID and OOD roles. That audit does not establish independent
geological sampling. The OOD parent is a single descriptive parent case.
Neither 20 slices nor 480 blocks are treated as independent sample size.

The cached physical model is converted to velocity, smoothed at sigma1.6
with reflect boundaries, and converted back to squared slowness. Receiver
observations come from the cached regenerated synthetic float32 wavefields.
These are not measured field data. Original geometry, preprocessing and file
hashes are checked; all24 rows of each slice must share exactly the same model.

## New phase admission without weakening earlier guards

Earlier B/C/POD/fifth freezes forbid new imported pipeline files. The new
historical adapter therefore has its own explicit phase freeze. A clean
`python -c` subprocess imports the **unchanged** old fifth validator and checks
all original source/runtime/receipt/weight guards without importing the new
historical module or CLI. It performs no predictions or historical-array
loads. The parent process also verifies that its actual public physics,
certificate, POD, FNO and historical modules execute from this checkout.

This prospective phase explicitly allows its three new source/config paths:
the protocol, module and CLI. All previously recorded source paths must keep
their checksums. Any other new imported source is rejected. Added unrelated,
unimported files are disclosed. The new freeze records the complete current
manifest and numerical runtime, snapshots all five model identities and the
prior receipt chain, and preserves the original artifacts. Its own evaluation
uses that full new manifest. No previous source or guard is modified.

The freeze is self-contained for model identity: it copies the three exact FNO
weight files and shared POD basis, binds the fixed coarse operator signature,
and snapshots B/C/POD/fifth receipts. It carries the interrupted training
admission and its unknown original runtime/extra100 spent updates forward.
The historical freeze reads metadata and weight bytes; it does not open test
NPZ arrays. The outcome runner checks this freeze and metadata before loading
the exact hash-pinned historical arrays.

## Outcomes and costs

Every model reports the actual common-model24-block aggregate objective,
gradient norm, cosine and true descent. FNOs use the shared-input batch VJP;
separately paid single-block gradient diagnostics verify its sum. Raw POD
sums its analytic block derivatives at that identical physical model. Mean
block cosine is marked secondary and cannot replace the aggregate gradient.

The corrected method uses the common normalized negative sum of its24 physical
state/adjoint gradient proposals. All block residual bounds use that same
direction; signed derivatives and nonnegative bounds are summed. The output
retains raw per-case JSON, model/frequency/operator contexts, per-block
components, aggregate vectors, availability, strict certification, realized
errors, false acceptance and fallback requirement. An unavailable frequency
keeps its paid raw POD proposal and skips correction; the aggregate abstains
with null unavailable bounds. No online exact recovery is performed. Numeric
aggregate overflow/nonfinite proposals preserve raw diagnostics and make the
attempt incomplete rather than yielding a complete comparison with a missing
case. All completed tables and partial cost counters remain available.

Exact reference verification is paid once per slice and shared across model
comparisons. Its per-slice and cumulative fine-solver counts are separate from
online model/proof work. POD projections and reduced counters, coarse setup,
correction/operator/norm work, and neural attempted/completed API calls are
recorded. Neural proposal batch work and additional offline block diagnostics
are explicitly separate. The old observation generation, model training, POD
SVD and calibration are prior paid work referenced through existing artifacts;
this phase does not measure an end-to-end amortized runtime or imply free
precomputation. There is no inversion or acceleration claim.

Parent summaries average each parent's own slices first. ID summaries then
average eight parent values; OOD reports its one parent's four-slice value.
No IID confidence intervals, binomial nominal coverage, pooled ID/OOD rate or
conformal guarantee is asserted. Statistical gates/scales are not evaluated
or fitted in this phase. Coarse residual inequalities concern the existing
discrete reference operator in exact arithmetic; ordinary floating-point
implementation is not a formally verified roundoff enclosure.

Metadata-only inspection:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/run_historical_transfer.py check-protocol
```

After protocol/source review and separate root authorization, `freeze` requires
all four prior freeze directories and a new output directory; `evaluate`
requires that historical freeze and a different new output directory. Neither
command mutates prior results. No project historical outcome run has been
executed as part of this implementation.
