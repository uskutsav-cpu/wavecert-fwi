# Fixed Devito boundary-truncation software verification

This is AI-authored software support and a record of executed diagnostics,
not a student's final research design, interpretation or competition paper.
The full scientific goal remains open.

The seven fixed cases completed with unchanged source, physical parameters,
source samples, time axes and coordinates. All software invariants passed.
**The receiver traces still show substantial boundary-truncation differences.**
Successful execution is not a physics-quality pass.

## Actual measurements

Every case uses homogeneous 2-D velocity 2 km/s, a 5 Hz Ricker source, 10 m
spacing, a fixed 1 ms step, fourth-order spatial differences, float64 and a
0–1100 ms record. Source and receiver are separated by 50 m. Physical source
coordinates are translated between boxes to preserve the relative geometry.
All use the installed Devito 4.8.23 backend, with `save=False`.

Here the full-record relative trace difference is
`norm(trace - reference_321) / norm(reference_321)`, using all 1,101 samples.
The largest reference has a 321×321 physical grid and 40 padding cells on each
side. These are finite-record receiver diagnostics, not reflection coefficients.

| Physical square grid | Damping padding cells per side | Full-record relative L2 difference |
|---|---:|---:|
| 61×61 | 0 | 2.407073263 |
| 61×61 | 20 | 0.569363532 |
| 61×61 | 40 | 0.249489294 |
| 101×101, matched exterior to padding 20 | 0 | 1.037217034 |
| 141×141, matched exterior to padding 40 | 0 | 0.686602500 |
| Enlarged reference, 241×241 | 40 | 4.224306702e−16 |
| Enlarged reference, 321×321 | 40 | Reference itself |

The two matched-exterior pairs have the same stencil-grid shape and relative
source/receiver indices; their operator difference is the damping profile.
Damping reduces mismatch relative to these undamped controls. The remaining
56.9% and 24.9% trace differences are retained without a post-outcome pass
threshold. Agreement between the enlarged references is a same-code numerical
check, not independent continuum truth.

![All seven fixed traces and their differences](/Users/swethasunilkumar/Documents/Codex/2026-11-07/wavecert-fwi/outputs/boundary_traces.png)

The early window is fixed at `t <= 250 ms`; its relative differences are at
most 6.09e−9. The late window uses `t > 250 ms`, normalizing its mismatch by
the **full** reference trace norm. Neither the window split nor a mirror-path
travel estimate establishes strict discrete causality. Ricker wavelets are not
compactly supported. Actual sampled wavelets and time axes are saved.

A subsequent [separate NumPy replay](numpy_boundary_replay.md) reproduces all
seven implemented traces within 3.42e−14 relative L2 error. Its source-level
audit also established the exact output convention: computation records indices
1–1099; stored receiver samples 0 and 1100 remain initialized zero. All original
metrics keep the unchanged 1,101 stored samples. Zero endpoints contribute no
signal energy; they are not computed pressure observations at the final time.

The separate `matched_exterior_contrasts` block in `metrics.json` compares each
damped trace directly with its matched undamped control. Its denominators are
those controls' full norms, 243.210246 and 204.832721; they are not the largest
reference norm, 168.873437. Those ratios must not be read as the remaining
error relative to the enlarged reference.

## Verification and retained artifacts

The fixed protocol and reviewed runner were frozen before any PDE execution:

- Production source: `9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da`.
- Runner: `f8a522f34a3ce1be68293944dbef362ea86ae9731f28fd802eabb0434fb57cb1`.
- Protocol: `a83f2de93543b16af428d1fbe562f4d7db8d9df982b203a04bb0d445e93306b8`.

Eleven synthetic tests passed with warnings as errors, independently repeated,
before execution. An adverse pre-execution review caught missing interruption
records. The original runner/review, minimal fix and mocked-interruption
regression are preserved. Each attempted case is now persisted before propagation.

The actual seven-case process terminated successfully. A separate AI audit
recomputed all 72 reported metric numbers using raw arrays and `math.hypot`,
checked 11 artifact hashes, the seven raw records, 119 recorded invariants and
source/configuration pins. Another physical audit independently checked the raw
metrics and interpretation limits. These are arithmetic/metadata audits, not a
second PDE implementation or independent human reproduction. The later NumPy
implementation replay is documented separately above.

The largest recorded field has three 401×401 slices: 3,859,224 logical bytes.
Its reported halo view is 4,014,744 bytes. These exclude model, compiler,
process and other allocations; they are not peak-RAM measurements. Full
time-history fields were not stored. Full damping/buffer arrays are not archived;
their hashes and metadata are retained alongside all receiver/source records.

All raw data, failed-review history, execution log, pre-execution freeze, audit
source/receipts and plotting code are in `results/devito_boundary_verification`.
The recorded post-auditor script assumes the original outer workspace layout;
it is archived as the actual executed audit, not a portable CLI.

## Reproduction scope

Use the same production source and installed dependency/source hashes recorded
by the protocol. The runner aborts on mismatches or an existing output path.
From the repository root, with the required Devito environment:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src \
python results/devito_boundary_verification/run_01/runner_source.py \
  --run --repo . --output results/boundary_reproduction_new \
  --expected-protocol-sha256 a83f2de93543b16af428d1fbe562f4d7db8d9df982b203a04bb0d445e93306b8 \
  --expected-source-sha256 f8a522f34a3ce1be68293944dbef362ea86ae9731f28fd802eabb0434fb57cb1
```

This check quantifies one homogeneous geometry and source band. It validates
neither independent geological environments nor PML, free surfaces, elasticity,
time-domain certificate transfer, reconstruction benefit or field accuracy.
The underlying damping formulation and example units are described in the
[official Devito modeling tutorial](https://www.devitoproject.org/examples/seismic/tutorials/01_modelling.html);
the protocol additionally pins the installed source actually used.
