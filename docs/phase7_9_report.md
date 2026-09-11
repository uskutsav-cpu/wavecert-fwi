# WaveCert-FWI Phase 7–9 report

## Executive summary

Phases 7–9 move WaveCert from single-direction certification to complete inversion trajectories and realistic-data readiness.

### Phase 7 — complete inversion trajectories

Four policies were run for 8 iterations on 4 independent held-out synthetic inverse problems, with four source-frequency blocks per iteration:

| policy | final model relative L2 | model-error reduction | true-objective reduction | mean exact gradient blocks |
|---|---:|---:|---:|---:|
| neural-only | 0.04989 | 0.97% | 75.67% | 0.0 |
| conformal-global | 0.04328 | 15.02% | 71.89% | 24.0 |
| conformal-selective | **0.04316** | **15.39%** | **87.41%** | **20.5** |
| exact | 0.04043 | 20.75% | 94.24% | 32.0 |

The selective policy therefore used **35.94% fewer exact gradient blocks** than exact FWI while substantially outperforming neural-only reconstruction error. Exact FWI remained the strongest reconstruction baseline, as expected.

True PDE objectives in the trajectory CSV are **offline evaluation quantities** and are not used to choose neural or WaveCert finite steps. Exact validation time is tracked separately from algorithm time.

## Phase 8 — production physics and Marmousi path

The repository now contains:

- an explicit production-environment report for the optional Devito backend;
- a Devito adapter aligned with the saved-forward-wavefield / receiver-residual / adjoint-gradient structure of Devito's public FWI tutorials;
- a loader for the public compact Marmousi binary `mar_big_117_567.bin` used in the reproducible SiameseFWI materials;
- `scripts/download_marmousi_public.sh`;
- a Phase-8 study that automatically uses the real public binary when present and otherwise falls back to a clearly labelled `marmousi-inspired-proxy`.

The current execution container does **not** include Devito and cannot make arbitrary network downloads. Therefore the checked-in Phase-8 result is intentionally labelled:

- `benchmark_source = marmousi-inspired-proxy`
- `production_backend_status = adapter-ready-runtime-unavailable`
- `public_marmousi_executed = false`
- `passed_full_external_validation = false`

The offline proxy path executed successfully. Over 6 iterations, selective WaveCert used 21 exact blocks versus 24 for exact FWI (12.5% savings), but the exact reference obtained a better final model error. These proxy numbers are engineering diagnostics, **not Marmousi claims**.

## Phase 9 — geology/OOD stress testing

Real external-data adapters were added for:

- OpenFWI 2-D velocity batches (`.npy`, canonical `(N,1,70,70)`), plus seismic arrays;
- SubsurfaceGen HDF5 2-D velocity slices (`velocity`), wavefields (`wavefield`) and shot-gather cubes (`shot_gather_cube`).

Because the field-scale external datasets are not attached to this offline environment, the executed stress test uses six compact **procedural proxies** named after the SubsurfaceGen setting classes:

- Penobscot
- F3
- Gulf of Mexico
- Fault
- Salt Canopy
- SEAM

They are not SubsurfaceGen samples and must never be reported as such.

Five cases per setting were evaluated. The forward-only FNO is strongly OOD on these models: mean receiver relative error is 0.487 and mean gradient cosine is approximately -0.020. This deliberately severe shift exposes an important limitation of Phase-5 calibration:

- conformal selective repair used 2.1 / 4 exact blocks on average;
- observed true-descent rate dropped to **80%** under the non-exchangeable shift;
- the deterministic certificate responded conservatively, requiring 4 / 4 exact blocks on every case and retained **100% observed descent safety**.

This is scientifically useful: split-conformal coverage is not claimed to survive arbitrary geology shift, while the deterministic reference-grid bound degrades to exact fallback rather than making a false safety claim.

## Remaining external validation

Phases 7–9 are code-complete for the current compact research harness, but two external experiments remain before a manuscript can call Phase 8/9 field-scale validated:

1. install Devito and execute the production time-domain FWI adapter;
2. attach the public Marmousi binary and representative OpenFWI/SubsurfaceGen samples, then rerun the prepared commands.

The repository reports these as pending rather than fabricating results.
