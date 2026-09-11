# Phase 0–3 Research Report

This document records the first completed empirical milestone of WaveCert-FWI. It is deliberately narrow: it establishes the research question, verifies the discrete exact reference, trains a real forward-only neural operator, and tests whether forward accuracy implies inversion-gradient accuracy.

## Phase 0 — frozen question

The project specification is frozen in [`research_spec.md`](../research_spec.md). The core deployment question is whether a pretrained neural wavefield surrogate can be certified *after training* for gradient-based FWI. The Phase-3 motivating hypothesis was pre-registered before the final held-out failure study:

- receiver forward relative L2 <= 0.15; and
- gradient relative L2 >= 0.50 **or** gradient cosine <= 0.90.

External mentor sign-off is intentionally marked pending. The repository does not claim approval that has not occurred.

## Phase 1 — exact discrete FWI reference verified

The exact backend is a damped 2-D complex Helmholtz finite-difference system with sparse direct solves, analytic tangent solves, and an adjoint-state gradient in squared-slowness coordinates.

During verification, a real adjoint bug was found and fixed: on very small grids, rounded receiver coordinates can repeat. Receiver restriction therefore has repeated samples, and its true adjoint must *accumulate* repeated contributions. The old injection implementation overwrote them. The corrected implementation uses indexed accumulation.

The checked-in Phase-1 run (`results/phase1/verification.json`) gives:

- maximum JVP/VJP duality relative error: **3.94e-15**;
- maximum centered finite-difference gradient relative error: **3.63e-08**;
- minimum fitted Taylor remainder slope: **1.99946**.

All frozen Phase-1 exit criteria pass.

## Phase 2 — real forward-only FNO baseline

A trainable Fourier Neural Operator is implemented in `wavecert.surrogates.fno`. It predicts the full complex wavefield from:

1. squared-slowness model,
2. source mask,
3. frequency,
4. depth coordinate,
5. horizontal coordinate.

The network receives **no derivative supervision**. Checkpoint selection uses validation forward loss only.

Reference checkpoint configuration:

- grid: 16 x 16;
- training samples: 256;
- validation samples: 48;
- held-out forward test samples: 80;
- spectral modes: 5 x 5;
- width: 20;
- depth: 3;
- best epoch: 24.

Held-out forward results:

- median receiver relative L2: **0.1369**;
- mean receiver relative L2: **0.1568**;
- median full-wavefield relative L2: **0.2161**;
- batched CPU throughput speedup versus repeated sparse exact solves: **2.46x** on the small reference grid.

The learned model's own autograd gradient is internally correct: a finite-difference directional check gives relative error **9.79e-05**. This matters because Phase 3 compares a *correct derivative of the learned approximation* against the derivative of the exact physical map, rather than confusing an autodiff implementation bug with model-Jacobian error.

All frozen Phase-2 exit criteria pass.

## Phase 3 — forward accuracy vs FWI-gradient accuracy

The final held-out study evaluates 80 independent inversion states, each using four source/frequency blocks (two sources x two frequencies). For every case the study computes exact and FNO wavefields, receiver data, objectives, and FWI gradients.

Aggregate results (`results/phase3/summary.json`):

- median receiver forward relative L2: **0.1651**;
- median full-wavefield relative L2: **0.1950**;
- median gradient relative L2: **1.7072**;
- 90th percentile gradient relative L2: **3.8461**;
- median exact-vs-neural gradient cosine: **0.3436**;
- 10th percentile cosine: **-0.2590**.

The pre-registered failure criterion is met in **31 / 80 cases (38.75%)**. These are cases with receiver forward error at or below 15% but a materially bad gradient by the frozen threshold.

Forward error is statistically related to derivative quality, but it is not sufficient to determine it:

- receiver forward error vs gradient relative error: Pearson r = **0.503**, Spearman rho = **0.453**;
- receiver forward error vs gradient cosine: Pearson r = **-0.328**, Spearman rho = **-0.324**.

The central empirical conclusion is therefore not that forward error is unrelated to gradient error. It is the more defensible statement:

> **low forward error does not guarantee a trustworthy FWI gradient.**

This is exactly the deployment-time failure mode that motivates the later a-posteriori certificate.

## What this result does and does not establish

It establishes a reproducible proof-of-concept on a controlled discrete acoustic problem using a real learned neural operator and exact adjoint reference.

It does **not** yet establish:

- field-scale seismic performance;
- Devito/JUDI production-scale timing;
- Marmousi/OpenFWI/SubsurfaceGen generalization;
- a rigorous scalable stability constant;
- a final residual-based certificate theorem;
- selective fallback gains in end-to-end FWI.

Those belong to the next research phases. The present result is the empirical motivation and validation platform needed before attacking those claims.

## Reproduction

```bash
PYTHONPATH=src python scripts/run_phase1_verification.py
PYTHONPATH=src python scripts/run_phase2_train_fno.py --epochs 25 --train 256 --val 48 --test 80 --shape 16 --width 20 --modes 5 --depth 3
PYTHONPATH=src python scripts/run_phase2_validate.py
PYTHONPATH=src python scripts/run_phase3_failure_study.py --cases 80
```

The FNO training run is deterministic to the extent supported by the installed PyTorch/CPU backend and uses explicit seeds. Machine-readable outputs are stored under `results/phase1`, `results/phase2`, and `results/phase3`.
