# WaveCert-FWI Research Specification — Frozen v0.2

**Status:** internal freeze complete; mentor scientific sign-off pending.

## 1. Research question

Can a pretrained neural wavefield operator be certified *after training* for use inside gradient-based full-waveform inversion (FWI), so that neural gradients are used when they are trustworthy and exact wave physics is invoked only when required?

The first manuscript-scale target is a 2-D constant-density acoustic setting. The verified reference problem is frequency-domain Helmholtz FWI; the production extension is time-domain acoustic FWI (e.g. Devito) with the same abstract interfaces.

## 2. Mathematical problem

For source `s` and angular frequency `ω`, let

\[
A(m,\omega)u_s=q_s,
\qquad d_s=P_su_s,
\]

where `m` is squared slowness, `A` is the discrete wave operator, `q_s` is a source, and `P_s` samples receivers. The exact least-squares objective is

\[
\Phi(m)=\frac12\sum_s\|P_su_s(m)-d_s^{obs}\|_2^2,
\]

with exact gradient `g = ∇Φ(m)` computed by the adjoint-state method.

A learned wavefield surrogate `N_θ` returns

\[
\hat u_s=N_\theta(m,q_s,\omega),
\]

which induces a surrogate objective `\hat Φ`, Jacobian `\hat J`, and gradient `\hat g` through automatic differentiation.

The deployment-time trust problem is to decide whether an inversion-relevant quantity from the learned derivative is safe to use without evaluating the complete exact gradient.

## 3. Primary hypotheses

### H1 — forward accuracy is insufficient for derivative trust
There exist test states for which receiver/wavefield forward error is small while gradient error is materially larger or gradient alignment is poor.

Operational failure-case definition for the Phase-3 study:

- receiver relative L2 error <= 0.15, and
- either gradient relative L2 error >= 0.50 **or** gradient cosine similarity <= 0.90.

These thresholds are declared before running the final failure study and must not be changed post hoc without versioning this document.

### H2 — physics-consistency residuals contain derivative information
The tangent residual

\[
r_t(v)=A(m)D\hat u(m)[v]+A_m(m)[v]\hat u(m)
\]

will be more informative about inversion-relevant derivative error than forward prediction error alone.

H2 is the bridge to the later certificate phases and is *not* required to declare Phase 3 complete.

### H3 — a posteriori certification can enable selective exact-physics fallback
A computable bound on directional-derivative or gradient error can be used to certify descent and trigger exact recomputation only for the uncertain source/frequency blocks.

H3 is reserved for Phases 4–7.

## 4. Primary endpoint for Phases 0–3

A reproducible held-out test study showing the relationship among:

1. wavefield relative L2 error,
2. receiver-data relative L2 error,
3. exact-vs-neural FWI gradient relative L2 error,
4. exact-vs-neural gradient cosine similarity,
5. exact vs neural directional derivative along the neural descent direction.

The study must save per-case metrics, aggregate statistics, and publication-ready scatter plots.

## 5. Reference and surrogate models

### Exact reference
- 2-D damped acoustic Helmholtz finite-difference discretization.
- Sparse direct linear solves.
- Squared slowness parameterization.
- Point sources and point receivers.
- Adjoint-state gradient.
- Verified JVP/VJP duality and second-order Taylor remainder.

### Learned surrogate
- Fourier Neural Operator (FNO)-style spectral convolution network.
- Inputs: normalized squared slowness, source map, normalized frequency, and spatial coordinates.
- Outputs: real and imaginary full wavefield channels.
- Training objective: forward wavefield MSE only. No derivative labels are used in the Phase-2 baseline.

## 6. Dataset policy

Synthetic models are generated from a deterministic base velocity gradient plus random smooth Gaussian anomalies and layering perturbations. Dataset generation uses fixed seeds and non-overlapping train/validation/test indices.

No test sample may be used for checkpoint selection or normalization statistics.

For Phase 2/3 the source/frequency support is shared between training and test so the study isolates model-distribution derivative error before later OOD experiments.

## 7. Baselines locked for later manuscript experiments

1. exact PDE FWI,
2. forward-trained neural-operator FWI,
3. forward-error threshold fallback,
4. fixed-period exact fallback,
5. random fallback at matched exact-solve budget,
6. WaveCert global fallback,
7. WaveCert source/frequency selective fallback,
8. derivative-informed / Jacobian-supervised neural surrogate when feasible.

## 8. Metrics

### Exact-solver verification
- JVP/VJP relative duality error,
- centered finite-difference gradient relative error,
- Taylor remainder slope.

### Surrogate forward metrics
- full-wavefield relative L2,
- receiver relative L2,
- normalized RMSE,
- inference wall time vs exact sparse solve wall time (single and batched).

### Derivative metrics
- gradient relative L2,
- gradient cosine similarity,
- directional derivative relative/absolute error.

### Later certificate metrics
- empirical coverage,
- effectivity index,
- exact forward/adjoint solves saved,
- wall-clock cost,
- final inversion quality.

## 9. Phase exit criteria

### Phase 0
- this specification committed and versioned;
- notation, hypotheses, metrics, split policy, and baselines frozen;
- mentor sign-off requested separately.

### Phase 1
On at least three random directions / dual vectors:
- JVP/VJP relative duality error < 1e-8,
- centered finite-difference gradient directional error < 5e-4,
- fitted Taylor slope >= 1.8 over the asymptotic range.

### Phase 2
- a real trainable neural operator is integrated (not a hand-designed low-fidelity surrogate),
- checkpoint is selected only on validation forward loss,
- held-out forward metrics and timing are written to disk,
- automatic differentiation produces finite model gradients.

A forward model is considered *usable for Phase 3* when median held-out receiver relative L2 <= 0.20. The speed comparison is reported honestly; the tiny validation grid is not required to beat a sparse direct solver in single-sample CPU latency.

### Phase 3
- >= 50 held-out inversion cases evaluated,
- forward-vs-gradient scatter data saved,
- at least one pre-registered failure case is found **or** the null result is reported without changing thresholds,
- aggregate forward/gradient correlation is reported,
- plots and machine-readable tables are generated by one command.

## 10. Reproducibility rules

- every random generator receives an explicit seed;
- configs and checkpoint metadata record code version, shape, frequencies, source positions, split seed, normalization statistics, and architecture;
- final manuscript figures are generated from saved tabular data, not manually edited;
- failed/null experiments remain documented rather than deleted;
- claims distinguish mathematically rigorous bounds from empirical indicators.

## 11. Scope exclusions for the Phase-0–3 freeze

The following are deliberately deferred:

- 3-D FWI,
- elastic/anisotropic physics,
- field data,
- rigorous field-scale stability constants,
- SubsurfaceGen OOD stress tests,
- full certificate theorem,
- end-to-end selective repair.

These are later phases, not prerequisites for establishing the learned-derivative failure mode.

## 12. Sign-off

- **Research specification / implementation review:** completed in-repo on 2026-09-10.
- **User/lead researcher:** Utsav Sunil Kumar — pending explicit acknowledgement in project history.
- **External seismic mentor:** pending review; no mentor approval is claimed in this repository.

---

## 13. Phase 4–6 extension — Frozen v0.3

This extension does not alter the Phase-0–3 thresholds above.

### Phase 4 — deterministic reference certificate

Primary object: a directional-derivative upper bound based on primal/tangent residuals and receiver-aware discrete stability constants.

Exit criteria:

- evaluate at least 50 held-out Phase-3-distribution states;
- realized directional error is below the deterministic direction-aware bound on every reference-grid case up to numerical tolerance;
- zero false descent certifications;
- save case-level and block-level machine-readable tables;
- report effectivity even if the result is too conservative to be useful.

### Phase 5 — calibration

The deterministic bound may not be shrunk and still called rigorous. Practical calibration is therefore a separate statistical layer.

Locked split-conformal policy:

- use a disjoint calibration/evaluation split by case ID;
- use the deterministic directional bound as the positive base score;
- report a 90% marginal operating point and a 95% sensitivity operating point;
- label all conformal claims as exchangeability-dependent statistical coverage, not deterministic PDE certification;
- calibrate source/frequency labels separately for selective repair.

### Phase 6 — adaptive fallback

Evaluate on a new random seed not used in Phase 4/5.

Policies:

1. neural-only direction;
2. case-level conformal global fallback;
3. blockwise conformal selective repair;
4. exact-gradient direction.

For selective repair, changing any exact/neural block contribution changes the hybrid gradient direction. All remaining directional certificates must therefore be recomputed after every repair.

Primary metrics:

- true-descent rate under offline exact validation;
- exact source/frequency block-gradient evaluations;
- cosine to the exact-gradient descent direction;
- fraction of blocks repaired.

Phase 6 is directional certification only. Finite-step line search and end-to-end inversion are deferred to Phase 7.
