# Experiment matrix

## Primary hypotheses

**H1 — Forward accuracy is insufficient.** There exist test models for which forward relative error is low but gradient relative error is high or gradient cosine is poor.

**H2 — Physics residuals predict derivative failure.** Primal/tangent residual features correlate with realized directional-derivative error substantially better than forward data error alone.

**H3 — The certificate is reliable.** The empirical coverage rate of the derived upper bound is 100% up to explicitly documented numerical tolerance in the regime where its assumptions hold.

**H4 — The certificate is useful, not merely correct.** Effectivity is small enough that many surrogate steps can be accepted without exact fallback.

**H5 — Selective fallback beats all-or-nothing fallback.** Shot/frequency-level exact verification reaches near-exact inversion quality using fewer high-fidelity evaluations than full-iteration fallback.

## Baselines

1. Exact PDE FWI — upper-quality / upper-cost reference.
2. Neural-only FWI — fastest but uncertified.
3. Forward-error threshold — tests whether conventional surrogate validation is enough.
4. Random fallback at matched PDE budget — controls for simply spending more exact solves.
5. Whole-iteration fallback — compares selective versus coarse fallback.
6. Jacobian-informed surrogate (when available) — tests complementarity with training-time derivative supervision.

## Stress axes

| Axis | In distribution | Shift / OOD |
|---|---|---|
| geology | training-style layered | held-out structural family |
| velocity contrast | moderate | high-contrast / salt-like |
| source frequency | training band | lower / higher band |
| geometry | nominal shots/receivers | shifted/sparser geometry |
| illumination | well illuminated | shadowed/poorly illuminated |
| noise | noiseless | controlled additive noise |

## Required plots

1. forward error vs gradient error scatter;
2. forward error vs gradient cosine;
3. certificate bound vs realized derivative error;
4. effectivity histogram;
5. fallback fraction vs exact-PDE solve budget;
6. reconstruction quality vs exact-PDE solve budget;
7. wall-clock vs reconstruction quality;
8. failure heatmap by shot/frequency;
9. OOD calibration plot;
10. ablation: primal residual only vs tangent residual only vs combined.

## Statistical reporting

- report medians and interquartile ranges across models/seeds;
- include worst-case and 95th-percentile certificate effectivity;
- report zero bound violations separately from “numerically within tolerance” violations;
- do not select thresholds on the OOD test split;
- publish exact configs and random seeds.
