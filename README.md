# WaveCert-FWI

**A posteriori certification and selective exact-physics fallback for neural-surrogate full-waveform inversion.**

WaveCert-FWI is a research codebase for a specific failure mode in scientific machine learning:

> A neural wave operator can predict wavefields accurately while its **Jacobian is wrong enough to produce a bad inversion gradient**.

The project asks whether a cheap, post-training certificate can decide—*at each inversion step, shot, or frequency block*—whether a neural gradient is safe to trust, and invoke exact wave physics only where needed.

## Phase 0–6 milestone: derivative failure, certification, calibration, and adaptive fallback

Phases 4–6 now extend the learned-surrogate milestone into an actual trust-and-fallback pipeline:

- **Phase 4:** a receiver/direction-aware deterministic residual certificate was evaluated on the same 80 held-out states. It achieved **100% coverage with zero violations and zero false descent certifications**, but was very conservative (median effectivity **8292.19**) and certified no neural directions outright.
- **Phase 5:** the rigorous bound is preserved unchanged, while a separate split-conformal deployment gate is calibrated on 50 cases and evaluated on 30 disjoint cases. The 90% gate achieved **90.0% evaluation coverage**, **2.34 median effectivity**, **13.3% immediate neural-direction acceptance**, and **0 observed false descent certifications**.
- **Phase 6:** on 30 new held-out states, neural-only directions were true descent **76.7%** of the time. Conformal selective repair reached **100% observed true-descent rate** while using **2.90/4 exact source-frequency gradient blocks on average**, a **27.5% reduction** versus exact direction construction.

See [`docs/phase4_6_report.md`](docs/phase4_6_report.md) and `results/phase4`, `results/phase5`, and `results/phase6`.

## Phase 0–3 milestone: completed proof-of-concept

The repository now includes an end-to-end learned-surrogate validation milestone, not only the original hand-designed low-fidelity demo:

- **Phase 0:** frozen research specification in [`research_spec.md`](research_spec.md); external mentor sign-off remains pending and is not claimed.
- **Phase 1:** exact Helmholtz JVP/VJP duality error **3.94e-15**, finite-difference gradient error **3.63e-08**, minimum Taylor slope **1.99946**.
- **Phase 2:** a real forward-only FNO trained on exact wavefields reaches **13.69% median receiver error**, **21.61% median full-wavefield error**, and **2.46x batched CPU throughput** relative to repeated sparse solves on the tiny reference grid. Its own autodiff gradient passes a finite-difference check at **9.79e-05** relative error.
- **Phase 3:** across **80 held-out inversion states**, the exact-vs-FNO gradient has median relative L2 error **1.707** and median cosine **0.344**. **31/80 (38.75%)** satisfy the pre-registered failure condition: receiver forward error <=15% but gradient relative error >=50% or gradient cosine <=0.90.

The result supports the specific motivation for WaveCert: **low forward error does not guarantee a trustworthy FWI gradient.** See [`docs/phase0_3_report.md`](docs/phase0_3_report.md) and the machine-readable outputs under `results/phase1`, `results/phase2`, and `results/phase3`.

---

## Research question

For a PDE-constrained least-squares objective

\[
\Phi(m)=\frac12\|P u(m)-d\|^2,\qquad A(m)u=q,
\]

suppose a pretrained neural operator produces an approximate state \(\hat u(m)\) and an autodiff tangent \(D\hat u(m)[v]\).

Can we compute an inexpensive **a posteriori** quantity \(\eta\) such that

\[
|D\Phi(m)[v]-D\hat\Phi(m)[v]|\le \eta,
\]

and therefore certify descent whenever

\[
D\hat\Phi(m)[v]+\eta<0?
\]

The long-term target is a shot/frequency-level hybrid inversion loop:

```text
neural proposal
      │
      ▼
cheap PDE-consistency certificate
      │
 ┌────┴────┐
 │         │
pass      fail
 │         │
 ▼         ▼
trust   exact physics only for uncertain blocks
 │         │
 └────┬────┘
      ▼
 certified inversion update
```

## Why this matters

Recent seismic neural-operator work shows that forward accuracy alone does not guarantee inversion-quality Jacobians. Park et al. (2026) improve this by adding Jacobian information **during training**. WaveCert-FWI explores the complementary question: can we **detect and certify** derivative reliability *after training*, without assuming the surrogate was trained with Jacobian supervision?

The broader mathematical precedent comes from certified reduced-order modeling for PDE-constrained optimization, where a posteriori cost/gradient bounds are used to decide when a low-fidelity model must be corrected by the high-fidelity PDE.

## What is already implemented

This repository is not only scaffolding. The default demo is a fully runnable, small **2-D complex Helmholtz inverse problem** with:

- exact acoustic frequency-domain forward solver;
- exact tangent/Jacobian-vector solve;
- exact adjoint-state gradient;
- controlled low-fidelity surrogate with its own differentiable tangent and gradient;
- primal and tangent residual evaluation against the exact PDE;
- numerical stability constant \(\beta=\sigma_{\min}(A)\) on the small validation grid;
- a derived directional derivative error certificate;
- a mathematically safe descent test;
- source/frequency block decomposition;
- selective exact verification, prioritized by uncertainty;
- exact-vs-surrogate objective and gradient metrics;
- gradient cosine similarity;
- certificate coverage/effectivity diagnostics;
- automated tests for adjoint gradients and bound validity;
- CI, configs, documentation, and experiment blueprints for Marmousi and SubsurfaceGen.

The small Helmholtz backend is intentionally transparent and testable. It is **not** presented as the final production seismic solver. The next backend is intended to be Devito/JUDI plus a neural-operator checkpoint.

---

## The current certificate

For one source/frequency block, let

\[
A(m)u=q,
\]

and let the exact tangent in direction \(v\) satisfy

\[
A(m)\,\delta u=\omega^2\operatorname{diag}(v)u.
\]

Given surrogate state \(\hat u\) and surrogate tangent \(\delta\hat u\), define

\[
r_p=A\hat u-q
\]

and

\[
r_t=A\,\delta\hat u-\omega^2\operatorname{diag}(v)\hat u.
\]

If

\[
\beta=\sigma_{\min}(A)>0,
\]

then

\[
\|u-\hat u\|\le \frac{\|r_p\|}{\beta}
\]

and

\[
\|\delta u-\delta\hat u\|
\le
\frac{\omega^2\|v\|_\infty\,\eta_u+\|r_t\|}{\beta}
=:\eta_{\delta u}.
\]

For

\[
D\Phi(m)[v]=\Re\langle Pu-d,P\delta u\rangle,
\]

this yields the computable directional certificate

\[
\boxed{
|D\Phi-D\hat\Phi|
\le
\eta_u\bigl(\|P\delta\hat u\|+\eta_{\delta u}\bigr)
+
\|P\hat u-d\|\eta_{\delta u}
}
\]

for the point-receiver restriction used here.

This is the baseline certificate exercised by the original demo. Phase 4 additionally implements a sharper receiver/direction-aware discrete reference bound based on `||P A^-1||` and `||P A^-1 diag(v) A^-1||`; see [`docs/theory.md`](docs/theory.md) and [`docs/phase4_6_report.md`](docs/phase4_6_report.md).

---

## Quick start

```bash
git clone https://github.com/uskutsav-cpu/wavecert-fwi.git
cd wavecert-fwi

python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

pytest
wavecert demo --output results/demo

# Learned-surrogate Phase 2/3 workflow
pip install -e '.[dev,torch]'
wavecert train-fno --epochs 25 --train 256 --val 48 --test 80 --shape 16
PYTHONPATH=src python scripts/run_phase2_validate.py
wavecert failure-study --cases 80
wavecert certificate-study --cases 80
wavecert calibrate-certificate --alpha 0.10
wavecert adaptive-study --cases 30
```

Or without installation:

```bash
PYTHONPATH=src python -m wavecert demo --output results/demo
```

The demo writes:

```text
results/demo/
├── summary.json
├── models.png
├── gradients.png
└── certificate_coverage.png
```

### Reference demo result

The checked-in reference run intentionally uses a moderately imperfect surrogate. On four source/frequency blocks it gives:

- surrogate objective relative error: **3.63%**;
- gradient cosine with the exact adjoint gradient: **0.967**;
- total realized directional-derivative error: **0.0465**;
- certified upper bound: **0.2943**;
- no-fallback certificate: **not yet sufficient**;
- exact blocks needed to certify descent: **1 / 4**.

That is the behavior the project is targeting: keep a neural proposal when it can be justified, and spend exact physics only where the certificate says uncertainty is too high.

Run the controlled surrogate-quality sweep with:

```bash
wavecert ablation --output results/ablation
```

The included sweep moves from a very accurate surrogate requiring **0/4** exact blocks to progressively poorer surrogates requiring more fallback, while checking bound coverage at every level.

## Repository map

```text
wavecert-fwi/
├── src/wavecert/
│   ├── physics/          # exact reference PDE backends
│   ├── surrogates/       # low-fidelity and PyTorch adapters
│   ├── certificates/     # a-posteriori estimators
│   ├── repair/           # selective exact-physics policies
│   ├── experiments/      # reproducible experiment runners
│   ├── metrics.py
│   └── cli.py
├── tests/                # gradient + certificate correctness tests
├── configs/              # demo, Marmousi, SubsurfaceGen blueprints
├── docs/
│   ├── theory.md
│   ├── literature.md
│   ├── experiment_matrix.md
│   └── roadmap.md
├── results/demo/         # generated reference run
└── .github/workflows/    # CI
```

## Planned production stack

The architecture is designed around two replaceable interfaces:

1. **Exact reference physics**
   - current: compact SciPy Helmholtz backend;
   - planned: Devito acoustic time-domain FWI / JUDI-scale workflows.

2. **Wavefield surrogate**
   - current validation baseline: controlled smoothed-physics surrogate;
   - current learned baseline: an in-repo differentiable forward-only FNO with saved Phase-2 checkpoint;
   - included generic adapter: arbitrary differentiable PyTorch model via `torch.func.jvp`;
   - later production comparison: maintained FNO/TFNO implementations from the official `neuraloperator` ecosystem.

No certificate code needs to know the internal surrogate architecture.

## Core evaluation protocol

Every surrogate should be evaluated on more than forward error:

| Metric | Why it matters |
|---|---|
| forward relative \(L_2\) | standard surrogate quality |
| gradient relative \(L_2\) | inversion sensitivity error |
| gradient cosine | update-direction alignment |
| true objective change | whether the proposed step actually helps |
| certificate coverage | fraction of realized errors below the bound |
| certificate effectivity | how conservative the bound is |
| fallback rate | how often exact physics is required |
| exact PDE solves | central compute/accuracy trade-off |
| wall-clock | practical speedup |
| final inversion error / SSIM | downstream reconstruction quality |

## Experimental ladder

### Stage 0 — mathematical validation

Small Helmholtz systems where \(\sigma_{\min}(A)\), exact gradients, tangent solves, and certificate errors can all be computed directly.

### Stage 1 — Marmousi 2-D

- constant-density acoustic FWI;
- multiple shots and frequency bands;
- compare forward error vs gradient error;
- identify cases with small forward error but poor gradient cosine;
- certify the proposed direction;
- selectively fall back on uncertain shot/frequency blocks.

### Stage 2 — realistic/OOD geology

Use SubsurfaceGen-style field-scale geology to stress test:

- held-out geological setting;
- acquisition geometry shift;
- frequency shift;
- sparse receiver geometry;
- high-contrast/salt-like structures;
- illumination changes.

### Stage 3 — optimization guarantees

Connect the online certificate to inexact-gradient/trust-region conditions and quantify convergence versus exact-PDE cost.

---

## Literature position

This project deliberately sits between three literatures:

1. **Seismic neural operators and derivative failure** — accurate forward prediction can coexist with poor inversion Jacobians.
2. **A posteriori PDE error estimation** — residuals can turn equation inconsistency into computable solution/quantity-of-interest bounds.
3. **Certified surrogate optimization** — high-fidelity solves are triggered only when the cheap model can no longer be trusted.

See [`docs/literature.md`](docs/literature.md) for a curated reading map.

## Scientific status

This repository is an **active research prototype**. The directional certificate implemented for the small Helmholtz backend is mathematically derived under explicit finite-dimensional assumptions and validated numerically by tests. It should not yet be described as a field-scale certified FWI method.

The major open problems are:

- obtaining practical **certified lower bounds** on the wave-operator stability constant at scale;
- extending the estimate to realistic time-domain/acoustic/elastic settings;
- controlling approximation errors introduced by finite precision, discretization, and approximate residual evaluation;
- determining whether a directional certificate is sufficiently tight to save exact PDE solves on realistic neural operators;
- deciding when full-gradient certification is necessary versus descent-direction certification;
- integrating blockwise fallback into a convergent optimization method.

## Reproducibility principles

- Exact quantities used only for validation are labeled as such.
- The online certificate never silently uses the unknown exact gradient.
- Numerical estimators are not called “certified” unless their lower/upper-bound role is explicit.
- Production data and checkpoints stay outside git; configs record their provenance.
- Every theorem-inspired code path gets a numerical unit test.

## License

MIT. See [`LICENSE`](LICENSE).
