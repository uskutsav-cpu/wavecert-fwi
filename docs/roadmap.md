# Research roadmap

## Milestone 0 — completed in this repository

- [x] finite-dimensional 2-D wave inverse problem;
- [x] exact state, tangent, and adjoint gradient;
- [x] surrogate state/JVP/gradient interface;
- [x] primal/tangent residuals;
- [x] directional derivative certificate;
- [x] proof-to-code correspondence;
- [x] finite-difference gradient tests;
- [x] empirical bound-coverage tests;
- [x] shot/frequency block abstraction;
- [x] selective exact-verification prototype;
- [x] reproducible demo artifacts and CI.

## Milestone 1 — Devito acoustic backend

- [ ] adapter for Devito forward wavefields;
- [ ] exact Devito adjoint-state gradient;
- [ ] Born/tangent operator adapter;
- [ ] receiver/source geometry abstraction compatible with Devito;
- [ ] reference Marmousi experiment;
- [ ] discretization-consistency tests against the compact backend.

## Milestone 2 — neural operator

- [ ] decide full-wavefield encoding (real/imag in frequency domain or space-time in time domain);
- [ ] FNO baseline using official `neuraloperator` package;
- [ ] checkpoint loading and inference API;
- [ ] `torch.func.jvp` tangent evaluation;
- [ ] forward-only training baseline;
- [ ] optional derivative-informed/Jacobian-informed training baseline.

## Milestone 3 — practical certificate

The current dense SVD stability constant is validation-only. Replace it with one of:

- [ ] analytically justified energy-norm stability estimate;
- [ ] certified reduced-order lower bound;
- [ ] inf-sup lower-bound method;
- [ ] problem-specific coercivity/stability surrogate with rigorous safety factor.

Then test:

- [ ] primal-only estimator;
- [ ] tangent-only estimator;
- [ ] combined estimator;
- [ ] adjoint-residual estimator;
- [ ] goal-oriented / dual-weighted directional estimator;
- [ ] randomized directional probes for approximate full-gradient control.

## Milestone 4 — blockwise adaptive inversion

- [ ] per-shot bound;
- [ ] per-frequency-band bound;
- [ ] priority queue for uncertain blocks;
- [ ] exact verification until descent is certified;
- [ ] optional exact-gradient replacement for failed blocks;
- [ ] trust-region or line-search wrapper with inexact-gradient conditions.

## Milestone 5 — realistic OOD study

- [ ] Marmousi proof-of-concept;
- [ ] SubsurfaceGen in-distribution split;
- [ ] held-out geology;
- [ ] acquisition shifts;
- [ ] frequency shifts;
- [ ] high-contrast structures;
- [ ] noise robustness;
- [ ] compute/accuracy Pareto frontier.

## Milestone 6 — paper-grade artifact

- [ ] frozen environment/lock file;
- [ ] public benchmark configs;
- [ ] pretrained checkpoints or download script;
- [ ] one-command figure reproduction;
- [ ] table-generation script;
- [ ] ablation suite;
- [ ] artifact-evaluation instructions;
- [ ] paper/preprint citation.
