# Changelog

## 0.2.0 — Phase 0–3 learned-surrogate milestone

- froze the research question, hypotheses, metrics, splits, and exit criteria in `research_spec.md`;
- added exact data-JVP and real-adjoint data-VJP operators;
- found and fixed repeated-receiver accumulation in the restriction adjoint;
- added automated JVP/VJP, finite-difference gradient, and Taylor verification;
- added deterministic synthetic wavefield dataset generation;
- implemented a trainable 2-D Fourier Neural Operator for complex Helmholtz wavefields;
- added saved-model normalization/checkpoint support and differentiable FWI objective gradients;
- trained and checked in the Phase-2 FNO reference checkpoint;
- added forward accuracy and throughput benchmarking;
- added an 80-case held-out forward-vs-gradient failure study with pre-registered thresholds;
- added manuscript-oriented Phase-1/2/3 figures, CSV/JSON outputs, and a research report;
- expanded the CLI with reference verification, FNO training, and failure-study commands.

## 0.1.0

- initial certified Helmholtz proof-of-concept;
- directional residual certificate;
- selective exact-physics verification;
- surrogate-quality ablation and initial documentation.
