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

## 0.3.0 — Phase 4–6 certification and adaptive fallback milestone

- derived a receiver/direction-aware deterministic residual certificate using `||P A^-1||` and `||P A^-1 diag(v) A^-1||`;
- replaced dense inverse construction in the main certificate study with sparse multi-RHS receiver-resolvent solves;
- validated deterministic coverage on all 80 Phase-3 held-out cases with zero violations;
- documented the deterministic certificate's conservatism rather than hiding it;
- added split-conformal calibration as a separately labeled statistical deployment gate;
- added 90% and 95% case-level operating points and per-block Bonferroni calibration;
- implemented global exact fallback and dynamically recomputed blockwise selective repair;
- evaluated Phase 6 on 30 new held-out cases;
- achieved 100% observed descent safety for selective repair while reducing exact block gradients by 27.5% versus exact direction construction;
- added Phase-4/5/6 CLI commands, scripts, tests, plots, CSV/JSON outputs, and documentation.
- resolved the Phase-4/5/6 Ruff quality gate without suppressing rules, including closure capture, callable defaults, import/export hygiene, and modern typing cleanup;
- added a CLI parser regression test covering all three Phase-4/5/6 commands.
