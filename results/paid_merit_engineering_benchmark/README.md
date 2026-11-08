# Frozen paid-merit software engineering comparison

**360 runs complete; both independent post-audits passed. No certificate
time-to-target or reliable reconstruction advantage is established.** This is
reused development data and AI-authored technical support, not a student final
research design or competition submission.

Read [the complete technical record](../../docs/paid_merit_engineering_benchmark.md)
before interpreting the tables or figures. Every original output is retained in
`run_01/`, with its producer's complete artifact receipt. `fixture/` preserves
the exact driver, addendum, tests and pre-outcome histories. `freezes/` preserves
actual source/environment/GO and terminal observations. The first CLI invocation
used an incorrect flag and exited before project execution; its log is retained.
The corrected actual invocation completed all 360. The original executor's exit
code was unavailable after tool-output truncation; absent PIDs, the complete
receipt and final CLI line establish observed termination, without inventing an
exit-code observation.

`audits/` contains both actual post-audit reports, source, independent source
reviews, logs and preserved earlier versions. `analysis/` contains every planned
run/target/internal checkpoint and conditional case summaries. `figures/`
contains the original descriptive figures and full caption/source.
`post_outcome_initial_baseline/` is an additive, explicitly post-outcome
diagnostic; it does not replace or change the original study.

All target events use actual fully returned time including the engine's finish
tail; incomplete, late and unreached cases receive no observed target time.
Internal checkpoints are not return evidence. Cumulative costs, preparation,
offline reference work and serialization scopes remain separate. All five
controls use paid LU reference-merit acceptance; none executes neural/POD
inference. Same-M GMRES is a missing-gradient-field control.

The source data are the existing CC BY 4.0 SubsurfaceGen-derived validation
fixtures. See `SOURCE_DATA_ATTRIBUTION.original.md` and the original dataset
manifest/immutable source revision under
[`data/reference/subsurfacegen_parent_split`](../../data/reference/subsurfacegen_parent_split).
The adjacent [ancestry correction](../../docs/ancestry_audit.md) supersedes any
interpretation of provider-ID disjointness as original-model/site independence.
Two SEAM IDs here share one ancestor; three repetitions do not create new samples.

Reproduce with the current frozen source, single-thread numerical libraries and
a fresh output path (approximately 35 minutes on the recorded host; other hosts
will differ):

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src \
python results/paid_merit_engineering_benchmark/fixture/run_engineering_benchmark.py \
  --execute-engineering-benchmark --repo "$PWD" --output /tmp/wavecert-paid-run-new \
  --expected-addendum-sha256 74039601d9785a63899aba406baeac7d21f39ce430cd08798a4e0473f169dce2 \
  --expected-driver-sha256 7fc7543294380fda9dbde2784e6c469425be06d9d36ee45be2cbc44a34bd0883
```

The archived run can be checked without rerunning its optimizer:

```sh
python results/paid_merit_engineering_benchmark/audits/receipt_auditor/audit_paid_merit.py \
  --repo "$PWD" --run "$PWD/results/paid_merit_engineering_benchmark/run_01" \
  --output /tmp/wavecert-receipt-audit-new.json --terminal-confirmed

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python results/paid_merit_engineering_benchmark/audits/objective_auditor/audit_saved_models.py \
  --execute-after-terminal-confirmation \
  --expected-script-sha256 807f4f2f0e0aeea1d2f61b64ee5f64a99819c38d4f6164529e664f0a235edeae \
  --repo "$PWD" --run "$PWD/results/paid_merit_engineering_benchmark/run_01" \
  --output /tmp/wavecert-objective-audit-new.json
```

These checks share NumPy/SciPy/SuperLU and evaluate ordinary floating-point
agreement. Compact intermediate vectors cannot be recovered from hashes.
Neither audit establishes continuum truth, new theory, independent sites,
external human review or a student's contribution/understanding. Full scope
notes and the actual AI assistance record remain part of the evidence.
