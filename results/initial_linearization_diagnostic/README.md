# Fixed initial-linearization diagnostic

All eight reused cases, all eight frequencies, 128 LU factorizations and 1,152 RHS solves completed. Six manufactured controls passed before execution. The independent receiver-only auditor passed 1,600 checks with zero issues. Lower-band relative remainders are smaller on seven cases; case 2 remains the exception. This is not an inversion result or a physical/causal acceptance test.

[Technical report and reproduction command](../../docs/initial_linearization_diagnostic.md) · [All-eight figure](figures/initial_linearization_all8.png)

- `fixture/`: exact runner/protocol/tests, original logs and lint-only history.
- `run_01/`: all 64 receiver-only records, status and complete receipt.
- `prefreeze.json`, `source_review.json`, `terminal_observation_01.json`: phase/source/actual-process evidence.
- `post_auditor/`, `post_audit_01/`: separate scalar source/criteria/tests and raw comparison receipts; no independent PDE re-solve.
- `figures/`: all scalar CSV values, source, caption and original clipped-layout history.
- `future_continuation*`: source/API advisory and restart-control addendum only; not an implemented or frozen study.

Staging receipts describe their original unexecuted phase. The later completed execution and audit are separate, without rewriting historical receipts. Production source remains9f/161. Shared SciPy/SuperLU, reused slices and SEAM ancestry limit independence. AI-authored engineering diagnosis, not student final design or confirmation.
