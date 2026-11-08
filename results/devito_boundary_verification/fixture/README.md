# Fixed boundary-truncation software verification

AI-authored support only: no student final research design, physics pass threshold,
independent geology, certificate transfer or goal-completion claim.

Seven fixed forward-only cases use the existing audited Devito adapter and
explicit actual-source `solver.forward(..., save=False, dt=1ms)`. Three small-box
padding cases, two matched-exterior undamped controls, and two enlarged references
retain all raw traces, source samples, time axes, coordinates and source hashes.
The matched-exterior pairs check actual stencil-grid and relative node identity.
Whole/early/late trace differences have explicit normalizers and no tuned
threshold. Geometric return estimates do not prove discrete causality or a clean
reference window. The metrics are not intrinsic reflection coefficients.

Default invocation prints hashes and runs no PDE. Actual execution requires
`--run --repo PATH --output NEW_DIRECTORY --expected-protocol-sha256 HASH
--expected-source-sha256 HASH`. Root review and GO are still required.
An exclusive new output directory retains running/failed records; only an atomic
`complete_receipt.json` marks completion. All seven cases remain accounted for.
Eleven synthetic metric/layout/guard tests and Ruff pass. No PDE has been executed.

The pre-execution retention fix records an attempted case before propagation.
A mock KeyboardInterrupt check verifies interrupted/failed status and absence
of a completion receipt; the pre-fix source is retained byte-exact in history.
