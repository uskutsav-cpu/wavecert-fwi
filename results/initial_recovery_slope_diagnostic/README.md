# Initial recovery slopes: post-outcome software diagnostic

All eight original initial/truth models satisfy the declared bounds. Both raw
proposals and normalized negative reference data gradients have positive
half-relative-squared velocity-error slopes in five cases and native-m-error
slopes in three, despite negative data derivatives everywhere. These local
numerical facts are not finite-step or complete causal findings.

Read [the technical scope](../../docs/recovery_and_physics_diagnostics.md).
`run_01/` retains all eight cases and all 1,440 original internal endpoint
comparisons, with modes/eligibility separated. The producer completed with
exit code 0, 4,483 checks and zero issues. The independent scalar post-audit
completed with 67,118 checks / 33,600 numerical comparisons and zero issues.
No new PDE, optimizer trajectory, truth-aware policy or parameter selection ran.

`fixture/` contains exact source/protocol, twelve pure controls and their actual
receipt, independent reviews, cosmetic-history preservation and validation
failures. `freezes/` retains actual root GO and an earlier metadata-only guard
failure before any diagnostic execution; numerical code/settings were unchanged.
`post_audit_source/` and `post_audit_01/report.json` preserve separate scalar
arithmetic, fixed criteria, pre-run freeze and actual command/exit evidence.
`figures/` contains all sixteen directions, full caption, CSV and plotting source.
`advisories/` distinguishes structural observation bounds and established FWI
methods from a claimed new contribution. No numerical Jacobian rank was computed.

Reproduce from a fresh output path with the unchanged source digest `9f30afa9`:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python results/initial_recovery_slope_diagnostic/fixture/diagnose_initial_slopes.py \
  --execute-reviewed-diagnostic --repo "$PWD" \
  --witness "$PWD/results/paid_merit_initial_direction_verification/run_01" \
  --benchmark "$PWD/results/paid_merit_engineering_benchmark/run_01" \
  --output /tmp/wavecert-recovery-slopes-new \
  --expected-source-sha256 7998631fc3542cdd42ebce0bcc236d87755a33647b1135327ebebda6f8dec46e \
  --expected-protocol-sha256 b1104b4159ffc0065fb5b3a47a1c552a8cb94fc329daa8f7b315418452d7a8d2
```

Source data inherit the [CC BY 4.0 fixture attribution](../../data/reference/subsurfacegen_parent_split/README.md)
and [SEAM ancestry correction](../../docs/ancestry_audit.md). Eight cases and
repeated endpoints are not independent sites. The regression in
`tests/test_recovery_physics_diagnostic_reproduction.py` reproduces known
numerical evidence; it is not a new confirmation study. Extensive AI authorship
and the student's unverified scientific ownership remain explicit.
