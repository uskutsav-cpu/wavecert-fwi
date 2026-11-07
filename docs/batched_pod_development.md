# Validation-only batched POD development

This is a post-confirmation engineering extension. Completed scalar timings
and earlier validation/confirmation results motivated reducing repeated Python,
small-solve and transform dispatch. It is not student preregistration, new
confirmation, new theory, a rank search on held-out outcomes, or an inversion
experiment. Original POD, coarse-stability, scalar certificate, Helmholtz and
exact-route sources remain unchanged and hash-pinned.

The fixed protocol is `configs/batched_pod_development_protocol.json`, SHA256
`a53600eff96f128193f813c077da29235d185fca473a24e060ca8fe75334ad1f`.
It was recorded before this new batch study. The only array inputs are the
existing eight development-validation inverse slices/four parents (all24
source/frequency blocks) and the frozen training-fit state+tangent POD64 basis.
The existing validation loader verifies their byte identity, acquisition,
parent IDs and shared historical teacher/basis provenance. No source
calibration, historical test, or fresh external arrays are opened.

## Fixed comparisons and parity gate

All six controls remain in the report:

1. Strongest implemented exact objective/full-gradient selector.
2. Existing scalar POD64 objective/gradient calls, one per block.
3. New frequency-grouped batched POD64 objective/gradient.
4. Existing scalar corrected POD64/coarse96 physical proposal/certificate.
5. New batched corrected POD64/coarse96 proposal/certificate.
6. Exploratory batched corrected POD64/coarse64 proposal/certificate.

Before comparative timing, every one of the eight cases compares raw scalar
and batch fields, then scalar versus batch corrected fields at rank96 and
rank64. Scalar corrected64 is an additional **untimed parity reference**, not a
seventh timed method. All its work is reported. Comparisons include objectives,
aggregate/block gradients, states, adjoints, common direction, actual residual
components, error components, bound terms, signed centers, bounds, contexts and
availability/certification/fallback decisions. Fixed tolerances are relative
1e-9 and absolute1e-10. A failed parity gate writes evidence and starts no
comparative timings.

The raw scalar public API already computes an adjoint but does not return its
field. For preflight only, an additional adjoint solve/projection/lift per block
reconstructs that field using the same unchanged POD adapter; these extra24
reduced RHS per slice are disclosed and charged separately. They are excluded
from the timed scalar public-API control. No work is claimed free.

Independent fine block objectives/gradients are recomputed on another solver
and checked against the stored aggregate teacher labels. They supply coverage
checks for all aggregate and block bounds. Fine verification and all scalar/
batch preflight setup/solve/transform work remain separate paid receipts.

## Bounded cold costs

Six repetitions revisit all eight cases. Each case rotates the six methods by
`(repetition+case_index)%6`, so every method occupies every position once. Every
call constructs new independent physics/POD/coarse/certifier objects, while
within-survey caching remains available as in each API. There are exactly288
cold calls,48 paired rows, and6,912 block-quality rows. No case or unavailable
candidate is silently omitted.

`time.perf_counter` brackets constructor setup and the complete current-model
query. Total cold includes POD basis copy/orthogonality check/projected
operators, damping DCT setup, model mass/DCT work, all small factors/norms/SVD/
eigenproblems, rank-column identity inverse RHS, receiver constants, corrections,
actual residuals, projections, lifts, physical gradients and result construction.
Quality checks, receipt extraction, common input I/O, offline verification,
output and teardown stay outside core times. Existing scalar APIs are called
unchanged; these are solver-cold calls in one process, not process/OS-cache cold.

RHS counts retain their matrix-column units. Actual new batch dispatch counts
are reported separately from fine/reduced/coarse RHS and real-equivalent
transform counts. A matrix RHS dispatch is not one RHS column; a batch FFT call
is not one real two-dimensional transform. Reduced and fine dimensions are
different and these counts are not interchangeable FLOPs or a speedup estimate.
In particular `coarse_projection_calls` and `coarse_lift_calls` retain logical
single-vector column units; `coarse_solve_calls` counts actual LU dispatches.
`batch_dispatches` separately counts actual grouped projection/lift/FFT calls.
Each receipt carries this unit map. The input audit verifies actual module paths
and hashes for this runner, both reused utilities, and all new batch helpers,
as well as the five unchanged production modules.
The exact selector minimizes implemented RHS alternatives, not universally
wall time. The information tasks also differ: exact gradient, approximate raw
POD gradient, and physical proposal plus directional bound.

All five thread environment values must be1 before Python starts. Actual
evaluation/timing requires root's source/protocol review and GO with other
task-owned compute idle. System load and caller context are recorded, not
treated as proof that an external computer was idle. Repeated missing or
nonfinite proposals cannot pass completion through None/None parity. Finite
unavailable candidates remain as explicit abstentions, with no online recovery.
Preflight separately requires non-null finite objectives, full gradients and
every state/adjoint/block field on both sides; optional unavailable bounds may
remain null. Matching missing proposals never start comparative timing.

## Fixed development recommendation

Only batched corrected96 and batched corrected64 are eligible candidates. A
candidate must strictly certify **all eight** cases and cover every aggregate
and every block's independent exact derivative, with all parity and repeat
checks passing. The cost score averages each slice's median of six **total
cold** times within its parent, then averages the four parents. Recommend the
lowest-score eligible candidate; exact-score ties prefer96. If neither is
eligible, retain existing96 and make no performance claim. A failed study
integrity/completeness check yields no recommendation.

This is a development recommendation only. Root must separately freeze any
variant and a new protocol before additional held-out outcomes. This runner
does not redeploy a model or reevaluate old fresh/historical roles. Ordinary
floating-point residual inequalities retain the unverified PSD allowance;
there is no formal roundoff enclosure or corrected-state autograd claim.
Eight correlated slices/four parents, and six repeated visits, do not create
additional independent geological samples. Summaries are descriptive with no
IID-eight/simultaneous interval, equal-accuracy, time-to-target, field-scale or
inversion acceleration claim.

Input-only command (no predictions or timings):

```sh
PYTHONPATH=src python scripts/run_batched_pod_development.py --check-inputs
```

The separate `--run-development` mode must not run until reviewed and approved.
Use a new output directory and set the protocol's five thread variables to1.
The runner retains hashes/runtime/source, preflight checks and costs, every
call/block/paired row, aggregate vectors, contexts, all candidate scores and
eligibility failures. Original studies and figures remain unchanged.
