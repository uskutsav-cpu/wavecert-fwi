# Bounded cold-query runtime controls

This study measures three existing implementations on the **same eight
previously used validation slices from four parents**, each with all 24
source/frequency blocks. It uses the frozen rank-64 state+tangent POD basis
and rank-96 coarse stability space. It adds no new geology, training,
parameter selection, optimization policy, or held-out outcome evaluation.

The three tasks differ:

| Control | Output |
| --- | --- |
| `best_exact_objective_gradient` | Exact discrete objective and full adjoint gradient, in ordinary floating point |
| `raw_POD_objective_gradient` | Approximate POD objective and full reduced-model gradient |
| `corrected_POD_directional_certificate` | Approximate physical objective/gradient proposal plus a fine-objective directional bound |

The exact implementation chooses the smallest implemented RHS count from
\(2s\), \(2r\), and \(r+s-c\) when transpose symmetry holds. This choice is
not asserted to minimize wall time among every possible exact solver. In
the fixed geometry, each frequency has six sources, twenty unique receiver
positions, and six collocations, so this chooser uses the ordinary route:
48 fine RHS and four fine factorizations for the complete survey.

The corrected proposal/bound has the scope described in
[`coarse_adjoint.md`](coarse_adjoint.md): exact-arithmetic inequalities
evaluated with ordinary float64/complex128 and an unverified cancellation
allowance. It is not a formally verified roundoff enclosure. Unavailable
bounds retain all proposal blocks and explicitly require exact fallback.
That recovery is not executed here. No objective-decrease, finite-step,
inversion, time-to-target, equal-accuracy, native-64, or field-scale speedup
claim follows from a query-time comparison.

## What is timed

Each method/case/repetition receives **new independent** physics, POD,
coarse-engine and certifier objects as applicable. Nothing reuses a model,
factor, receiver or mass-projection cache across measured invocations.
Caching *within* a 24-block survey remains enabled as in the existing APIs.
These are cold **per-model solver caches in one process**, not fresh Python
processes or cold operating-system/instruction/data caches. The offline
reference solver runs first, so its code/library initialization may warm
process-level state before the measured invocations.

Wall time uses `time.perf_counter`. Three adjacent timestamps give:

1. **Setup:** new Helmholtz object and, as applicable, the POD constructor
   (basis copy, orthogonality Gram, projected stiffness/damping) and coarse
   engine/certifier construction (including both damping DCT-I transforms).
2. **Query:** the actual existing API call, including current-model mass
   projection, fine or reduced factorizations/solves, model DCTs, small
   SVD/eigenproblems, rank-column inverse RHS, receiver constants, corrections,
   residual stencils, physical gradients, and certificate result construction.
3. **Total cold:** setup plus query. It excludes receipt extraction, common
   quality checks/JSON/CSV output, and object teardown after the query.

The raw POD survey calls its existing `objective_and_gradient` once per block
and sums all objectives/gradients. The corrected method calls the existing
`PODCoarseAdjointCertifier.certify_survey` once, including its shared survey
direction and bound. The exact control calls the existing strongest
implemented full-gradient selector once.

Counters accompany times. Setup separately retains the POD stiffness/damping
projection receipts, one basis orthogonality Gram, and **all** coarse-engine
constructor count deltas. Query/cumulative receipts preserve every currently
counted fine RHS/factorization/SVD, POD mass/projection/lift/solve, and coarse
DCT/DST/IDST, small factor/eigen/SVD/inverse-RHS/correction/receiver operation.
Matrix columns count as RHS columns. Zero fine solver work for the two
approximations is checked from actual counters; they still perform full-grid
mass projections, lifts, stencils and vector contractions. Operation counts
are not interchangeable FLOP units.

## Common and excluded work

The identical fixture/basis bytes are loaded and audited once before queries;
their shared I/O/audit time is reported separately. Model/block arrays are
prepared once, with their preparation time also reported. A separate fine
solver recomputes all 192 block objective/gradients outside core time and
checks the stored exact aggregate labels; its wall time and fine work are
reported separately. The original shared teacher-label generation, POD basis
fit and validation-selection history remain in input provenance. They are
reused development costs, not free or charged once to each query.

The order is frozen: repetition outermost, then all eight cases; the three
methods rotate by `(repetition + case_index) % 3`. Every method occupies each
position once for each slice across three repetitions. Output writes happen
only between complete triples. All **72** cold calls remain in
`cold_calls.csv`, with **24** directly paired rows in `paired_cold_calls.csv`.
All **576** corrected-certificate block checks remain in
`certificate_blocks.csv`. Cost receipts bind model/operator contexts, and
source/input hashes are checked again at completion.

Objective/gradient agreement for the exact method, repeat agreement for all
three methods, corrected physical-gradient/direction/center consistency,
actual fine solver counts and all available directional bounds are checked.
Numerically failed objective/gradient proposals prevent a `complete` result,
even if repeated failures agree; ordinary finite unavailable bounds remain
reported as legitimate abstentions. There are no warm diagnostics, selective
case omissions, or changing stopping/acceptance thresholds in this study.

## Execution and interpretation

`scripts/benchmark_coarse_runtime.py --check-inputs` performs no prediction,
fine solve, timing, or output write. Actual timing requires `--run-timings`
in an otherwise idle compute session. Set `OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and
`NUMEXPR_NUM_THREADS` to `1` **before starting Python**. The runner enforces
these environment values, records package/source/environment/clock metadata,
and records system load averages and the caller's `--compute-context` note.
These metadata do not prove that an external machine was idle.

Summaries describe each of the four parents and retain raw paired times.
Three repetitions revisit the same eight correlated slices. They do not
create additional geological samples; no IID-eight confidence interval or
generalization inference is reported. Faster measured query time, if observed,
would still require a separate complete inversion experiment with common
termination/quality targets before supporting an inversion acceleration claim.
