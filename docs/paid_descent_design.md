# Paid-merit trajectory implementation scope

This additive prototype is staged outside the repository. It has no dataset,
experiment protocol, optimizer default change, runtime result, or claim of
inversion acceleration. Its tests use toy objectives or tiny synthetic native
Helmholtz problems; clocks in integration tests are fixed abstract clocks.

`ExactMeritOracle` owns one cold native sparse-LU solver. At each frequency its
merit route uses the smaller of the source and unique-receiver counts. A paid
receipt retains the actual fields and LU factors, including an accepted point
through rejected trials. The incremental full gradient selects the cheaper
source/receiver-coded route and reuses collocated fields only after checking
the actual operator's transpose symmetry. The cold RHS total is
`min(2*s, 2*r, s+r-c)` for this supported reference operator. Receiver slots
retain their multiplicities and observed values. Repeated residuals are summed.

Full source, operator arrays/configuration, model, ordered survey and data hashes
bind reuse. Mutating these inputs fails before a cache hit or paid solve.
`export_merit_fields(receipt)` exposes immutable U or C arrays without another
physics RHS. The issuing oracle must validate an actual issued snapshot before
use. Foreign, copied, modified or expired snapshots fail closed. Exports do not
keep an expired receipt's LU alive, and contain no factor handle. C means
`P_unique @ inv(A)`; using `C.T` as Z requires the measured transpose symmetry.
Ordinary floating-point LU fields have numerical residuals; consumers must
measure them rather than declaring them zero. Copy/hash/symmetry work is counted.

`run_paid_descent` accepts an oracle factory and optionally a
`proxy_factory(own_oracle)`. Constructors and initial accurate merit are paid
inside the online clock. All policies use an infinity-normalized direction,
the same box projection, and the same backtracking rule requiring strict
accurate-objective decrease. A failed/abstaining certificate, invalid or zero
proxy direction, or exhausted proxy line search permits one paid exact-gradient
recovery. A zero proxy never establishes stationarity. Only the paid exact
projected gradient mapping may establish constrained stationarity. A certificate
on the raw direction does not certify the clipped displacement or finite step.

The engine records all rejected trials, paid work that overruns budget,
fallbacks, failures, cumulative counters, and accepted checkpoints. It checks
budget before and after expensive calls. Completed work past budget is paid
but receives no new endpoint credit. The runner must choose the budget,
targets, step schedule, cases and proposal parameters before project outcomes;
none are selected by this generic implementation. Truth/offline verification
is never passed to the policy.

`run_exact_lbfgsb` is the stronger native SciPy L-BFGS-B comparator, using the
same merit/gradient receipt cache. It preserves native termination and all
function/Jacobian events. A native callback creates an eligible endpoint only
on strict accurate-objective decrease within budget. Equal-objective callbacks
remain paid events and do not change the native solver algorithm. This
comparator has its own native line-search policy; it is not relabeled as the
common projected-descent policy.

The current oracle only enables the audited native sparse-LU backend. The
`LinearSolveContext` protocol is a future adapter boundary, not an enabled
custom-solver factory. A future accurate GMRES adapter must retain per-column
convergence/failure diagnostics, actual iteration/setup costs, operator identity,
and nonzero numerical field residuals before it can become an active backend.

The owning experiment must count proposal/stability constructors, all current-
model setup, accurate line-search work, and offline fitting separately. Accurate
full gradients, approximate proposals, and approximate proposals plus a
directional certificate are distinct tasks. No method should be credited with
offline validation fields, free warm caches, or omitted setup. A trajectory
study must also report an uncertified control using the exact same approximate
fields, as well as exact L-BFGS-B and the strongest appropriate linear-solver
baseline. No outcome of those unrun comparisons is implied here.
