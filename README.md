# WaveCert-FWI

**Residual checks, classical reduced models, and selective exact-physics fallback for seismic inversion.**

A neural wave simulator can predict useful wavefields while producing derivatives
that send an inversion in the wrong direction. WaveCert investigates when a
post-training physics check can justify a neural direction, and when exact wave
physics should replace unreliable source/frequency contributions.

This is a **research prototype**. Its tested certificate applies to a discrete
2-D Helmholtz system. Field-scale acceleration, time-domain certification, and
reliability on measured seismic data remain unestablished.

## Start here

```bash
git clone https://github.com/uskutsav-cpu/wavecert-fwi.git
cd wavecert-fwi
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,torch,data]'
pytest -ra
wavecert demo --output results/demo
wavecert verify-physics --output results/independent_physics
```

The core demo needs only base dependencies. PyTorch and data extras support the
learned studies. Devito additionally needs its extra and a working C compiler.

```bash
# Short integrated check.
wavecert evidence-study --seeds 731 --cases-per-seed 1 --iterations 2 \
  --calibration-cases 10 --alpha 0.2 --scenarios clean --output results/quick_evidence

# Larger development protocol.
wavecert evidence-study --cases-per-seed 20 --iterations 12 --output results/evidence_study
```

The study compares exact FWI, neural-only, fixed-budget random/residual repair,
periodic exact updates, calibrated repair, acquisition-guarded repair, deterministic
receiver-aware repair, and analytical-bound repair. It saves raw trajectories,
paired model-level summaries, uncertainty intervals, all online exact costs,
checkpoint/source hashes, and quality-versus-time figures. `status.json` marks
running, failed, and complete runs; source changes during execution mark a run stale.

## Ongoing scientific phase

The next scientific objectives remain open. The new
[ancestry audit](docs/ancestry_audit.md) finds that the twelve SEAM IDs share one
original model across research roles; model-ID separation is not independent
ancestry/site separation. Historical outcomes remain, with narrower interpretation.
See [continuing research](docs/continuing_research.md) for the full active goal,
new controlled development evidence and the gates before any success claim.
The new paid-merit, cached-field bound, norm-free GMRES and trajectory components
pass the full CI suite and 89 focused checks with warnings treated as errors.
The [360-run paid-merit engineering comparison](docs/paid_merit_engineering_benchmark.md)
is now complete and independently checked. Exact L-BFGS-B returns both targets
in 48/48 planned attempts versus 30/48 for the certified arm. At internal
8-second checkpoints, certified velocity error is worse than initialization
on 7/8 cases. A useful certificate advantage remains unestablished.
Student scientific ownership and final design decisions remain open; see the
[assistance record](docs/assistance_record.md).
The new [boundary diagnostic](docs/devito_boundary_verification.md) also exposes
substantial finite-record trace errors in a fixed homogeneous Devito fixture;
passing gradient checks alone does not establish boundary fidelity.
Its [separate NumPy replay](docs/numpy_boundary_replay.md) reproduces all seven
implemented traces in the local run, including the substantial boundary effects.
The fixed [BP2004 acquisition](results/bp2004_acquisition/README.md) adds one
synthetic ancestor with original terms/provenance retained; it has not been used
to evaluate the method. Three crops do not create three independent sites.
The [recovery/physics diagnostics](docs/recovery_and_physics_diagnostics.md)
now show that reference data-descent itself initially increases velocity error
in 5/8 reused cases, with feasible initial/truth models. Separately, the enlarged
homogeneous timedomain reference agrees with an analytical Green solution to
0.0305142% relative L2; the small 40-cell-padding case remains at 24.9504%.
Both are independently audited, scoped diagnostics, not a recovery win or a
completion of realistic geological validation.

The [48-run paired coordinate comparison](docs/coordinate_cost_engineering_comparison.md)
also completes within its full API-return cap, with an independent 4,523-check
post-audit. Physical-velocity coordinates reduce error relative to native m on
five cases, but both methods remain worse than initialization on those same five.
Changing coordinates alone does not resolve this fixture's recovery problem.

The [fixed initial-linearization diagnostic](docs/initial_linearization_diagnostic.md)
now evaluates all eight cases at eight frequencies. Lower frequencies reduce
the relative remainder in seven cases, with case 2 retained as the exception.
This is a diagnostic rather than an inversion remedy. A separate
[anchor-reuse feasibility check](docs/stability_anchor_feasibility.md) returns
zero sufficient bounds at all 192 old endpoint/frequency instances. Neither
finding establishes a new method or completes the scientific phase. The
[BP heterogeneous numerical verification](docs/bp_heterogeneous_verification.md)
now completes all 57 calls, with separate NumPy agreement on three probe traces
and six directional derivatives. Substantial padding sensitivity remains;
physical acceptance and broader independent geological validation remain open.


The new [96-run frequency-continuation comparison](docs/frequency_continuation_development.md)
completes all three repetitions under the full return cap and passes a separate
41,398-check scalar audit. Continuation still worsens velocity recovery versus
initialization on 7/8 case medians; lower data misfit is insufficient. The interrupted
85/1/10 predecessor and the checker's failed first schema audit remain archived.
All four scientific objectives remain open.

## Latest controlled comparison

A train-only classical POD model substantially outperforms the tested FNOs on
inversion-relevant gradients. On 200 previously unevaluated original model slots
from two known generated batches, raw POD has mean gradient cosine 0.99745 and
0.99953; coarse-corrected POD has 0.99866 and 0.99975. The new neural model has
0.05583 and -0.01178. All five fixed comparators and all 4,000 model-block evaluations (800 distinct
source/frequency blocks) are retained.

The new coarse state/adjoint certificate accepts 168/200 directions, observes
zero false certifications, and abstains on 32. Its guarantee is a pointwise,
discrete, exact-arithmetic inequality evaluated in ordinary floating point;
it is not a universal safety or trajectory claim. Its scalar implementation
is **5.43× slower than exact** on the eight-slice small-grid cold-query study,
including setup. A later fixed development study cuts corrected cost by 1.55×
through batching, but it remains 3.52× slower than exact. Raw batched POD costs
4.54 ms versus 12.06 ms exact for a different approximate-gradient task; no
inversion speedup is established. Lowering the coarse rank to 64 certifies 0/8
validation cases and is ineligible. All outcomes remain in the
[batching study](docs/batched_pod_development.md).

See the [evidence ledger](docs/strengthening_status.md),
[coarse certificate](docs/coarse_adjoint.md), [classical control](docs/pod_controls.md),
[fixed confirmation](docs/gradient_confirmation.md), and
[prior-work boundaries](docs/literature.md). The
[research assessment and next milestones](docs/jshs_research_plan.md) distinguish
completed software work from open scientific outcomes.

## Current evidence and limits

- Independent sine-transform and manufactured-solution checks verify constant-
  coefficient reference solves and second-order refinement. This does not
  validate absorbing boundaries or real field physics.
- Bounded, content-keyed LU reuse reduces repeated factorization work. The
  exact baseline receives the same improvement; no acceleration over exact FWI
  follows merely from this optimization or fewer repaired gradient blocks.
- Optional true-objective decrease and projected Armijo checks count their
  online forward/gradient work. Surrogate-merit acceptance alone can increase
  the true objective.
- The newer split separates provider model IDs across research roles, but its
  SEAM IDs share one original ancestor across roles. See the ancestry audit;
  model-ID disjointness does not establish site or original-model independence.
- The fresh parent-held-out FNO still has substantial error. Negative results
  are retained rather than described as successful production validation.
- Training/validation diagnostics separate receiver-residual error from neural
  Jacobian error. Both remain substantial at smoothed inversion states.
- Exact receiver-map and reciprocity controls use information already present
  in the reference solves. They provide stronger comparisons than counting
  repaired blocks alone; see [`docs/receiver_controls.md`](docs/receiver_controls.md).

The evidence ledger and remaining research gaps are in
[`docs/strengthening_status.md`](docs/strengthening_status.md). Historical phase
reports remain under `docs/`; they describe earlier implementations and protocols.

## Mathematical decision

For squared slowness `m`, state `A(m)u=q`, receiver restriction `P`, and observed
data `d`, the objective is `Phi(m)=0.5 ||Pu-d||^2`. The surrogate supplies `u_hat`
and `du_hat=D u_hat(m)[v]`. Define

```text
r_p = A u_hat - q
r_t = A du_hat - omega^2 diag(v) u_hat
```

A stability bound converts these residuals into a directional error bound `eta`:

```text
D Phi_hat(m)[v] + eta < 0  =>  D Phi(m)[v] < 0
```

After repairing a hybrid gradient, remaining certificates are recomputed for
the changed direction. Greedy ordering does not establish a globally minimal
repair count. See [`docs/theory.md`](docs/theory.md) for the derivation.

The reference mode computes receiver-aware norms with exact discrete solves.
The optional `analytic` mode combines Dirichlet coercivity and a normal-reference
perturbation bound without an online solve or SVD for its stability constant.
If that sufficient bound is unavailable, the policy abstains or falls back to
exact physics before evaluating the neural operator. It was unavailable on the
tested heterogeneous 64-by-64 problems; it is not a field-scale solution.
Ordinary floating-point evaluation is not formally verified roundoff certification.

## Guarantee scope and cost accounting

| Mechanism | Scope and limitation |
|---|---|
| Deterministic discrete certificate | A direction at one model under the stated operator/stability assumptions; no arbitrary finite-step or continuous-PDE guarantee |
| Analytical stability bound | The current regular-grid Dirichlet system; can be unavailable |
| Split-conformal scale | Exchangeable calibration/test units with the same sampling protocol; statistical, not deterministic |
| Full repair-path calibration | All greedy rounds at one fixed model; no later-iterate or whole-trajectory coverage |
| Acquisition support guard | Declared frequency/source/receiver support; does not detect every geology/noise shift |
| `exact-decrease` acceptance | Paid true-objective forward checks; accepted steps decrease the discrete objective, but may stall |
| `exact-armijo` acceptance | Paid true-gradient check of the actual projected displacement, unless that gradient is already available |

Report factorizations, forward/adjoint/tangent/stability RHS, SVDs, and total
online time. Repaired gradient blocks are only one cost component. Offline
validation uses a separate solver/cache. A fixed two-block random policy is not
automatically matched in total computational cost to WaveCert.

The guarded policy requires a calibration-domain contract containing
`frequencies_hz`, `source_indices`, and ordered `receiver_indices`. Observable
mismatches trigger exact fallback. A matching contract cannot establish geological
exchangeability. Legacy Phase-5 block scales were calibrated on the original
neural direction; they do not establish adaptive-path coverage.

A zero or invalid neural/hybrid gradient is not evidence of true stationarity.
With a full exact budget, repair recovers through exact gradients and paid
exact merit; otherwise it explicitly abstains. Iteration records distinguish
the requested `step_acceptance` from `effective_step_acceptance`, including
these recovery overrides.

## Reproduce parent-held-out geology evaluation

The compact fixture at
[`data/reference/subsurfacegen_parent_split`](data/reference/subsurfacegen_parent_split)
contains **24-by-24 synthetic Helmholtz fields derived from SubsurfaceGen velocity
slices**, not original field-scale wavefields or measured seismic data. It has
17 training, four validation, 12 calibration, eight ID-test, and one OOD parent.
OOD is descriptive; one parent cannot support across-parent uncertainty estimates.

```bash
wavecert real-wavecert-study --dataset data/reference/subsurfacegen_parent_split \
  --checkpoint checkpoints/fno_parent_grouped.pt --output results/grouped_wavecert

# Forward-only training; calibration never selects a checkpoint.
wavecert train-real-fno --dataset data/reference/subsurfacegen_parent_split \
  --checkpoint checkpoints/fno_parent_grouped.pt --output results/grouped_geology \
  --epochs 40 --patience 10 --width 24 --modes 8 --depth 3 --seed 20261107 --threads 1

# Established derivative-informed baseline, not a new WaveCert claim.
wavecert train-real-fno --dataset data/reference/subsurfacegen_parent_split \
  --checkpoint checkpoints/fno_parent_grouped_jvp.pt --output results/grouped_geology_jvp \
  --epochs 40 --patience 10 --width 24 --modes 8 --depth 3 --seed 20261107 --threads 1 \
  --jacobian-weight 0.1 --jacobian-seed 20261107
```

Checkpoint/dataset manifest mismatches are rejected. Report every explored model
rather than selecting the best test result. To regenerate raw-derived data, use
`scripts/prepare_grouped_subsurfacegen.py` with the immutable revision in
`configs/strengthening_protocol.json`, then `scripts/build_subsurfacegen_wavecert.py`.
Raw downloads require `--download`. New compressed-byte hashes require new
provenance and retraining, not bypassing the integrity check.

## External generated-velocity stress test

```bash
# First preparation: fetch only two bounded velocity members from the author archive.
python scripts/prepare_openfwi_stress.py
python scripts/evaluate_openfwi_stress.py
```

The frozen development protocol selects 100 original models from each of two
500-model author-published OpenFWI batches. It evaluates both existing geological
checkpoints with regenerated 24-by-24 reference Helmholtz fields. These are not
the original OpenFWI seismic records or an official benchmark evaluation.
Selections, checkpoint/calibration hashes, layout correction, attribution and
input provenance are recorded. Preparation preserves existing caches and refuses
unbounded archive downloads. Input license provenance is separate from MIT code.

The completed stress results retain high forward/gradient errors and false
source-calibrated acceptances. Deterministic coverage with no accepted neural
directions is abstention, not a useful accelerated inversion. See
[`results/openfwi_external_stress`](results/openfwi_external_stress).

## Independent Devito baseline

```bash
bash scripts/download_marmousi_public.sh
wavecert devito-study --marmousi data/external/marmousi/mar_big_117_567.bin \
  --iterations 5 --output results/devito_verified
```

This uses a common fixed time axis, CFL-safe bounds, explicit observed sampling,
the edge-padding gradient pullback, and finite-difference checks. It evaluates
exact time-domain FWI on one downsampled Marmousi model with synthetic data.
**No neural WaveCert time-domain certificate is exercised.**

## Research context and ownership

Forward accuracy versus derivative accuracy and certified surrogate optimization
are established research. WaveCert's opportunity is a useful post-training,
inversion-aware residual check with selective seismic fallback. Closest work is
mapped in [`docs/literature.md`](docs/literature.md).

The AI assistance record is in [`docs/assistance_record.md`](docs/assistance_record.md).
The student must independently review scientific decisions, analyze and interpret
the evidence, understand the derivations, author their competition submission,
and disclose assistance. No mentor approval or JSHS outcome is claimed.

Code: MIT. Bundled SubsurfaceGen-derived data: CC BY 4.0, with separate attribution
in the fixture README.

The [Overthrust partial-acquisition record](docs/overthrust_partial_acquisition.md) adds a fixed raw model cross-section for future physics checks. No wave/inversion validation has been run on it. A classical prior component is staged with manufactured checks; its proposed recovery study remains unexecuted.
