# Velocity-variable paid-merit baseline candidate

This is a standard coordinate adapter, **not a new method or a project-data
recovery/performance result**. It was prepared after inspecting the development
diagnostics. No final student question/design, new optimizer, prior selection or
confirmation claim is supplied by it.

The frozen native solver uses squared slowness `m=v^-2`. The wrapper uses
positive velocity optimizer coordinates, with exact chain rule
`g_v=-2*v^-3*g_m` and tangent pushforward `d_m=-2*v^-3*d_v`. It forwards native
evaluate/gradient/accept operations, preserving legal cache/factor reuse and
physics counters, while reporting its own coordinate-work counters separately.
Reversed bounds, native-owner authentication, immutable receipt bindings,
problem/units fingerprints and floating-point roundtrip ambiguity are explicit.

**Receipt, checkpoint and final optimizer models are velocities.** Their
gradient/model hashes cannot be relabeled as native slowness. Receipts expose
the actual converted native model and its hash; `start_from_slowness` exposes
requested versus actual native initialization. A future paid comparison must
use a common actual physical start, convert/rehash returned native models inside
the measured API, and apply quality metrics afterward. Existing m-only endpoint
helpers are unsafe on unconverted wrapper trajectories.

The exact source is `fixture/velocity_oracle.py`, SHA256
`61dc9b2c80f7adba64154a92e4e63d716c127b81cdfb161c8267e43a08734d5c`.
Two AI source reviewers found no material mapping/ownership/cache/cost blocker.
Actual validation passed 31 manufactured tiny-grid tests with warnings as errors,
Ruff and Python 3.10 grammar checks. This includes a real two-iteration existing
L-BFGS-B control with three strictly decreasing paid-merit checkpoints and an
explicit injected-failure case. Its max-iteration/unsuccessful native optimizer
status is retained rather than called convergence. No rejected trial occurred
in that integration control; separate cache tests cover unaccepted evaluations.

Run the manufactured controls from the repository root:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q -W error \
  results/velocity_parameterized_baseline_candidate/fixture/test_velocity_oracle.py
```

`fixture/VALIDATION_RECEIPT_02.json` and
`fixture/independent_source_review_v1.json` retain exact commands, outputs,
source/test/scope hashes and source-invariance evidence. Earlier 30-test and
preformat versions are preserved under `fixture/history/`. The original 161-file
production source digest remains `9f30afa9`; this candidate is not integrated
into that frozen source tree. Project-data execution requires a separately
reviewed fully paid engineering protocol. No useful reconstruction, speed,
certificate-transfer, novelty or student-ownership conclusion follows from
manufactured controls alone.
