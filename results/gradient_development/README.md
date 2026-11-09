# Frozen four-candidate development comparison

This is assistant-assisted model development on **training and validation only**,
not independent confirmation or a student-authored competition submission.
All four candidates completed the frozen 2,000-update budget. Selection uses
the shared physical receiver error, tangent error and aggregate-gradient
direction criterion in
[the protocol](../../configs/gradient_development_protocol.json).

| Candidate | Selected update | Validation score | Mean aggregate gradient cosine | Descent slices |
|---|---:|---:|---:|---:|
| Raw, forward only | 100 | 3.197331 | -0.166595 | 2/8 |
| Raw, derivative supervised | 100 | 2.808487 | -0.131176 | 3/8 |
| Raw + smoothed, forward only | 100 | 3.080468 | -0.063016 | 4/8 |
| Raw + smoothed, derivative supervised | 400 | **2.439511** | **0.161247** | **5/8** |

The eight slices belong to four validation parents. Scores and cosines average
within parents, then weight parents equally. A zero-state/zero-tangent/
zero-gradient control has score **3 exactly**, cosine zero and no descent
directions; its report and the earlier checkpoints' common-criterion reports
are retained in `controls/`. A lower selection score by itself does not
establish reliable gradients. The selected neural model's alignment remains
modest, and all later validation histories, including degradation, are retained.

The two augmentation groups share raw-training-only normalization, model
initialization and update budgets. Raw candidates have 816 training states;
augmented candidates have 1,632. Equal updates are not equal epochs or equal
wall time. The tangent targets use one fixed independently seeded direction
per state, rather than the full Jacobian. The saved labels and input manifests
make the actual frozen comparison inspectable.

## Interrupted execution and recovery

The original run stopped with `ENOSPC` while saving the fourth candidate's
first eligible checkpoint. Its three completed candidates, failed status and
100-update partial remain intact. The fourth candidate was restarted from the
same seed using the unchanged training function and protocol. Its first 100
optimizer records and initial validation baseline exactly matched the
preserved partial. All candidates retain a successful 2,000-update history;
actual work was **8,100 updates**, including the failed attempt's extra 100.

The recovery changed atomic file writing only, adding a bounded disk-space
retry policy; no retries were needed. The reviewed driver bytes are archived
at `recovery_attempts/20261107T214641503789Z/recovery_driver.py`, alongside the
execution receipt. Original failure evidence is in `failure_attempts/`.
The successful fourth candidate and its selected weights are under that
recovery attempt, while `candidates/augmented_derivative/` deliberately retains
the original partial. Read `selection.json` and the receipts to locate the
selected artifact; do not mistake the original partial for the completed run.

Training source was commit `be6e51e090b4722d0610baaf812b56d4d355ae40`, source
digest `5e56226791eb0d2297667bc3502f3d24813c81e27c4e4d3ed317b9f556871f98`.
Checkpoint-recorded, end and recovery source identities agree. Recovery-time
source/input identities and a contemporaneous numerical-runtime observation
were checked. The failed runner did not persist its original full start-runtime
or end-to-end timing receipt. Those whole-run fields remain unknown in the
summary; the recovery audit does not invent uninterrupted provenance or a
complete training-time benchmark.

The selected weights have SHA256
`b1786eb61fa6d0b7fec3afba7fd35954eb30cea3ac40db6a099d61f9afc89d02`.
They require a separate immutable selection freeze and matching calibration
before the planned fresh comparison. This directory alone does not authorize
deployment or establish generalization, acceleration or a competition result.
