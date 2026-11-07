# Prospective fresh-index confirmation after development selection

This workflow fixes a new evaluation cohort before the development winner is
known. It is an assistant-development engineering record, not a student
preregistration or proof that researchers never inspected other results.
The main training checkout remains unchanged while training runs.

The Stage A plan is `configs/gradient_confirmation_plan.json`, recorded at
2026-11-07 21:03:28 UTC while training status was running and `selection.json`
was absent. Its SHA256 is
`8330f33ee38255dbb14f54e2ddcd6f154bbe7d5f4a40b11f2560b652f563eb41`.
The tools reject other plan bytes. The comparison parameters are not selected
by maximizing confirmation performance.

## Freshness and finite population

The old external study inspected 100 original model indices from each named
500-model author batch. The published protocol and historical preparation
receipt fix those exclusions. From each remaining 400-index complement,
`SeedSequence([2026110731, family_code])` selects 100 indices without replacement.
The explicit old indices, eligible populations, new selections and hashes are
saved in Stage A. All four source/frequency blocks remain together per original
model index.

One-sided hypergeometric bounds use **N=400, n=100**, separately for each family
and model label. They describe only that named complement. Repeated raw arrays
remain original index slots with fixed outcomes; they are not silently collapsed
into a different sampling design. Conditional generated-draw bootstrap intervals
require an additional unverified independence assumption and are withheld on
within-population raw duplicate clusters. Neither uncertainty summary is a
real-site, field-data, new-family, or universal geological claim. Bounds are
individual, without simultaneous coverage across groups and outcomes.

These are fresh model outcomes relative to the old external 100, within two
already known generated families and batches. They are not new field data,
original OpenFWI seismic records, or a claim that the entire research direction
was chosen without prior test observations.

The native NPY payload SHA256, ZIP-member CRC32, shape, dtype and original model
identities are verified against published pins. A newly generated local
preparation receipt can have different timestamps, provenance and snapshot
paths. Its structural identities are checked and its own hash is recorded;
its bytes are not required to equal the historical published receipt.

## Stage B: freeze selection and weights first

The new freeze tool requires the original training status and summary to be
complete, with unchanged source and inputs. It verifies all four candidates,
2,000-update histories, scheduled validation histories, best eligible update,
checkpoint hashes and the existing validation-only winner rule. Initial
untrained checkpoints are ineligible. Current recorded training source bytes
must match their declared Git commit; subsequently added tools are distinguished
from edits to the training files.

The output must be a **new directory**. It copies the selected weights, both
legacy weights, both pinned legacy calibration reports, the plan and original
training receipts. An immutable `freeze.json` binds those copies and all four
candidate checkpoint identities. Original training results and selection files
are never rewritten. An incomplete or changed training run cannot authorize a
new held-out prediction through this workflow.

Each freeze records the complete source file manifest and numerical runtime.
Before Stage C, every source/config path recorded by B must still have the same
checksum. Before Stage D, the manifests from both B and C are checked again.
The Python, NumPy, SciPy and PyTorch versions and planned CPU device must agree
across stages. Other environment information is recorded without requiring
unrelated optional packages to match exactly.

New unused analysis files may be added, with their paths disclosed in the
next stage's audit. A new source file imported into the pipeline is not treated
as an unrelated addition. Editing recorded pipeline, physics, model, statistical
or configuration files, or changing a bound runtime version, requires a **new
immutable freeze attempt before new outcomes**, not rewriting the old receipt.
These guards support reproducibility; they are not a security proof against
arbitrary dynamic code outside the recorded workflow.

## Stage C: selected-checkpoint calibration

Only after Stage B exists does the wrapper call the existing
`run_real_wavecert_study` for the selected checkpoint. This is a new calibration
computation on the original **12 dedicated source calibration parents**, not
12 newly collected or previously uninspected parents. Training/model selection
does not load those arrays.

The existing function also reports the historical eight ID-test parents and
one OOD parent. Those outcomes are explicitly **development reuse**, not pristine
confirmation. The wrapper preserves its new output in a separate directory,
keeps both legacy calibration reports unchanged, and writes a second immutable
`calibration_freeze.json`. It requires the planned dataset/role identity,
12 calibration parents, matching selected weights and a positive finite scale.
Failed attempts leave a failure receipt and no authorization for fresh
confirmation. Calibration outcomes cannot choose another candidate.

## Stage D: all three fixed models and actual aggregate gradients

Confirmation requires both freeze stages, the exact plan, exact weight and
calibration hashes, and validated raw payloads **before model construction or
prediction**. It reports `legacy_forward`, `legacy_jvp`, and
`development_winner` on both families: 200 original models, 600 model/checkpoint
cases and **2,400 block rows**. A missing model label, selected index or block
prevents a complete result.

The primary quantities are the cosine and true descent direction of the
**actual summed four-block inverse gradient at one shared smoothed model**.
The neural aggregate is obtained by differentiating the summed survey objective
with respect to a shared model input and checked against the sum of individual
block gradients. It is not inferred from the mean of block cosines. The table
retains exact/neural objectives, norms, zero flags and directional derivatives.
Exact and neural aggregate vectors are saved with family, model index and
checkpoint identities for independent inspection.

Secondary block tables retain forward/gradient errors, deterministic and
source-calibrated directional bounds, coverage, acceptances and false
acceptances. Transferring a source-calibrated scale to these external models
is descriptive OOD analysis, without a nominal external conformal guarantee.
Neither a block direction nor the aggregate direction is a finite-step,
adaptive-path, inversion-trajectory, or acceleration result.

Partial or nonfinite execution is marked failed with the affected unit and
completed rows retained; it is not reported as a completed 200-model study.
No automatic retraining, recalibration, model substitution or test-based
selection occurs.

## Commands

Read-only plan validation performs no predictions:

```bash
PYTHONPATH=src python scripts/gradient_confirmation.py check-plan
```

Only after training completes, freeze its recorded source checkout and outputs:

```bash
PYTHONPATH=src python scripts/gradient_confirmation.py freeze-selection \
  --training /path/to/completed/results/gradient_development \
  --training-source-root /path/to/recorded/training/checkout \
  --output results/confirmation_selection_freeze
```

Then calibrate the selected checkpoint in a new output directory:

```bash
PYTHONPATH=src python scripts/gradient_confirmation.py calibrate-selected \
  --selection-freeze results/confirmation_selection_freeze \
  --output results/confirmation_calibration_freeze
```

Finally, after the second freeze, evaluate all three models on the new indices:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/gradient_confirmation.py evaluate \
  --selection-freeze results/confirmation_selection_freeze \
  --calibration-freeze results/confirmation_calibration_freeze \
  --data data/external/openfwi_stress \
  --output results/gradient_confirmation
```

The native velocity cache can be obtained with the existing bounded
`prepare_openfwi_stress.py` workflow. These confirmation tools download nothing
and have no training option. Runtime/source/input hashes, raw tables, offline
physics costs, aggregate vectors and completion/failure status are saved.

## Explicit admission of an interrupted training study

An ENOSPC-interrupted run is not silently treated as uninterrupted. The
optional recovery admission checks the separately reviewed I/O-only driver
SHA256, preserved original failure and all three completed candidates,
unchanged pinned teacher archives, recomputed shared raw-training normalization,
all four checkpoint metadata, finite optimizer histories and exact learning-rate
schedule. The restarted fourth candidate must reproduce the preserved first
100 optimizer records and update-zero validation baseline exactly, then finish
the original 2,000-update chain. Its alternate recovery directory is explicit.

The Stage B freeze copies the driver, failure/recovery receipts, original partial
and completed histories. It preserves `source_unchanged_during_run=null`,
`inputs_unchanged_during_run=null` and the absence of an original start-runtime
receipt. Matching initial/end/recovery source identities and unchanged recovery
inputs are reported separately. Actual work is **2,100 fourth-candidate updates,
8,100 total updates** including the preserved failed prefix; no end-to-end wall
time is reconstructed. The original selection/protocol and all partial outcomes
remain intact. This admission enables later separately authorized stages; it
makes no new unseen-data or uninterrupted-runtime claim.

Recovery evidence is portable without changing historical receipt strings.
The resolver prefers the adjacent, hash-pinned `recovery_driver.py` archive
inside the recovery attempt and falls back to the original driver path only
when that archive is absent. Stage B copies those exact reviewed bytes.
Checkpoint files are always read from known contained candidate/recovery
locations. Historical absolute checkpoint paths are preserved as metadata and
checked against their exact expected relative suffix; their old external
locations are never dereferenced.
