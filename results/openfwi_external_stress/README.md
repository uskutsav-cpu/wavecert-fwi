# Frozen-checkpoint external stress evidence

This is an assistant-supported development experiment, not a student-authored
competition paper, student preregistration, official OpenFWI benchmark or
field-data validation. Source commit: `3554dc5e0f7dd8c30da341f0ace9767b592b9d02`.
The run completed with unchanged source and input hashes.

The two checkpoints were frozen before evaluating the 100 selected original
models from each author-published 500-model velocity batch. Four source/frequency
blocks per model and two checkpoints produce 1,600 block rows and 400
model/checkpoint rows. Both batches have distinct raw arrays. All selections
and outcomes are retained. The original failed singleton-channel assumption
and its layout-only correction are preserved in `preparation_history/`.

| Family | Checkpoint | Accepted source-calibrated blocks | False acceptances among those blocks |
|---|---|---:|---:|
| FlatVel-A | Forward-only | 13 | 7 |
| FlatVel-A | Derivative-informed | 11 | 5 |
| CurveFault-A | Forward-only | 1 | 1 |
| CurveFault-A | Derivative-informed | 0 | 0 |

Each row evaluates 400 blocks from 100 original models. Counts are descriptive;
the correlated blocks are not independent statistical units. Zero accepted
blocks provide no evidence of useful safe neural updates. Deterministic bounds
cover all measured directional errors and accept no neural directions.
Median forward relative errors exceed one and median gradient cosines remain
near zero. The source-calibrated gate has no nominal OOD coverage guarantee.

`blocks.csv` retains objectives, gradient norms, signs, bounds and identities.
`models.csv` groups the four blocks by original model. `summary.json` reports
each family/checkpoint separately, plus input/source hashes and offline
evaluation cost. `run_status.json` establishes terminal completion. The plot
shows one point per original model in each checkpoint panel.

Fixed-batch hypergeometric bounds describe only the named 500-entry batch under
uniform sampling without replacement. The binomial/bootstrap alternatives
additionally assume independent generated-model draws. Neither establishes
independent real-site sampling; intervals are not simultaneous across groups.
There is no optimizer trajectory, finite-step or acceleration experiment here.

Only velocity maps from the [author release](https://zenodo.org/records/7272136)
were used. The reference observations were regenerated after documented resizing
and unit conversion. The original seismic records were not used. Attribution,
upstream license provenance, bounded ZIP-member checks and the unverified
full-archive MD5 limitation are in `protocol.json` and `preparation_manifest.json`.
The small historical source-domain calibration snapshots are bundled separately
under `data/reference/openfwi_source_calibration`; raw OpenFWI batches are not
bundled in the repository.
