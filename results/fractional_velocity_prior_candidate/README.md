# Classical fractional-velocity prior candidate

This staging-only AI engineering component evaluates a standard quadratic penalty
on fractional physical-velocity updates from a supplied initial squared-slowness
model. It contains no wave solver, file/data access, project arrays, optimizer,
training, truth input, tuning or stopping policy. It is outside production source.
It is neither a new regularization method nor the student's final design.

`fractional_velocity_prior.py` returns the penalty, its zeroth-order/roughness
parts, and the analytic native-m gradient. The scalar composer uses the same
positive initial-data normalization in both lambda=0 and lambda>0 arms. The
lambda=0 arm does not evaluate the prior. The caller must include all actual
preparation, construction and evaluation costs in its run clock.

`UNEXECUTED_STUDY_PROPOSAL.json` records a proposed paired 48-slot development
comparison and meaningful recovery gates against both initialization and a paired
exact baseline. No 48-run driver has been implemented or executed. The proposal
is not an execution registration or permission to run project PDEs. Exact input
hashes, slot order, reviewed driver, source freeze and root execution decision are
still required. Reused development slices and timing repetitions supply no new
independent geological samples.

Manufactured controls use only arrays of at most 35 entries. They compare values
against scalar `math.sqrt`/`math.fsum` arithmetic and directional gradients
against centered differences of that scalar reference, including corners and
edges. The controls also verify exact initial zero, copied initial anchor,
constant-update scaling, actual graph degrees, zero smoothing, a one-row grid,
both normalized objective arms, and invalid input rejection. All arithmetic is
ordinary float64, not a verified error enclosure.

Attempt 01 retained 38 passes and one finite-difference check failure: its
coarser 2e-5 perturbation had 3.02e-8 truncation error against a 2e-8 absolute
tolerance. The original source/tests/log/XML remain preserved. Attempt 02 halves
the two step sizes to 1e-5 and 5e-6, keeps both tolerances and the required
second-order convergence rule unchanged, and passes all 39 controls. Numerical
penalty/gradient source is unchanged; other edits sort imports, use TypeError for
the wrong prior object type, and use a literal test dictionary for lint.

Run the small controls from this folder:

```sh
/opt/miniconda3/bin/python -B -I -m pytest -q -p no:cacheprovider test_fractional_velocity_prior.py
```

The penalty's gradient is zero at initialization; it does not change the already
adverse initial data-gradient directions. A model that stays near initialization
fails the proposed 10% recovery-improvement and data-fit gates. The fixed weight
is an engineering scale, not a noiseless discrepancy rule or a truth-derived
prior radius. Recovery benefit and generalization remain untested.
