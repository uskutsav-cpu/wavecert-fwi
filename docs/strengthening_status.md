# Evidence ledger for the 0.6 development work

This is a technical development record, not a competition research paper,
student interpretation, mentor endorsement, or claim of a 10/10 project.

## Completed implementation improvements

- Correct finite-sample conformal order statistics, explicit unsupported ranks,
  zero-bound failures, and consistent zero-scale handling.
- Dedicated calibration data separated from checkpoint model selection.
- Whole-parent splitting, overlap audits, parent-level simultaneous coverage,
  appropriate uncertainty units, and checkpoint/dataset integrity links.
- Complete fixed-state repair-path scores, with no unsupported trajectory claim.
- Bounded model-content-keyed LU reuse, shared adjoint factors, per-frequency
  directional stability work, and complete exact-work counters.
- Optional analytical stability lower bounds with explicit unavailability and
  exact fallback before neural work.
- True-objective decrease and projected Armijo checks, explicit online costs,
  and offline validation on an isolated solver/cache.
- Seeded random/residual/periodic trajectory baselines and an acquisition-domain
  guarded policy that does not claim to detect every geological shift.
- Independent spectral/manufactured-solution verification.
- Corrected Devito time sampling, padding pullback, and final model restoration;
  actual fixed-dt finite-difference checks and an independent Marmousi baseline.
- A compact attributed SubsurfaceGen-derived fixture with separate training,
  validation, calibration, and test parents; forward-only and optional established
  derivative-informed training paths.
- Source/data/checkpoint hashes, runtime environment records, failure/completion
  status, raw tables, confidence summaries, and CI coverage of learned/data paths.

## Evidence to inspect

| Artifact | What it establishes | What it does not establish |
|---|---|---|
| `results/independent_physics` | Sparse/spectral agreement and second-order manufactured-solution refinement | Absorbing-boundary or field-data fidelity |
| `results/reference_cache_benchmark` | Same gradients with fewer repeated factorizations | WaveCert speedup over an equally optimized exact baseline |
| `results/stability_benchmark` | Valid small-grid comparisons and analytical-bound availability | A universally useful field-scale stability bound |
| `results/cost_break_even` | The current receiver-map setup costs `2rf` RHS versus `2sf` for an exact gradient | A universal impossibility result or a measured runtime lower bound |
| `results/receiver_oracle_controls` | Direct exact directional information in the maps and a cheaper reciprocity gradient control | A neural inversion speedup or new certification theorem |
| `results/research_audit` | Historical split overlap and parent-aware descriptive reanalysis | Repair of the archived experiment's design |
| `results/evidence_study` | Complete synthetic trajectories, controls, uncertainty and full cost accounting | Field-scale advantage, universal OOD safety, or student-approved confirmation |
| `results/evidence_audit` | Raw Cartesian completeness, trajectory joins, aggregate arithmetic, and declared source/checkpoint bytes | Independent reconstruction of every executed RHS or proof of scientific validity |
| `results/grouped_geology` | Fresh forward-only training on the new parent-isolated fixture | A sufficiently accurate production wave surrogate |
| `results/grouped_wavecert` | Parent-aware derivative/certificate evaluation on that fixture | An entire-trajectory guarantee or broad OOD geological validation |
| `results/development_generalization` | Train/validation prediction, fresh-direction and residual/Jacobian decomposition diagnostics | Untouched confirmation, a full Jacobian norm estimate or successful generalization |
| `results/openfwi_external_stress` | Frozen-checkpoint derivative/certificate stress results on 200 selected generated models from two author-published batches | Original OpenFWI seismic benchmark performance, field data, or broad independent geological validation |
| `results/devito_verified` | Independently checked exact time-domain baseline on downsampled Marmousi | Neural WaveCert time-domain certification |

The fresh forward-only FNO's held-out receiver prediction error remains high
(about 68% median on the forward test states). At the smoothed inversion states,
the calibrated gate produces observed false acceptances. These results are
retained. Do not describe the new hard geological evaluation as perfect safety
or successful acceleration.

Development-only decomposition checks confirm that both receiver-residual
prediction and neural Jacobian accuracy need improvement. On the existing
model-selection parents, the two checkpoints' median smoothed-state gradient
cosines are about `0.016` and `0.015`. Holding the true residual fixed still
leaves low alignment. These diagnostics use training/validation arrays only;
they do not create a new untouched test set or justify tuning on the final
test results. More slices from existing parents cannot increase independent
geological replication.

The external generated-velocity stress study completed on source `3554dc5`
with unchanged source and input hashes: 200 selected original models, two
unchanged checkpoints and four blocks each (1,600 block measurements). The
receiver forward-error medians range from about 1.03 to 1.09 and gradient
cosine medians from 0.014 to 0.049. Deterministic bounds cover every measured
directional error but accept no neural directions. The source-calibrated
gate's accepted/false-accepted block counts are 13/7 and 11/5 on FlatVel-A
(forward/JVP checkpoints), and 1/1 and 0/0 on CurveFault-A. Zero acceptances
are abstention, not demonstrated useful safe neural updates. Nominal
source-domain conformal coverage is not asserted for these OOD outcomes.

Every original model retains its four correlated blocks. Fixed-batch
hypergeometric intervals cover only the named 500-entry batch under the
declared uniform 100-entry sample; additional binomial/bootstrap summaries
require an unverified independent-generated-draw assumption. The two families
are not pooled into independent environments. Both primary batches contain
distinct raw arrays, but this does not establish independent real sites.

The completed frozen-source study uses 60 independent generated models in each
of four scenarios, nine policies and 12 iterations: 2,160 case rows and 28,080
trajectory rows. The separate raw-table audit passed, including historical
source/checkpoint verification against `dad24d3`. These are development results.

On clean cases, calibrated selective repair uses 40.98 versus 48 exact gradient
blocks, but 1,301.90 versus 179.67 total online RHS. Its mean paired runtime
ratio to exact FWI is **46.74** (model-bootstrap 95% interval 44.18–49.28).
The paired final model-error difference is `+0.000404` (0.000197–0.000616),
slightly worse. It improves model error over the two-block random control by
about 0.00559 while taking 6.00 times as long; this is not matched-cost
superiority. Deterministic repair reproduces exact trajectories after paying
substantial extra certification work.

The frozen calibration realization covers only 45 of 60 clean initial repair
paths (75%). Its one-sided 95% failure-probability upper limit is 35.85%.
This is weaker than the target observed coverage and must be retained. A
marginal split-conformal theorem does not promise the target conditional
coverage for every particular calibration sample. No whole-trajectory coverage
is established.

Under frequency shift, unguarded calibrated repair has mean objective ratio
0.749 versus 0.0286 for exact FWI. The acquisition guard routes all shifted
frequency and sparse-receiver iterations to exact fallback and exactly matches
their reference objective trajectories. Its frequency-shift paired runtime
ratio is 1.011 (1.001–1.023). This establishes useful fail-closed routing,
not improved reconstruction or a general OOD detector. Zero objective increases
across the experiment follow paid exact-decrease acceptance, not calibration
alone.

The earlier 12-model synthetic pilot showed near-exact reconstruction with
calibrated selective repair on clean cases, but more elapsed time than exact FWI.
It also exposed severe frequency-shift stalling. The guarded policy was added
after that development observation and is evaluated as a new exploratory policy,
not retroactively called preregistered. Its frequency/geometry checks cannot
establish exchangeability for unknown geology or noise.

The corrected five-iteration Devito baseline reduced the discrete objective by
about 75% and passed full-domain/interior finite differences below `1e-9` best
relative error. It uses one source geometry, one downsampled model, and synthetic
observations. Optimizer termination at its iteration limit is reported rather
than relabeled convergence.

## Work that still determines scientific competitiveness

1. Establish a useful, affordable certificate on realistic wave operators while
   comparing against an equally optimized exact solver and simpler controls.
2. Improve forward/derivative generalization on many independently held-out
   geological parents. Derivative-informed training is an established comparison,
   not by itself a novel WaveCert contribution.
3. Demonstrate reconstruction or time-to-target benefit at matched total cost.
   A direction cosine or repaired-block reduction alone is insufficient.
4. Extend mathematical scope to the intended production backend before claiming
   time-domain, elastic, continuous-PDE, or field-scale certification.
5. Evaluate broader OOD geology. The available Penobscot stress set has one parent.
6. Have the student independently analyze the evidence, defend the methods,
   obtain substantive human criticism, and write the actual submission.

These are open research outcomes. They cannot be completed by changing a status
flag, adding documentation, or assigning a higher score.
