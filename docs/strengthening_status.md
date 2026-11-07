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
| `results/research_audit` | Historical split overlap and parent-aware descriptive reanalysis | Repair of the archived experiment's design |
| `results/evidence_study` | Complete synthetic trajectories, controls, uncertainty and full cost accounting | Field-scale advantage, universal OOD safety, or student-approved confirmation |
| `results/grouped_geology` | Fresh forward-only training on the new parent-isolated fixture | A sufficiently accurate production wave surrogate |
| `results/grouped_wavecert` | Parent-aware derivative/certificate evaluation on that fixture | An entire-trajectory guarantee or broad OOD geological validation |
| `results/devito_verified` | Independently checked exact time-domain baseline on downsampled Marmousi | Neural WaveCert time-domain certification |

The fresh forward-only FNO's held-out receiver prediction error remains high
(about 68% median on the forward test states). At the smoothed inversion states,
the calibrated gate produces observed false acceptances. These results are
retained. Do not describe the new hard geological evaluation as perfect safety
or successful acceleration.

The initial 12-model synthetic pilot showed near-exact reconstruction with
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
