# Research roadmap

This roadmap distinguishes implemented tools from achieved scientific outcomes.
The current evidence ledger is in [strengthening_status.md](strengthening_status.md).
Historical phase reports describe older implementations and experimental designs.

## Verified development foundations

- [x] Transparent discrete 2-D Helmholtz states, tangents and adjoint gradients.
- [x] Directional residual bounds and recertification after hybrid repairs.
- [x] Correct finite-sample quantiles, whole-parent splitting and fixed-state
  repair-path calibration with explicit limits on interpretation.
- [x] Bounded LU reuse, complete online case-level cost counters and isolated
  offline validation.
- [x] Paid true-objective decrease and projected Armijo options.
- [x] Independent spectral/manufactured-solution reference verification.
- [x] Learned FNO state/JVP/VJP interfaces and forward/derivative training controls.
- [x] Parent-isolated attributed geological fixture, saved checkpoints and
  retained negative generalization results.
- [x] Corrected Devito exact baseline, fixed time sampling, boundary pullback and
  finite-difference checks on a downsampled Marmousi example.
- [x] Seeded heuristic controls, acquisition shifts, provenance and CI.

## Scientific outcomes still open

- [ ] A cheap, useful certificate for the intended realistic wave operator.
  Dense receiver maps are reference tools; the analytical sufficient bound
  can be unavailable. Compare with equally optimized exact methods.
- [ ] Strong learned forward and derivative generalization across independently
  held-out geological parents, including smoothed inversion states.
- [ ] Reconstruction or time-to-target improvement at matched **total** cost.
  Fewer repaired gradient blocks alone does not meet this requirement.
- [ ] A separately justified confirmation dataset after development choices are
  frozen. Additional correlated slices cannot substitute for new parents.
- [ ] Broader independent OOD geology; the bundled OOD stress set has one parent.
- [ ] A backend-specific derivation and experiments before claiming time-domain,
  elastic, continuous-PDE or field-scale certification.
- [ ] Student-owned scientific interpretation, substantive human criticism,
  independent reproduction and a student-authored submission with assistance
  disclosed under the current competition rules.

## Next controlled development experiment

Diagnose the observed prediction and Jacobian failures before increasing model
size. A possible development comparison is train-only smoothing/perturbation
augmentation with stronger, appropriately normalized derivative supervision.
Choose any settings exclusively on development validation, preserve all explored
models, and freeze the eventual confirmation protocol before accessing new
confirmation data. This is a proposed experiment, not a demonstrated improvement.

The competition outcome is not a software milestone and is not promised here.
