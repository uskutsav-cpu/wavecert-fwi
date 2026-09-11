# Mentor sign-off checklist for the Phase-0 research freeze

This file exists so the project does not silently convert internal assumptions into mentor-approved claims.

Ask the seismic/FWI mentor to review only the points that materially change the project:

1. **Physics scope:** Is 2-D constant-density acoustic FWI an appropriate first system before elastic/3-D work?
2. **Parameterization:** Is squared slowness the right primary optimization variable for the initial derivation and reference implementation?
3. **Core novelty statement:** Is the distinction clear between improving Jacobians during training and certifying an already-trained surrogate during inversion?
4. **Primary theoretical target:** Is a directional-descent certificate sufficient for the first result, or should the project prioritize a full gradient-norm bound?
5. **Experimental unit:** Should selective fallback first be decomposed by shot, frequency band, or both?
6. **Reference backend:** Is Devito's acoustic FWI stack an acceptable high-fidelity reference for the next scale-up?
7. **Stress-test priority:** Which matters most first: source geometry, frequency shift, illumination, salt/fault complexity, or held-out geology?
8. **Meaningful compute benefit:** What reduction in exact forward/adjoint solves would be considered practically significant in this seismic setting?

## Sign-off record

- Reviewer:
- Date:
- Approved as written / approved with changes / not approved:
- Requested changes:

No approval should be recorded here unless the reviewer actually provides it.
