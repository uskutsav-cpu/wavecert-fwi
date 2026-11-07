# Additive batched POD and coarse-adjoint execution

These adapters preserve the recorded scalar implementations in `pod.py`,
`coarse_stability.py`, and `coarse_adjoint.py` byte for byte. They introduce no
new stability estimate, acceptance threshold, optimizer default or theorem.
The physical directional bound and ordinary-floating-point limitations remain
those in [coarse_adjoint.md](coarse_adjoint.md).

`batched_pod_survey_fields(surrogate, m, blocks, receiver_indices)` returns
`PODSurveyFields`. Its `block_states`, `block_adjoints`, and `block_gradients`
have shape `(fine_dimension, n_blocks)`; `block_objectives` has one value per
block. All columns follow the original input order. `objective` and
`aggregate_gradient` (also `gradient`) aggregate every block. Source/frequency
pairs must be unique, while repeated receiver slots retain their full weight
through summed receiver injection.

The adapter groups blocks by frequency, prepares the unchanged reduced
operator once per group, and uses matrix RHS for the forward and Hermitian
adjoint reduced solves. It lifts the resulting coefficients with matrix
products. All source/adjoint RHS columns, fine lifts and operator/mass work
remain charged to the original POD counters. Model caches retain their
existing content-based keys; a copied model reuses valid factors while changed
model bytes require new model projections/factors.

`BatchedPODCoarseAdjointCertifier(physics, surrogate, stability)` has the same
`certify_survey` interface and three method names as the scalar certifier. It
returns the same `CoarseAdjointSurveyResult` and per-block certificate types.
The model/operator signatures and frequency stability contexts are unchanged.
An internal field-only POD adapter avoids computing a raw physical gradient
that would be discarded before correction. The final fields are contracted
once per block.

Spatial DST-I/IDST-I transforms use only axes `(0, 1)` of a
`(nz, nx, n_source_columns)` tensor. The source axis is never transformed.
The stencil applies the same five-point operator to each column; each residual
is recomputed after correction. The state correction precedes recomputation
of the adjoint RHS, using the corrected receiver state. Coarse solves use
matrix RHS and the unchanged frequency factorization. Measured residual
projections/components are reused between diagnostics and the oriented solve;
no component is replaced by an assumed zero. Column norms use scaled
Euclidean evaluation to avoid intermediate overflow. Small SVD/eigenvalue
stability evaluation in `FastCoarseStability.evaluate` is unchanged.

The aggregate direction is still the unit negative sum of every physical
block gradient, with the same small/zero-gradient rule. Every block is bounded
at that one direction. Unavailable frequencies retain raw POD fields and
explicitly skip correction; any unavailable block makes the complete bound
and upper derivative null and requests exact fallback. Available bounds that
do not prove descent also request fallback. Numeric overflow abstains with
strict JSON instead of exporting an infinite bound or partial derivative.
No exact recovery occurs inside this API.

`costs` retains the original `pod`, `coarse_stability`, `certificate`,
`reference_solver`, and `dimensions` groups. The new `batch_dispatches` group
counts actual source assembly, matrix projection/solve/lift, receiver
injection, stencil, norm and FFT dispatches. Existing reduced/coarse RHS
counters count matrix **columns**. `coarse_solve_calls`,
`coarse_projection_calls`, and `coarse_lift_calls` count actual calls, while
`full_operator_applications` counts applied field columns. Existing
`dst_I_real_transforms`/`idst_I_real_transforms` count real-equivalent spatial
transforms per field column (two for a complex field), using the same units
as the scalar implementation. Dispatch reduction is not a reduction in those
mathematical transform columns. Reusing an already computed residual
decomposition can also reduce transform columns. Constructor setup, model
mass/DCT work, and small inverse/SVD/eigenvalue work remain paid and must be
reported separately by any study. No wall-time or inversion-speedup claim is
made by these counts.

The wrappers use explicit frozen private adapter entry points. Import-time
checks reject cached imported reference modules/classes from another checkout
and verify SHA-256 of their actual source paths. They bind the three reference
adapter files and their underlying `helmholtz.py` operator; both adapters expose
`reference_source_sha256`. A change to those reference files requires a
reviewed adapter update rather than silent compatibility. Experiment-level
source manifests still bind the entire implementation during a run.

Synthetic tests cover non-square grids, complex fields, multiple/interleaved
frequencies, repeated receiver entries with different observations, all three
methods, matrix-RHS accounting, source-axis isolation, model cache content,
mixed unavailability, exact-limit acceptance, independent objective finite
differences, and zero/nonfinite/overflow abstentions. Scalar/batch arithmetic
can differ by normal BLAS/FFT rounding; parity is tested tightly and decisions
are compared away from the gate threshold. The corrected physical proposal
remains distinct from differentiating a composite corrected state map.
