# Independent saved-model objective and reconstruction audit — source only

This auditor was staged before benchmark output inspection. It has not been run,
imported, syntax-checked or tested during the live timing job. Root must independently
prove that job terminal, review this source, then explicitly authorize execution.
It does not change repository files or the frozen engineering benchmark.

The source imports only standard-library modules at import/default-CLI time. Actual
execution additionally requires the exact reviewed script SHA, a new output filename,
and a terminal benchmark status before numerical imports or project input arrays.
No WaveCert, ExactMeritOracle, adapter or verification helper is imported, even when
executed. It uses the original source text only as the declared equation definition.

The independent route constructs the 49-grid truth with explicit bilinear arithmetic,
rebuilds the initial Gaussian-smoothed velocity using generic SciPy, builds the
five-point matrix by direct neighbor graph assembly, creates its own source columns
and noiseless observations, and solves with SciPy sparse LU. NumPy/SciPy/SuperLU are
shared dependencies, so this is a second equation implementation, not an independent
linear algebra library. Source/driver/addendum/protocol/input/retained-artifact hashes
and source-path manifest are checked.

All saved initial, returned-final and four internal endpoint model slots are checked
for Phi and squared-slowness/physical-velocity relative L2. Numerical work deduplicates
only identical (case, little-endian float64 model SHA), while preserving every slot
occurrence and its first occurrence. Missing/failed/unattempted rows remain explicit.
True linear residuals are checked. Online/after-return Phi, endpoint ratios and stored
quality are compared where those diagnostics exist. No unavailable intermediate
directions, fields or certificate derivatives are reconstructed.

Eligibility is separately recomputed from terminal row status plus final-study flag,
selected paid checkpoint and recorded elapsed/budget, or actual fully returned target
status/cost/return-wall conditions. An initial fallback without a paid checkpoint is
never credited. Internal endpoint evidence is explicitly internal state only. Every
saved model can still be verified numerically as an uncredited diagnostic. If any
check fails, the resulting report is not a passed numerical basis for positive use.

The frozen ordinary-floating-point tolerances are in TOLERANCES: Phi relative2e-9
plus2e-12 times initial-Phi scale (floor1e-12); quality relative5e-11 plus2e-13;
ratios relative4e-9 plus4e-12; rebuilt initial model relative2e-12 plus2e-14;
true linear residual relative1e-9. Explicit interpolation, sparse assembly and scalar
reductions change arithmetic order. These tolerances are fixed before inspecting
outcomes; failure must be retained and investigated, not hidden by loosening them.
They are numerical agreement tolerances, not formal error enclosures.

After terminal confirmation and review, invoke the CLI using single-thread BLAS
environment and these explicit options:

```
python audit_saved_models.py --execute-after-terminal-confirmation \
  --expected-script-sha256 REVIEWED_HASH --repo ABSOLUTE_REPO \
  --run ABSOLUTE_TERMINAL_RUN_DIRECTORY --output NEW_ABSOLUTE_JSON
```

This is AI-authored software verification on reused geology. It supplies no independent
site inference, external human reproduction, student scientific ownership, physical
validation, novel theorem, acceleration evidence or complete competition design.
