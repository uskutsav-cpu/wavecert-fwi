# Exact receiver-map and reciprocity controls

These controls strengthen the reference comparison. They are established
linear-algebra consequences, not a new neural method or evidence of WaveCert
acceleration. The implementation uses the small `Helmholtz2D` backend and
ordinary floating-point sparse solves.

## Stronger exact reference added after this control study

The saved `receiver_oracle_controls` study below compares its stated ordinary
and reciprocity implementations. The later
[residual-coded full-gradient control](residual_coded_reference.md) adds a
`2r` RHS route without transpose symmetry and preserves every historical
result. The stronger chooser uses `min(2s, 2r, r+s-c)` when the symmetry option
is available. The two-map directional oracle therefore has no initial RHS
advantage over that complete exact gradient for any tested geometry. This is
an implementation-specific operation-count result, not a runtime inequality.

## Exact directional information already present in the maps

For a fixed model, frequency, and proposed model direction `v`, define

```
C = P A^-1
T = C diag(v) A^-1
```

For any point-source vector `q`, the exact discrete receiver state is `C q`
and its tangent is `omega² T q`. Therefore the objective and directional
derivative are

```
Phi = 1/2 sum_sources ||C q - d||²
D Phi[v] = sum_sources Re <C q - d, omega² T q>.
```

The current `directional` certificate already constructs these two matrices
before taking their spectral norms. Exposing `T` lets an exact directional
control use that information directly. It requires the same initial `2r`
receiver-map RHS per frequency, without an SVD, neural state/JVP, or residual
bound. The implementation deduplicates receiver locations for solves while
preserving every repeated observation in the objective. The identities do
not require operator symmetry; their implementation here is scoped to the
reference backend. Numerical agreement is not a formally verified roundoff
enclosure. A negative derivative needs a separate finite-step check.

## A stronger exact full-gradient baseline

With `C`, every adjoint field is `lambda = C^H (P u-d)`. The reference
operator additionally satisfies `A.T == A`, so a source collocated with a
receiver obtains its entire forward state from the corresponding transposed
row of `C` with the source-amplitude scaling. This is **transpose symmetry**,
not Hermitian symmetry: the imaginary damping remains present.

For `s` distinct source locations, `r` unique receiver locations, and `c`
sources covered by receivers, the receiver route uses `r+s-c` RHS. The
implementation selects it only when cheaper in RHS than the ordinary `2s`
forward/adjoint route, independently at each frequency. It retains ordinary
adjoint-state evaluation on ties and explicitly checks transpose symmetry
before reciprocal source-state reuse. Counts are not runtime predictions.

Green-function reciprocity and its use in frequency-domain inversion are
established in the literature; see [Tao and Sen, 2013, equations 6–7](https://academic.oup.com/jge/article/10/6/065008/5111276).
This control should be considered before interpreting low-receiver,
many-source certificate savings against a `2s` baseline. The original
`cost_break_even` artifact measures the stated ordinary implementation; it
does not cover every possible exact-solver optimization.

## Reproduce and inspect

```
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/run_receiver_controls.py
```

`results/receiver_oracle_controls/summary.json` records source-file hashes,
environment, all timings and counters, numerical agreement, and centered
finite differences of the objective. `cases.csv` gives one row per geometry
and control. The script refuses to call results complete if source files
change during its run.

The recorded run uses a 32×32 heterogeneous model, frequencies 3 and 4 Hz,
nine source/receiver geometries, and three independent cold solver repetitions
per control in rotated order. The four controls are ordinary exact gradient,
the cheaper exact full-gradient route, exact receiver-map directional
evaluation, and the initial norm-only map computation. The last control
computes neither a derivative nor a complete gradient. Observation generation
and finite differences occur outside every timed branch.

All executed RHS/factorization/SVD counters match the stated formulas. Maximum
objective disagreement is `4.52e-17`, directional disagreement `1.69e-16`, and
gradient L2 disagreement `6.50e-15`; the worst case's best centered finite
difference relative error is `2.22e-7`. These are numerical checks on these
specific models, not universal error bounds.

For 12 sources and 2 receivers at two frequencies, the ordinary full gradient
uses 48 RHS, the reciprocity full gradient 24, and the directional maps 8.
For 2 sources and 12 receivers, full gradients use 8 RHS and directional maps
48. For 12 collocated sources/receivers, the reciprocal full gradient uses 24
RHS while directional maps use 48. A directional oracle supplies less
information than a complete gradient, and a lower RHS count need not produce
the lowest local elapsed time.

This benchmark does not generate useful neural directions, run an inversion,
measure time to a reconstruction target, include training/line-search costs,
or validate field data, Devito, or production-scale behavior. Dense maps and
their matrix multiplications also consume memory and work beyond RHS counts.
