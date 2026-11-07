# Exact full gradients from residual-coded receiver solves

This control strengthens the exact reference. It is a reassociation of the
ordinary least-squares adjoint gradient, with no neural model, random encoding,
source subsampling, or approximation. No novelty claim is made. Earlier
`cost_break_even` and `receiver_oracle_controls` artifacts are preserved with
their original implementations and scope; this additional comparison does not
quietly rewrite their historical numbers.

## Discrete identity and conjugation

At one frequency, let the fixed source columns form `Q`, let common fixed
linear receiver sampling be `P`, and let observed columns form `D`. For the
reference operator

```
A(m) = K - omega² diag(m) + i omega diag(damping),
U = A^-1 Q,     C = P A^-1,     R = C Q - D,
Phi = 1/2 ||R||_F².
```

The ordinary adjoint states are `Lambda = A^-H P^H R = C^H R`, and the real
squared-slowness gradient satisfies

```
g_j = omega² Re sum_s conj(Lambda[j,s]) U[j,s]
    = omega² Re sum_l C[l,j] sum_s U[j,s] conj(R[l,s]).
```

Because the state equation is linear, define

```
B = Q R^H,
V = A^-1 B = U R^H,
g_j = omega² Re sum_l C[l,j] V[j,l]
    = omega² Re diag(V C)_j.
```

The last expression uses `C.T * V`, not `C.conj().T * V`. The conjugation is
already in `R^H`. The sign is positive because the reference model derivative
of `A` is `-omega² diag(dm)`. Source columns may be complex or distributed.
Neither transpose symmetry nor Hermitian symmetry of `A` was used.

Construct `C` by solving `A^H X=P^H` and taking `C=X^H`: `r` adjoint RHS.
Then solve `A V=Q R^H`: `r` forward RHS. The complete objective and gradient
therefore require `2r` RHS and one reusable factorization per frequency.
The matrix products also require work: arbitrary dense `Q` can cost
`O(n_grid*n_sources*n_receivers)` arithmetic, even though the number of solves
is independent of source count. Dense working storage scales as
`O(n_grid*n_sources + n_grid*n_receivers + n_receivers*n_sources)`, excluding
the operator, factorization, and solver workspace. The code does not form the
grid-by-grid product `V C`. These operations must not be described as free or
treated as a wall-clock bound.

For the point-source `SurveyBlock` API, `Q` is never materialized in the coded
route. Each source contributes its conjugated receiver residuals, multiplied
by the existing `1/spacing²` source amplitude, to the corresponding rows of
the coded RHS. This uses receiver-by-source residuals plus dense
receiver-by-grid/coded-field storage, without storing every source wavefield.

Repeated receiver slots retain distinct observed values and all their misfit
terms. Before coding, their residuals are **summed**, not averaged, by unique
receiver position. This yields `2r_unique` RHS with unchanged objective and
gradient. The general matrix helper retains all supplied rows; it does not
silently approximate a low-rank or redundant sampling matrix.

## The implemented exact choices

For `s` distinct point-source blocks, `r` unique receiver locations, and `c`
sources covered by receiver locations, the new chooser evaluates these RHS
counts separately at each frequency:

| Exact full-gradient route | RHS | Additional condition |
|---|---:|---|
| Batched ordinary forward/adjoint | `2s` | Reference derivative and fixed sources/sampling |
| Receiver/source-state reuse | `r+s-c` | Explicitly checked `A.T == A` |
| Residual-coded receiver gradient | `2r` | No transpose-symmetry assumption |

`best_exact_objective_gradient` selects the least RHS among these options,
preferring ordinary then symmetry on ties. This minimizes the stated counts
among these three implementations; it is not a claim of global optimality or
the fastest runtime. The separate existing `reciprocity_objective_gradient`
API and all optimizer defaults remain unchanged.

## Consequence for the current map certificate

The current receiver-map directional certificate constructs both
`C=P A^-1` and `T=C diag(v) A^-1` before taking their spectral norms. That
initial complete map construction costs `2r` RHS per frequency. An exact
directional oracle based on those two maps also costs `2r` and supplies only
one derivative.

The new exact full-gradient chooser always includes a `2r` option. Hence,
for **any admissible source/receiver count in this implementation**, the
complete initial two-map construction cannot use fewer RHS than the selected
exact full gradient. Spending the second `r` RHS on `V` instead of `T` obtains
the whole gradient; this does not claim an already computed `T` can be turned
into a gradient without further work. The norm-only certificate additionally
computes SVDs and uses neural inference, residual applications, and potentially
more repair rounds, all of which require separate accounting.

This is an implementation-specific RHS comparison, not a universal
impossibility result for certification, a FLOP/runtime inequality, or a proof
that useful neural search directions cannot improve an optimization method.
The solve-free analytical stability mode is a different construction. A new
certificate, another backend, a different acquisition/parameterization, or a
complete time-to-reconstruction comparison needs its own evidence.

## Established context

[Tao and Sen (2013)](https://academic.oup.com/jge/article/10/6/065008/5111276)
derive frequency-domain source/receiver Green-function gradient expressions.
[Liu et al. (2015), section 6](https://academic.oup.com/gji/article/202/3/1827/612210)
explicitly account for common-receiver reciprocal adjoint FWI when sources
outnumber receivers, narrowing their proposed method's apparent advantage.
[Gholami (2024), equations 14–16](https://dspace.igf.edu.pl/xmlui/bitstream/handle/123456789/133/2024_EGN.pdf?isAllowed=y&sequence=1)
expresses the multisource gradient as a diagonal of matrix products. The
residual-coded reassociation above follows the ordinary adjoint identity;
these citations establish related formulations, not a claim that every paper
states this exact nonsymmetric `2r` implementation or cost count.

## Scope, verification, and reproduction

The identity requires an invertible linear state operator, fixed common
linear sampling, sources independent of `m`, and the stated diagonal mass
derivative for a real squared-slowness model. Model-dependent acquisition,
another parameterization, attenuation derivatives, elastic/time-domain
operators, and field data are not validated here. Sparse solves use ordinary
floating point, without a formal rounding-error enclosure. A negative gradient
direction still needs an appropriate finite-step acceptance rule.

Material tests use complex distributed source matrices, complex sampling,
a genuinely nonsymmetric/non-Hermitian operator, distinct repeated-receiver
observations, and independent objective finite differences. The complex case
also verifies that an incorrect extra conjugation of `C` would fail. Executed
RHS counters count matrix columns, and all routes are compared with the
existing per-block exact adjoint implementation.

Reproduce operation counts and numerical checks without reporting timings:

```
PYTHONPATH=src MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/benchmark_residual_coded.py
```

The nine-geometry 32×32 sweep saves source/environment hashes, raw counters,
objective/gradient/directional agreement, finite-difference checks, a CSV,
and a count plot in `results/residual_coded_controls/`. Timings are deliberately
off by default so a run concurrent with model training cannot be mistaken for
an idle-machine performance result. After other compute has finished, optional
`--timing-repeats 3 --output results/residual_coded_controls_timed` records cold-solver invocation timings, with separate
factor caches and rotated method order. Such timings still do not establish a
full inversion acceleration or field-scale result. The runner requires a new
or empty output directory so a later run cannot overwrite existing evidence.
