# Cached-merit derivative proposals: developer scope

This is AI-authored provisional software documentation. It is not a student's
final research question, hypothesis, experimental design, submission, or proof
of completing a scientific milestone. Student choices and contributions need
their own accurate account.

This module implements classical adjoint and duality estimates for the current
finite discrete reference operator. It introduces no new general certification,
coarse-correction, or Krylov theorem. It has no measured acceleration result.

The supported state equation is

\[
A(m)u=q,\qquad
A=K-\omega^2\operatorname{diag}(m)+i\omega\operatorname{diag}(d).
\]

Here `K` is the constant-spacing, five-point, real Dirichlet stiffness; `m` is
real positive squared slowness; sources, receivers and damping are independent
of `m`. The native source amplitude is `a=1/h²`. A time-domain acoustic backend,
PML, variable grid, elastic model, or another parameterization needs a separate
operator/derivative/error analysis. The current low-mode component bounds depend
on complex transpose symmetry, not Hermitian symmetry.

## Receipt and data contract

`ExactMeritOracle.export_merit_fields(receipt)` exposes immutable paid fields,
with no factorization handle. Every proposal/bound call supplies its expected
owning oracle and calls `validate_merit_fields` before using the export. The
oracle checks issued identity, receipt lifetime, actual operator, model,
geometry, observed data and every exposed array's fingerprint. Independent
policies must own independent receipts and work. A saved field is not free work.

At one frequency, `Qsrc` contains all physical source columns. The objective is
`Phi = 1/2 ||P_slots U - D_slots||_F²`, without averaging. Duplicate receiver
slots keep their individual observations and objective terms. With expansion
matrix `J`, collapse residuals by **sum**, `R_unique=J.T R_slots`.
`W=J.T J=diag(w)` records their multiplicities. Replacing the full objective by
`1/2 ||R_unique||²` would generally change the problem.

## Two cached-field routes

When merit evaluation stores forward fields `Uhat`, the missing field is a
physical adjoint `Lambdahat`. Its proposal is
`ghat = omega² Re sum(conj(Lambdahat) * Uhat)`.
Measure both `rU=A Uhat-Qsrc` and
`rLambda=A.H Lambdahat-P_slots.H Rhat_slots`. Retain the coupled adjoint forcing
from state error. No state generation is repeated, but its numerical residual
is not declared zero.

When merit evaluation stores `C=P_unique A^-1`, define `Zhat=C.T` only after the
native transpose-symmetry check. The missing field is
`Vhat ≈ A^-1(Qsrc Rhat_unique.H)`, and the proposal is
`ghat = omega² Re sum(Zhat * Vhat)` with **no conjugation of Zhat**.
Conjugation is already in the coded residual RHS.

The paid map residual `rZ=A Zhat-P_unique.T` is measured. For native point
sources `Qsrc=a Esource`, its error drives the missing field through

\[
A eV_l=w_l a^2 E_{source}E_{source}^T\overline{eZ_l}-rV_l.
\]

The source-space forcing coefficient is bounded using the existing coarse
component estimates and the sampling constants for the **unscaled** point
source positions. General complex source columns instead produce
`Qsrc Qsrc.H conj(eZ) W`; the native `a²` specialization must not be extended
silently to arbitrary sources. A complex common amplitude would use `|a|²`.

The evaluator also includes affine residual defects. In the U route,
`F=Rhat_slots-(P_slots Uhat-D)` adds `P_slots.H F` to the effective adjoint
residual. In the C route,
`F=Rhat_unique-(W Zhat.T Qsrc-J.T D)` adds `Qsrc F.H` to the effective V
residual. These terms are included rather than accepted under a tolerance and
discarded. Export tampering still fails the owning oracle's validation.

For real shared direction `v`, let `c(w)` be its low/complement component
norms. Both routes retain the three directional-error terms

\[
\omega^2\sum_l\left[
c(v X_l)^T\eta_{Y_l}+c(v Y_l)^T\eta_{X_l}
+\|v\|_\infty\|\eta_{X_l}\|_2\|\eta_{Y_l}\|_2\right].
\]

Use `(X,Y)=(Uhat,Lambdahat)` or `(Zhat,Vhat)` with the corresponding center
above. Only in the mathematical zero-error cached-field limit do the other
terms disappear. Sparse LU and tolerance-converged GMRES are ordinary
floating-point approximations, not analytically exact fields.

## Direction, failure, and acceptance

All source/frequency contributions form one summed gradient proposal. The
default direction is its normalized negative, and every bound is evaluated at
that same direction. A zero proposal does not prove true stationarity.
Unavailable stability, invalid numeric arithmetic or incomplete processing
cannot yield a complete certified upper derivative.

All inequalities are exact-arithmetic derivations evaluated using the existing
ordinary-FP foundation and its unverified PSD allowance. The implementation is
not a verified roundoff enclosure. A strictly negative upper derivative
concerns the current unprojected direction. Box clipping changes that
direction; finite steps require the separate paid reference-merit acceptance
rule. It is not the derivative obtained by differentiating the complete
fixed-sweep approximate forward solver.

## Proposal and cost API

`fixed_sweep_missing_fields` constructs the missing adjoint or coded forward
field with a coarse Galerkin initialization followed by a fixed number of
norm-free coarse/DST inverse actions. The same fields can be used by an
uncertified control. This helper does not evaluate certificate norms/SVDs.

`evaluate_cached_merit` evaluates the coupled bound from the issued export,
provided missing fields and matching stability contexts. Its output binds the
safe export manifest and reports query work only. The caller must additionally
charge all merit, proposal, constructor, context-prepare, trial, rejected-step
and fallback work. Cumulative engine counters should be read once; do not add
their per-call deltas again. Scope flags inside result metadata are not integer
work counters and must not be passed directly to a counter-only trajectory
ledger.

Compare a cached U against the implemented exact incremental choice
`min(s,r-c)` RHS and a cached C against `min(r,s-c)`, including legal
collocation reuse. Those are counts among implemented routes, not global
runtime lower bounds. Include native L-BFGS-B with the same paid-gradient cache
and a tolerance-converged same-preconditioner right-GMRES control. Every arm
needs the same strict paid-reference objective criterion for credited endpoints.
Fixed-sweep and certificate-stopped comparisons must report failures and full
costs rather than exclude unavailable cases.

The integrated synthetic checks run from the repository root with

```sh
PYTHONPATH=src \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest tests/test_cached_merit_certificate.py -q -W error
python -m ruff check src tests scripts
```

They include independent LU/finite differences, substantial deliberately
perturbed paid fields, affine defects, duplicate receivers, both merit routes,
issued-owner admission and explicit numeric/availability failures. They do not
establish performance or geological generalization.

## Established context

[Qian et al. (2017), Theorem 5](https://kiwi.oden.utexas.edu/papers/multifidelity-optimization-adaptive-reduced-model-qian-grepl-veroy-willcox.pdf)
already gives coupled state/adjoint gradient-error bounds including their
product. [Keil et al. (2021), Proposition 2.7](https://arxiv.org/pdf/2006.09297)
distinguishes inexact physical gradients from reduced-objective derivatives.
[Giacomini, Pantz and Trabelsi (2017), section 2.4](https://www.numdam.org/item/10.1051/cocv/2016021.pdf)
already uses a negative directional upper bound to certify descent; its
shape-gradient linearization assumption should be retained when comparing it.

[Tao and Sen (2013)](https://academic.oup.com/jge/article/10/6/065008/5111276)
establishes source/receiver Green-function FWI gradient formulations.
[Ekre et al. (2022)](https://link.springer.com/article/10.1007/s00466-021-02113-2)
combines spectral and POD structure for improved computable error estimates in
a different application. [Southworth, Sivas and Rhebergen (2020)](https://arxiv.org/pdf/1911.02664)
analyzes the relation between block fixed-point/Krylov and Schur-complement
iterations. The specific engineering question here is whether reusing already
paid merit fields makes a useful bound economical under strong exact controls.
That remains an experimental question; these connections establish neither
priority nor acceleration.
