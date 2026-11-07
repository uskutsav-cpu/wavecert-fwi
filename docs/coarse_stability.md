# Low-Laplacian coarse stability foundation

`wavecert.physics.coarse_stability.FastCoarseStability` supports exactly the
current `Helmholtz2D` constant-spacing, five-point, real Dirichlet stiffness:

\[
A=K-\omega^2\operatorname{diag}(m)+i\omega\operatorname{diag}(d),\qquad m>0.
\]

It snapshots the grid and the reference damping configuration. It does not
support PML, variable spacing, Devito, a modified stiffness, or a different
boundary condition. It does not change optimization defaults or itself
provide an inversion certificate. This is an established spectral coarse-space
and block small-gain construction, with an implementation for this operator;
no novelty claim is made.

All inequalities below are exact-arithmetic statements. Their implementation
uses ordinary floating point, numerical inverses/norms and a heuristic PSD
allowance. It is **not a formally verified roundoff enclosure**. Independent
small-grid comparisons test the implementation; they do not prove numerical
coverage for every input.

## Basis and fast assembly

Order the real tensor sine eigenvectors of K by `(eigenvalue, z-mode, x-mode)`;
take the first `1 <= r < N` vectors as B, and set Q=I−BBᵀ. A cut through an
eigenvalue multiplicity remains valid: the first omitted eigenvalue is still
the minimum eigenvalue on Q. Projection/lifting use orthonormal DST-I without
storing an N×r basis. A deterministic signature hashes the configuration,
selected integer modes, ordered eigenvalues and damping vector.

For any real grid field f, set M_f=Bᵀdiag(f)B. Products of two sines give four
cosine terms. On a zero-padded `(nz+2)×(nx+2)` grid, unnormalized DCT-I T gives

\[
(M_f)_{ab}=
\frac{T_{|k_a-k_b|,|p_a-p_b|}-T_{k_a+k_b,|p_a-p_b|}
-T_{|k_a-k_b|,p_a+p_b}+T_{k_a+k_b,p_a+p_b}}
{4(n_z+1)(n_x+1)}.
\]

Cosine indices are folded modulo twice the corresponding interval count;
indices beyond its Nyquist value reflect back. This folding matters for high
selected modes. Two real DCT-I transforms for d,d² are paid at construction;
two for m,m² are paid per uncached model. Model cache keys contain actual
float64 model bytes, are bounded by `model_cache_size`, and do not depend on
array identity. No online N×r matrix product or fine-grid dense matrix occurs.

Write V_r=−ω²M_m+iωM_d and A_r=diag(λ₁,…,λᵣ)+V_r. With L=QAB=Qdiag(V)B,

\[
G=L^HL=\omega^4M_{m^2}+\omega^2M_{d^2}-V_r^HV_r.
\]

The implementation Hermitianizes computed G. With
`scale=||moment||F+||Vr||F²`, its engineering allowance is
`128*eps*r*max(scale,1)`. A minimum eigenvalue below `−10*allowance` produces
an unavailable result. Otherwise the diagonal shift is
`max(0,−minimum_eigenvalue)+allowance`. This repairs computed PSD cancellation,
including the constant-potential case, but is **not proof that shifted G
dominates the exact Gram**. No separate verified error bounds for DCT, inverse,
norm or eigenvalue evaluation are claimed.

## Component inequalities

Define

\[
\gamma=\lambda_{r+1}-\omega^2\max(m),\quad
a=\|A_r^{-1}\|_2,\quad \ell=\|L\|_2,\quad
k=\|A_r^{-1}L^T\|_2,\quad \tau=k\ell/\gamma.
\]

On Q, the real part of A is coercive with lower bound γ. For γ>0 and τ<1,
the two error components are bounded by M times the two residual components:

\[
M=\frac1{1-\tau}
\begin{pmatrix}a&k/\gamma\\ \ell a/\gamma&1/\gamma\end{pmatrix}.
\]

Thus `1/||M||2` is a discrete stability lower bound in exact arithmetic. The
implementation evaluates k² from the largest eigenvalue of
`Ar_inverse @ conjugate(G) @ Ar_inverse.H`. The conjugation is essential:
A is complex symmetric, so its upper off-diagonal block is Lᵀ, not Lᴴ.

For a particular residual r, the oriented estimate replaces a||r_B|| by the
actual small-solve response s=||A_r⁻¹r_B||:

\[
\eta_B=(s+k\|r_Q\|/\gamma)/(1-\tau),\qquad
\eta_Q=(\|r_Q\|+\ell\eta_B)/\gamma.
\]

The adjoint variant must use A_r⁻ᴴr_B. The same k is valid because A_r is
complex symmetric and spectral norms are invariant under transpose. Residual
complements are formed as actual vectors r−B(Bᵀr), avoiding subtraction of
squared norms.

For a physical adjoint, Aᴴe_λ=PᵀPe_u−r_a. If η_Pu bounds ||Pe_u||, let
`pB=||PB||`, `pQ=||PQ||`, and `rho_adjoint=||Ar^-H(PB).T||`. Then

\[
\eta_{\lambda B}=
\frac{\|A_r^{-H}r_{aB}\|+\rho\eta_{Pu}
 +(k/\gamma)(\|r_{aQ}\|+p_Q\eta_{Pu})}{1-\tau},
\quad
\eta_{\lambda Q}=
\frac{\|r_{aQ}\|+p_Q\eta_{Pu}+\ell\eta_{\lambda B}}\gamma.
\]

Receiver repeats count with their multiplicities. For receiver count R<=r,
PB is evaluated analytically and R×R Grams give pB,pQ; rho costs R reduced
adjoint RHS columns. A disclosed small ordinary-FP allowance is applied to
receiver Gram eigenvalues. For R>r the implementation avoids a potentially
full N×r sample matrix and returns the conservative sampling norm
`sqrt(maximum_receiver_multiplicity)` for pB,pQ and a times that norm for rho.
Receiver caches are bounded, content keyed and return copies.

## Corrections, availability and accounting

`correct(field,rhs,adjoint=False)` subtracts B A_r⁻¹ Bᵀ(A field−rhs), or its
adjoint counterpart. It recomputes the actual stencil residual after the
correction; it never sets coarse residual components to zero by assignment.
A physical adjoint caller must recompute its RHS from the corrected state
before correcting the adjoint. Such fields may be used to bound a physical
adjoint gradient proposal; they are not automatically the autograd derivative
of a corrected reduced model.

Nonpositive γ, unresolved/singular A_r, strongly indefinite computed G,
τ>=1 or nonfinite arithmetic return explicit unavailable reasons. The
singularity threshold is `eps*r*max(sigma_max(Ar),1)`. Unavailable results have
beta=0; inverse/bound/correction methods raise. Projection, lifting and actual
operator application remain available for reporting.

`cost_snapshot()` contains cumulative integer operation counts, not timings,
FLOPs or fine-grid RHS equivalents. It includes all real DCT/DST transforms,
small SVD/eigenproblems/LU factors, **r reduced RHS columns to form Ar⁻¹**,
orientation/correction/receiver solve columns, and full stencil applications.
Complex DST calls count as two real transforms. Fine-grid LU, RHS and dense
SVD counters stay zero. Resetting counts excludes already paid constructor
work; each result still discloses the two offline damping DCT transforms.
Results share engine counters, so use before/after snapshots for per-call
costs. Model moments may be cached; reduced LU factors are assembled and
charged afresh for each `evaluate` call. No wall-time acceleration claim is
implied by these counts.
