# Experimental coarse-adjoint directional bounds

`PODCoarseAdjointCertifier` constructs a physical adjoint proposal from the
unchanged, fixed real-basis `PODGalerkinSurrogate`. It bounds the fine discrete
objective's derivative at that proposal's shared survey direction using
[`FastCoarseStability`](coarse_stability.md). It performs no fine factorization,
fine inverse-map solve, neural inference, or optimizer step. It is a research
API for the current real Dirichlet five-point stiffness plus complex diagonal
potential. It is not a generic backend certificate.

The inequalities below hold in exact arithmetic. The implementation evaluates
them with ordinary float64/complex128 and the foundation's unverified
cancellation allowance. There is no formally verified floating-point
enclosure. Empirical checks against exact discrete adjoints do not change
that limitation. A negative directional upper bound concerns the derivative
at the current point; finite-step decrease still requires a separate step
rule or check.

## Error identity

Write the linear state equation and receiver objective as

\[
 Au=q,\qquad \phi=\tfrac12\|Pu-d\|_2^2,\qquad
 A^H\lambda=P^H(Pu-d).
\]

For real squared slowness with mass derivative
\(A'[v]=-\omega^2\operatorname{diag}(v)\), the exact physical gradient is
\(g_j=\omega^2\Re(\overline{\lambda_j}u_j)\). Sources and receivers are fixed
and independent of the model. Point-receiver injection uses summation, so
repeated receiver entries retain their full objective weight.

For arbitrary approximate fields, use the actual full-grid residuals

\[
 r_p=A\widehat u-q,\qquad
 r_a=A^H\widehat\lambda-P^H(P\widehat u-d).
\]

Then \(e_u=u-\widehat u\) and \(e_\lambda=\lambda-\widehat\lambda\) obey
\(Ae_u=-r_p\) and
\(A^H e_\lambda=P^HP e_u-r_a\). The physical proposal gradient is
\(\widehat g=\omega^2\Re(\overline{\widehat\lambda}\widehat u)\).
For any real direction \(v\), expansion into the two linear error terms and
their product gives

\[
 |g^Tv-\widehat g^Tv|\le\omega^2\left[
 c(v\widehat u)^T\eta_\lambda+
 c(v\widehat\lambda)^T\eta_u+
 \|v\|_\infty\|\eta_u\|_2\|\eta_\lambda\|_2\right],
\]

where \(c(w)=(\|B_0^Tw\|_2,\|Qw\|_2)^T\),
\(Q=I-B_0B_0^T\), and \(\eta_u,\eta_\lambda\) bound the corresponding
two error-component norms. The first two terms follow from orthogonal
decomposition and Cauchy–Schwarz. The last uses
\(\|\operatorname{diag}(v)\|_2=\|v\|_\infty\).

## Three fixed proposals

The spectral space \(B_0\) used to bound errors is independent of the POD
space used to propose fields. Let \(A_r=B_0^TAB_0\),
\(L=QAB_0\), \(a=\|A_r^{-1}\|_2\),
\(\ell=\|L\|_2\), \(k=\|A_r^{-1}L^T\|_2\),
and \(\tau=k\ell/\gamma<1\), with the foundation's positive complement
coercivity margin \(\gamma\). Its component response matrix is \(M\).
Because the current operator is complex symmetric and \(B_0,Q\) are real,
the same coupling norms bound its Hermitian adjoint.

`uncorrected_generic_M` keeps raw POD state/adjoint fields and uses

\[
 \eta_u=M c(r_p),\quad
 \eta_{Pu}=(p_B,p_Q)\eta_u,\quad
 \eta_\lambda=M[c(r_a)+(p_B,p_Q)^T\eta_{Pu}],
\]

where \(p_B=\|PB_0\|_2\) and \(p_Q=\|PQ\|_2\).

`uncorrected_oriented` keeps those same fields but retains the orientation
of the residual's coarse component in its reduced solve:

\[
 \eta_{uB}=\frac{\|A_r^{-1}B_0^Tr_p\|_2+k\|Qr_p\|_2/\gamma}{1-\tau},
 \qquad \eta_{uQ}=\frac{\|Qr_p\|_2+\ell\eta_{uB}}{\gamma}.
\]

Set \(\rho=\|A_r^{-H}(PB_0)^T\|_2\). For the adjoint use

\[
 \eta_{\lambda B}=\frac{\|A_r^{-H}B_0^Tr_a\|_2+
 \rho\eta_{Pu}+(k/\gamma)(\|Qr_a\|_2+p_Q\eta_{Pu})}{1-\tau},
\]
\[
 \eta_{\lambda Q}=\frac{\|Qr_a\|_2+p_Q\eta_{Pu}+\ell\eta_{\lambda B}}{\gamma}.
\]

`coarse_corrected_oriented` first updates the raw POD fields sequentially:

\[
 \widetilde u=\widehat u-B_0A_r^{-1}B_0^Tr_p,
 \qquad
 \widetilde\lambda=\widehat\lambda-
 B_0A_r^{-H}B_0^T[A^H\widehat\lambda-P^H(P\widetilde u-d)].
\]

It recomputes both actual residuals after correction, then uses the oriented
bounds and physical gradient of the corrected fields. In exact arithmetic
the coarse residual components vanish, since multiplication by \(B_0^T A B_0\)
or its Hermitian adjoint cancels the solved coarse residual. The code measures
the remaining floating-point components instead of assigning zero.
The corrected physical gradient is a proposal; it is **not claimed to be the
derivative obtained by differentiating the composite corrected state map**.
Raw uncorrected POD gradients do agree with the unchanged reduced model's
objective derivative, which is tested separately.

## Survey and abstention rules

Every block uses one direction formed from the unit negative **sum** of all
physical proposal gradients. The survey derivative center and error bound
sum all block centers and bounds at that same direction. A strict negative
upper derivative certifies descent within the stated numerical scope.

If any frequency's stability bound is unavailable, that frequency retains
raw POD fields, requests no correction, and reports `correction_skipped`.
All blocks remain in the descriptive aggregate. The complete survey bound
and upper derivative become JSON `null`, `available=False`, and
`exact_fallback_required=True`. Available but nonnegative upper derivatives,
zero/small proposal gradients, and nonfinite derived quantities likewise
require fallback. No stationarity is inferred from a zero proposal. The API
does not execute exact recovery. Invalid finite-domain inputs raise explicit
errors. Numeric overflow cannot serialize an infinite certificate.

Exports bind the model's canonical squared-slowness SHA-256 and each actual
frequency context, including the operator signature, gamma/tau, component
matrix, and numerical diagnostics. Block field arrays remain accessible
separately for checking; compact JSON omits the full complex fields.

## Fixed development reproduction and costs

`scripts/run_coarse_adjoint_validation.py --check-inputs` verifies the compact
fixture, frozen rank-64 POD basis, original POD source hash, and shared
teacher/basis provenance without predictions or fine solves. Without that
flag the runner evaluates the three methods above at spectral rank 96 on
**all eight existing validation slices, all 24 blocks per slice**. Its
protocol explicitly records that earlier private exploratory outcomes were
already inspected. This is a public-source development reproduction, not
prospective confirmation or held-out evaluation. It reports four-parent
summaries; eight correlated slices are not eight independent geological
environments. There is no rank search, method selection, or change to any
optimizer/default/checkpoint.

Every case/method receives independent cold POD/coarse-engine instances.
Constructor POD operator projections and two damping DCTs are reported as
setup. Online receipts separately count model DCTs, small factorizations,
matrix-column inverse RHS, receiver/coarse solves, real-equivalent DST/IDST
transforms, full-grid stencil applications, POD mass projections/lifts, and
physical contractions. These are operation counts, not equivalent FLOPs or
runtime. Fine online zero-work assertions use actual counter deltas.
Historical teacher generation and basis fit remain visible in input
provenance. Independently recomputed exact validation fields/gradients use a
separate fine solver with separately reported costs. No runtime or
full-inversion acceleration claim follows from this pilot.

The underlying ingredients are standard Galerkin, adjoint residual and
block-coercivity arguments, assembled here for this discrete reference
operator. This implementation makes no novel-theorem or new-method claim.
