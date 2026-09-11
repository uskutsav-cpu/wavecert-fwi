# Theory note: residual certification of a directional FWI derivative

## 1. Exact problem

For one source/frequency block, consider

\[
A(m)u=q,
\]

where \(m\in\mathbb R^n\) is squared slowness and, in the included 2-D frequency-domain backend,

\[
A(m) = K-\omega^2\operatorname{diag}(m)+i\omega D.
\]

Receiver restriction is \(P\), observed data is \(d\), and

\[
\Phi(m)=\tfrac12\|Pu-d\|_2^2.
\]

For a direction \(v\), differentiating the state equation gives

\[
A(m)\delta u = \omega^2\operatorname{diag}(v)u.
\]

The directional derivative is

\[
D\Phi(m)[v] = \Re\langle Pu-d,P\delta u\rangle.
\]

The adjoint-state gradient is equivalent, but the directional form is especially convenient for a posteriori certification.

## 2. Surrogate consistency residuals

Let a neural operator or other surrogate return \(\hat u\) and a JVP \(\delta\hat u=D\hat u(m)[v]\).

Define the **primal residual**

\[
r_p=A\hat u-q,
\]

and **tangent residual**

\[
r_t=A\delta\hat u-\omega^2\operatorname{diag}(v)\hat u.
\]

These are post-training quantities. They ask whether the surrogate state and its derivative satisfy the exact physical equations, rather than whether they merely match a finite training set.

## 3. State and tangent error bounds

Let

\[
\beta=\sigma_{\min}(A)>0.
\]

Then

\[
\|A^{-1}\|_2=1/\beta.
\]

Because

\[
A(u-\hat u)=-r_p,
\]

we have

\[
\eta_u:=\frac{\|r_p\|}{\beta}
\quad\Longrightarrow\quad
\|u-\hat u\|\le\eta_u.
\]

Similarly,

\[
A(\delta u-\delta\hat u)
=
\omega^2\operatorname{diag}(v)(u-\hat u)-r_t,
\]

so

\[
\eta_{\delta u}
:=
\frac{\omega^2\|v\|_\infty\eta_u+\|r_t\|}{\beta}
\]

satisfies

\[
\|\delta u-\delta\hat u\|\le\eta_{\delta u}.
\]

## 4. Directional objective certificate

Write

\[
r=Pu-d,
\qquad
\hat r=P\hat u-d.
\]

Then

\[
D\Phi-D\hat\Phi
=
\Re\bigl[
\langle r-\hat r,P\delta u\rangle
+
\langle \hat r,P(\delta u-\delta\hat u)\rangle
\bigr].
\]

For point receiver restriction, \(\|P\|_2\le1\). Therefore

\[
\|r-\hat r\|\le\eta_u,
\]

and

\[
\|P\delta u\|
\le
\|P\delta\hat u\|+\eta_{\delta u}.
\]

Hence

\[
\boxed{
|D\Phi-D\hat\Phi|
\le
\eta_u(\|P\delta\hat u\|+\eta_{\delta u})
+
\|\hat r\|\eta_{\delta u}
=:\eta_D.
}
\]

The exact directional derivative therefore lies in

\[
[D\hat\Phi-\eta_D,\; D\hat\Phi+\eta_D].
\]

A sufficient condition for the proposed direction \(v\) to be a true descent direction is

\[
\boxed{D\hat\Phi(m)[v]+\eta_D<0.}
\]

## 5. Block decomposition

For an objective summed over source/frequency blocks \(b\),

\[
\Phi=\sum_b \Phi_b,
\qquad
D\Phi[v]=\sum_b D\Phi_b[v].
\]

If each block has

\[
|D\Phi_b-D\hat\Phi_b|\le\eta_b,
\]

then

\[
D\Phi[v]
\le
\sum_b(D\hat\Phi_b[v]+\eta_b).
\]

This makes blockwise fallback natural: evaluate exact physics for the largest-uncertainty blocks, replace their upper bounds by the exact directional derivatives, and stop once the sum is negative.

## 6. What is rigorous here — and what is not yet

For the included small discrete Helmholtz system, the algebra above is finite-dimensional and the code computes \(\beta\) from a dense singular-value decomposition. The tests verify that the realized derivative error falls below the computed bound.

Production FWI introduces harder issues:

1. A dense SVD is impossible at field scale; a genuinely certified lower bound for \(\beta\) is needed.
2. Hyperbolic time-domain formulations require the appropriate stability/energy norm, not a naive copy of this Euclidean derivation.
3. A full wavefield may be required to evaluate residuals; receiver-only surrogates do not expose enough information.
4. Discretization error and the distinction between the continuous and discrete PDE must be accounted for if the claim is continuous-level certification.
5. An exact tangent JVP of the neural operator is assumed available; PyTorch autodiff provides this for differentiable surrogates.
6. Tightness matters. A correct but huge upper bound produces no computational savings.

## 7. Why start directional instead of full-gradient

A full bound

\[
\|\nabla\Phi-\nabla\hat\Phi\|
\]

is stronger but may be substantially more expensive. Optimization only needs enough information to justify an update. Certifying the scalar directional derivative along the proposed step is therefore a natural first target and aligns with goal-oriented error estimation.

---

## Phase 4 extension: receiver- and direction-aware residual bound

The original `1/beta` certificate is mathematically valid but can be highly conservative because it controls full-state errors in the global Euclidean norm. The Phase-4 implementation instead bounds only the quantities seen by the FWI receiver functional.

With primal residual `r_p = A u_hat - q` and tangent residual

`r_t = A du_hat - omega^2 diag(v) u_hat`,

the exact errors satisfy

`e_u = -A^{-1} r_p`

and

`e_t = -A^{-1} r_t - omega^2 A^{-1} diag(v) A^{-1} r_p`.

For receiver restriction `P`, define

`alpha = ||P A^{-1}||_2`

and

`kappa(v) = ||P A^{-1} diag(v) A^{-1}||_2`.

Then

`||P e_u|| <= alpha ||r_p||`

and

`||P e_t|| <= alpha ||r_t|| + omega^2 kappa(v) ||r_p||`.

Writing `r_hat = P u_hat - d`, the directional-derivative error obeys

`|D Phi - D Phi_hat| <= eta_Pu (||P du_hat|| + eta_Pt) + ||r_hat|| eta_Pt`.

The code forms `P A^{-1}` using sparse adjoint solves with one right-hand side per receiver, rather than constructing the complete dense inverse. This remains a reference-grid method because obtaining these exact stability maps is still too expensive for field-scale FWI.

## Phase 5 extension: split-conformal scale calibration

Because the deterministic bound is valid but loose for the Phase-2 FNO, WaveCert keeps it unchanged and adds a separate statistical deployment layer.

On a calibration set, define

`s_i = realized_error_i / deterministic_bound_i`.

The split-conformal scale is the finite-sample upper quantile `q_(1-alpha)` of `s_i`, and the practical bound on an exchangeable new case is

`eta_conf = q_(1-alpha) eta_det`.

This statement is distribution-free in the standard split-conformal sense under exchangeability. It is **not** a deterministic PDE theorem, and the repository deliberately uses different terminology for the two guarantees.

For blockwise repair, each source/frequency label is calibrated independently at a Bonferroni allocation `alpha / n_blocks`.

## Phase 6 extension: changing the hybrid direction requires recertification

Suppose selected source/frequency blocks are replaced by exact gradients. The total hybrid gradient changes, hence the descent direction changes. Any certificate computed for the old direction is therefore irrelevant to the new one.

The adaptive algorithm consequently follows this loop:

1. form the current exact/neural hybrid gradient;
2. normalize its negative to obtain `v`;
3. recompute all unrepaired block certificates along `v`;
4. accept if the summed upper directional derivative is negative;
5. otherwise replace the largest-uncertainty block by its exact gradient and repeat.

This is the central correctness invariant of the Phase-6 implementation.
