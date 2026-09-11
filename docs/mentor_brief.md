# Mentor-call technical brief

## One-sentence project

**WaveCert-FWI asks whether the gradient produced by a pretrained neural wave operator can be certified online using PDE-consistency residuals, so exact wave physics is invoked only for uncertain source/frequency blocks.**

## Core failure mode

A small forward error

\[
\|\hat F(m)-F(m)\|
\]

does not imply a small Jacobian error

\[
\|D\hat F(m)-DF(m)\|.
\]

FWI updates depend on the adjoint Jacobian, so this distinction matters directly for optimization.

## What the repository proves in the small discrete setting

For the included 2-D Helmholtz model, the code defines primal and tangent residuals

\[
r_p=A\hat u-q,
\qquad
r_t=A\delta\hat u-\omega^2\operatorname{diag}(v)\hat u
\]

and uses \(\beta=\sigma_{\min}(A)\) to upper-bound state, tangent, and finally directional-objective error.

The actual online decision is:

\[
D\hat\Phi[v]+\eta_D<0
\quad\Longrightarrow\quad
D\Phi[v]<0.
\]

The test suite independently computes the exact derivative and verifies coverage.

## What is deliberately *not* claimed yet

- no field-scale stability constant yet;
- no theorem yet for the Devito time-domain wave equation;
- no claim that the current bound will be tight on a real FNO;
- no claim of wall-clock speedup from the controlled low-fidelity surrogate;
- no continuous-PDE certification beyond the discrete system currently implemented.

## Immediate research questions for the mentor

1. Is directional certification enough for a strong first result, or is a full gradient-norm certificate necessary?
2. Which exact state variable should a neural operator predict so that primal/tangent residuals are practical—full space-time wavefield or another representation?
3. Should the first production backend be time-domain Devito acoustic FWI, or a frequency-domain formulation closer to the validation mathematics?
4. Which SubsurfaceGen shift would be most scientifically meaningful for the first OOD study?
5. Is shot-level fallback more operationally natural than frequency-level fallback in his workflow?
6. What exact-solve reduction would count as practically significant?

## Most relevant novelty comparison

- **Park et al. 2026:** improve neural Jacobians by supervising them during training.
- **Qian et al. 2017:** certify reduced-order optimization models and fall back to high fidelity.
- **Cao/Jha 2023–24:** use PDE residuals to improve neural-operator reliability.
- **WaveCert-FWI:** combine post-training derivative consistency, seismic inversion, and adaptive blockwise exact-physics verification.
