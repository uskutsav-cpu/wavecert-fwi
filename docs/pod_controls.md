# Classical POD Galerkin development controls

This is a training/validation-only engineering comparison of an established
projection method. It is not a new certification theorem, a student-authored
submission, or an evaluation on held-out geology. Projection-based parametric
model reduction and POD are established methods; see
[Benner, Gugercin and Willcox (2015)](https://epubs.siam.org/doi/abs/10.1137/130932715).

The frozen protocol is `configs/pod_development_protocol.json`. It compares
state-only and state-plus-tangent real bases at ranks 8, 16, 32 and 64. Every
basis uses only the already generated raw and velocity-smoothed training fields.
Each complex snapshot receives a single normalization shared by its real and
imaginary components. The second family applies the same normalization to
training tangent snapshots. One real SVD per family supplies all rank prefixes.
The basis does not depend on the evaluation model or validation outcomes.

For a real orthonormal basis `B`, the model solves

```
K_r = B.T K B                      (offline)
D_r = B.T diag(damping) B          (offline)
M_r(m) = B.T diag(m) B             (online full-grid projection)
A_r(m, omega) = K_r - omega² M_r(m) + i omega D_r
u_hat = B solve(A_r, B.T q)
du_hat = B solve(A_r, omega² B.T diag(direction) u_hat)
```

Receiver least-squares gradients use `A_r.H`, the repeated-sampling injection
`P.T`, and the complex conjugate in
`omega² Re(conj(B lambda_r) * u_hat)`. These are derivatives of the reduced
model's objective, subject to ordinary numerical solve error. A consistent
reduced derivative can still differ substantially from the fine-PDE derivative.
No fine-PDE accuracy certificate follows from reduced differentiation alone.

The same frozen parent-weighted criterion as the neural training comparison
evaluates all eight controls: receiver relative squared L2, full-field JVP
relative squared L2, and one minus aggregate inverse-gradient cosine. Every
validation slice and all its source/frequency blocks are retained. The
generic evaluator is checked against the FNO-specific evaluator on an
independent tiny problem. The original four-FNO selection is unchanged.

Counters distinguish reduced dense factorizations/RHS solves, full-grid mass
assemblies, source/tangent/adjoint projections, and state lifts. A reduced RHS
is not a fine-grid RHS equivalent. The shared teacher generation costs are
preserved, including exact training tangents for the augmented basis. This
comparison makes no timing, equal-SGD-budget, field-scale, or acceleration claim.

Input verification only, from the isolated worktree:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/run_pod_development.py --check-inputs \
  --teacher-study ../wavecert-fwi/results/gradient_development
```

The same command without `--check-inputs` fits the frozen bases and evaluates
the controls. Outputs must stay inside the executing repository and cannot
overwrite the read-only teacher study. Each family saves only its rank-64
basis; candidate records identify and hash the lower-rank prefixes. A separate
prospective comparator freeze is required before any held-out evaluation.
