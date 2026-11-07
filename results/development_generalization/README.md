# Development generalization diagnostics

This is an engineering diagnostic of the two existing geological FNO
checkpoints. It uses only training and model-selection validation wavefield
arrays. It was designed after earlier development outcomes were inspected;
it is not a fresh confirmation experiment or a competition submission.

Reproduce from the repository root:

```bash
PYTHONPATH=src MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/diagnose_development_generalization.py
```

The default compares all raw forward examples, 48 freshly sampled unit-L2
random squared-slowness tangents per role, and 24 smoothed inversion states
per role. Both checkpoints use identical indices and tangent directions.
Seeds, indices, parent IDs, checkpoint hashes, dataset identity, executed
offline physics costs, and source/environment provenance are retained.

The smoothed-state comparison separates the exact objective gradient
`g = J* r` from the neural gradient `g_hat = J_hat* r_hat`. Here `r` is the
exact receiver residual, `r_hat` is the predicted receiver residual, and `*`
is the real adjoint for complex data and real model parameters. Two further
comparisons hold one source of error fixed:

- `J_hat* r` uses the exact residual with the neural Jacobian.
- `J* r_hat` uses the predicted residual with the exact Jacobian.

Every sampled neural gradient is also checked against `J_hat* r_hat` using
a separate fixed-residual vector-Jacobian product. This identity check tests
the decomposition calculation; it does not demonstrate physical accuracy.

`summary.json` contains block medians and separate parent medians.
`raw_forward.csv`, `fresh_random_tangents.csv`, and
`smoothed_gradient_decomposition.csv` preserve individual measurements.
`development_diagnostics.png` shows the principal negative findings.

Training outcomes are in-sample. Validation parents were already used for
checkpoint selection. Repeated blocks and slices do not increase the number
of independent parents, and no population confidence interval is asserted.
Fresh random tangents do not estimate the entire Jacobian operator norm.
These results cannot establish field-data fidelity, a certificate, or faster
inversion.
