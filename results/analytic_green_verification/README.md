# Analytical 2D homogeneous wave comparison

The enlarged-reference trace differs from the declared infinite-domain Green
solution by **0.0305142% relative L2**. The small 40-padding-cell case still
differs by **24.9504%**. All seven cases, all three quadrature orders and the
original traces are retained. This supplies a homogeneous forward reference,
not completion of realistic physics across independent geology.

[Technical scope](../../docs/recovery_and_physics_diagnostics.md) and
[`publication/TECHNICAL_SCOPE.md`](publication/TECHNICAL_SCOPE.md) give the
normalization, causal source, startup discrepancy, recorded-sample mask and
limits. No amplitude, shift, window, quadrature order or width was fitted.
`physical_quality_pass` is null; no physical acceptance tolerance is implied.

The producer completed with exit code 0 after 14 pure controls, source/math
review and exact freeze. Its fixed 128/256/512-point calculations are in
`run_01/`. A different `z²` substitution with scalar source evaluation and
QUADPACK agrees with the 512-point result at 1.8515e-15 relative L2; 228
independent post-audit checks pass. This is ordinary FP agreement, not a formal
error enclosure. No PDE was rerun or original seven/fourteen-run artifact changed.

`fixture/` preserves exact code/protocol, primary metadata, pure tests and
histories including the pre-GO planned-prefix retention fix. `post_audit_source/`
and `post_audit_01/` retain alternate source/criteria/freeze, raw curve and actual
execution evidence. `publication/` contains source, all-seven PNG/SVG, caption,
recorded-metric CSV and visual inspection receipt. Original source notices are
retained; native fields are not calibrated pressure measurements.

Reproduce the comparison without a new wave solve:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python results/analytic_green_verification/fixture/compare_green.py \
  --execute-analytical-comparison \
  --original-run "$PWD/results/devito_boundary_verification/run_01" \
  --output /tmp/wavecert-analytic-green-new \
  --expected-source-sha256 ba7b03027d837f6e834e47ebf13c79768287d6d2963b1df3b898a6cbce822b69 \
  --expected-protocol-sha256 729e23e1a8a5373ba5492a47f2259d0bcb9a0618b25ddc87f3d73e0ea44337e5
```

The CI regression explicitly checks that a physical-quality pass remains unset.
Neither this analytical fixture nor its portability checks transfer the
certificate to Devito/PML, establish heterogeneous-geology recovery or student
ownership, or create a novel Green/quadrature theorem.
