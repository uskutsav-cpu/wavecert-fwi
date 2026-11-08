# Analytical homogeneous forward-wave comparison

This AI-authored software verification record measures computational fidelity
against an infinite-domain analytical solution. It is not a student paper,
final scientific interpretation or completion of the realistic-physics goal.

The enlarged 321×321 reference differs from the analytical solution by
**0.0305142% relative L2** over the actually recorded samples. The 241×241
reference has the same displayed error. This supplies a physical-equation
reference beyond agreement between two implementations of the same stencil.
The 61×61 case with 40 damping cells still differs by **24.9504%**; the earlier
approximately 25% same-backend finite-domain finding remains intact.

The plotted field is absolute in the native solver/source convention, with no
amplitude or time fitting. Here `c = 2 m/ms`, source–receiver distance is 50 m,
grid spacing is 10 m, Ricker frequency is `0.005/ms` and delay is 200 ms. The
aligned source node adds `dt²*c²*q[t]` without division by cell area. Matching
integrated forcing therefore gives continuum point strength `A = h² = 100 m²`.
These native field units are not calibrated pressure measurements.

For `L = c⁻² ∂tt − Δ`, the causal two-dimensional Green function is
`H(s − r/c)/(2π sqrt(s² − (r/c)²))`. Convolution with the causal Ricker source
was evaluated at fixed Gauss–Legendre orders 128, 256 and 512 after substituting
`s = (r/c) cosh(v)`. The normalization follows the
[University of Toronto's two-dimensional wave formula](https://www.math.toronto.edu/courses/apm346h1/20181/PDE-textbook/Chapter9/S9.1.html),
with a corroborating [Brown University presentation](https://www.cfm.brown.edu/people/dobrush/am34/Mathematica/ch6/2wave.html).
The quadrature nodes and weights use the
[NIST Jacobi-matrix construction](https://dlmf.nist.gov/3.5.vi).

All three fixed-order calculations and step-forcing `acosh` controls were
retained. A separate AI post-audit used `s = r/c + z²`, a standard-library Ricker
evaluation and adaptive QUADPACK integration. Its valid-sample curve agreed
with GL512 at **1.8515×10⁻¹⁵ relative L2**, and all 228 post-audit checks passed.
That alternate computation occurred after the initial result. Convergence,
QUADPACK error estimates and cross-quadrature agreement are ordinary
floating-point diagnostics, not rigorous enclosures. `physical_quality_pass`
remains null; no student-selected physical acceptance tolerance exists.

The operator records receiver samples 1 through 1099. Slots 0 and 1100 remain
initialized zeros and were never computed; they are excluded by the frozen
comparison mask. All 1101 original and analytical samples remain retained.
The numerical stencil starts with `u[0] = u[1] = 0`, omits source sample 0 and
uses source samples 1 through 1099. The continuum convention instead has zero
initial state at time 0 with `Ricker(t) H(t)`; its nonzero value at time 0 creates
a known startup-discretization mismatch. No shift or source activation change
was used to hide it.

The receiver is only five grid spacings from the source. Matching point-source
strength does not make a finite-grid node an exact continuum delta. Startup,
spatial and temporal discretization, the finite-grid source, and finite-domain
effects can all contribute to the measured difference. This comparison does
not isolate their individual errors and does not define an intrinsic absorber
reflection coefficient.

All seven configurations, all three quadrature orders, the original seven
Devito records and fourteen NumPy replays remain unchanged. The figure keeps
both enlarged references and every undamped/damped control. All configurations
belong to one homogeneous scene. This advances a bounded forward-fidelity
check; it does not establish heterogeneous geology, independent environments,
PML/free-surface or adjoint correctness, certificate transfer, inverse recovery,
or student ownership. The original 161-file production digest remains
`9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da`.

The plotting source reads stored arrays and recorded metrics only. Its CSV,
caption, PNG/SVG, input/output hashes and render receipt accompany the figure.
