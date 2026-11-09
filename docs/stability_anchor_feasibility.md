# Initial stability-bound reuse: endpoint feasibility

The simplest stability-anchor reuse rule gives a zero sufficient lower bound
at **all 192 frequency/endpoint instances** in the existing 48-run coordinate
comparison. It is unusable unchanged at those final states. This does not show
that the actual operators are singular or that refreshed anchors cannot work.

For fixed stiffness and model-independent damping,
`A(m)-A(m0)=-omega^2 diag(m-m0)`. The triangle inequality gives the conditional
exact-arithmetic bound

```text
beta_candidate = max(0, beta_anchor - omega^2 ||m-m_anchor||_infinity).
```

This requires a valid anchor lower bound. The stored anchors use ordinary
floating-point arithmetic with an unverified cancellation allowance, so this
calculation does not turn them into formal enclosures. The formula is a standard
perturbation argument, not a new theorem.

All 32 initial case/frequency anchors were available. All 48 returned native-m
models and all four original frequencies were included, with no endpoint
selection. Every unclamped margin was negative; the closest was approximately
`-4.63250722613706`. A separately coded 95-digit Decimal calculation using
exactly decoded binary64 inputs reproduced all 192 negative margins. This is a
high-precision consistency check, not directed interval arithmetic. Its review
also checks source, fixed operator, data, geometry and damping bindings.

The anchor is the stored requested initial model. The coordinate comparison's
actual common start differs slightly through the velocity roundtrip; distance
is explicitly measured from the stored anchor. Only final endpoints were
examined. No intermediate reuse lifetime, refresh policy, timing, reconstruction
benefit or SCM comparison was measured. No PDE, optimizer or BP calculation ran.

Rapid parameter-dependent stability estimation and offline/online local-bound
construction already appear in [natural-norm SCM work](https://doi.org/10.1016/j.cma.2010.02.011).
Controlling lower-bound quality is also addressed by
[certified natural-norm SCM](https://arxiv.org/abs/1503.04760).
A difference between two reduced approximations is not automatically a
guaranteed estimator: the [hierarchical reduced-basis estimator](https://arxiv.org/abs/1802.03298)
requires a saturation condition. The retained primary-source scope note states
which sections were actually checked; this was a targeted review, not an
exhaustive novelty search.

Before developing an anchor controller, sharper local information or paid
refresh would need to preserve enough coverage to beat the cached exact
baseline in total cost. These endpoint results do not establish that. This is
AI-authored exploratory analysis of already inspected development data, not
new confirmation or the student's final contribution.

Exact source, registration, root freeze, raw report, independent review and its
two failed metadata-adaptation attempts are in the
[artifact archive](../results/stability_anchor_feasibility/README.md).
