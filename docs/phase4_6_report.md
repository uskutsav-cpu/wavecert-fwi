# WaveCert-FWI Phases 4–6 Report

## Executive result

Phases 4–6 turn the Phase-3 derivative-failure observation into a deployment-time trust mechanism.

1. **Phase 4 — deterministic residual certificate.** A direction-aware a-posteriori upper bound is derived from the primal and tangent PDE residuals. On all **80** Phase-3 held-out states it covers the realized directional-derivative error (**100% empirical coverage; 0 violations; 0 false descent certifications**). It is, however, very conservative: median effectivity is **8292.19**, and it certifies **0/80** neural directions without fallback.
2. **Phase 5 — split-conformal calibration.** The rigorous bound is left unchanged and a separate statistical deployment gate is calibrated on the first **50** cases and evaluated on the disjoint final **30** cases. At the 90% operating point, empirical evaluation coverage is **90.0%**, median effectivity falls to **2.34**, **13.3%** of neural directions are accepted immediately, and there are **0 observed false descent certifications**. The 95% gate obtains **96.7%** empirical coverage and accepts **3.3%** immediately.
3. **Phase 6 — adaptive exact-physics fallback.** On **30 new held-out states** (new seed), neural-only directions are true descent directions **76.7%** of the time. The global conformal fallback reaches **96.7%** true-descent rate while using **3.07/4** exact blocks on average. The blockwise adaptive repair policy reaches **100% observed true-descent rate** while using **2.90/4** exact blocks on average, a **27.5% reduction in exact source-frequency gradient blocks versus exact direction construction**.

These results are a proof-of-concept on the 16×16 Helmholtz reference system. They are not yet a claim of field-scale certified FWI.

## Phase 4: deterministic direction-aware residual certificate

For one source/frequency block,

\[
A u=q,\qquad A\,\delta u=\omega^2\operatorname{diag}(v)u.
\]

For a neural state \(\hat u\) and neural tangent \(\delta\hat u\), define

\[
r_p=A\hat u-q,
\qquad
r_t=A\delta\hat u-\omega^2\operatorname{diag}(v)\hat u.
\]

The state and tangent errors satisfy the exact discrete identities

\[
e_u=u-\hat u=-A^{-1}r_p,
\]

and

\[
e_t=\delta u-\delta\hat u
=-A^{-1}r_t-\omega^2 A^{-1}\operatorname{diag}(v)A^{-1}r_p.
\]

Let \(P\) be the receiver restriction and define

\[
\alpha=\|PA^{-1}\|_2,
\qquad
\kappa(v)=\|PA^{-1}\operatorname{diag}(v)A^{-1}\|_2.
\]

Then

\[
\|Pe_u\|_2\le \alpha\|r_p\|_2
=:\eta_{Pu},
\]

and

\[
\|Pe_t\|_2
\le
\alpha\|r_t\|_2+\omega^2\kappa(v)\|r_p\|_2
=: \eta_{Pt}.
\]

For

\[
D\Phi(m)[v]=\Re\langle Pu-d,P\delta u\rangle,
\]

write \(\hat r=P\hat u-d\). Expanding around the neural state/tangent gives

\[
\boxed{
|D\Phi-D\hat\Phi|
\le
\eta_{Pu}(\|P\delta\hat u\|+\eta_{Pt})
+
\|\hat r\|\eta_{Pt}.
}
\]

This is the strongest deterministic reference-grid certificate currently implemented in the repository.

### Why the implementation does not form a dense inverse

Although the expression contains \(A^{-1}\), the code constructs only the receiver map \(PA^{-1}\) through sparse multi-right-hand-side solves of \(A^H\). The direction-specific map is built through another receiver-sized multi-RHS solve. This preserves the exact discrete reference constant without explicitly forming the full dense inverse.

### Phase-4 results

- cases: **80**;
- source/frequency blocks per case: **4**;
- Phase-3 failure cases reproduced: **31**;
- neural direction is actually true descent in **73.75%** of cases;
- deterministic certificate coverage: **100%**;
- violations: **0**;
- false descent certifications: **0**;
- median effectivity: **8292.19**;
- 90th-percentile effectivity: **17055.80**;
- neural directions certified without fallback: **0%**.

The scientific conclusion is two-sided: the residual derivation is numerically reliable, but the worst-case operator-norm bound is too conservative for deployment with the current forward-only FNO.

## Phase 5: statistical calibration without weakening the theorem

The deterministic certificate is retained exactly as derived. A second, explicitly statistical gate is constructed by split conformal calibration.

For each calibration case let

\[
s_i=\frac{|D\Phi_i-D\hat\Phi_i|}{\eta_i^{\rm det}},
\]

where \(\eta_i^{\rm det}\) is the deterministic Phase-4 bound. Let \(q_{1-\alpha}\) be the finite-sample split-conformal upper quantile of \(s_i\). On a new exchangeable case, the practical bound is

\[
\eta_i^{\rm conf}=q_{1-\alpha}\eta_i^{\rm det}.
\]

This is **not** a deterministic PDE certificate. Its interpretation is marginal finite-sample coverage under exchangeability.

### Data split

- deterministic Phase-4 study: 80 cases total;
- conformal calibration: cases **0–49**;
- untouched Phase-5 evaluation: cases **50–79**.

### 90% operating point

- scale: **2.601748×10⁻⁴**;
- target marginal coverage: **90%**;
- observed evaluation coverage: **90.0%**;
- immediate neural-direction acceptance: **13.3%**;
- observed false descent certifications: **0**;
- median effectivity: **2.34**.

### 95% operating point

- scale: **4.196615×10⁻⁴**;
- observed evaluation coverage: **96.7%**;
- immediate neural-direction acceptance: **3.3%**;
- observed false descent certifications: **0**.

### Blockwise calibration

For selective repair, each of the four source/frequency labels is calibrated separately at a Bonferroni allocation \(\alpha/4\). On the 30-case Phase-5 evaluation set, simultaneous coverage of all four block errors is **93.3%**. This is an empirical check, not a stronger deterministic theorem.

## Phase 6: adaptive fallback

Two deployment policies are implemented.

### Global fallback

1. build the neural gradient and neural descent direction;
2. evaluate the conformal case-level upper bound;
3. if the upper derivative is negative, accept the neural direction;
4. otherwise compute all exact block gradients and use the exact gradient direction.

### Selective repair

1. start from the summed neural gradient;
2. certify each unrepaired block along the **current** hybrid direction;
3. if the summed upper derivative is negative, accept;
4. otherwise repair the block with the largest uncertainty using its exact PDE gradient;
5. recompute the hybrid direction;
6. recompute all remaining certificates for that new direction;
7. repeat until accepted or all blocks are exact.

Recomputing the certificate after the direction changes is essential. Reusing a certificate for the old direction would not be valid.

### Phase-6 results on 30 new cases

| Policy | True descent rate | Mean exact blocks | Mean exact fraction | Mean cosine to exact direction |
|---|---:|---:|---:|---:|
| neural only | 76.7% | 0.00 | 0.0% | 0.309 |
| conformal global fallback | 96.7% | 3.07 | 76.7% | 0.867 |
| conformal selective repair | **100.0% observed** | **2.90** | **72.5%** | **0.881** |
| exact | 100.0% | 4.00 | 100% | 1.000 |

The selective policy reduces exact block-gradient evaluations by **27.5% relative to exact direction construction**, and by **5.43% relative to global conformal fallback**, while achieving 100% observed descent safety on this 30-case test.

The global gate accepted the neural direction with zero exact blocks in **7/30** cases and fell back to all four exact blocks in **23/30**; one of the seven statistically accepted neural directions was not a true descent direction, giving the 96.7% overall safety rate. The selective policy performed a partial repair in **27/30** cases and required all four exact blocks in only **3/30**; no unsafe selected direction was observed.

## Important limitations

1. **The deterministic bound is not yet practical at scale.** The exact receiver stability constants still require sparse solves with multiple right-hand sides. Field-scale work needs scalable certified or well-controlled estimators.
2. **The conformal gate is statistical.** Its guarantee depends on exchangeability between calibration and deployment distributions and is not valid under arbitrary OOD geology/acquisition shift.
3. **Thirty Phase-6 cases are a proof-of-concept, not a final uncertainty statement.** Larger repeated studies are needed.
4. **Phase 6 certifies directions, not finite step lengths.** End-to-end inversion and step-size control are Phase 7.
5. **The current FNO is deliberately forward-only.** Derivative-informed training remains an important comparator for later manuscript experiments.

## Reproduction

```bash
pip install -e '.[dev,torch]'

wavecert certificate-study --cases 80
wavecert calibrate-certificate --alpha 0.10
wavecert adaptive-study --cases 30
```

Outputs are written to `results/phase4`, `results/phase5`, and `results/phase6`.
