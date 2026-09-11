# Literature map

This is the reading map that motivated the repository architecture.

## A. Seismic forward modeling, adjoints, and FWI

### Devito seismic modeling and FWI tutorials
- Devito, *Introduction to seismic modelling*: https://www.devitoproject.org/examples/seismic/tutorials/01_modelling.html
- Devito, *Full-Waveform Inversion*: https://www.devitoproject.org/examples/seismic/tutorials/03_fwi.html

Why it matters: provides an open, modern reference implementation of the acoustic wave equation, acquisition geometry, adjoint-state gradient, and multi-shot FWI workflow. The Devito FWI tutorial writes the gradient as \(J^T\delta d\), exactly the object a neural surrogate can corrupt through an inaccurate Jacobian.

### Virieux & Operto (2009)
*An overview of full-waveform inversion in exploration geophysics.*

Why it matters: classic reference for FWI physics, optimization, frequency continuation, acquisition, conditioning, and failure modes.

## B. Neural operators and derivative accuracy

### Kovachki et al. (2023)
*Neural Operator: Learning Maps Between Function Spaces with Applications to PDEs.* JMLR.

Why it matters: broad mathematical/operator-learning framework.

### Official NeuralOperator library
https://github.com/neuraloperator/neuraloperator

Why it matters: maintained PyTorch implementation of FNO/TFNO and related operator-learning infrastructure. WaveCert intentionally wraps arbitrary differentiable PyTorch surrogates instead of reimplementing FNO.

### Park et al. (2026)
*Neural Operators with Accurate Jacobian for High-Fidelity Image-Domain Seismic Inversion.*
https://slim.gatech.edu/Publications/Public/Submitted/2026/park2026IMAGEnoa/abstract.html

Why it matters: closest seismic evidence for the core failure mode. Forward-only neural operators can have inaccurate inversion-relevant Jacobians; Jacobian-informed training improves gradient structure and reconstruction quality.

**Difference from WaveCert:** Park et al. improve derivatives during training. WaveCert asks whether an already-trained surrogate's derivative can be certified online and selectively rejected.

## C. Residual-based neural-operator reliability

### Cao et al. (2023)
*Residual-based error correction for neural operator accelerated infinite-dimensional Bayesian inverse problems.* Journal of Computational Physics 486, 112104.
https://doi.org/10.1016/j.jcp.2023.112104

Why it matters: uses PDE residual information to correct neural-operator predictions in inverse problems. It strongly motivates post-training physical-consistency checks.

### Jha (2024)
*Residual-based error corrector operator to enhance accuracy and reliability of neural operator surrogates of nonlinear variational boundary-value problems.* CMAME 419, 116595.
https://doi.org/10.1016/j.cma.2023.116595

Why it matters: develops the residual-corrector perspective further and demonstrates large downstream optimization errors when neural surrogates are used naively.

## D. Certified surrogate optimization

### Qian, Grepl, Veroy & Willcox (2017)
*A Certified Trust Region Reduced Basis Approach to PDE-Constrained Optimization.* SIAM Journal on Scientific Computing 39(5), S434–S460.
https://doi.org/10.1137/16M1081981

Why it matters: probably the most important abstract precedent for this project. It derives a posteriori bounds for surrogate cost **and cost gradient**, embeds them in a trust-region method, and calls the high-fidelity PDE only when the reduced model is not sufficiently accurate.

**Novelty implication:** WaveCert should not claim to invent “gradient certification plus fallback” in general. The research opportunity is the neural-wave-operator / seismic / residual-Jacobian setting and potentially blockwise shot-frequency certification.

## E. Realistic seismic ML stress tests

### Stitt et al. (2026)
*SubsurfaceGen: Procedural Generation of Field-Scale Earth Models and Seismic Data.*
https://arxiv.org/abs/2605.30541

Why it matters: provides geologically diverse field-scale models, wavefields, and shot gathers, including a held-out geological setting for OOD evaluation. It is a natural second-stage benchmark after Marmousi.

## F. The conceptual synthesis

The working hypothesis is:

```text
Park 2026:
forward accuracy != Jacobian accuracy
             +
Cao/Jha:
PDE residuals expose/correct neural-operator inconsistency
             +
Qian 2017:
a-posteriori gradient bounds can govern adaptive high-fidelity fallback
             ↓
WaveCert-FWI:
post-training residual certificate for neural seismic derivative reliability
+ shot/frequency selective exact-physics verification
```
