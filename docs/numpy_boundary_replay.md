# Separate NumPy replay of the fixed boundary cases

This AI-authored software verification uses a separate NumPy propagation loop,
without importing or calling Devito. It is reference-informed: the primary
Devito results were already known, and the equation, generated constants and
time bounds were inspected before the replay. It is not blind confirmation,
independent geological validation or a student's personal interpretation.

**In the local run, all seven faithful traces reproduce the implemented Devito backend within
3.42e−14 relative L2 error.** All fourteen planned replays completed, with
unchanged source and criteria. The independently generated source samples
match exactly in this local environment; sample times match exactly.

## Frozen comparison and actual results

Two variants were fixed before NumPy propagation: generated-C spatial literals,
and independently derived exact rational fourth-order spatial coefficients.
Both use the same actual time update, source and boundary convention. The
engineering agreement gates were fixed at receiver relative L2 error <= 1e−9,
source relative L2 error <= 1e−14, and exact sample times. Bitwise source equality
is retained as a diagnostic because platform implementations of `exp` can differ.
These are numerical-replay criteria, not physics-quality thresholds or verified
roundoff enclosures.

| Case | Generated-literal relative L2 error | Rational-coefficient relative L2 error |
|---|---:|---:|
| Small box, no damping | 5.449e−15 | 1.337e−7 |
| Small box, padding 20 | 3.411e−14 | 9.968e−8 |
| Small box, padding 40 | 3.004e−14 | 6.285e−8 |
| Matched exterior 20, undamped | 9.283e−15 | 1.358e−7 |
| Matched exterior 40, undamped | 6.813e−15 | 1.153e−7 |
| Enlarged reference 241 | 1.553e−14 | 1.095e−8 |
| Enlarged reference 321 | 1.547e−14 | 1.095e−8 |

The faithful variant meets the fixed agreement criterion in 7/7 cases. The
rational variant meets it in 0/7; every mismatch remains in the archive. This
criterion identifies agreement with the **implemented** generated operator.
It does not establish which variant is more accurate against continuum physics.
Only spatial coefficient values change between these two variants.

The faithful NumPy traces also reproduce the original boundary-truncation
measurements: full-record relative differences from the largest reference are
2.407073263, 0.569363532 and 0.249489294 for padding 0, 20 and 40. Their changes
from the original reported ratios are at most 3.56e−14. The substantial boundary
effects therefore persist in the separately coded implementation. These remain
receiver-trace diagnostics, not reflection coefficients.

## The discrete equation actually replayed

Inspection of the pinned generated C and symbolic/time-argument metadata showed
a centered second time derivative and a forward first difference for damping.
The update before source injection is

```text
(m/dt² + d/dt) u_next = (2m/dt² + d/dt) u_current
                      - (m/dt²) u_previous + Laplacian4(u_current).
```

At the aligned undamped source node, `u_next += dt² * source / m`. The receiver
then samples `u_current`. The implementation uses cyclic storage and zero exterior
ghost values; logical outer grid nodes are updated rather than forcibly zeroed.
The rational coefficients are derived with exact polynomial moments instead of
copying the generated floating-point constants.

For the 1,101 stored samples, actual operator arguments are `time_m=1` and
`time_M=1099`. Receiver samples 0 and 1100 remain initialized zero; `u[1100]`
is computed but not recorded. Source samples 0 and 1100 are unused. This was
established through an argument-only query with all operator execution forbidden;
the generated C matched the cached primary kernels. No alignment or endpoint
changes were made to improve agreement. The original trace metrics retain all
stored samples, including those zero endpoints.

## Evidence and reproducibility

- Runner SHA: `4832612882cb7b484777e309fb14ccc291685ad1afb5efd030332d84df31375c`.
- Protocol SHA: `53fbdd57a119ec68dd07e8c3ce833857e281acce13c0ff9001456ec114b59550`.
- Production SHA: `9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da`.

Twelve pure tests and separate pre-execution review passed. All fourteen own
solves finished before reference numeric arrays were opened for comparison.
A separate AI post-audit recomputed 98 reported numbers, checked 19 replay and
11 primary artifact hashes, all fourteen raw records and 21 source pins, and
reconstructed damping profiles with scalar arithmetic. The post-auditor's own
pre-numeric metadata-key correction is retained with its original source.

The portable audit CLI and every actual raw trace/profile, protocol, source
snapshot, review and execution log are in `results/numpy_boundary_replay`.
The immutable `fixture/` retains its pre-execution documentation and test receipt;
the parent README describes the executed run. Copied Devito sources are
attributed under the included original `DEVITO_LICENSE.md`.

The propagation runtime requires only Python and NumPy. From the repository root:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python results/numpy_boundary_replay/fixture/replay_numpy.py \
  --run --repo . \
  --reference-run results/devito_boundary_verification/run_01 \
  --output results/numpy_reproduction_new \
  --expected-protocol-sha256 53fbdd57a119ec68dd07e8c3ce833857e281acce13c0ff9001456ec114b59550 \
  --expected-source-sha256 4832612882cb7b484777e309fb14ccc291685ad1afb5efd030332d84df31375c
```

The new repository test reproduces all seven faithful traces before loading
targets, checks the frozen tolerances, and prints each result into CI logs.
It passed locally with warnings as errors. All four Linux CI jobs at `f8ecf74`
then passed, with the seven numerical comparisons printed into each job log:

| Python | NumPy | Maximum trace relative L2 | Maximum source relative L2 |
|---|---|---:|---:|
| 3.10.22 | 2.2.6 | 3.411e−14 | 1.015e−17 |
| 3.11.18 | 2.4.6 | 3.423e−14 | 1.470e−17 |
| 3.12.16 | 2.5.4 | 3.423e−14 | 1.470e−17 |

The learned/data job separately repeated Python 3.12 with the same reported
values. All exact time checks passed. Source samples were not bitwise identical
to the ARM reference in Linux, while remaining well inside the **predeclared**
1e−14 excitation tolerance. Full case reports and job identities are archived
in `ci_reports_f8ecf74.json` and [CI run 42445601820](https://github.com/uskutsav-cpu/wavecert-fwi/actions/runs/42445601820).
These are repeated known numerical fixtures, not additional geological samples.
The existing CI workflow and original 156 production files remain unchanged.

This verifies the fixed discrete implementation and its conditional trace
measurements. Matched-cost reconstruction benefit, broader physical fidelity,
independent geology, a specific novel contribution and the student's own
scientific interpretation/defense remain open.
