# Optional IID tolerance calibration and the saved-score diagnostic

This is an established statistical comparison added after inspecting the
development study. It changes neither the original marginal conformal result
nor an inversion policy. It is not a new WaveCert theorem or a confirmatory
experiment.

## Two different probability statements

Ordinary split conformal calibration controls coverage averaged over calibration
samples and a new exchangeable score unit. It does not promise that every
realized calibration sample has population coverage at least the target.

For a scoring rule fixed independently of calibration, a one-sided IID tolerance
threshold instead gives this PAC statement:

```text
P_calibration{ P_new(S_new <= q | calibration, frozen scoring rule) >= 1-alpha }
    >= 1-delta.
```

Here `1-alpha` is the target population coverage, and `1-delta` is confidence
over calibration samples. This is a property of the procedure before calibration
is observed, not a way to identify with certainty whether a particular fitted
threshold met its target. It is not feature-conditional coverage.

The tolerance result requires IID calibration and future score units conditional
on frozen training, model selection, and score construction. Mere metadata
isolation or a claim of exchangeability does not establish these assumptions.
Correlated slices, blocks, repair rounds, or future inversion iterates must not
be counted as independent calibration units. A fixed-state repair-path maximum
remains one score per independently sampled model/state under its declared
protocol.

## Rank and edge cases

With `n` calibration scores sorted ascending, choose the smallest 1-based rank
`k` in `1,...,n` satisfying

```text
BinomCDF(k-1; n, 1-alpha) >= 1-delta,
q = S_(k).
```

For a continuous score law, `F(S_(k))` has the `Beta(k,n+1-k)` distribution.
The displayed binomial probability is exactly the probability that population
coverage is at least `1-alpha`. Ties make this bound conservative when coverage
uses `S_new <= q` and failures use `S_new > q`.

The implementation compares `BinomSF(k-1; n, 1-alpha) <= delta`, avoiding loss of
the confidence request when `1-delta` rounds to one. If no sample rank qualifies,
the mathematical threshold is positive infinity. It must not be replaced by the
largest finite calibration score.

Very small `delta` requests additionally use log-PMF tail summation to avoid
internal survival-function underflow. A positive tail is never reported as
zero risk because its exponential underflows; its log probability is retained.

A supported rank can also select an infinite score. The helper distinguishes
that case from an unsupported rank. Strict JSON uses `null` together with
`finite_rank_supported`, `finite_threshold`, and `threshold_kind`; `null` never
means a zero bound. Infinite deployment thresholds require abstention or exact
fallback. Zero remains the raw mathematical quantile here; a strictly positive
repair scale would require a separately disclosed conservative substitution.

At target coverage 90% and calibration confidence 95%:

| Calibration units | Smallest qualifying rank |
|---:|---:|
| 12 | None; positive infinity |
| 28 | None; positive infinity |
| 29 | 29 |
| 30 | 30 |
| 60 | 59 |
| 100 | 96 |

Thus the current 12-parent geology calibration cannot support a finite 90%/95%
IID tolerance threshold. This differs from the weaker marginal conformal
request that can have a finite rank at that sample size.

## Retained development results

The frozen 60-model study at source commit
`dad24d37b3fa51c16f46534c955161b744e5e009` has 30 calibration path scores. Its
ordinary marginal rank is 28, with scale `0.0007159541652124627` and observed
held-out initial-path coverage **45/60 = 75%**. Those artifacts remain unchanged.

The conventional, fixed comparison `alpha=0.1, delta=0.05` selects rank 30 and
threshold `0.0009185145222390499`, covering **52/60 = 86.67%** of the same saved
held-out scores. It does not achieve 90% observed coverage in this finite test
set. Neither probability statement guarantees a particular finite test fraction.
The parameters were chosen for this post-hoc comparison after the original
results were inspected; this is not independently validated improvement.

For context, under a continuous IID score law, ordinary rank 28 at `n=30` has
probability `0.5886487604` of population coverage at least 90%, despite marginal
coverage `28/31 = 90.32%`. Rank 30 raises the former probability to
`0.9576088417`. These are theoretical sampling-law calculations, not proof that
the current geological data or adaptive inversion states meet the assumptions.

Reproduce the separate diagnostic with:

```bash
PYTHONPATH=src python scripts/diagnose_tolerance_calibration.py \
  --study results/evidence_study \
  --output results/tolerance_calibration_diagnostic
```

The output must be new or empty and outside the original study directory. The
script checks original completion metadata, calibration/test identities, saved
marginal quantile and coverage flags. It records input hashes, source hashes,
Git revision, runtime versions, and completion/failure status. It performs no
new PDE solves, training, inversion, recalibration of the external stress study,
or test-based threshold optimization. No new default optimizer policy is added.

## References

The exact binomial calibration condition is established in
[Vovk (2012), Proposition 2b](https://proceedings.mlr.press/v25/vovk12/vovk12.pdf).
The tolerance-region interpretation and order-statistic construction are also
presented in [Hulsman (2022), Proposition 2 and Equations 3.6–3.7](https://arxiv.org/pdf/2210.14735).
These are existing methods used here as a comparator, not a novelty claim.
