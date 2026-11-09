# Engineering analysis: the receiver-map RHS break-even condition

This note documents a limitation of the current discrete `directional`
certificate implementation. It is an engineering analysis and reproducibility
record, not a student-authored competition paper or evidence of acceleration.

The stronger [residual-coded exact reference](residual_coded_reference.md)
now includes a complete exact-gradient route costing `2rf` RHS. Consequently,
the initial two-map certificate cannot save RHS against the best implemented
exact reference for **any** source/receiver count. The ordinary `2sf`
comparison and its saved sweep below remain a historical, narrower audit;
they do not identify an advantageous acquisition regime by themselves.

## The implementation being counted

Let `s` be the number of source positions per frequency, `r` the number of
receiver samples, and `f` the number of frequencies. The current implementation
forms the receiver map `P A^-1` using one adjoint-system right-hand side (RHS)
per receiver. The first proposed direction also needs
`P A^-1 diag(v) A^-1`, requiring another RHS per receiver.

This counts the standard adaptive path constructing maps at the current model.
It does not treat a previously supplied inverse or receiver map as free; any
amortization of those constructions requires an explicitly accounted protocol.
Thus the initial map-based certificate pays:

```
receiver map:             r f RHS
first directional map: + r f RHS
                       ---------
                         2 r f RHS
```

A complete exact gradient evaluates one forward and one adjoint solve per
source/frequency block:

```
exact forward states:    s f RHS
exact adjoint states: +  s f RHS
                       ---------
                         2 s f RHS
```

Both paths can reuse one factorization per frequency at a fixed model. The
receiver-map path additionally evaluates spectral norms. Its initial RHS ratio
is `r/s`. If `r >= s`, these maps alone cannot reduce RHS count relative to the
complete exact gradient, even if the neural direction is accepted immediately
and no blocks are repaired. Neural inference, residual evaluation, further
direction-dependent maps, repairs, and finite-step checks add work beyond this
initial comparison.

If `r < s`, the maps use fewer RHS than this particular ordinary `2sf` route.
The residual-coded exact full gradient also uses `2rf`, so this does not
provide an RHS advantage over the stronger exact reference.

## What this does not establish

- An RHS count is not a runtime or FLOP count. Batched and single-RHS solves
  have different overhead and arithmetic intensity; hardware changes timings.
- This is not a universal lower bound on certification or an impossibility
  result for all certificates. The analytical stability mode uses another
  construction and does not form these receiver maps.
- The analysis concerns a fixed model. Reusing stability information after the
  model changes requires a separate mathematically justified argument.
- Exact FWI may also benefit from reusing accepted forward states or other
  optimizations. Fair comparisons must improve and account for both methods.

For the 16×16 development pilot, `s=2`, `r=12`, and `f=2`: the initial maps
require 48 RHS solves versus 8 for the entire exact gradient, a ratio of six.
Those counts explain why fewer repaired gradient blocks did not establish
lower total computation. Pilot runtimes are separate, version-specific
measurements and are not inferred from this formula.

## Counter verification and reproduction

Run from the repository root:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  python scripts/run_cost_break_even.py
```

The script sweeps source counts `(2, 6, 12, 24)` and receiver counts `(2, 6,
12)` on a 32×32 grid at frequencies `(3, 4)`. Every geometry uses the same
seeded heterogeneous model and normalized direction. Observations are exact
current receiver values with 5% normalized complex noise, ensuring nonzero
residuals for the full exact-gradient calculation. Data synthesis uses a third
solver; both measured branches start with independent empty factor caches.

The audit records executed stability, forward, and adjoint RHS counts,
factorizations, and SVDs and asserts agreement with the formulas. It does not
load a neural operator or report timings. Varying geometry on one model is not
independent geological replication.

Artifacts are saved in `results/cost_break_even/`:

- `summary.json`: assumptions, per-geometry counters, and source/environment provenance.
- `cases.csv`: flat machine-readable counts.
- `rhs_break_even.png`: initial RHS ratios; the region at or above one cannot
  save RHS work through initial map computation alone.

## Research implications

The [exact receiver controls](receiver_controls.md) show that the same two
maps contain the exact discrete directional derivative without neural
residual bounds or SVDs. The stronger residual-coded route obtains the whole
gradient with `2r` RHS, independent of transpose symmetry. The exact chooser
therefore uses `min(2s, 2r, r+s-c)` RHS per frequency, with the last option
available only for the explicitly checked transpose-symmetric operator; `r`
counts unique point receivers and `c` counts sources at those positions.
This minimizes counts among the implemented routes, without asserting a
universal runtime lower bound. Historical `2sf` artifacts retain their
original numbers and declared scope.

A practical next experiment should first establish an acquisition and
certificate construction where the full online budget can plausibly beat an
optimized exact baseline. A deterministic cascade can try an available cheap
analytical bound before expensive receiver maps; accepting either valid bound
preserves its mathematical direction guarantee. However, changing a
statistically calibrated repair policy requires a new matching calibration
protocol. No runtime gain from such a cascade is asserted here.
