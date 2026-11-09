# Initial linearization at fixed frequency bands

AI technical diagnostic draft, 8 November 2026. This records post-outcome software development evidence; it is not a student conclusion or final research design.

All eight retained initial/truth witnesses completed the fixed eight-frequency probe: **64 case/frequency rows,128 LU factorizations and1152 RHS solves**, with zero optimizer calls. Original-band tangent contractions agree with the saved independent gradient under the fixed numerical rule. Lower-band remainder ratios are smaller in seven cases; **case2 is the retained exception**. The pooled lower-band ratio remains above1, so its more favorable aggregate alignment does not establish a reliable full truth-correction linearization across all eight cases.

For each frequency, the independently assembled old49×49 equation is `A(m,f)=K−ω²diag(m)+iωdiag(d)`, with unchanged damping, six sources,20 receivers and source normalization from the807 witness. The direction is fixed as `δm=m_truth−m_initial`, and the tangent satisfies `A_initial δu=ω²diag(δm)u_initial`. With receiver data change `b=d_truth−d_initial` and tangent `t=J_initial δm`, the relative remainder is `ρ=||b−t||₂/||b||₂`, alignment is `Re⟨b,t⟩/(||b||₂||t||₂)`, and `g_data·δm=Re⟨−b,t⟩`. There are no added quadrature weights. These evaluate one initial linearization along the full known truth correction; no parameter update was performed.

The fixed lower band is `[0.25,0.5,1,2]Hz`; the original band is `[2.5,3,3.5,4]Hz`. Each band concatenates all four frequencies and all source/receiver samples. The table includes every case; displayed values are rounded, and the receipt retains full precision.

| Case / retained slice | Lower ρ | Original ρ | Lower alignment | Original alignment | Lower g·δm | Original g·δm |
|---|---:|---:|---:|---:|---:|---:|
| 0 / gom_148_il_0236 | 0.322249 | 1.811285 | 0.992377 | −0.030084 | −0.849671 | +8.869163 |
| 1 / gom_148_xl_0768 | 0.260111 | 2.556165 | 0.965638 | −0.125107 | −0.485308 | +74.399075 |
| **2 / seam_011_il_0522** | **1.173215** | **1.133888** | **0.212818** | **−0.160268** | **−48.309719** | **+13.825311** |
| 3 / seam_011_xl_0866 | 0.099323 | 0.354069 | 0.995463 | 0.950887 | −0.116566 | −6.280822 |
| 4 / gom_151_il_0247 | 0.477861 | 2.631139 | 0.990904 | 0.098816 | −2.596345 | −40.928532 |
| 5 / gom_151_xl_0857 | 0.523133 | 1.859052 | 0.977593 | 0.124377 | −4.747441 | −35.615083 |
| 6 / seam_005_il_0596 | 0.108675 | 0.467507 | 0.995336 | 0.904889 | −0.235571 | −8.556632 |
| 7 / seam_005_xl_0795 | 0.110901 | 0.474032 | 0.995596 | 0.903082 | −0.240828 | −8.679106 |

All-eight pooling concatenates every case and frequency in its band; it is a descriptive norm calculation, not an average over independent samples.

| Fixed band, all8/8 cases | ‖b‖₂ | ‖t‖₂ | ‖b−t‖₂ | ρ | Alignment | g·δm |
|---|---:|---:|---:|---:|---:|---:|
| Lower | 16.740919 | 14.185996 | 19.139957 | 1.143304 | 0.242462 | −57.581449 |
| Original | 32.218362 | 57.976521 | 66.282476 | 2.057289 | 0.001588 | −2.966626 |

Case2 has lower-band remainder19.036270 against data change16.225733, givingρ1.173215, compared with original-bandρ1.133888. Its lower-band alignment0.212818 differs markedly from the other seven lower-band alignments0.965638–0.995596. Its large data-change norm remains in the pooled result. There was no exclusion, frequency selection or case-specific adjustment.

The original-band contraction check uses `|DD_tangent−g_saved·δm| ≤ 1e−9 max(|DD_tangent|,|g_saved·δm|,||g_saved||₂||δm||₂,1e−12)`. All8/8 checks passed; the largest absolute discrepancy is4.263256414560601e−14. Every truth/initial/tangent solve passed the inherited1e−9 true-relative-residual guard. The run marks numerical integrity true and physical acceptance null. These are ordinary floating-point consistency checks, not formal enclosures.

This is reused noiseless, self-consistent synthetic data under the same discrete operator. The eight slices are correlated development inputs, and nominal SEAM identities share an ancestor. Frequency continuation is established prior FWI work. This diagnostic supplies no tested multiscale control, optimization result, cycle-skipping cause, remedy, physical validation, independent-cohort evidence, certificate transfer or student interpretation. The BP/Devito verification has a separate physical scope; see [the BP record](bp_heterogeneous_verification.md).

The [complete receipt](/Users/swethasunilkumar/Documents/Codex/2026-11-07/wavecert-fwi/work/wavecert-fwi/results/initial_linearization_diagnostic/run_01/complete_receipt.json) has SHA256 `6d7189953db68a59e3be220c81d86115f100d8f57442ef0ee8357c9b3d112563`. The64 receiver-only NPZ records and two JSON records total814928 bytes; no full fields were saved. Root confirmed actual process terminal exit0 before this draft was written. A separate receiver-only scalar audit passed **1,600 checks with zero issues**, recomputing all 64 raw receiver triplets, 16 case bands, two pooled bands and eight saved-gradient contractions. Its work counts and linear-solve residual checks read metadata; it did not independently rerun the PDE solves.

The [artifact archive](/Users/swethasunilkumar/Documents/Codex/2026-11-07/wavecert-fwi/work/wavecert-fwi/results/initial_linearization_diagnostic/README.md) contains exact sources, protocols, six manufactured controls, raw receiver records, source review, actual terminal evidence and the independent scalar auditor. Staging receipts retain their original pre-execution labels; the completed run and post-audit are separate.

![All eight fixed-band remainder comparisons](/Users/swethasunilkumar/Documents/Codex/2026-11-07/wavecert-fwi/work/wavecert-fwi/results/initial_linearization_diagnostic/figures/initial_linearization_all8.png)

The figure retains the case 2 exception and uses a logarithmic axis. It has no physical pass threshold. Full caption, all scalar CSV values, rendering source and the first clipped-annotation layout are retained.

From the repository root, a new numerical reproduction can use:

```sh
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -I results/initial_linearization_diagnostic/fixture/run_probe.py \
  --execute-probe --repo . \
  --witness results/paid_merit_initial_direction_verification/run_01 \
  --output ../initial_linearization_reproduction_02 \
  --expected-source-sha256 44e07c0a846847e396a8db9d213c3346cfa654724ae0f55862fb668d6dde62d5 \
  --expected-protocol-sha256 736d6136c97c84de9dccb240d492c788684e59887675e303fe69a1b2e0a92cf3
```

This requires NumPy/SciPy and the unchanged 161-file production-source contract `9f30afa905cac4e2b3605e98ea85274968ce9797652ede55eab9b0b79811b9da`. It executes 1,152 new small RHS solves; it is separate from replaying saved receiver arithmetic. No BP, Devito, optimizer or full-wavefield files are involved. The student's own interpretation and final design remain pending.
