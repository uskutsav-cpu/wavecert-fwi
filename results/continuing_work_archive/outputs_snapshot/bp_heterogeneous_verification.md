# BP heterogeneous numerical verification

The fixed native run completed all 54 forwards and3 adjoints across three retained BP2004 crops. The independent retained-record scalar audit passed2,618 checks with 0 integrity issues. All six prescribed middle-step gradient checks were informative and met the fixed numerical error criterion. Physical acceptance remains null. These are separate findings: record integrity and gradient consistency do not make the waveforms physically adequate.

The aggregate trace differences below include every crop and all ten receivers. Each entry is100×||candidate−reference||₂/||reference||₂ over exact common computed times, with the second named configuration as reference. No selected window, lag or amplitude fitting is used. Full-precision ratios, norms and sample counts are retained in aggregate_sensitivities.csv.

| Crop | coarse40→coarse80 | coarse80→coarse160 | coarse80→fine160 | coarse80→coarse80_halfdt |
|---|---:|---:|---:|---:|
| upper_left | 13.945763% | 5.825061% | 2.766556% | 0.017813% |
| middle_center | 37.854666% | 18.520556% | 0.975378% | 0.043298% |
| lower_right | 49.825439% | 22.846015% | 1.240897% | 0.056562% |

Coarse spacing is12.5m, so padding40/80/160 is500/1,000/2,000m per side. The fine grid has6.25m spacing and padding160, matching the 1km coarse80 absorber width. It samples the same piecewise-bilinear velocity field defined by the retained192×96 coarse nodes, with the exact 2,387.5×1,187.5m physical extent. It adds no native BP samples. Native damping samples differ with the grid. The half-step control changes0.5ms to0.25ms. Padding changes also alter exterior paths and spreading. The table therefore measures conditional sensitivity; its entries are neither reflection coefficients nor an additive error budget, and they establish no convergence rate or physical tolerance.

Padding differences remain substantial in this fixture. The larger domain is not an analytical truth reference, and the smaller temporal differences do not isolate all remaining error to a boundary mechanism. No width, mesh or time step is selected as physically acceptable from these results.

The following are all prescribed middle-step checks at relative step3×10⁻⁴. The fixed criterion is relative directional discrepancy≤10⁻³ for an informative direction. All three steps per direction remain in the native and audit records; this table does not select the best step. Full values are in middle_gradient_checks.csv.

| Crop | Direction | adjoint g·d | central difference | relative discrepancy | informative / criterion |
|---|---|---:|---:|---:|---|
| upper_left | interior | 8239.057329 | 8239.03721 | 2.44186846e-06 | true / met |
| upper_left | edge | -1800.104684 | -1800.106802 | 1.17632042e-06 | true / met |
| middle_center | interior | 10834.59027 | 10834.56881 | 1.98051141e-06 | true / met |
| middle_center | edge | 4971.683641 | 4971.681368 | 4.57199185e-07 | true / met |
| lower_right | interior | 14560.3069 | 14560.2771 | 2.04626001e-06 | true / met |
| lower_right | edge | 7240.528631 | 7240.526118 | 3.47129927e-07 | true / met |

The probe velocity is0.99 times each raw BP crop. Its observed trace is the declared coarse40 raw-model forward, reused for this engineering gradient check. The objective is one-half the unweighted sum of squared computed receiver residuals. Directions are predetermined positive squared-slowness perturbations on the interior or the four physical edges. The audit independently recomputes sampled objectives, gradient contractions, central differences and edge/corner Eᵀ accumulation from raw padded gradients, retaining naive cropping and exact restored-model witnesses. This is numerical verification of the chosen discrete gradient and extension map, not certificate transfer, inverse recovery or a formal floating-point enclosure.

The acoustic controls use the pinned Devito OT2/space-order4 implementation, float64, fixed constant-density physics, no free surface and constant-edge exterior velocity. Sources and receivers are artificial local-subdomain stress geometry200m below crop tops located about2.4,5.4 and8.4km deep in the BP composite. They are not original BP surface shots. All three crops remain one synthetic ancestor; the nearly homogeneous lower_right crop remains included.

A held Ricker Source is passed explicitly to each forward:5Hz,200ms delay and0–1,000ms record. Coarse source amplitude1 at 12.5m and fine amplitude4 at 6.25m preserve native point strength156.25m². Actual source/time/coordinate/model hashes and arrays are retained. Native receiver indices1..nt−2 are computed; initialized endpoints0/last remain raw but are excluded by the fixed masks. Half-step comparisons use only exact shared computed times0.5–999.5ms. Nonzero Ricker startup and discrete-stencil conventions remain part of these controls; no causal clean-window claim is made.

Only one coarse40 saved field per crop is used for its adjoint; its materialized allocation is824,732,160 bytes. Other forwards use three rolling slices. No wavefield history is written. The independent audit validates retained records and scalar reductions; it does not independently reproduce the native PDE from discarded wavefields. A separate NumPy replay has now completed; its results are recorded below as distinct evidence.

This is AI-authored engineering verification on a previously acquired synthetic parent, not a student final design, independent geological confirmation, inversion result, method novelty, accepted physical accuracy or competition conclusion. Old homogeneous7-call, NumPy 14-call, analytical Green, coordinate-comparison and acquisition artifacts remain immutable. No source, tolerance, duration or boundary profile was tuned after these outcomes.

Traceability: native complete receipt ae2ca931876b6c1159af151394bfb0e9c0352ea0c384e879ac398e5bca019206; independent scalar receipt d84b72d45ca5361bd9b9db0a883c0d444a459c4c17e31c436707cc64d67203d1. Producer73b617773c76a35e847add60d4b9f2200517d49d68a0f9af63b1098bc9d0137d; protocol63d8338f6a023f17365b8fa0d706f9f3a04e0b1e96006381e49309c3856444f0; auditor8b8bb021ca292c9f56104c8929f183bab05434c8ece726c149cf9a352e1f329c; criteria9ab520f7e3c35a7c6619b5fddae6a3e750cecd80aaeb18e4ef6c70b4b3d30c6e. Production source 9f remains unchanged.

## Original NumPy forward and tangent replay

All three independent rolling forward/tangent calculations completed before the reader opened native numeric witnesses. The implementation uses Python and NumPy, with no Devito or production import at replay time. Its recurrence is informed by pinned primary source and fresh generated C. There was no coefficient, amplitude, shift, mask or tolerance fit. It is a second implementation, not a blind derivation or human reproduction.

| Crop | Receiver relative L2 difference | Interior DD relative difference | Edge DD relative difference |
|---|---:|---:|---:|
| upper_left | 4.8713e-14 | 2.6493e-15 | 1.7431e-14 |
| middle_center | 2.6670e-14 | 2.6862e-14 | 6.4759e-14 |
| lower_right | 2.3527e-14 | 3.2606e-14 | 5.4139e-14 |

All six directions were informative and met the predeclared `1e-3` derivative criterion; all three traces met `1e-9`. These are relative errors, not percentages. Directions are normalized on the physical grid, then extended through constant-edge padding before the tangent solve. The edge result makes the pullback important: on upper_left, the full folded gradient gives approximately `-1800.10`, the independent tangent gives the same value, and naive physical cropping gives approximately `-101.31`. This verifies the implemented extension derivative; it does not introduce a new padding-adjoint formula.

The replay retains its own small receiver/tangent records and input/direction witnesses, all 57 native call statuses and all six comparisons. Completion receipt: `207bc209e59e6e07d8a1da1d26afd15ab1a92fcb01feef7793bd29d66438e89a`. A further independent stored-record review recomputed the trace norms, all six contractions and edge/corner folds in 132 checks with zero issues, without propagating again. The native and replay processes actually returned exit code zero; that is separate from the numerical comparisons and from physical acceptance, which remains null.

The [archive README](../work/wavecert-fwi/results/bp_heterogeneous_verification/README.md) provides exact sources, test histories, freezes, all 57 native calls, all 18 finite-difference checks, scalar audit, replay records and reproduction commands. Source-only staging receipts retain their historical unexecuted/resource-held labels. Storage was initially below the fixed guard and briefly prevented file writes. Later measured free space and RAM passed the unchanged guards before the one native run; no guard was lowered and no native run was restarted.

The first remote known-fixture run is mixed: one Linux learned/data job passes
all three traces/six derivatives, while the Python 3.12 matrix replay returns2
and the remaining matrix jobs are cancelled. Its same-version passing and
failing jobs are retained in the archive. Failure-only test diagnostics are
being added before drawing a cause or cross-environment success conclusion.
No numerical criterion, core implementation or native protocol is changed.

The diagnostic-only revision subsequently passes all four jobs in
[CI42467661756](https://github.com/uskutsav-cpu/wavecert-fwi/actions/runs/42467661756),
with all six known-fixture numerical reports per job retained. Across the BP
repetitions, maximum trace/DD relative differences are4.4091e-14/6.8235e-14.
These are three unique inputs repeated across jobs, not new geological samples.
The replay, success criteria and tolerances were unchanged; these passing jobs
do not establish the cause of the earlier failure. Its anomaly remains open.

## V2: exact shared numerical inputs

The later [97f8905 CI run](https://github.com/uskutsav-cpu/wavecert-fwi/actions/runs/42515747409) fails the original replay's input-direction guard. Its Python 3.11 and learned/data jobs fail; the cancelled 3.10 log also records this failure, while 3.12 has no completed result. All three trace comparisons agree, but the upper-left interior input-direction guard prevents both upper-left derivative comparisons. The archived witness context masses and the CI's saved mass hashes agree exactly, which excludes the mass branch of this particular guard. The failing direction norm/vector was not retained remotely, so a specific CPU/BLAS reduction mechanism is unproved. All four [original job logs](../work/wavecert-fwi/results/bp_heterogeneous_verification/CI_history_97f8905/summary.json) remain intact. The new 108-control frequency test was not reached before failfast.

The separately versioned [V2 replay](../work/wavecert-fwi/results/bp_heterogeneous_verification/numpy_replay_v2/protocol.json) uses exact recorded probe velocity, physical-m context and two direction arrays as shared numerical **inputs**. The exporter selectively decodes those four allowed members per crop, binding the original aggregate, worker status/completion, probe call/container, all six witnesses and actual array hashes. It exports no receiver, residual, gradient or wavefield outputs. No direction normalization, velocity roundtrip, clipping or fitted adjustment occurs. All three independent primal/two-tangent calculations finish before native **solution outputs** are decoded. This is source-informed reproduction with shared inputs; it no longer claims independent numerical input generation.

The numerical kernel remains byte-identical (`b217dd6d…`), and every numerical criterion remains unchanged. All 27 manufactured corruption/provenance/phase controls pass. The actual V2 replay returns exit 0: all three traces and all six informative derivatives agree, with maximum relative differences **4.8713e-14** and **6.4759e-14**. All six prescribed-direction input differences are exactly zero. See the [root source review](../work/wavecert-fwi/results/bp_heterogeneous_verification/bp_numpy_canonical_v2_root_source_review_01.json), [actual terminal observation](../work/wavecert-fwi/results/bp_heterogeneous_verification/bp_numpy_canonical_v2_terminal_observation_01.json) and [canonical input manifest](../work/wavecert-fwi/results/bp_heterogeneous_verification/canonical_inputs_v2_01/manifest.json).

A separate [saved-result audit](../work/wavecert-fwi/results/bp_heterogeneous_verification/numpy_replay_v2_independent_scalar_audit/report_02.json) passes 305 checks, 87 scalar numerical comparisons and 36 Boolean comparisons with zero issues. It rechecks all three traces/six contractions, exact input bindings and unchanged V1/kernel/criteria pins without propagating again. Source order and recorded phase flags agree; the audit does not independently prove runtime I/O chronology.

The revised public parent regression also passes locally with warnings treated as errors. Its first invocation failed during conftest import because an editable installation resolved another checkout; setting `PYTHONPATH=src` selected this checkout, with no source or numerical change. The earlier fixture, all failed CI/control attempts, and the current kernel remain archived. The actual new [CI42540137515](https://github.com/uskutsav-cpu/wavecert-fwi/actions/runs/42540137515) at `8d7b7fc` subsequently passes all four jobs on Python 3.10.22/3.11.18/3.12.16. All four full decoded logs contain all seven numerical reports, including the prescribed V2 input-manifest identity, three traces/six informative derivatives and all 108 frequency child controls. Across these Linux BP repeats, maximum trace/DD relative differences are4.4091e-14/6.8235e-14. Matrix jobs each report513 passed/34 skipped; learned/data reports564 passed/4 skipped/1 warning. The raw [CI evidence](../work/wavecert-fwi/results/reproducibility_v2_CI_8d7b7fc/summary.json) is separate from the frozen pre-CI BP archive inventory. This actual V2 success does not retroactively pass97f8905 or prove its specific CPU/BLAS mechanism; known-fixture repeats add no geology.

```bash
PYTHONPATH=src python -m pytest -q -s -W error tests/test_bp_numpy_tangent_reproduction.py
```

Physical acceptance remains null. These repeated known BP crops add no geological ancestor, recovery result, new certification contribution, human reproduction or student-owned interpretation.
