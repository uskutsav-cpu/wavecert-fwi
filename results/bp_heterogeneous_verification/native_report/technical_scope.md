Native BP engineering guard: record integrity and conditional sensitivities

The fixed native run completed all54 forwards and3 adjoints across three retained BP2004 crops. The independent retained-record scalar audit passed2,618 checks with0 integrity issues. All six prescribed middle-step gradient checks were informative and met the fixed numerical error criterion. Physical acceptance remains null. These are separate findings: record integrity and gradient consistency do not make the waveforms physically adequate.

The aggregate trace differences below include every crop and all ten receivers. Each entry is100×||candidate−reference||₂/||reference||₂ over exact common computed times, with the second named configuration as reference. No selected window, lag or amplitude fitting is used. Full-precision ratios, norms and sample counts are retained in aggregate_sensitivities.csv.

| Crop | coarse40→coarse80 | coarse80→coarse160 | coarse80→fine160 | coarse80→coarse80_halfdt |
|---|---:|---:|---:|---:|
| upper_left | 13.945763% | 5.825061% | 2.766556% | 0.017813% |
| middle_center | 37.854666% | 18.520556% | 0.975378% | 0.043298% |
| lower_right | 49.825439% | 22.846015% | 1.240897% | 0.056562% |

Coarse spacing is12.5m, so padding40/80/160 is500/1,000/2,000m per side. The fine grid has6.25m spacing and padding160, matching the1km coarse80 absorber width. It samples the same piecewise-bilinear velocity field defined by the retained192×96 coarse nodes, with the exact2,387.5×1,187.5m physical extent. It adds no native BP samples. Native damping samples differ with the grid. The half-step control changes0.5ms to0.25ms. Padding changes also alter exterior paths and spreading. The table therefore measures conditional sensitivity; its entries are neither reflection coefficients nor an additive error budget, and they establish no convergence rate or physical tolerance.

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

A held Ricker Source is passed explicitly to each forward:5Hz,200ms delay and0–1,000ms record. Coarse source amplitude1 at12.5m and fine amplitude4 at6.25m preserve native point strength156.25m². Actual source/time/coordinate/model hashes and arrays are retained. Native receiver indices1..nt−2 are computed; initialized endpoints0/last remain raw but are excluded by the fixed masks. Half-step comparisons use only exact shared computed times0.5–999.5ms. Nonzero Ricker startup and discrete-stencil conventions remain part of these controls; no causal clean-window claim is made.

Only one coarse40 saved field per crop is used for its adjoint; its materialized allocation is824,732,160bytes. Other forwards use three rolling slices. No wavefield history is written. The independent audit validates retained records and scalar reductions; it does not independently reproduce the native PDE from discarded wavefields. A separate NumPy replay is separate evidence and is not incorporated here while it is live.

This is AI-authored engineering verification on a previously acquired synthetic parent, not a student final design, independent geological confirmation, inversion result, method novelty, accepted physical accuracy or competition conclusion. Old homogeneous7-call, NumPy14-call, analytical Green, coordinate-comparison and acquisition artifacts remain immutable. No source, tolerance, duration or boundary profile was tuned after these outcomes.

Traceability: native complete receipt ae2ca931876b6c1159af151394bfb0e9c0352ea0c384e879ac398e5bca019206; independent scalar receipt d84b72d45ca5361bd9b9db0a883c0d444a459c4c17e31c436707cc64d67203d1. Producer73b617773c76a35e847add60d4b9f2200517d49d68a0f9af63b1098bc9d0137d; protocol63d8338f6a023f17365b8fa0d706f9f3a04e0b1e96006381e49309c3856444f0; auditor8b8bb021ca292c9f56104c8929f183bab05434c8ece726c149cf9a352e1f329c; criteria9ab520f7e3c35a7c6619b5fddae6a3e750cecd80aaeb18e4ef6c70b4b3d30c6e. Production source9f remains unchanged.
