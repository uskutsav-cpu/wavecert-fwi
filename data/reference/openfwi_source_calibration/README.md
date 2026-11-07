# Frozen source-domain calibration summaries

These are exact historical WaveCert-generated summary bytes, retained so the
OpenFWI external stress protocol can reproduce its original calibration scales
after later reproductions overwrite the working results summaries.

| Snapshot | Original results path | SHA256 |
|---|---|---|
| `forward_only.json` | `results/grouped_wavecert/summary.json` | `c3c3d425a2e0cdb75ecb9021e13e77f9afbfff5df21ea9b42e87891702e8081d` |
| `derivative_informed.json` | `results/grouped_wavecert_jvp/summary.json` | `8116e3756ff57d133a0bfec1a607cde6a66918d557c2ba85cd4869ab8abac10a` |

The hashes, checkpoint identities, and scales were frozen in
`configs/openfwi_external_stress.json` before evaluating the external models.
Preparation prefers the original results path when its bytes match the frozen
hash. Otherwise, it accepts a bundled snapshot only when the same hash matches.
The generated preparation manifest records both the original reference path and
the actual resolved snapshot path. These files do not authorize recalibration,
new scale selection, or an external-distribution coverage guarantee.

The summaries describe synthetic reference Helmholtz fields derived from the
parent-isolated SubsurfaceGen fixture in
`data/reference/subsurfacegen_parent_split`, whose attribution and CC BY 4.0
source-data terms remain applicable. Source: Stitt et al. (2026),
[SubsurfaceGen](https://arxiv.org/abs/2605.30541),
[revision-pinned dataset](https://huggingface.co/datasets/subsurfacegen/field-scale-dataset/tree/c612dc2255f6dc4a139bb667f12eaaedf61a8d3d).
They are engineering experiment records, not a student-authored submission or
evidence of field-data validation. No OpenFWI velocity arrays or seismic records
are bundled here; OpenFWI source data retain their separately documented terms.
