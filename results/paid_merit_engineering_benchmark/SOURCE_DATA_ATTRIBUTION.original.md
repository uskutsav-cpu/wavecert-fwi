# Compact SubsurfaceGen-derived research fixture

This directory contains downsampled synthetic velocity models and newly generated
frequency-domain Helmholtz fields. It does not contain measured field data or
the original field-scale SubsurfaceGen wavefields.

Source: Joseph Stitt, Pratik Rathore, Madeleine Udell, and Ching-Yao Lai (2026),
*SubsurfaceGen: Procedural Generation of Field-Scale Earth Models and Seismic Data*,
[arXiv:2605.30541](https://arxiv.org/abs/2605.30541).
Original dataset: [subsurfacegen/field-scale-dataset](https://huggingface.co/datasets/subsurfacegen/field-scale-dataset),
revision `c612dc2255f6dc4a139bb667f12eaaedf61a8d3d`.

The source velocity data and these derived data are distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
WaveCert code retains its separate MIT license.

Changes: two slices per ID parent and four slices from the OOD parent were selected,
resized to 24 by 24, converted to km/s, clipped to [1.35, 5.5] km/s, and passed
through WaveCert's discrete solver at six sources and four frequencies. Training,
model-selection validation, calibration, and ID testing use disjoint parent models
assigned from a common ID pool. This is a new research split, not the official
benchmark split. OOD contains only one parent and is descriptive.

`source_selection.json` records the immutable source revision, original file
checksums, selection seed, and parent assignments. `dataset_manifest.json`
records array/metadata hashes and preprocessing. The fixture was atomically
reassigned from an earlier stratified development cache before the new checkpoint
was trained; that derivation is retained in its manifest. No field values were
changed during reassignment.

The prepared arrays are intentionally bundled so checkpoint-linked evaluation
works without downloading large raw data or requiring cross-platform sparse
solves to reproduce byte-identical compressed arrays. Regenerating raw-derived
data may produce a different artifact hash; report that new provenance and
retrain against it instead of overriding the checkpoint integrity check.
