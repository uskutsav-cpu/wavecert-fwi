# Existing validation inverse cases

This small archive is a byte-identical copy of the existing
`validation_inverse_cases.npz` produced during the gradient/POD development
study. Its SHA-256 is
`be429def325b741d415546167b8e13b5d52a3304d79f36efd3945a58363177db`.
`provenance.json` records the original source, identities, numerical operator,
and attribution. It contains eight existing validation slices from four
parents, their smoothed float32 model inputs, 24 synthetic receiver blocks
per slice, and previously computed exact aggregate gradients/objectives.
These are development data, not held-out observations or field measurements.

The underlying earth models are derived from Joseph Stitt, Pratik Rathore,
Madeleine Udell, and Ching-Yao Lai's *SubsurfaceGen: Procedural Generation of
Field-Scale Earth Models and Seismic Data* (2026),
[paper](https://arxiv.org/abs/2605.30541),
[dataset](https://huggingface.co/datasets/subsurfacegen/field-scale-dataset),
revision `c612dc2255f6dc4a139bb667f12eaaedf61a8d3d`.
The model data are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
the upstream code is separately MIT licensed. The WaveCert development fixture
resizes the models to 24 by 24, converts velocities to km/s, applies the
previously recorded velocity clipping, and generates its own discrete
Helmholtz data. The inverse inputs here use the existing Gaussian-smoothed
initial velocities. No new cases were selected for the certificate pilot.

The compact copy makes the bounded certificate reproduction self-contained;
it does not replace the training snapshots needed to refit the POD basis.
Shared teacher-generation costs and the frozen basis-fit record remain in
`results/pod_development/input_provenance.json` and `summary.json`.
