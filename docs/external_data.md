# External benchmark data runbook

WaveCert's CI and compact mathematical studies are self-contained. Phase 8/9 also provide adapters for public external benchmarks, but large data are intentionally not committed to Git.

## Compact public Marmousi model

The Phase-8 loader supports the `117 x 567` float32 Marmousi binary used by the public SiameseFWI reproducibility repository.

```bash
bash scripts/download_marmousi_public.sh
wavecert production-study --marmousi data/external/marmousi/mar_big_117_567.bin
```

The result JSON records whether the real binary or the offline proxy was used.

## OpenFWI

Official OpenFWI 2-D Vel/Fault/Style velocity files are NumPy batches. The public benchmark repository documents velocity batches as `(500,1,70,70)` and seismic batches as `(500,5,1000,70)`.

WaveCert adapters:

```python
from wavecert.benchmarks.openfwi import (
    load_openfwi_seismic_batch,
    load_openfwi_velocity_batch,
)
```

Keep the official train/test split by file family when reproducing OpenFWI results. Do not randomly mix samples across the provided train/test files.

## SubsurfaceGen

The 2026 field-scale release uses HDF5 files:

- velocity slice: key `velocity`, shape `(nz,nx)`;
- wavefield: key `wavefield`;
- shot-gather cube: key `shot_gather_cube`.

The full dataset index uses `train`, `test_in_dist`, and `test_out_dist` split labels. Preserve those labels rather than constructing a random pixel/slice split.

The preview repository can be downloaded with:

```bash
python -m pip install huggingface_hub h5py hdf5plugin pandas pyarrow
huggingface-cli download subsurfacegen/field-scale-dataset-preview \
  --repo-type=dataset --local-dir=data/external/subsurfacegen-preview
```

The preview itself is multi-GB, so it is not appropriate for ordinary CI.

## Reproducibility rule

Every result using external data must record:

1. exact dataset/repository identifier;
2. split name;
3. model/slice identifier;
4. frequency band;
5. source/receiver geometry;
6. preprocessing/downsampling;
7. whether a result used a real public sample or an offline procedural proxy.
