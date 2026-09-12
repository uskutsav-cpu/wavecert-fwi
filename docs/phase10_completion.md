# WaveCert-FWI completion layer (0.5.0)

This layer turns the remaining benchmark work into reproducible commands.

## Real geology

1. Build a denser exact-physics operator-learning dataset from selected SubsurfaceGen slices:

```bash
python scripts/build_subsurfacegen_wavecert.py \
  --selected-root data/external/subsurfacegen-selected \
  --output-root data/external/subsurfacegen-wavecert-v2 \
  --train-models 800 \
  --validation-models 200
```

By default every selected model is paired with every configured source/frequency
combination.

2. Train the stronger FNO:

```bash
wavecert train-real-fno \
  --dataset data/external/subsurfacegen-wavecert-v2 \
  --checkpoint checkpoints/fno_subsurfacegen_v2.pt
```

3. Calibrate/evaluate WaveCert on real geology:

```bash
wavecert real-wavecert-study \
  --dataset data/external/subsurfacegen-wavecert-v2 \
  --checkpoint checkpoints/fno_subsurfacegen_v2.pt
```

Conformal coverage is interpreted only under exchangeability. OOD coverage is
reported diagnostically.

## Devito / Marmousi

```bash
wavecert devito-study \
  --marmousi data/external/marmousi/mar_big_117_567.bin \
  --iterations 5
```

This command actually executes Devito forward, adjoint-gradient, and L-BFGS-B.
Import availability alone is not counted as full external validation.

## OpenFWI

Download one official velocity batch and run:

```bash
wavecert openfwi-study \
  --velocity-file data/external/openfwi/FlatVel_A/model1.npy \
  --checkpoint checkpoints/fno_subsurfacegen_v2.pt \
  --max-models 100
```

The official Vel/Fault/Style format is `(N,1,70,70)` for velocity maps and
`(N,5,1000,70)` for seismic batches.

## Baseline repetition

```bash
wavecert baseline-study --cases 4 --iterations 8
```

This repeats the four principal trajectory policies across independent seeds and
collects common accuracy/cost statistics.


## Expand the SubsurfaceGen slice cache first

If the current cache contains only the earlier 200/25/25 selection, expand it
before using the 800/200 defaults:

```bash
python scripts/select_subsurfacegen_slices.py \
  --index-root data/external/subsurfacegen-index \
  --output-root data/external/subsurfacegen-selected-v2 \
  --train 1000 \
  --id 50 \
  --ood 50
```

Then point `build_subsurfacegen_wavecert.py` at
`data/external/subsurfacegen-selected-v2`.

## Heuristic baselines / ablations

```bash
python scripts/run_heuristic_baselines.py
```

This compares neural-only, exact, one-block random repair, periodic exact
fallback, a diagnostic forward-error threshold, and WaveCert selective repair
on matched cases.
