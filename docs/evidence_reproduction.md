# Inspecting and reproducing the strengthening evidence

Use the `research-strengthening` branch / draft PR5 for the new implementation
and archived outputs. The default `main` branch remains the original project.
Historical reports describe their recorded source, not every later adapter.

## Current software checks

```sh
python -m pip install -e '.[dev,torch,data]'
PYTHONPATH=src python -m pytest -ra
python -m ruff check src tests scripts
```

The base-only installation omits learned/data extras and skips those genuinely
optional paths. Devito checks additionally require Devito and a working compiler.
`PYTHONPATH=src` binds the selected checkout when multiple editable installations
exist. Source-bound adapters reject foreign cached imports.

## Read-only evidence analysis

A full-history Git checkout (ordinary `git clone`, without `--depth`) is required
for audits that compare recorded source bytes to historical commits.

```sh
python results/strengthening_audit/audit-five-model-results.py \
  --repository . --output /tmp/wavecert-five-model-audit.json
python results/strengthening_figures/plot-strengthening-evidence.py \
  --repository . --output /tmp/wavecert-new-figures
```

Choose a new plot output directory outside the repository. The audit reads all
saved aggregate vectors and raw tables, checks arithmetic/cohorts/flags, and
verifies the 144 recorded source files. It does not rerun the PDE, reconstruct
every operation counter, establish formal roundoff control, or replace human
reproduction. Original and portable-script rerun receipts are distinguished in
the audit README. Plot captions and manifests retain statistical units and scope.

## Reexecuting a historical experiment

Start from its actual frozen source revision, use a separate full-history
checkout/worktree and new output directories, and follow that revision's study
documentation and input preparation. Do not overwrite archived receipts or edit
source/configuration during a run. Later additive code is deliberately not
retroactively admitted into an earlier freeze. A shallow CI synthetic fixture is
not a substitute for the real historical source audit.

| Actual experiment | Frozen Git source | Documentation / output |
|---|---|---|
| Four FNO training candidates | `be6e51e090b4722d0610baaf812b56d4d355ae40` | `results/gradient_development/README.md`, including interrupted run and I/O-only recovery |
| Eight classical POD candidates | `6470e4d41052eb4537a38fc23cd65452540a9e6e` | `docs/pod_controls.md`, `results/pod_development` |
| Public coarse validation and fixed five-model comparison | `8b3c1f1854dc94ccdad1b3305a3bc83ee672c41c` | `docs/coarse_adjoint.md`, `docs/gradient_confirmation.md`, `docs/pod_confirmation.md` |
| Separately frozen historical transfer | `3871987718777bcb453b7e090d6d5a62db9e6081` | `docs/historical_transfer.md`, `results/historical_transfer` |
| Scalar cold-query costs | `312c65922f50cd0176c9f2096adbb64cd0b6e7b4` | `docs/coarse_runtime_controls.md`, `results/coarse_runtime_controls` |

Some output snapshots and preparation artifacts were published after the source
commit that generated them. Keep a copy of the current archived evidence outside
the historical checkout when inspecting those artifacts. Preserve pinned input
bytes, licenses and whole-parent roles; external raw caches are not silently
bundled or relabeled as field observations. Runtime variation across machines
is expected, and a new run must record its own environment and source receipts.

Private/exploratory scripts under `results/coarse_development_exploration` retain
their original bytes and sometimes absolute paths. They are discovery records;
the public fixed-protocol implementations are the portable reproduction paths.
The later batched/rank study is validation-only engineering development, not a
rerun or replacement of the earlier fresh-index comparison.
