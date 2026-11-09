The manufactured control bundle needs **16 computational files** in the two literal sibling directories below. Paths in `PORTABLE_CONTROL_FILE_MANIFEST.json` are relative to their common parent. The layout document and manifest are descriptors; they are additional to those 16 files.

```text
bundle-root/
  fractional_velocity_prior48_post_audit/
    audit_retained.py
    trace_checks.py
    criteria.json
    test_scalar_audit_manufactured.py
    test_trace_checks.py
    export_manufactured_producer_schema.py
    test_producer_schema_interop.py
  fractional_velocity_prior48_driver_candidate/
    run_prior48.py
    protocol.json
    FROZEN48_SCHEDULE.json
    bounded_exact.py
    composed_exact_adapter.py
    fractional_velocity_prior.py
    independent_saved_models.py
    original_prior_proposal.json
    original_protocol.json
```

Keep the directory names and file bytes exact. Scalar controls read the sibling protocol during import and also import the trace-control fixture module. The isolated schema exporter loads `run_prior48.py` and the composed adapter; `driver.load_protocol` verifies all seven files in `protocol.local_dependency_sha256`, including files that the scalar control never executes. Supplying just the producer driver, adapter and protocol would leave required metadata dependencies missing.

Python 3.10 or newer, NumPy, SciPy and pytest must already be available in the selected environment. From `bundle-root`, verify the manifest before running the controls:

```python
from pathlib import Path
import hashlib
import json

root = Path.cwd().resolve()
manifest = json.loads((root / "fractional_velocity_prior48_post_audit" /
                       "PORTABLE_CONTROL_FILE_MANIFEST.json").read_text())
assert manifest["required_file_count"] == len(manifest["required_files"]) == 16
for row in manifest["required_files"]:
    path = (root / row["path"]).resolve()
    assert path.is_relative_to(root) and path.is_file()
    assert path.stat().st_size == row["bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
```

Then run the three explicitly selected control files:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -I -B -m pytest -q -c /dev/null --import-mode=prepend \
  fractional_velocity_prior48_post_audit/test_scalar_audit_manufactured.py \
  fractional_velocity_prior48_post_audit/test_trace_checks.py \
  fractional_velocity_prior48_post_audit/test_producer_schema_interop.py
```

No project input arrays, full WaveCert repository, original preparation helper package, actual study results, `synthetic_temp` tree or historical control outputs are required for this control command. The graph helper and original protocol/proposal files in the sibling folder are retained because metadata pin validation requires their exact bytes. This minimal layout is for manufactured controls; auditing real retained study results is a separate terminal-gated operation requiring its source/input/result evidence.

The interoperability control starts a fresh `-B -I` subprocess. That process calls the byte-frozen bounded engine, composed adapter, `driver.execute_policy` and real SciPy minimizer with a one-dimensional manufactured scalar owner. It exports the producer's actual nonzero-query record schema. The independent auditor process receives JSON and imports no producer or project physics module. Its fake clock tests schema accounting and cannot establish performance or native PDE correctness.

The initial producer-interoperability extension passed 70 controls at tool chunk `e9efec`, exit 0, in 3.38 seconds. A prospective terminal-credit reporting fix then passed 75 controls at `4491f7`, exit 0, in 3.46 seconds. The five added cases cover unknown/nonzero/Boolean exit status, missing complete receipt, and the positive verified-completion path. Row/case arithmetic and fixed numerical criteria remain unchanged. The unqualified continuation gate requires verified full-study credit; its arithmetic-only precursor is explicitly conditional. The two actual scalar exports retained 6/16 library events, 2/5 callback point records and 3/8 gradient dispatches for lambda 0/100. The initial layout review verified dependency paths and exact file hashes without rerunning controls or establishing Linux/all-Python portability. This descriptor was refreshed after the source/reporting fix; the original 70-control descriptor is retained in `history_before_terminal_credit_fix_70`. Updating these descriptors changes no source or numerical criteria.
