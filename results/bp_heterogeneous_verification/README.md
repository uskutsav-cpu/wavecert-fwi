# Heterogeneous BP numerical verification

One fixed native run completed all 54 forwards and three adjoints across all three retained BP crops. The scalar post-audit passed 2,618 checks with no issues. A separately coded NumPy replay reproduced all three probe forwards and six informative directional derivatives under fixed criteria. Padding sensitivities remain substantial, and physical acceptance is null.

[Technical report](../../docs/bp_heterogeneous_verification.md) contains every crop, padding/mesh/time comparison and prescribed middle-step gradient check. All 18 FD checks and raw witnesses remain in `native_run_01/`. `numpy_run_01/` retains the three own rolling calculations and all six DD comparisons. Original source-only receipts still describe the pre-execution phase; actual terminal receipts and completed outputs are separate.

`native_fixture/` has exact source/protocol and 15 controls including one real tiny manufactured API check. `post_auditor/` has the fixed scalar criteria and seven controls. `numpy_replay/` has primary source/C references, derivation, fourteen distinct manufactured/phase controls and NumPy-only replay code. Failure/style/warning histories remain. The historical primary-C query needs its pinned Devito installation/cache path; replay itself imports neither Devito nor production code. Native execution requires the source-pinned Devito files and a compiler.

From the repository root, the NumPy-only known-fixture replay uses a fresh outer output:

```sh
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 python -I results/bp_heterogeneous_verification/numpy_replay/run_replay.py \
 --execute-after-native-terminal --inputs results/bp2004_acquisition/run_01 \
 --native-run results/bp_heterogeneous_verification/native_run_01 \
 --terminal-observation results/bp_heterogeneous_verification/native_terminal_observation_01.json \
 --output ../bp_numpy_reproduction_02 \
 --expected-source-sha256 c51e570934071eade9e59989cd57daddd064252cd930e410573db0365944d9e0 \
 --expected-protocol-sha256 f0b9db69a434116f212182897f4fa34ffd714782cf0e8dd30adb0410cc139e8c \
 --expected-native-complete-sha256 ae2ca931876b6c1159af151394bfb0e9c0352ea0c384e879ac398e5bca019206
```

A new native execution is a separate, heavier operation. It requires the unchanged 256MiB disk guard, saved-field RAM/reserve guard and exact production source9f. Do not substitute this replay for a new native run. Full native CLI paths/settings and terminal evidence are retained with its freeze/execution records.

All three crops are parts of one synthetic BP ancestor. Artificial sources lie200m below deep crop tops; constant-edge exteriors do not reproduce original BP surface surveys. There is no inverse method, certificate transfer, new theorem or accepted continuum/field accuracy here. No time-by-grid history is written. Original BP data terms and modification notice in [the acquisition archive](../bp2004_acquisition/README.md) continue to apply to derived arrays; they are not relicensed as MIT code. Copied Devito implementation references carry their license in `numpy_replay/DEVITO_LICENSE.md`.

AI-authored implementation verification. All four scientific goals and student-owned interpretation remain incomplete.

## First remote regression outcome

The initial c197 publication passed the full learned/data Linux job, including
all three BP traces and six derivatives, but the Python3.12 matrix replay
returned2. The other matrix jobs were cancelled. The same reported runtime
versions in a passing and failing job do not establish a version-only cause.
The [first CI history](CI_history_c197/README.md) retains the failure and the
successful known-fixture reports. A diagnostic-only test change exposes saved
comparison/model details; numerical criteria and the frozen replay remain
unchanged. Full cross-environment verification is unresolved at this point.

## Subsequent source/test CI

Diagnostic-only commit d2c34fcf8028507b8f13111bd56bfcf8e5515837 passed all four
jobs in [CI42467661756](https://github.com/uskutsav-cpu/wavecert-fwi/actions/runs/42467661756).
All six known-fixture numerical reports from every job are archived in
`cross_environment_CI_reports_d2c34fc.json`. The BP portions repeat three unique
inputs:12 trace comparisons and24 informative directional comparisons across
jobs, without adding geological samples. Their maximum relative trace/DD
errors are4.4091e-14/6.8235e-14.

This revision only adds failure diagnostics. Its success checks, frozen replay
source/core/protocol and numerical tolerances are unchanged. Therefore the four
passing jobs do not identify or repair the original c197 failure. That anomaly
remains unresolved and both outcomes remain visible. No native57-call study,
new recovery cohort or physical acceptance test was repeated by CI.


## Separately versioned exact-input replay

The original `numpy_replay` and failed CI42515747409 remain historical evidence. The live regression now uses `numpy_replay_v2` with `canonical_inputs_v2_01`: exact native-derived velocity/context/direction inputs, with no native solution outputs decoded until all three own solves finish. The numerical kernel and tolerances are unchanged. The actual local V2 replay and public parent test pass all three traces/six informative derivatives; Linux V2 verification remains pending. See the [V2 report](../../docs/bp_heterogeneous_verification.md#v2-exact-shared-numerical-inputs), source/control receipt, root review and terminal observation. This is shared-input, source-informed AI reproduction, not blind input generation or accepted physical accuracy.

From the repository root:

```bash
PYTHONPATH=src python -m pytest -q -s -W error tests/test_bp_numpy_tangent_reproduction.py
```

The native 57-call study is not rerun. Source/history/control failures, exact canonical input provenance and the original public parent test are preserved in the V2 archive.

Inventory scope: `archive_manifest.json` is the preserved236-entry pre-V2 snapshot. Its original README is retained under `history_manifest_97f8905`, with a scope addendum; the other235 listed artifacts remain byte-identical. The current V2 publication inventory is `archive_manifest_v2_publication.json`, which includes current README and all additive assets and excludes itself and generated runtime cache/bytecode files.
