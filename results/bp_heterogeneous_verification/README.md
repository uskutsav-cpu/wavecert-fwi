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
