# Eight initial-state raw direction witnesses

This is a separately frozen software-verification fixture for **exactly the
eight original initial models** from the paid-merit engineering benchmark. It
does not rerun its optimizer or reconstruct unavailable intermediate vectors.
There is no new geology, parameter selection, performance study, final student
design or formal floating-point enclosure.

Actual producer execution completed with exit code 0: eight cases, 552 checks,
zero issues. It calls the real certified adapter once per initial state with the
unchanged 49-grid/rank-64/four-repair settings. A transparent pass-through capture
retains the actual cached/missing fields and certificate result; the original
function reference is restored in `finally`. Production source is unchanged.

The true physical gradient is assembled from independent graph-matrix state and
conjugate-transpose adjoint solves, with independently generated observations,
using pinned objective-auditor source `807f4f2f…235edeae`. NumPy/SciPy/SuperLU
remain shared dependencies; this is a separate equation implementation, not
independent numerical libraries. Raw initial model, damping, directions,
approximate/true gradients, states, adjoints, observations, missing/cached fields,
source/receiver maps and complete scalar receipts are retained in `run_01/`.

Two actual postchecks completed without new PDE solves or production imports:

- The copied NumPy-only raw replayer passed **1,289 checks**, zero issues.
- A second NumPy checker passed **728 checks**, zero issues. Maximum relative
  gradient/direction reconstruction discrepancy was `4.1585e-16`; maximum direct
  stencil residual was `7.8127e-14`.

All eight directions were available and certified, all eight bounds covered
literally and with the separately reported historical helper allowance, and no
false certification was observed. All 22 producer artifacts were rehashed
unchanged. These eight dependent initial fixtures do not establish a statistical
risk guarantee. Coarse stability constants are supplied by the ordinary-FP
production implementation; the replay does not independently rederive/enclose
those constants or prove the bound in floating-point arithmetic.

`fixture/` preserves current source, protocol, all pre-GO histories/adverse reviews,
the 14 focused mock tests and their actual validation receipt, three source-review
versions, and both post-audit commands/logs/receipts. Material pre-GO fixes preserve
partial arrays on handled interrupts, reject clear status on adverse flags, and
bind actual model/damping/source/receiver inputs. The tests include those failure
paths. The combined raw post-audit receipt is
`fixture/raw_witness_post_audit_receipt_01.json`.

Replay the archived raw witnesses with only NumPy plus the standard library:

```sh
python results/paid_merit_initial_direction_verification/run_01/replay_raw_witnesses.py \
  --execute-after-terminal-confirmation \
  --run "$PWD/results/paid_merit_initial_direction_verification/run_01" \
  --output /tmp/wavecert-initial-raw-replay-new.json \
  --expected-source-sha256 1e9af84e461861c98dfc69f34d85d88c539b40f6658d3f1924cd32cb10729ccc
```

Recompute the eight queries with the unchanged production source and a new output:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src \
python results/paid_merit_initial_direction_verification/fixture/run_initial_direction_verification.py \
  --execute-after-terminal-confirmation --repo "$PWD" \
  --benchmark "$PWD/results/paid_merit_engineering_benchmark/run_01" \
  --objective-auditor "$PWD/results/paid_merit_initial_direction_verification/run_01/independent_assembly_source.py" \
  --output /tmp/wavecert-initial-queries-new \
  --expected-source-sha256 52cdbb46e8337b955cbdec13a0b1ed96637c8c34da337e1f3269ad423ffe38f9 \
  --expected-protocol-sha256 e53db61db1a6c3d83a5fdf3cba195c357f37f54e125e793980a76e33d5923e46
```

The regression in `tests/test_paid_initial_witness_replay.py` checks the archived
raw arithmetic in a separate isolated Python process. It is a known-fixture
software regression, not a new confirmation study. Source data attribution and
the SEAM ancestry correction are inherited from the original
[CC BY 4.0 derived fixtures](../../data/reference/subsurfacegen_parent_split/README.md).
The [engineering comparison](../../docs/paid_merit_engineering_benchmark.md)
still shows no useful certificate advantage. Independent human review, student
ownership and personal scientific interpretation are separate requirements.

All four jobs in [CI run 42451566612](https://github.com/uskutsav-cpu/wavecert-fwi/actions/runs/42451566612)
passed at source/test commit `9dcce6a`. Actual per-case JSON reports are retained
in [`ci_reports_9dcce6a.json`](ci_reports_9dcce6a.json): Linux Python 3.10.22,
3.11.18 and 3.12.16 with NumPy 2.2.6, 2.4.6 and 2.5.4. Each job repeats all eight
raw witnesses (all literal bounds cover, zero false certifications) and the
seven known homogeneous traces. The maximum raw independent-gradient
reconstruction discrepancy across those jobs is `1.1815e-16` relative L2.
These are repeated known fixtures, not 32 new directions or geological samples.
