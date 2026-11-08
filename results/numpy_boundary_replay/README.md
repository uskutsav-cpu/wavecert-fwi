# Executed reference-informed NumPy replay

All fourteen fixed local replays completed. Generated-literal agreement is 7/7
(maximum relative L2 error 3.411e−14); rational-coefficient agreement is 0/7 at
the unchanged 1e−9 threshold. All mismatches are retained. Source and time samples
match exactly in this local run. The original substantial boundary errors are
reproduced. See [scope and reproduction](../../docs/numpy_boundary_replay.md).

- `run_01/` contains every actual raw NumPy trace/profile, frozen runner/protocol,
  source ledger, agreement record and completion receipt.
- `fixture/` is the byte-exact pre-execution bundle, including primary C/source
  and argument metadata. Its README/test receipt describe that earlier state.
- `review/` retains the pre-execution freeze, actual execution log, pre/post
  audits and portable post-auditor source, including its correction history.
- `DEVITO_LICENSE.md` is the original installed Devito license accompanying
  the attributed source copies. Replay propagation uses only stdlib and NumPy.

This is AI-authored implementation verification, not blind research confirmation,
independent geology, a student's final research design or full goal completion.
