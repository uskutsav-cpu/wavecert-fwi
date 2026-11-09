# Portable reproduction test scope

`test_numpy_boundary_reproduction.py` is staged for the repository's existing
pytest CI. It explicitly loads source48326128 and protocol53fbdd57 from
`results/numpy_boundary_replay/fixture`, with no import fallback. The same frozen
source API supplies `load_protocol` and pure NumPy `solve_case`.

All seven faithful traces are computed before any primary target NPZ is opened.
Comparison requires exact times, source relative L2<=1e-14, and receiver relative
L2<=1e-9. Bitwise source equality is diagnostic. Per-case errors and environment
details are printed into the existing CI log; no workflow changes are required.
The rational-coefficient negative control remains in the archived14-run evidence
and is deliberately not a passing-CI requirement.

This reproduces a reference-informed finite-grid numerical implementation.
It establishes neither continuum accuracy, independent human reproduction,
student scientific ownership, independent geology nor completion of goal4.
The staged CI test has not been executed here; root owns integration and runs.
