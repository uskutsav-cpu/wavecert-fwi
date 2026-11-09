# Reference-informed NumPy boundary replay

This AI-authored software check independently implements the same seven fixed
finite-grid cases. It is not blind, continuum truth, a student final design,
independent human reproduction, or completion of the research goals.

Propagation uses only Python standard library and NumPy, with independently
generated analytic damping and Ricker samples. No installed/imported Devito,
fields, damping arrays, cached sources or receiver values enter propagation.
The copied primary sources, generated C and trapped metadata query specify the
actual one-sided damping derivative, centered second derivative, spatial weights,
and time1..1099 update/inject/current-receiver order. Last receiver sample1100
remains zero, as required by the primary operator bounds.

Two declared variants use generated C literals or exact rational spatial weights;
all14 raw replays are retained before any reference numeric array is opened.
Both use the actual same one-sided damping recurrence. Trace agreement is fixed
at relative L2<=1e-9, with exact times and source relative L2<=1e-14. Source
bitwise equality is diagnostic across ARM/x86. All errors and failed flags remain.
No waveform alignment, coefficient tuning, or absorber changes are permitted.

Default invocation prints hashes without propagation. Actual execution requires
`--run`, repository/reference/new-output paths and both reviewed SHA256 values.
Attempt prefixes and interruptions are retained; reference access becomes unknown
when comparison starts and true only on successful comparison.12 pure tests and
Ruff pass. No fourteen-case numerical replay has run; root retains GO.
