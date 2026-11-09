# Independent saved BP V2 scalar audit

This audit reads saved V2 NumPy output, frozen canonical inputs and original native
BP records. It imports no producer, kernel, solver, source exporter or previous
auditor numerics. NumPy is shared for NPZ decoding, array identity and output
allocation. Independently written scalar `math.fsum` and `math.hypot` arithmetic
recomputes trace/source norms, the padding pullback, six tangent/gradient/padded/
naive-crop contractions, informative-direction rules and original success gates.
No PDE, tangent, adjoint or inversion is executed.

The final audit passes 305 checks with zero issues, including 87 exact numerical
scalar comparisons and 36 exact Boolean-field comparisons. It verifies all three
trace comparisons and six informative derivatives, exact prescribed canonical
input identity to both own and native arrays, container/worker/aggregate-call
bindings, source identities and truthful scoped provenance flags. The old f0
protocol and c51 runner are explicitly hash-pinned, the b217 numerical kernel is
unchanged, and every numerical criterion remains fixed. All bound files retain
their hashes after the audit.

The largest trace relative L2 discrepancy is 4.871289597762567e-14; the largest
directional derivative relative discrepancy is 6.475897255156544e-14. Those are
same-discrete-equation verification results, not realistic physical accuracy or
inverse reconstruction evidence. All three crops share one BP synthetic ancestor.
Physical acceptance remains null. V2 shares native-derived numeric INPUTS and
therefore does not independently generate inputs or provide blind validation.

Source inspection confirms that the runner's three propagation calls finish
before target-array comparisons, while the exporter selects only the prescribed
input entries. Saved scoped flags agree. This audit does not provide a separate
operating-system I/O trace proving runtime chronology, reconstruct final rolling
state hashes, retroactively pass the failed 97f CI run, or establish Linux V2 CI.

The first successful audit report and its source/log are preserved. Its count of
123 exact saved scalar/flag comparisons combined numerical values with Boolean
fields. The final version separates those into 87 numerical and 36 Boolean
comparisons and adds explicit historical source pins/native-completion binding;
it changes no numerical formula or threshold. Both audits use only saved arrays.
