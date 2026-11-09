# Four-arm frequency-schedule engineering comparison

This is post-outcome AI engineering development on eight already inspected
synthetic slices. The lower-band initial-linearization result motivated the
candidate; it is not an untouched hypothesis test or the student's final design.
No new FWI optimizer, certification theorem or reconstruction guarantee is
introduced. All four scientific requirements remain open.

The four fixed workflows are joint inversion on eight frequencies, low four
followed by cumulative eight, joint eight followed by a cold joint-eight restart,
and the original four-frequency contextual reference. All receive the same
available eight-frequency observations, repeated preparation and native-m start.
The original raw-input metadata is still validated with its unchanged four-
frequency schema. Only observation preparation and independent reference
evaluation use the explicit eight-frequency clone.

Original-four versus eight-frequency workflows also changes the information
actually used. Continuation versus restarted joint-eight holds the restart
mechanism fixed, but changes the early objective and the amount of work possible
before the absolute deadline. This is a conditional workflow comparison, not a
pure causal frequency-order experiment.

Each measured API starts before input/source checks, interpolation, smoothing,
all 48 synthetic observation RHS and cold oracle construction. It includes all
trial/gradient/receipt validation, transitions, owner cleanup, common-eight-
frequency endpoint work, model hashes and strict output metadata. The outer
driver stamps the complete measurement/controller return, including its wrapper
tail. Artifact storage, integrity checks and fresh offline truth/graph audits
follow that return; their work does not enter policy decisions or stand in for
algorithm work. Shared imports are reported once outside the per-call windows.

All stage clocks refer to the same true start. Stage one has an absolute ceiling
of 3.75 seconds; stage two/single-stage work has a 7.5-second ceiling. An early
native stop hands off immediately. Actual stage durations need not be equal,
and global setup/elapsed ages are never added as separate durations. A returned
state needs the actual complete API return at or before eight seconds. The
half-second reserve is not a guarantee; late returns stay visible and ineligible.

The standard SciPy L-BFGS-B settings retain the old tolerances and line search.
New 4096/2048 iteration limits are safety caps, replacing a known binding 200-
iteration control. Their nonbinding behavior must be checked from actual runs,
along with the native default maximum function evaluations and termination
messages. The bounded recorder changes storage and measurement, with paired
manufactured equivalence controls; it does not change the optimizer equations.

Each policy stage owns a new cold physics/oracle context. Only the last eligible
model crosses stages, with no fields, LU factors, receipts or optimizer history.
Within-owner paid reuse remains enabled. Joint-eight workflows obtain the
initial common objective from genuine paid initialization; low-start/original-
four workflows pay a separate full-eight baseline owner. A queried baseline
owner cannot become cold stage two. Final common metrics use public owner
validation, with legitimate reuse where available. Final owner counters are
summed once; observation preparation is separate. Unknown partial totals are
null rather than zero.

A late initial checkpoint cannot seed a handoff. Without a stage-one eligible
paid point, the workflow gets no initial-model fallback credit. A stage-two
budget stop without a new eligible point may retain the previous paid state;
the final common query is still paid. Failed methods remain ineligible. Model
selection uses the last chronological eligible point, without truth or a
best-common-objective checkpoint search. Half/quarter ratios are returned-final
diagnostics, never target stopping or first-passage times.

Only initial/final vectors are raw model witnesses. Intermediate scalar streams
retain the first eight and last eight records, all counts and ordered digests;
native callback invocations, written point records and retained samples are
distinct. Digests commit to omitted records; those records cannot be replayed
from samples. Full final counters and failures remain. The 24 MiB retained-output
envelope includes duplicate failure prefixes and the final receipt. Writes are
size-checked with a two-MiB final reservation; the physical guards remain
256 MiB initially and 64 MiB between runs. No wavefield history is written.

All 96 planned rows are saved before project arrays are read. Unexpected
infrastructure/source/input/disk errors stop after preserving the current
prefix; remaining rows stay unattempted. No automatic retry or case substitution
occurs. Numerical audit failures remain failed rows. The independent graph audit
shares NumPy/SciPy/SuperLU and is not a continuum or field-data validation.

Repetitions do not add geological samples. Nominal SEAM identities share an
ancestor; the eight slices are correlated development inputs. A useful returned
reconstruction advantage must be assessed against initialization, both strong
full-eight controls and actual consumed cost. A shared eight-second allocation,
passing controls or a complete archive alone does not establish that advantage.
