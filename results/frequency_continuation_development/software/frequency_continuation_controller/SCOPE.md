# Provisional frequency-continuation controller

This is AI-authored engineering baseline software, developed after existing
diagnostics were inspected. Frequency continuation and restarted L-BFGS-B are
standard methods. This is not student preregistration, a final scientific
question/design, a new optimization method, recovery evidence or project GO.

`controller.py` has no project-data runner or truth-model input. The root owns
the separate proposed benchmark protocol/driver. Production source9f is not
edited. Settings are configurable;4096/2048 are proposed safety caps, not a
claim that caps or SciPy's unchanged15000 function-evaluation default cannot
bind. Native termination, cap/line-search messages and failure records remain.

The public API is:

- `load_native_api(repo, engine_run=..., engine_source_path=..., engine_source_sha256=...)`;
  source paths/hashes are checked explicitly, with no fallback engine.
- `PreparedProblem(physics, initial_model, blocks, receiver_indices)`;
  physics is a configuration dictionary containing shape, spacing,
  damping_width and damping_strength, with disabled factor caches.
- `prepare(context)` registers actual preparation-cost providers through
  `context.register_preparation_costs(name, provider)` and returns that problem.
  No preparation solver, fields, factors or truth is transferred into policy.
  If an existing helper only exposes its executed snapshot after returning,
  register that snapshot then; interruption inside it remains unknown work.
- `measure_controller_return(arm=..., prepare=..., api=..., settings=...,
  clock=..., attempt=..., progress=...)` returns payload, timing and runtime_gate.

Arms are joint8, low4_then8, restart8 and original4. All receive all eight
prepared frequency observations. The original4 arm deliberately optimizes a
smaller objective and is contextual; comparing it with eight-frequency arms
does not isolate algorithm effects. Joint8→joint8 shares the model-only cold
restart and absolute phase ceilings with low4→all8. These are ceilings, not
equal actual stage durations; early native termination hands off immediately.

The single actual start precedes preparation. A fresh AnchorClock is passed
only to each engine, whose first call sets its ledger start to that same true
global start. The default first/stage-final ceilings are3.75/7.5 seconds; the
caller stamps `execute_timed` returning its complete payload at the default8s
gate. Preparation, constructors, transitions, source/receipt validation,
public common-eight-frequency queries, cost snapshots, model hashes, cleanup
and strict metadata work are inside that API. Measurement classification,
artifact persistence and independent truth/objective audits occur afterward.
Global ages/setup timestamps and nested snapshots are not additive costs.

Every stage/metric gets a distinct cold native owner and fresh native physics
template. Only accepted model bytes cross stages. The transparent delegate
stores authentic first/current receipt headers, never extra receipt, LU or
field handles. Joint8/restart8 obtain their common initial objective from the
genuine first engine initialization. Low4/original4 pay a separate full8 metric
owner. An owner queried for a metric is never called a cold stage2 owner.
Final full8 assessment invokes public evaluate/accept validation, using a
legitimate full8 owner and its legal within-owner cache where available.

Only the last chronological within-budget checkpoint may supply a model.
Its SHA and objective must match the authentic native current receipt and
engine final model/objective. A late initial checkpoint cannot seed handoff;
a separate metric receipt cannot provide a free initial fallback. If stage2
has no eligible point after a budget stop, only the previous stage1 paid model
may remain, followed by a paid full8 final query. A failed method is ineligible
even when its last paid model is retained for diagnostics. Interruptions retain
all stage/owner slots, available engine prefix and counters, then propagate.
An unfinalized pending engine prefix is labeled separately and is never treated
as a completed API return; the engine does not query an oracle or clock merely
to manufacture that prefix. Outer failure-counter snapshots remain distinct.

Latest owner snapshots are summed once per distinct owner, separately from
preparation. Unknown aggregates are null with explicit known partial sums;
they are not reported as zero. A runtime gate is only necessary timing/API
evidence: full-study credit also needs root's terminal, source/input integrity,
artifact and offline independent-objective requirements.

The bounded engine is a separately reviewed measurement adaptation. Initial
and final vectors remain O(n); intermediate vectors are not retained. Its
head/tail scalar sampling, full event counts and ordered hash-chain summaries
must not be described as replaying all intermediate events. The controller
uses native_callback_invocations (all entries, including entry-guard halts)
for caps. native_callback_point_records counts actually written point records,
including equal/late records. native_callback_count is a labeled invocation
alias, never equated to complete point records or sampled-step length.
Eligible descriptor counts are explicitly retained
counts. Authentic initial and last eligible descriptors remain necessary.

The first20 controller controls, including four tiny4×5 real native-oracle
integrations with abstract clocks, passed before that metadata adaptation;
their exact sources/logs/receipt are preserved in history/first20_controls.
Manufactured controls are software evidence, not project optimization,
performance, physical acceptance, independent geological sites or student
scientific conclusions. No project arrays/trajectories have been executed by
this controller during staging.
