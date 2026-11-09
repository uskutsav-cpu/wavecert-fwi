# Standard velocity-coordinate exact baseline candidate

This is AI-authored software support for an established change of coordinates,
not novelty, a recovery guarantee, a new optimizer, a scientific question or a
student's final design. No project data, project-data optimizer steps or
benchmark outcomes are executed here. Tests use manufactured 5×6 native grids.
A two-step existing L-BFGS-B control checks gradients, strict paid-merit decrease,
velocity states, native-model exposure, ownership and serialization; a native
gradient failure checks the explicit failed status and retained paid costs.
All use a constant abstract clock and supply no runtime or recovery evidence.
A separate zero-step test isolates factory/initial-receipt compatibility.
The current validation record is `VALIDATION_RECEIPT_02.json`; the earlier
30-test `TEST_RECEIPT.json` and matching sources are preserved as history.

The existing paid engine accepts a structural oracle interface, so no runner or
engine edit is necessary. Construct a fresh `ExactMeritOracle`, then its
`VelocityMeritOracle`, **inside** the existing `oracle_factory`. The constructor
rejects a shared or already queried native oracle. Native merit, gradient and
acceptance calls still perform their ordinary ownership, source, geometry/data
and model validation. Each evaluation dispatches to the native oracle even on
a cache hit. The accepted receipt's LU/fields survive rejected trials under the
existing native rules. Native physics/oracle counter keys remain intact; the
additional `velocity_coordinates` branch records coordinate calls, entries and
bytes rather than invented physics solves. Actual factory/call wall time pays
the conversion, copying, hashing and validation work; no seconds are added by
formula. Native exceptions propagate with their already incurred counters.

The optimizer variable is physical velocity `v`, in explicitly declared units
consistent with the native length/time convention (normally km/s for native
kilometres and Hz). Unit labels do not rescale input values. The exact mapping
implemented in float64 is `m = np.power(v, -2.)`; the chain rules are
`g_v = -2*v**(-3)*g_m` and `d_m = -2*v**(-3)*d_v`. The direction helper only
maps coordinates; it performs no tangent solve and supplies no certificate.
Positive velocity bounds `[v_low,v_high]` map to reversed native bounds
`[v_high**(-2),v_low**(-2)]`. Fixed coordinates are allowed. Nonpositive,
nonfinite, complex, Boolean, malformed or unrepresentable transforms fail
without clipping. A derivative that cannot be represented as a finite negative
float64 value fails rather than becoming a spurious zero gradient.
Bound conversions are ordinary float64 transforms, without outward rounding;
an m→v→m endpoint can likewise differ by ULPs. A future paired protocol must
record its actual coordinate bounds rather than assume bitwise inverse boxes.

**`receipt.model`, every engine `checkpoint.model`, and `final_model` are
VELOCITY under this wrapper.** They must never be passed directly to the old
`matched_cost_pilot.budget_endpoints` or offline-quality helpers, which assume
squared slowness. `receipt.native_model` and `native_model_sha256` expose the
actual immutable native model paid for. `to_native_model(v)` supplies an
explicit counted conversion plus the velocity/native hashes and problem
fingerprint for a future endpoint adapter. Such an adapter must convert and
rehash native states before native-model quality evaluation, include any
required output conversions and provenance work before measuring its actual
API return, and retain fresh counters/timing after that work. The current
engine's finish snapshot does not automatically account for a future wrapper
around its return. Offline truth/quality audits remain after the timed API.
No benchmark adapter, timing protocol or experimental settings are supplied.

The parameterization/problem fingerprint binds mapping, adapter/native source
hashes, explicit units and native operator/survey/data identity. Distinct
velocity bytes have distinct coordinate receipts even if float64 maps them to
the same native model; genuine native fields may then be reused only within
that same owned policy. Pullbacks always use the receipt's velocity. Foreign,
copied, mutated or mismatched coordinate/native receipts fail closed.

Converting a requested native initial model with `v = 1/sqrt(m)` and then back
to `m = v**(-2)` can change its bytes. `start_from_slowness` returns velocity,
the actual converted native model, both hashes, the roundtrip flag and its
maximum absolute difference. It performs no ULP search or native-model
replacement. A future paired comparison can choose this common actual native
start for both arms and record both original and converted hashes. This code
does not claim bitwise equality with the original initial model or borrow its
receipt. The first issued velocity receipt's actual initial native hash remains
available as metadata without retaining its old LU factors indefinitely.

Changing coordinates changes optimization geometry and the meaning of step,
stationarity and solver tolerances. Existing m-coordinate numerical settings
are not automatically equivalent v-coordinate settings. This candidate makes
no new choices for them and has not been run on the reused validation cases.
No regularization, prior, certificate pullback proof or scientific contribution
claim is added. Project execution remains held for independent review, source
freeze and a separately controlled protocol. Production source9f is untouched.
