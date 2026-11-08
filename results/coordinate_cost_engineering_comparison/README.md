# Fully paid paired coordinate engineering comparison

Read the [technical report](../../docs/coordinate_cost_engineering_comparison.md).
All 48 fresh runs return within the same 8-second cap. Both native m and
physical v still worsen velocity error relative to initialization on the same
five of eight reused cases. This is an AI-authored development comparison, not
scientific confirmation, a new method, a certificate win or a student submission.

- `run_02/` is the complete fresh 48-run study. Its complete receipt SHA256 is
  `2e7b62662a9a0342842226376c78b3f7a6f70b36d24d15b7b7419599602160d7`.
  Engine status is 43 budget stops and 5 native iteration-limit stops; no
  optimizer convergence is claimed. Initial/final native and optimizer vectors
  are retained, with explicit coordinate tags. Intermediate vectors are hashes.
- `post_audit_02/report.json` passes 4,523 checks / 672 numerical comparisons,
  with zero issues. It verifies all 48 returned rows, 24 pairs and 16 three-
  repetition groups. Its SHA256 is
  `e494b94086608c0cd2ab79dfc4a2b5c09fe04a9b22570fad504fcaf759472dcf`.
- `run_01_invalid/` preserves the first attempted schedule, with 14 offline
  checker failures, 1 interrupted policy and 33 unattempted rows. It has zero
  primary credit and was not resumed or overwritten. Raw cores/vectors for
  returned engines remain, with failure/counter completeness flags.
- `fixture_v2/` preserves exact V2 source, protocol, 11 pre-run controls and
  receipts. Its history contains the original V1 source/logs/review and a
  byte-exact first failure artifact; the root-level change receipt and patch
  document the exact driver-only import fix. The actual
  imported-auditor integration control uses manufactured data and reference
  solves; the ten older controls are pure mocks.
- `reviews/`, `freezes/` and `logs/` retain original and adverse source reviews,
  both registered attempts, the explicit root interrupt/terminal observations,
  fresh V2 GO and actual execution evidence. No settings/thresholds/budget were
  changed from method outcomes.
- `post_audit_source/` contains the separately frozen scalar collector and
  criteria. It reconstructs truth independently with scalar half-cell bilinear
  interpolation, and checks saved vectors/costs/timing/artifacts. It does not
  rerun the embedded PDE audit or independently measure the runtime.
- `figures/` contains the visually checked all-eight-case figure, all 48 rows,
  case summaries, plotting source and exact execution/format-history receipts.
  Layout revisions change tick labels only; their CSV bytes are unchanged.

The original 161-file production source remains `9f30afa9`. The standard
velocity wrapper is `61dc9b2c`, imported sparse auditor `807f4f2f`, V2 driver
`351b5271`, and byte-identical numerical protocol `f0306803`. Ordinary floating
point, shared library, background-machine, coordinate-dependent gtol and
inverse-bound caveats are explicit in the report and frozen scope.

Reproduce from the unchanged repository with a **fresh output outside it**:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python results/coordinate_cost_engineering_comparison/fixture_v2/run_coordinate_comparison.py \
  --execute-coordinate-engineering-comparison --repo "$PWD" \
  --output /tmp/wavecert-coordinate-new \
  --expected-protocol-sha256 f0306803ed34c03318e9a64d2848a8ce57900a6b4e5998c05d88d2bcbfa523f4 \
  --expected-driver-sha256 351b527170751d958939dfbc1f17b091770696ba043cf522be0db5f238e19c63
```

The default CLI is metadata-only. After an actual terminal exit, record its
exit code/status in a terminal-observation artifact before using the post-audit
CLI. Its default CLI lists frozen hashes and its explicit execution flag
requires that observation and a fresh output; no live benchmark may be audited.
Keep numerical workloads separate from the timing study. The manufactured CI
integration regression uses no project arrays or benchmark trajectories.

Underlying models inherit the [CC BY 4.0 source attribution](../../data/reference/subsurfacegen_parent_split/README.md)
and [ancestry correction](../../docs/ancestry_audit.md). Repeated slices and SEAM
IDs are not independent sites. AI authorship and student ownership limits are
recorded in the [assistance record](../../docs/assistance_record.md). All four
scientific requirements remain open.
