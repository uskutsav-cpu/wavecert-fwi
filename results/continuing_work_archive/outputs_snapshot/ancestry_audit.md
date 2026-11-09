# Ancestry audit: model IDs are not independent sites

This metadata-only audit supersedes any interpretation of the existing
**model-ID parent isolation** as **original-ancestor or real-site isolation**.
The old split files remain disjoint by model ID, and historical receipts and
outcomes remain unchanged. Those facts do not establish independent geological
environments. No velocity, wavefield, seismic, checkpoint, or new model outcome
was loaded for this audit.

The complete machine-readable map is
[`parent_ancestry.csv`](/Users/swethasunilkumar/Documents/Codex/2026-11-07/wavecert-fwi/work/wavecert-fwi/results/new_ancestry_audit/parent_ancestry.csv), with
42 explicit model-ID rows. The accompanying
[`ancestry_audit.json`](/Users/swethasunilkumar/Documents/Codex/2026-11-07/wavecert-fwi/work/wavecert-fwi/results/new_ancestry_audit/ancestry_audit.json) records
input hashes, every family's membership/role, limitations, primary sources,
resource calculations and external-source access checks. Audit inputs are the
cached index Parquets and research manifests at the pinned SubsurfaceGen
revision `c612dc2255f6dc4a139bb667f12eaaedf61a8d3d`.

## What is actually available

| Released IDs | Count | Shared recipe or original ancestor | Supported interpretation |
| --- | ---: | --- | --- |
| `f3_042`–`f3_051` | 10 | F3-inspired procedural recipe | Different delivered volumes; not measured fields |
| `gom_142`–`gom_151` | 10 | GoM-inspired procedural recipe | Different delivered volumes; not independent sites |
| `fault_000`–`fault_004` | 5 | Procedural fault stress recipe | Synthetic stress realizations |
| `salt_canopy_000`–`salt_canopy_003` | 4 | Legacy `pysaltmodel` recipe | Procedural analogues; draw independence unverified |
| `seam_000`–`seam_011` | 12 | **One SEG SEAM Phase-I original model** | Subregions, not twelve original-earth-model samples |
| `penobscot_000` | 1 | Penobscot-inspired procedural recipe | One synthetic realization, irrespective of slice count |

Recipe/ancestor descriptions are from the authors' [paper, Section 3.2 and
Appendix C.1–C.6](https://arxiv.org/html/2605.30541v1). Two short primary excerpts
make the distinction explicit:

> “The twelve SEAM models are reused, not generated.”

Appendix C.6, first paragraph; the next paragraph describes twelve subregions
on a 4-by-3 grid. The individual grid positions are **not** recorded in the
cached Parquet, so this audit does not invent their locations or overlap.

> “The Fault setting is a stress test, not a field.”

Appendix C.3, first sentence. These two excerpts total 18 words from that paper.
Its Section 3.2 describes sampling around matched setting templates. The cached
index has no per-volume seed/parameter-draw identifiers; IID status of the
specific delivered IDs is therefore not verified by this audit.

The locally hashed metadata show:

- Official training has 4,096 slices from 41 IDs. Official ID testing has 100
  other slices from **the same 41 IDs**. Official OOD has 80 slices from one
  Penobscot ID. These are not 4,276 independent environments.
- Every one of the 42 IDs already occurs in the bundled research roles. There
  are **zero unused model IDs** in this release for fresh-parent confirmation.
- Collapsing only the known SEAM common volume leaves 31 base-volume clusters.
  **31 is not an established independent sample size.** Shared settings,
  templates and unverified draws remain relevant.
- Independently sampled generator realizations can support conditional
  synthetic-draw inference under a frozen distribution. Sharing a generator
  alone neither proves nor refutes their statistical independence. It does
  not make them independently sampled real-world sites.

### The SEAM ancestor crosses existing roles

| Existing research role | SEAM IDs from the same original model |
| --- | --- |
| Training | `003`, `006`, `007`, `010` |
| Model-selection validation | `005`, `011` |
| Calibration | `000`, `004` |
| ID testing | `001`, `002`, `008`, `009` |

This is a stronger relationship than the previous manifest audit checked.
Future ancestor-disjoint work must put **all twelve** in one role. The present
audit changes the interpretation of isolation; it does not erase observations,
alter old assignments, recompute intervals, or assert that every explicitly
conditional finite-pool statement in the old reports is invalid.

## New roles and sample precision

For a new physics study, treat all current 42 IDs, all their additional slices,
and the already exercised Marmousi-family binary as **legacy development or
historical transfer**. Retrieving a different slice or interpolating 24-by-24
arrays does not create a new geological ancestor. A new native-grid experiment
on those models can test physics and resolution, but cannot be relabeled fresh
site confirmation.

Before new payloads or outcomes, freeze a source/ancestor ledger with license,
source revision, original model identity, transformations, crops, generator
code/config/seed, and roles. Every slice, crop, source, frequency, initialization
and inversion iteration inherits its original ancestor's role. Reserve named
external models as separate source-family stress cases. For procedural studies,
use new independently drawn parameters/seeds from registered distributions;
do not count a fixed fitted center as a random confirmation draw. Keep whole
settings out when the target claim is unseen-setting transfer. Known settings
with new draws remain known-setting synthetic confirmation.

Use separate development/model-selection/calibration/confirmation streams.
Freeze the model and any statistical scale before confirmation; if a pilot is
used to tune anything, its parents become development and need replacement.
Do not pool heterogeneous source families into a site-IID failure estimate.
Report named benchmark cases descriptively and generated-draw results by
registered stratum. Group all known common original models before defining
the inference unit. Record duplicate outputs explicitly: independent RNG-draw
slots may remain valid draw units, while deduplicating a selected finite-pool
sample changes its sampling design and cannot retain the original pool size.

For planning only, zero failures in `n` genuinely IID model-event trials gives
the one-sided 95% binomial upper limit `1 - 0.05**(1/n)`:

| Independent trials | Best zero-failure upper limit |
| ---: | ---: |
| 4 | 52.7% |
| 8 | 31.2% |
| 12 | 22.1% |
| 20 | 13.9% |
| 30 | 9.5% |
| 59 | 5.0% |

Thus an eight-parent pilot can be useful engineering evidence but cannot
support a strong low-failure claim. A pointwise 5% upper target requires at
least 59 zero-failure independent trials per defined stratum. With a fixed
Bonferroni family of four strata, the analogous target requires 86 per stratum;
six require 94. These calculations are **not intervals for current results**
and do not supply an independence assumption. Slices, blocks and repeated
visits cannot replace the required original-model trials.

## Native-grid physics and cache feasibility

The pinned release's delivered 2-D slice grid is 619-by-1000 at 10 m; it is
already provider-smoothed/resampled data, not an unprocessed field truth. The
dataset card describes eighth-order space differences, a free surface,
60-cell sponge and fixed 1 ms computational steps. See its
[immutable dataset card](https://huggingface.co/datasets/subsurfacegen/field-scale-dataset/blob/c612dc2255f6dc4a139bb667f12eaaedf61a8d3d/README.md).
The current cache contains only 24-by-24 derived models; its raw slice payloads
were deleted. The 86 previously selected H5 files total 95.64 MiB compressed.
Do not download 3-D volumes or wavefield/shot collections to recover velocity.
A small, bounded velocity-slice cache is sufficient for a registered crop study.

The current frequency backend is a constant-density five-point Dirichlet
operator with diagonal damping. Its coarse proof is restricted to that exact
operator. The Devito adapter is a separate, more realistic time-domain acoustic
baseline with fixed time sampling and an edge-padding gradient pullback. It
currently uses `bcs="damp"`, does not request `fs=True`, and saves the full
forward history for gradients. Its existing Marmousi study resizes to a small
grid and declares 10 m spacing; it is not the original full physical extent.
The [Devito primary tutorial](https://www.devitoproject.org/examples/seismic/tutorials/01_modelling.html)
explains the acoustic equation and sponge treatment. Neither baseline is a
variable-density/elastic/anisotropic field-data validation, and the frequency
certificate cannot simply be transferred to Devito.

The local machine has 8 GiB RAM, Devito 4.8.23 and no installed `pyrevolve`.
One logical float32 saved history for 619-by-1000 plus 60 cells on every side
and 5,001 steps is about **15.42 GiB**, before other arrays/halos; float64 is
30.84 GiB. Free-surface padding details change that estimate, not the resource
conclusion. Full native adjoints need checkpoint/recomputation or larger RAM.
The installed upstream acoustic solver has a checkpointing branch; the current
WaveCert adapter does not use it. An additive adapter must be verified before
claiming support.

A controlled **native-release-resolution 128-by-256 or 96-by-192 crop at 10 m**,
bounded recording time and sequential shots is a feasible next physics test.
It remains a crop of its original ancestor. The larger crop's analogous
float32 five-second history is approximately 1.30 GiB with 40-cell all-side
padding, excluding other allocations. Register coordinates/time/band/physics
before outcomes. Require boundary/reflection and dispersion checks plus
adjoint dot/Taylor/finite-difference checks, including parameter padding.
Do not call an interpolation probe native geology or a full-field run.

Geometry alone also limits the existing proof. At a declared minimum velocity
of 1.5 km/s, positivity of its complement `gamma` requires at least 748 low
modes at 3 Hz, 1,344 at 4 Hz and 3,068 at 6 Hz on the full native 10 m rectangle.
For the 128-by-256 crop the floors are 35, 64 and 154. These are eigenvalue
counts, not model evaluations or sufficient certification conditions. A fixed
coarse96 method cannot be advertised as scaling unchanged to the full domain.

## Historical additional-source preflight, before acquisition

The table and registration details below retain the pre-acquisition state.
Since that snapshot, the fixed [BP2004 acquisition](/Users/swethasunilkumar/Documents/Codex/2026-11-07/wavecert-fwi/outputs/bp2004_acquisition.md)
has completed: three native crops and three stride views, one ancestor, original
terms/source headers retained and 84 retained-artifact checks passed. Actual
full-stream SHA256 values are now recorded in its receipt. There is no BP PDE or
method evaluation, and no additional-site inference. The earlier “no payload”
statement below describes the historical preflight, not the current acquisition.

| Candidate | Provenance/rights evidence | Role and bounded route |
| --- | --- | --- |
| BP 2004 exact velocity | [Original BP documentation](https://s3.amazonaws.com/open.source.geoscience/open_data/bpvelanal2004/2004_Benchmark_READMES.pdf), pp. 1–4 | First feasible new numerical ancestor: compressed velocity is 19,533,256 bytes by HEAD; freeze source/crops/terms first |
| SEG/EAGE Overthrust | [Publisher catalog/license](https://wiki.seg.org/wiki/SEG/EAGE_Salt_and_Overthrust_Models) reports CC BY 4.0 | Different original numerical model; official CD archive HEAD currently returns 403; no verified byte budget or payload yet |
| SEG/EAGE Salt | Same publisher catalog and license | Separate original model within that project; also 403; not an IID-site replicate of other salt models |
| Marmousi2 | [AGL catalog](https://www.agl.uh.edu/resources-data.php), Marmousi II section, currently labels “Level 3 - Limited” | Resolve original version/data grant before a new bundle; the existing adaptation repo's MIT does not verify upstream data rights |
| SEAM Phase I | [SEG selected open-data page](https://seg.org/seam/open-data/) | Already represented by the twelve SEAM IDs; another crop adds no new ancestor; direct package terms vary |

The existing small binary comes from the authors'
[SiameseFWI reproduction repository](https://github.com/DeepWave-KAUST/Siamese_FWI-pub),
whose README identifies Marmousi2 and Overthrust data. That is an adapted-data
route, not verification of the original data grant. No new rights claim is
made here. Potential archival mirrors must retain a documented chain to the
publisher and be verified separately; a mirror's description is not a new license.

### BP 2004 registration details

Official catalog-listed velocity URL:
`https://s3.amazonaws.com/open.source.geoscience/open_data/bpvelanal2004/vel_z6.25m_x12.5m_exact.segy.gz`.
HEAD returned 200 with `Content-Length=19533256`, byte ranges and ETag
`e2c4ad66a0553c9f108ffcf05506da82`. **No payload or SHA256 was verified**;
an ETag is not a SHA256. The future preparation must pin/check its full bytes
and preserve the original documentation and disclaimer.

BP's README requires BP/Frederic Billette credit, the Billette–Brandsberg-Dahl
2005 reference, retained documentation on redistribution, and a documented
modification notice. It describes a 2-D acoustic mixed-domain calculation.
The scalar velocity model is `(z,x)=(1911,5395)` at `(6.25,12.5)` m, spanning
67,425 m laterally and 11,937.5 m deep. Original shots number 1,348, separated
by 50 m; each has 1,201 receivers at 12.5 m, with offsets −15 km to zero and
source/receiver depth 12.5 m. Sampling is 2,001 points at 6 ms; dominant and
maximum usable frequencies are stated as 27 and about 54 Hz.

This is **not** the separate [2007 BP TTI benchmark](https://wiki.seg.org/wiki/2007_BP_Anisotropic_Velocity_Benchmark).
The 2004 README does not specify a VTI/TTI parameterization. Using its scalar
velocity for newly regenerated constant-density isotropic acoustic observations
would be a new controlled physics benchmark, not reproduction of its original
shots. Its left/center/right pieces and every crop remain **one composite
synthetic ancestor**, not several independent geological sites.

No implementation, old receipt, model outcome or timing was changed by this
audit. New confirmation needs genuinely new original ancestors or registered
independent procedural draws with explicitly synthetic limits.
