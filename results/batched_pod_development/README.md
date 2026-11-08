# Completed validation-only batching/rank development

Source 3f569eb8d104e416dff04567b0195904818c76a6, production-source SHA b403390e70e83b086c509fe3f43289008af689b6e28b5016a54d585bb985edec, protocol a53600eff96f128193f813c077da29235d185fca473a24e060ca8fe75334ad1f. All 288 calls, 6,912 block rows and 24 preflight comparisons complete; source and inputs unchanged. All checks passed. See `summary.json`, the complete tables/vectors/cost receipts, and `docs/batched_pod_development.md`.

The score averages six-visit slice-median total cold setup+query times within each parent, then across four parents. Exact 12.06 ms; raw scalar 12.96 ms; raw batch 4.54 ms; corrected scalar96 65.99 ms; corrected batch96 42.51 ms; corrected batch64 27.93 ms. Corrected96 certifies 8/8 in every repetition, corrected64 certifies 0/8 while all bounds remain available/covering. Its faster times do not override failed eligibility. The fixed rule recommends batch96 for future development only; no default deployment/new held-out evaluation.

The batch corrected method is 1.55× faster than scalar but still 3.52× slower than exact. Raw batched POD is 2.66× faster than exact for its different approximate-gradient task. No equal-accuracy, full-inversion, time-to-target or field-scale speedup is claimed. Reused 8 slices / 4 parents and 6 repetitions are descriptive development evidence, not additional independent samples. Ordinary-floating-point proof limitations remain.

A separate saved-vector/raw-table/source audit passed and is archived in `results/strengthening_audit`. It verifies the declared fixed selection and all 156 historical/current source blobs without replaying the PDE.
