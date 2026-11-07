# Separate development evidence audits

`audit-five-model-results.py` is an AI-assisted, read-only arithmetic/provenance audit of all 1,000 saved aggregate vectors and tables. Run from a full Git-history checkout:

```sh
python results/strengthening_audit/audit-five-model-results.py --repository . --output /tmp/five-model-audit.json
```

The original `five-model-audit.json` preceded a portability-only CLI edit to the audit script. Its recorded script hash therefore differs from the archived portable script. The separate `five-model-audit-portable.json` is the successful rerun with the archived script and matching hash. Neither modifies the experiment results. Both verify the 144 recorded production files against experiment commit 8b3c1f1 and current bytes; later additive source files do not replace those historical files.

The historical receipt records a separate saved-vector/arithmetic check, not an independently packaged PDE replay. The primary-literature JSON records a targeted source review, not novelty clearance. All reviews are AI assistance, not independent human review. See each receipt's limits.
