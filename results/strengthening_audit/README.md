# Separate development evidence audits

`audit-five-model-results.py` is an AI-assisted, read-only arithmetic/provenance audit of all 1,000 saved aggregate vectors and tables. Run from a full Git-history checkout:

```sh
python results/strengthening_audit/audit-five-model-results.py --repository . --output /tmp/five-model-audit.json
```

The original `five-model-audit.json` preceded a portability-only CLI edit to the audit script. Its recorded script hash therefore differs from the archived portable script. The separate `five-model-audit-portable.json` is the successful rerun with the archived script and matching hash. Neither modifies the experiment results. Both verify the 144 recorded production files against experiment commit 8b3c1f1 and current bytes; later additive source files do not replace those historical files.

The historical receipt records a separate saved-vector/arithmetic check, not an independently packaged PDE replay. The primary-literature JSON records a targeted source review, not novelty clearance. All reviews are AI assistance, not independent human review. See each receipt's limits.

## Completed batching audit

The separate AI reviewer’s byte-exact v1 audit and passing receipt are preserved. The readable v2 changes import sorting/formatting only; its non-import AST is identical and its separate rerun also passed. Both check all 288 calls, 6,912 block rows, 48 paired rows, 24 preflight comparisons, 288 saved vectors, 156 Git-bound source files, input identities, arithmetic/counter receipts and the fixed parent-weighted rank choice. Full per-block field/gradient arrays are not saved, so their parity relies on recorded preflight diagnostics and tests rather than a new field replay. Neither audit reruns the PDE or constitutes human reproduction.

```sh
python results/strengthening_audit/independent_batched_pod_raw_audit_v2.py --repo . --output /tmp/new-wavecert-batch-audit.json
```

Use a new receipt path. `verification_receipt.json` records the 454-test full local pass and all-green Linux CI on the actual frozen code, along with the original disk-full synthetic-fixture failure and cleanup-only rerun.
