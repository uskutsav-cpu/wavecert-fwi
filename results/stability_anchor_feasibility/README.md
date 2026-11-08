# Standard stability-anchor endpoint feasibility

All 48 returned endpoints × four frequencies were included: 0 positive and 192 zero sufficient bounds. This does not imply actual singularity. The fixed-grid triangle-inequality formula is standard and conditional on valid anchors; ordinary-FP stored anchors are not formal enclosures.

See [the technical scope](../../docs/stability_anchor_feasibility.md). `fixture/` contains exact source/registration and primary-literature scope notes. `run_01/report.json` is the raw exploratory calculation. `prefreeze.json` predates execution. The independent review recomputes all 192 signs in 95-digit Decimal arithmetic; its initial two metadata-adaptation failures and corrected source are retained.

From the repository root, with NumPy installed and a fresh outer output:

```sh
python results/stability_anchor_feasibility/fixture/assess_anchor.py --repo . \
  --registration results/stability_anchor_feasibility/fixture/registration.json \
  --output ../anchor_feasibility_reproduction_02
```

No PDE, optimizer, BP array, new geological sample, useful controller or novelty finding is supplied by this diagnostic. AI development analysis; all four scientific goals remain open.
