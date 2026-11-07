# WaveCert-FWI: research assessment and route to a stronger project

This is an AI-assisted technical assessment and planning record, not a JSHS
paper, abstract, presentation script, student conclusion, or eligibility ruling.
Scores below are subjective project-readiness judgments, not official judging
scores, national rankings, or probabilities of winning.

## Current assessment

**About 7.5/10 for research readiness**, improved from the initial roughly 6.5/10
assessment. The repository is substantially more defensible. A national-winning
result remains unestablished; polishing the repository alone cannot create one.

| Area | Judgment | Evidence / remaining issue |
|---|---:|---|
| Problem and motivation | 9/10 | Wrong surrogate derivatives can damage inverse problems; useful and clearly testable |
| Mathematical and numerical care | 8.5/10 | Explicit operator assumptions, adjoint/FD checks, availability and fallback; no formally verified floating-point enclosure |
| Reproducibility and experimental integrity | 9/10 | Frozen protocols/source/data, retained failures, raw vectors/tables, classical/exact controls, separate audits; independent human reproduction still needed |
| Demonstrated practical benefit | 6/10 | Excellent classical gradient alignment and useful pointwise certification; no matched-cost reconstruction advantage |
| Novel contribution | 6.5/10 | Specific implementation/evidence is defensible; core gradient-bound/coarse/POD ideas have direct published precedents |
| Physical realism and breadth | 5.5/10 | Small discrete Dirichlet system and generated velocities; one OOD parent, no certified time-domain/field result |
| Student ownership and oral defense | Not assessed | Code history cannot establish the student's contribution or ability to explain the work |

The overall judgment is qualitative, not an arithmetic average. National
placement cannot be inferred without the competing projects and judges.

## What the completed work actually supports

The controlled FNO training comparison retained all four candidates, fixed budgets,
validation histories, and a disk-full restart with a documented, I/O-only recovery.
Smoothing plus JVP supervision improved development validation, but the selected
model still had mean aggregate gradient cosines 0.0558 and -0.0118 on the two
previously unused-index cohorts. Making the neural model larger is not yet the
best-supported next action.

The train-only classical POD baseline was far stronger: mean cosines 0.99745 and
0.99953, with true descent on all 200 original model slots. Coarse correction
increased these to 0.99866 and 0.99975. The certificate accepted 168/200 directions,
with no observed false certifications and 32 abstentions. All five fixed models,
all 4,000 source/frequency blocks, and negative outcomes are retained. The two
batches were known during development; unused indices are not new geological
families or measured field data. No finite-step/inversion benefit follows from
these direction measurements.

The scalar corrected certificate costs a median 68.30 ms including setup versus
12.57 ms for exact objective/full gradient in the fixed small-grid study. Raw POD
costs 13.66 ms. Different information tasks are being compared, and zero online
fine solves is not a speedup. A separately frozen batching/rank development
comparison is used to measure whether implementation overhead can be reduced;
it cannot retroactively change the earlier confirmation protocol.

The strongest current scientific direction is a careful comparison of classical
reduced models and residual-certified physical directions, with neural methods
retained as meaningful negative controls. Neural acceleration remains a hypothesis.

## Milestones that would materially raise the assessment

| Priority | Work and decision | Evidence required to call it successful |
|---|---|---|
| 1 | Define a precise contribution against the closest published methods | A method-by-method comparison of assumptions, bounds, online/offline cost, and what changes here; a human numerical-analysis reviewer identifies a defensible specific contribution |
| 2 | Establish a cost regime where the certificate is useful | Fixed mesh/source/frequency/rank controls with identical numerical objectives, all setup/cache/fallback costs counted; speed ratios and failure/availability rates reported together |
| 3 | Demonstrate an inversion outcome | Same initialization, constraints, stopping target and exact acceptance policy; compare equally optimized exact FWI, raw POD, corrected POD, and fixed-budget heuristics at matched total wall time and at a fixed target |
| 4 | Obtain meaningful independent geological replication | Train/validation/confirmation separation by actual parent; several independent OOD parents; choose sample size from a declared precision/power objective before outcomes, not from slice count |
| 5 | Extend the operator scope deliberately | Derive and test the stability/adjoint result for the actual intended boundary treatment/backend; retain explicit unavailability and compare against an independently checked exact solver |
| 6 | Establish student ownership and independent reproduction | Student independently reproduces a decisive result, explains the assumptions and failed controls, receives substantive human criticism, and writes their own conclusions/submission with actual assistance disclosed |

Do priorities 1–3 before another broad neural hyperparameter search. If the
certificate loses to an optimized exact method at every relevant tested scale,
report that boundary and revise the research question. A strong negative
computational study can still be valuable; it cannot support an acceleration claim.

### A concrete cost experiment to design next

Separate mesh refinement on a fixed physical domain from increasing domain size
or physical frequency. A fixed coarse rank can remain useful under refinement
while failing when physical wave content grows. Do not label the former
field-scale scalability. Keep sources/receivers at matched physical coordinates,
train POD only on declared training snapshots, and pay fitting/assembly costs.
Measure exact, raw reduced, corrected/certified, and fallback work separately.
Use a predeclared rank/availability rule; retain every unavailable case.

Then evaluate full inversion with paid true-objective acceptance. Reuse accepted
forward states and LU factors in the exact baseline as well. Count rejected steps,
line searches, derivative checks, certificate setup, and exact fallbacks. Record
both time-to-target and final quality under the same time budget. A cosine close
to one is a diagnostic, not this endpoint. If a physical corrected proposal is
not the derivative of the reduced objective, do not use a line search that assumes
that identity.

### A concrete generalization experiment to design next

Choose independent parent models and geological shifts before evaluating them.
Separate model selection, statistical calibration and final confirmation roles.
Treat slices, sources, frequencies and repeated timings as correlated measures
within parents. Predeclare the primary outcome, an uncertainty method appropriate
to the sampling design, and the tolerated false-certification risk. Keep OOD
results descriptive when exchangeability is unsupported. Save all outcomes,
including abstentions and exact recovery costs.

Stress cases should include a range where raw approximations become unreliable;
otherwise 200/200 raw-POD descent cannot establish that certification improves
safety in practice. Define these stresses on development data, then freeze a new
confirmation design. Do not select stresses after seeing final test failures.

## Questions the student should be able to answer

1. What does the residual bound prove about one direction, and what does it not
   prove about a finite step, the entire inversion, or the continuous wave equation?
2. Why does accurate wavefield prediction not guarantee an accurate gradient?
3. Why did POD beat the neural models, and what data/protocol could falsify that
   explanation rather than merely repeat the observation?
4. Why do real coarse bases and complex symmetry matter in this derivation?
5. What does an unavailable certificate do, and where is its actual fallback cost?
6. What is new relative to Qian et al., and what is established prior work?
7. Which choices and work were yours, which were assisted, and which conclusions
   have you independently verified?

The official [2026–2027 JSHS rules](https://www.jshs.org/wp-content/uploads/2026/08/JSHS-Core-Rules-of-Competition-2026-27.pdf)
include research problem, scientific thought, design, conclusions and communication
in judging, and require disclosure of AI/outside assistance. They restrict
AI-authored submission prose. Use this record as technical support; the student
must author their actual paper/abstract and represent contributions accurately.
See [assistance_record.md](assistance_record.md).

## What would justify a near-10 assessment

A clearly bounded and defensible contribution, an independently reproducible
positive practical outcome or unusually informative negative finding, realistic
operator evidence, meaningful independent confirmation, transparent limitations,
and a student who can explain and defend every important decision. More tests,
more features, or a higher self-assigned score cannot substitute for those results.
