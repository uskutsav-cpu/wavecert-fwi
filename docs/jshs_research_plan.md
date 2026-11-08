# WaveCert-FWI: research assessment and route to a stronger project

This is an AI-assisted technical assessment and planning record, not a JSHS
paper, abstract, presentation script, student conclusion, or eligibility ruling.
Scores below are subjective project-readiness judgments, not official judging
scores, national rankings, or probabilities of winning.

## Current evidence and next work

The newer [fixed-band diagnostic](initial_linearization_diagnostic.md) finds
smaller linearization remainders at lower frequencies in seven of eight cases,
with the exception retained. This motivates a future standard continuation
control, not a demonstrated remedy or a new method. Its independent scalar
audit passes 1,600 checks. The [BP numerical record](bp_heterogeneous_verification.md)
adds 57 completed native calls, three separate NumPy probe forwards and six
derivatives. Numerical agreement is strong, while padding sensitivity remains
large and physical acceptance is null. The [simple anchor-reuse diagnostic](stability_anchor_feasibility.md)
is negative at all 192 existing endpoints/frequencies. These development
findings do not update the historical numeric rating.

The additional [48-run coordinate comparison](coordinate_cost_engineering_comparison.md)
passes its independent numerical audit and full return-time cap. Both coordinate
methods still worsen reconstruction relative to initialization on the same five
of eight reused cases. Together with the
[initial-direction and analytical wave diagnostics](recovery_and_physics_diagnostics.md),
this makes the remaining priorities concrete:

1. Establish useful reconstruction against initialization and strong exact
   controls, with every cost and failure retained. Parameterization alone did
   not resolve this development fixture. Any final scientific explanation,
   prior/acquisition choice and confirmation design must be the student's own.
2. Validate heterogeneous forward/adjoint physics and physical sampling before
   a larger inverse-method claim. The bounded BP cropped-domain check now
   verifies its discrete forward/gradient implementation, while substantial
   padding sensitivity leaves physical accuracy unresolved. All three crops
   share one ancestor and do not reproduce the original surface survey.
3. Delimit a contribution beyond the existing adjoint/coarse/POD/certification
   literature, then test a falsifiable benefit. More standard components alone
   do not establish novelty.
4. Reproduce the decisive scientific result independently and demonstrate the
   student's own interpretation, decisions and contribution account. Separate
   AI audits and manufactured CI controls support verification but do not
   establish those human requirements.

All four requirements remain open. The older ratings below are preserved as
historical assessments and should not be used as an updated readiness score.

## Historical assessment before the continuing scientific phase

**About 7.5/10 for research readiness**, improved from the initial roughly 6.5/10
assessment. The repository is substantially more defensible. A national-winning
result remains unestablished; polishing the repository alone cannot create one.

This historical score predates the [ancestry correction](ancestry_audit.md) and
the continuing scientific phase. It must not be read as evidence that shared
SEAM subregions are independent original models, that the method improves
matched-cost inversion, or that student ownership has been established. The
[continuing research record](continuing_research.md) tracks those open issues.
Its implementation benchmarks are provisional AI-authored development tools;
the final scientific decisions must originate with the student under the
[current JSHS rules](https://www.jshs.org/wp-content/uploads/2026/08/JSHS-Core-Rules-of-Competition-2026-27.pdf).

The later [360-run comparison](paid_merit_engineering_benchmark.md) now supplies
negative practical evidence: exact L-BFGS-B meets both targets in 48/48 attempts,
the certified arm in 30/48, and certified internal velocity error is worse than
initialization on 7/8 cases. Independent raw initial-direction replay improves
software credibility; it does not repair those outcomes. The historical 7.5
score is not an updated national-readiness rating. No new 10/10 score or national
placement is justified. The next scientific priority is a student-owned
interpretation of the fit-versus-recovery gap and a defensible contribution,
with the strong controls and initialization baseline retained.

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
all 4,000 model-block evaluations (800 distinct source/frequency blocks), and negative outcomes are retained. The two
batches were known during development; unused indices are not new geological
families or measured field data. No finite-step/inversion benefit follows from
these direction measurements.

The scalar corrected certificate costs a median 68.30 ms including setup versus
12.57 ms for exact objective/full gradient in the fixed small-grid study. Raw POD
costs 13.66 ms. Different information tasks are being compared, and zero online
fine solves is not a speedup. A separately frozen batching/rank development
comparison is complete. With setup included, the batched corrected method at coarse rank 96 costs 42.51 ms
versus 65.99 ms scalar and 12.06 ms exact under the fixed equal-parent score:
a 1.55× implementation gain, still 3.52× slower than exact. The cheaper coarse 64
variant certifies 0/8 and fails the fixed eligibility rule. Raw batched POD costs
4.54 ms (2.66× faster than exact for its approximate-gradient task). These are
reused validation outcomes, not matched-accuracy inversion results or a change
to the earlier confirmation protocol.

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

The historical technical recommendation was to investigate priorities 1–3 before
another broad neural hyperparameter search. If the
certificate loses to an optimized exact method at every relevant tested scale,
report that boundary and revise the research question. A strong negative
computational study can still be valuable; it cannot support an acceleration claim.

### Historical AI-developed cost-study alternative

This and the next alternative document AI-generated technical planning for
critique. They are not a student's final experimental design. The student-origin
question, prediction, comparison and contribution account remain pending;
subsequent approval or paraphrasing does not establish original ownership.

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

### Historical AI-developed generalization-study alternative

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


## Updated recovery gate after the full comparison

Frequency continuation is not promoted as the recovery remedy: it worsens the initial velocity model in 7/8 case medians, despite substantial common-eight data-fit reduction. The next work must establish a recovery-oriented assumption or acquisition with physically justified information content, compare strong classical controls at fully paid cost, and retain initialization and every adverse case. The conditional reciprocity/dimension ceiling is a source/math diagnostic, not a new method or proof of the observed cause.

Physical acceptance remains open after substantial BP padding sensitivity; additional ancestors require verified source, geometry and transformation provenance. The newly quarantined Overthrust demo is lossy and has not been decoded or validated. A distinct useful contribution beyond prior certification and the student's own question, interpretation, defense and writing are still required. The historical 7.5 assessment is not an updated national-readiness score; no 10/10 or winning rank is established.
