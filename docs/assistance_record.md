# AI assistance record — 2026-11-07 development work

This record documents assistance; it is not a student declaration or a signed
mentor statement. The student must review it, correct omissions, and use the
current JSHS Statement of Outside Assistance to disclose the actual work.

Codex assistance in this development session included repository review,
literature/source lookup, mathematical/code checks, debugging, software changes,
test authoring, dataset preparation and provenance support, experiment execution,
statistical analysis support, and repository documentation. Parallel assistant
workers reviewed physics/performance, calibration/data integrity, and optimizer/
Devito correctness. Their work is AI assistance, not independent human review.

The assistant introduced implementation choices and exploratory comparison
configurations. These are not represented as student-originated scientific
decisions or student-approved preregistration. Machine-readable protocols state
their development status. Experiment results are generated from executable code;
null/negative results and historical artifacts are retained.

Verification includes executable tests, independent spectral/manufactured-
solution checks, finite differences, source/data/checkpoint hashes, and explicit
accounting of online versus offline exact physics. These checks support the
specific claims they cover; they do not establish complete physical realism,
scientific novelty, or a competition result.

The assistant did not obtain mentor approval, contact outside researchers, write
a competition abstract/paper/presentation script as the student's original work,
or determine the student's personal contribution. No such contribution or
approval should be inferred from repository activity.

Before submission, the student should independently explain the residual and
adjoint derivations, review the final research question/design, reproduce and
analyze the results, assess novelty against primary sources, draw their own
conclusions, and author the submitted content. Review the
[2026–2027 JSHS rules](https://www.jshs.org/wp-content/uploads/2026/08/JSHS-Core-Rules-of-Competition-2026-27.pdf),
especially the AI and outside-assistance sections.

## Additional assistance in the controlled strengthening phase

Codex designed and implemented the four-candidate raw/augmented and forward/JVP
comparison, executed its fixed budgets, diagnosed a disk-full interruption,
provided the bounded I/O-only recovery driver, and audited preserved prefix
identity and model selection. Codex designed the eight-candidate POD control,
trained its bases from train-only teacher fields, and performed validation
selection. These choices and mathematical work are AI assistance.

Codex derived/implemented the component coarse stability and physical-adjoint
certificate, DCT moment assembly, classical exact receiver controls, and scalar
then batched development implementations. It designed and executed the
validation, immutable freezes, five-model unused-index comparison, historical
transfer, runtime studies, and private feasibility probes. It authored code and
tests, performed statistical/vector/source audits, reviewed primary literature,
and generated research figures and technical development plans. Multiple AI
workers provided separate checks, not independent human reproduction.

The saved protocols explicitly distinguish development choices, prior inspected
test roles, known data batches, and later additions. They are not student-originated
preregistrations. Null findings, failed/restarted jobs, unknown whole-run training
timestamps, abstentions, poor neural generalization, and unfavorable runtimes
are preserved. No final competition submission has been authored here.

The student must determine which contributions they actually made, independently
verify and understand assisted work, and disclose the actual assistance. A
repository score cannot assess their ownership or oral defense.
