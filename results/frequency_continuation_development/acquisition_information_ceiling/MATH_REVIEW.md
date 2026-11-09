# Conditional acquisition ceiling

The frozen reference source constructs a real symmetric five-point Kronecker
stiffness and adds a complex diagonal mass/damping term. Therefore its matrix is
complex symmetric, `A.T == A` in exact arithmetic; Hermitian symmetry is not
required and is generally absent. Both the preparation helper and exact merit
oracle use the same nonzero point-source amplitude `1/h²`, with point receivers.
The producer changes only the preparation frequency list and block count to
eight frequencies and 48 blocks. Geometry stays fixed.

At any real model where all selected frequency matrices are nonsingular,
`A⁻¹` is transpose symmetric and the forward map is differentiable. Each
measurement has the form `F_(r,s)=e_rᵀ A⁻¹ e_s/h²`. Exchanging source and receiver
therefore leaves it unchanged. Differentiating gives
`DF[dm]_(r,s)=(ω²/h²)e_rᵀ A⁻¹ diag(dm) A⁻¹ e_s`; exchanging the endpoints leaves
this expression unchanged as well. These are plain-transpose identities, with
no complex conjugation.

All six distinct source points occur among the 20 distinct receiver points.
Their 15 unordered off-diagonal pairs each create one duplicate complex row per
frequency. There are at most `120-15=105` potentially distinct complex slots per
frequency: 84 source-to-other-receiver slots plus the 21 upper-triangular slots
of the six-by-six source submatrix. The metadata lists all 15 exact integer
pairs and all full-eight frequency offsets. This is a count of possible slots,
not a proof that any of their Jacobian rows are independent.

For eight frequencies, the real/imaginary Jacobian of the 2401 unrestricted real
squared-slowness parameters has rank at most `2*8*105=1680`, hence a real linear
kernel dimension of at least `2401-1680=721`. The original four-frequency
measurement submap has the analogous upper bound 840 and kernel lower bound
1561. The full experiment makes eight-frequency observations available; the
four-frequency number refers specifically to that measurement submap, not to
all information accessed by the contextual controller's metric queries.

This is an exact-arithmetic structural ceiling derived from the pinned operator
and declared geometry. No data, wavefield, Jacobian, numerical rank, singular
value, condition number, observed reciprocity error or propagated model was
computed. Floating-point solves and noisy measurements need not have bitwise
reciprocity. The ceiling supplies no attained rank, finite-perturbation null
direction, nonlinear uniqueness theorem, cause of the observed recovery errors,
or recovery guarantee. It is not a new FWI technique.

The ambient count is for 2401 unrestricted real variables. Active box constraints
alter feasible directions, and a restricted prior/parameterization can reduce
the effective dimension; ambient kernel vectors need not be feasible at an
active bound. A smooth invertible positive-velocity reparameterization alone
preserves Jacobian rank, while a lower-dimensional model prior can change the
parameter count. Different source/receiver/frequency acquisition can change the
ceiling. None of these observations selects a remedy or proves one will work.

All acquired residual terms and their original weights remain in the producer.
An information-count bound does not authorize dropping reciprocal measurements:
doing so can change loss weighting and noise treatment. No source, objective,
measurement, production setting or numerical criterion was changed here.

Independent source/math review is required before this advisory is marked clear.
This is AI engineering assistance after earlier outcomes, not the student's
final question, hypothesis, experimental design, authorship or competition
evidence. Physical acceptance and all four scientific goals remain incomplete.
