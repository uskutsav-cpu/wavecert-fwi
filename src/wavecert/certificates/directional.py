from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from wavecert.physics.helmholtz import Helmholtz2D
from wavecert.surrogates.base import WavefieldSurrogate

Array = np.ndarray


@dataclass(frozen=True)
class DirectionalCertificate:
    """A posteriori bound for one objective directional derivative."""

    beta: float
    primal_residual_norm: float
    tangent_residual_norm: float
    state_error_bound: float
    tangent_error_bound: float
    surrogate_directional_derivative: float
    directional_error_bound: float
    certified_upper_derivative: float
    certified_descent: bool
    exact_directional_derivative: float | None = None
    realized_error: float | None = None
    effectivity: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def certify_direction(
    *,
    physics: Helmholtz2D,
    surrogate: WavefieldSurrogate,
    m: Array,
    q: Array,
    observed: Array,
    receiver_indices: tuple[int, ...],
    frequency_hz: float,
    direction: Array,
    beta: float | None = None,
    validate_exact: bool = False,
) -> DirectionalCertificate:
    r"""Certify the surrogate directional derivative for one source/frequency.

    Let ``A u = q`` and ``A du = ω² diag(v) u`` be the exact state and tangent
    equations.  For surrogate state ``û`` and tangent action ``dû``, define

        r_p = A û - q
        r_t = A dû - ω² diag(v) û.

    With β = σ_min(A),

        ||u-û|| <= ||r_p|| / β

    and

        ||du-dû|| <= (ω² ||v||_∞ ||u-û|| + ||r_t||) / β.

    Because receiver restriction has operator norm at most one, these state and
    tangent bounds induce a computable bound on the error in

        DΦ(m)[v] = Re <P u-d, P du>.

    The returned ``certified_descent`` is true exactly when the upper bound on
    the *true* directional derivative is negative.
    """

    m = physics.validate_model(m)
    direction = np.asarray(direction, dtype=float).reshape(-1)
    if direction.size != physics.n:
        raise ValueError("direction has wrong size")

    u_hat = surrogate.state(m, q, frequency_hz)
    du_hat = surrogate.jvp(m, direction, q, frequency_hz)
    a = physics.operator(m, frequency_hz)
    omega = 2.0 * np.pi * float(frequency_hz)

    r_primal = a @ u_hat - q
    r_tangent = a @ du_hat - (omega**2) * direction * u_hat

    if beta is None:
        beta = physics.smallest_singular_value(m, frequency_hz)
    beta = float(beta)
    if beta <= 0:
        raise ValueError("beta must be positive")

    eta_u = float(np.linalg.norm(r_primal) / beta)
    eta_du = float(
        ((omega**2) * np.linalg.norm(direction, ord=np.inf) * eta_u + np.linalg.norm(r_tangent))
        / beta
    )

    p_u_hat = physics.restrict(u_hat, receiver_indices)
    p_du_hat = physics.restrict(du_hat, receiver_indices)
    r_hat = p_u_hat - np.asarray(observed)
    d_hat = float(np.real(np.vdot(r_hat, p_du_hat)))

    # |dΦ-dΦhat| <= ||r-rhat|| ||P du|| + ||rhat|| ||P(du-duhat)||
    # with ||r-rhat|| <= eta_u and ||P du|| <= ||P duhat|| + eta_du.
    eta_dir = float(
        eta_u * (np.linalg.norm(p_du_hat) + eta_du)
        + np.linalg.norm(r_hat) * eta_du
    )
    upper = d_hat + eta_dir

    exact_dd = realized = effectivity = None
    if validate_exact:
        exact_dd = physics.directional_derivative(
            m,
            q,
            observed,
            receiver_indices,
            frequency_hz,
            direction,
        )
        realized = abs(exact_dd - d_hat)
        if realized > 0:
            effectivity = eta_dir / realized
        else:
            effectivity = 1.0 if eta_dir == 0 else float("inf")

    return DirectionalCertificate(
        beta=beta,
        primal_residual_norm=float(np.linalg.norm(r_primal)),
        tangent_residual_norm=float(np.linalg.norm(r_tangent)),
        state_error_bound=eta_u,
        tangent_error_bound=eta_du,
        surrogate_directional_derivative=d_hat,
        directional_error_bound=eta_dir,
        certified_upper_derivative=upper,
        certified_descent=bool(upper < 0.0),
        exact_directional_derivative=exact_dd,
        realized_error=realized,
        effectivity=effectivity,
    )
