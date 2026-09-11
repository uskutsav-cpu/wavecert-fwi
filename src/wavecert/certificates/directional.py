from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from wavecert.physics.helmholtz import Helmholtz2D
from wavecert.surrogates.base import WavefieldSurrogate

Array = np.ndarray
StabilityMode = Literal["beta", "receiver", "directional"]


@dataclass(frozen=True)
class DirectionalCertificate:
    """A posteriori bound for one objective directional derivative.

    Three nested stability modes are supported:

    ``beta``
        Uses only the global stability constant ``beta = sigma_min(A)``.
    ``receiver``
        Replaces the outer ``1/beta`` receiver estimate by the sharper
        ``alpha = ||P A^{-1}||``.
    ``directional``
        Additionally uses ``kappa(v) = ||P A^{-1} diag(v) A^{-1}||`` to bound
        the receiver-space tangent error directly.

    The latter two are reference-grid tools in the current repository because
    their exact computation forms a dense inverse. They are useful for testing
    the *mathematical certificate* before a scalable stability estimator is
    introduced.
    """

    stability_mode: str
    beta: float
    receiver_resolvent_norm: float | None
    directional_tangent_resolvent_norm: float | None
    primal_residual_norm: float
    tangent_residual_norm: float
    state_error_bound: float
    tangent_error_bound: float
    receiver_state_error_bound: float
    receiver_tangent_error_bound: float
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
    stability_mode: StabilityMode = "beta",
    inverse_operator: Array | None = None,
    receiver_resolvent_matrix: Array | None = None,
    receiver_resolvent_norm: float | None = None,
    directional_tangent_resolvent_norm: float | None = None,
    validate_exact: bool = False,
) -> DirectionalCertificate:
    r"""Certify the surrogate directional derivative for one source/frequency.

    Let ``A u = q`` and ``A du = omega^2 diag(v) u``. For surrogate state
    ``u_hat`` and tangent action ``du_hat`` define

    ``r_p = A u_hat - q``
    ``r_t = A du_hat - omega^2 diag(v) u_hat``.

    The baseline stability estimate is

    ``||u-u_hat|| <= ||r_p|| / beta``

    and

    ``||du-du_hat|| <= (omega^2 ||v||_inf ||u-u_hat|| + ||r_t||) / beta``.

    The receiver-aware modes sharpen only the quantities actually needed by
    the FWI directional derivative.  Writing ``P`` for receiver restriction,

    ``||P(u-u_hat)|| <= alpha ||r_p||``, ``alpha = ||P A^{-1}||``.

    For ``stability_mode='directional'`` the exact discrete identity

    ``P(du-du_hat) = -P A^{-1} r_t - omega^2 P A^{-1} diag(v) A^{-1} r_p``

    yields

    ``||P(du-du_hat)|| <= alpha ||r_t|| + omega^2 kappa(v) ||r_p||``.

    These receiver-space errors induce a rigorous bound on

    ``D Phi(m)[v] = Re <P u-d, P du>``.
    """

    if stability_mode not in {"beta", "receiver", "directional"}:
        raise ValueError(f"unknown stability_mode={stability_mode!r}")

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
    rp_norm = float(np.linalg.norm(r_primal))
    rt_norm = float(np.linalg.norm(r_tangent))

    if beta is None and stability_mode != "directional":
        beta = physics.smallest_singular_value(m, frequency_hz)
    if beta is None:
        beta_value = float("nan")
        eta_u = float("nan")
        eta_du = float("nan")
    else:
        beta_value = float(beta)
        if beta_value <= 0:
            raise ValueError("beta must be positive")
        eta_u = float(rp_norm / beta_value)
        eta_du = float(
            ((omega**2) * np.linalg.norm(direction, ord=np.inf) * eta_u + rt_norm)
            / beta_value
        )

    sampling_norm = physics.receiver_sampling_norm(receiver_indices)
    alpha = receiver_resolvent_norm
    kappa = directional_tangent_resolvent_norm

    if stability_mode == "beta":
        eta_pu = sampling_norm * eta_u
        eta_pdu = sampling_norm * eta_du
    else:
        if alpha is None:
            alpha = physics.receiver_resolvent_norm(
                m,
                frequency_hz,
                receiver_indices,
                inverse_operator=inverse_operator,
                receiver_resolvent_matrix=receiver_resolvent_matrix,
            )
        alpha = float(alpha)
        eta_pu = alpha * rp_norm
        if stability_mode == "receiver":
            eta_pdu = alpha * (
                rt_norm + (omega**2) * np.linalg.norm(direction, ord=np.inf) * eta_u
            )
        else:
            if kappa is None:
                kappa = physics.directional_receiver_tangent_resolvent_norm(
                    m,
                    frequency_hz,
                    receiver_indices,
                    direction,
                    inverse_operator=inverse_operator,
                    receiver_resolvent_matrix=receiver_resolvent_matrix,
                )
            kappa = float(kappa)
            eta_pdu = alpha * rt_norm + (omega**2) * kappa * rp_norm

    p_u_hat = physics.restrict(u_hat, receiver_indices)
    p_du_hat = physics.restrict(du_hat, receiver_indices)
    r_hat = p_u_hat - np.asarray(observed)
    d_hat = float(np.real(np.vdot(r_hat, p_du_hat)))

    # Expand r = r_hat + P e_u and du = du_hat + P e_du:
    # |dPhi-dPhi_hat|
    #   <= ||P e_u|| (||P du_hat|| + ||P e_du||)
    #      + ||r_hat|| ||P e_du||.
    eta_dir = float(
        eta_pu * (np.linalg.norm(p_du_hat) + eta_pdu)
        + np.linalg.norm(r_hat) * eta_pdu
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
        stability_mode=stability_mode,
        beta=beta_value,
        receiver_resolvent_norm=alpha,
        directional_tangent_resolvent_norm=kappa,
        primal_residual_norm=rp_norm,
        tangent_residual_norm=rt_norm,
        state_error_bound=eta_u,
        tangent_error_bound=eta_du,
        receiver_state_error_bound=float(eta_pu),
        receiver_tangent_error_bound=float(eta_pdu),
        surrogate_directional_derivative=d_hat,
        directional_error_bound=eta_dir,
        certified_upper_derivative=upper,
        certified_descent=bool(upper < 0.0),
        exact_directional_derivative=exact_dd,
        realized_error=realized,
        effectivity=effectivity,
    )
