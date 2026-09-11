from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from wavecert.physics.helmholtz import Helmholtz2D

Array = np.ndarray


@dataclass
class SmoothedHelmholtzSurrogate:
    """Controlled low-fidelity surrogate used to validate certificate logic.

    The surrogate solves the same discrete wave family using a smoothed and
    biased model.  It is *not* intended as a speed benchmark.  Its purpose is
    to create realistic state/Jacobian mismatch with an analytically available
    JVP and gradient.  A neural-operator adapter can later replace this class
    without changing the certifier.
    """

    physics: Helmholtz2D
    smoothing_sigma: float = 1.0
    model_scale: float = 1.02
    model_offset: float = 0.002

    def _smooth(self, x: Array) -> Array:
        grid = np.asarray(x, dtype=float).reshape(self.physics.shape)
        # mode='wrap' makes the convolution operator symmetric/self-adjoint,
        # which lets us write the exact surrogate gradient compactly.
        return ndi.gaussian_filter(grid, sigma=self.smoothing_sigma, mode="wrap").reshape(-1)

    def effective_model(self, m: Array) -> Array:
        return self.model_scale * self._smooth(m) + self.model_offset

    def _operator(self, m: Array, frequency_hz: float) -> sp.csr_matrix:
        return self.physics.operator(self.effective_model(m), frequency_hz)

    def state(self, m: Array, q: Array, frequency_hz: float) -> Array:
        return np.asarray(spla.spsolve(self._operator(m, frequency_hz), q), dtype=np.complex128)

    def jvp(self, m: Array, direction: Array, q: Array, frequency_hz: float) -> Array:
        u = self.state(m, q, frequency_hz)
        omega = 2.0 * np.pi * float(frequency_hz)
        dm_eff = self.model_scale * self._smooth(direction)
        rhs = (omega**2) * dm_eff * u
        return np.asarray(spla.spsolve(self._operator(m, frequency_hz), rhs), dtype=np.complex128)

    def objective_and_gradient(
        self,
        m: Array,
        q: Array,
        observed: Array,
        receiver_indices: tuple[int, ...],
        frequency_hz: float,
    ) -> tuple[float, Array, Array, Array]:
        u = self.state(m, q, frequency_hz)
        pred = self.physics.restrict(u, receiver_indices)
        residual = pred - np.asarray(observed)
        objective = 0.5 * float(np.vdot(residual, residual).real)
        rhs_adj = self.physics.inject_receivers(residual, receiver_indices)
        a = self._operator(m, frequency_hz)
        lam = np.asarray(spla.spsolve(a.conjugate().transpose(), rhs_adj), dtype=np.complex128)
        omega = 2.0 * np.pi * float(frequency_hz)
        g_eff = (omega**2) * np.real(np.conjugate(lam) * u)
        # d m_eff / d m = scale * S and S is self-adjoint for the chosen filter.
        gradient = self.model_scale * self._smooth(g_eff)
        return objective, gradient, u, residual
