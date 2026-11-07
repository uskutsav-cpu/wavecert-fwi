"""Matrix-column execution of the frozen spectral coarse operations.

No stability norm or inequality is changed. FFT axes are only the two spatial
axes; source columns never become a transform dimension. The exact same
coarse residual is reused for its diagnostics and oriented solve.
"""

from __future__ import annotations

import numpy as np
import scipy.fft as fft

from wavecert.surrogates.pod_batch import REFERENCE_SOURCE_SHA256  # noqa: F401


def column_norms(values):
    """Stable Euclidean norms per column, including complex columns."""
    magnitude = np.abs(values)
    scale = np.max(magnitude, axis=0)
    safe = np.where(scale == 0, 1., scale)
    with np.errstate(over="ignore", invalid="ignore"):
        return scale*np.sqrt(np.sum((magnitude/safe[None, :])**2, axis=0))


class BatchedCoarseOperations:
    def __init__(self, context, dispatches):
        self.context, self.engine, self.dispatches = context, context._engine, dispatches

    def _dispatch(self, name):
        self.dispatches[name] = self.dispatches.get(name, 0)+1

    def _matrix(self, values, rows):
        raw = np.asarray(values)
        if raw.ndim != 2 or raw.shape[0] != rows or raw.shape[1] < 1 or not np.all(np.isfinite(raw)):
            raise ValueError(f"batched field must be finite with shape ({rows}, n_columns)")
        return raw

    def project(self, values):
        values = self._matrix(values, self.context.n)
        columns = values.shape[1]
        self.engine._count("coarse_projection_calls", columns)
        self.engine._count("dst_I_real_transforms", columns*(2 if np.iscomplexobj(values) else 1))
        self._dispatch("coarse_projection_dispatches")
        self._dispatch("dst_I_dispatches")
        tensor = values.reshape(*self.engine._shape, columns)
        transformed = fft.dstn(tensor, type=1, axes=(0, 1), norm="ortho", workers=1)
        modes = self.engine._modes
        return transformed[modes[:, 0]-1, modes[:, 1]-1, :]

    def lift(self, coefficients):
        coefficients = self._matrix(coefficients, self.context.rank)
        columns = coefficients.shape[1]
        self.engine._count("coarse_lift_calls", columns)
        self.engine._count("idst_I_real_transforms", columns*(2 if np.iscomplexobj(coefficients) else 1))
        self._dispatch("coarse_lift_dispatches")
        self._dispatch("idst_I_dispatches")
        tensor = np.zeros((*self.engine._shape, columns), dtype=coefficients.dtype)
        modes = self.engine._modes
        tensor[modes[:, 0]-1, modes[:, 1]-1, :] = coefficients
        return fft.idstn(tensor, type=1, axes=(0, 1), norm="ortho", workers=1).reshape(self.context.n, columns)

    def decompose(self, values):
        values = self._matrix(values, self.context.n)
        rb = self.project(values)
        components = np.stack((self.norm_columns(rb), self.norm_columns(values-self.lift(rb))))
        return rb, components

    def norm_columns(self, values):
        self._dispatch("column_norm_dispatches")
        return column_norms(values)

    def component_norms(self, values):
        return self.decompose(values)[1]

    def apply_operator(self, values, *, adjoint=False):
        values = self._matrix(values, self.context.n)
        columns = values.shape[1]
        values = values.reshape(*self.engine._shape, columns)
        with np.errstate(over="ignore", invalid="ignore"):
            out = 4*values.astype(np.complex128)
            out[1:, :, :] -= values[:-1, :, :]
            out[:-1, :, :] -= values[1:, :, :]
            out[:, 1:, :] -= values[:, :-1, :]
            out[:, :-1, :] -= values[:, 1:, :]
            out /= self.engine._config["spacing"]**2
            potential = -self.context.omega**2*self.context._model+1j*self.context.omega*self.engine._damping
            if adjoint:
                potential = potential.conjugate()
            out += potential.reshape(*self.engine._shape, 1)*values
        self.engine._count("full_operator_applications", columns)
        self._dispatch("full_operator_dispatches")
        if not np.all(np.isfinite(out)):
            raise ValueError("nonfinite batched stencil application")
        return out.reshape(self.context.n, columns)

    def solve(self, rhs, *, adjoint=False, category="coarse_rhs"):
        self._dispatch("coarse_solve_dispatches")
        return self.context.solve_coarse(rhs, adjoint=adjoint, category=category)

    def correct(self, field, rhs, *, adjoint=False):
        residual = self.apply_operator(field, adjoint=adjoint)-rhs
        response = self.solve(self.project(residual), adjoint=adjoint, category="coarse_correction_rhs")
        corrected = field-self.lift(response)
        return corrected, self.apply_operator(corrected, adjoint=adjoint)-rhs

    def oriented_error_bounds(self, rb, components):
        response = self.norm_columns(self.solve(rb, category="coarse_oriented_rhs"))
        c = self.context
        eb = (response+c.k*components[1]/c.gamma)/(1-c.tau)
        return np.stack((eb, (components[1]+c.ell*eb)/c.gamma))

    def oriented_adjoint_error_bounds(self, rb, components, eta_receiver_state, constants):
        response = self.norm_columns(self.solve(rb, adjoint=True, category="coarse_oriented_rhs"))
        c = self.context
        forcing = components[1]+constants["pQ"]*eta_receiver_state
        eb = (response+constants["rho_adjoint"]*eta_receiver_state+c.k*forcing/c.gamma)/(1-c.tau)
        return np.stack((eb, (forcing+c.ell*eb)/c.gamma))
