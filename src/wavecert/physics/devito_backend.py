from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DevitoAcousticConfig:
    """Configuration for the optional production time-domain acoustic backend.

    This wrapper follows the public Devito seismic tutorial API. Devito is an
    optional dependency because the compact Helmholtz backend is sufficient for
    mathematical unit tests and CI.
    """

    spacing: tuple[float, float] = (10.0, 10.0)
    origin: tuple[float, float] = (0.0, 0.0)
    nbl: int = 40
    t0: float = 0.0
    tn: float = 1000.0
    f0: float = 0.010
    space_order: int = 4


class DevitoAcousticFWI:
    """Thin adapter around Devito's ``AcousticWaveSolver``.

    The interface intentionally mirrors the official Devito FWI tutorials:
    ``solver.forward(..., save=True)`` generates predicted data and the saved
    wavefield, and ``solver.gradient`` back-propagates the receiver residual.

    It is kept optional so WaveCert's test suite remains usable on machines
    without a Devito/JIT toolchain. Install with ``pip install -e '.[devito]'``.
    """

    def __init__(
        self,
        vp: np.ndarray,
        *,
        source_coordinates: np.ndarray,
        receiver_coordinates: np.ndarray,
        config: DevitoAcousticConfig = DevitoAcousticConfig(),
    ) -> None:
        try:
            from examples.seismic import AcquisitionGeometry, Model
            from examples.seismic.acoustic import AcousticWaveSolver
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Devito seismic examples are required. Install the 'devito' extra."
            ) from exc

        vp = np.asarray(vp, dtype=np.float32)
        self.config = config
        self.Model = Model
        self.AcquisitionGeometry = AcquisitionGeometry
        self.AcousticWaveSolver = AcousticWaveSolver
        self.model = Model(
            vp=vp,
            origin=config.origin,
            shape=vp.shape,
            spacing=config.spacing,
            nbl=config.nbl,
            space_order=config.space_order,
            bcs="damp",
        )
        self.geometry = AcquisitionGeometry(
            self.model,
            np.asarray(receiver_coordinates, dtype=np.float32),
            np.asarray(source_coordinates, dtype=np.float32),
            config.t0,
            config.tn,
            f0=config.f0,
            src_type="Ricker",
        )
        self.solver = AcousticWaveSolver(
            self.model, self.geometry, space_order=config.space_order
        )

    def update_velocity(self, vp: np.ndarray) -> None:
        self.model.update("vp", np.asarray(vp, dtype=np.float32).reshape(self.model.shape))

    def forward(self, *, save: bool = False) -> tuple[np.ndarray, Any]:
        rec, u = self.solver.forward(vp=self.model.vp, save=save)[0:2]
        return np.array(rec.data[:], copy=True), u

    def objective_and_gradient(self, observed: Any) -> tuple[float, np.ndarray]:
        """Return one-geometry FWI objective and squared-slowness gradient.

        ``observed`` may be a Devito receiver object or a NumPy shot record with
        the same sampling as the geometry.
        """

        try:
            from devito import Function
            from examples.seismic import Receiver
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Devito is required") from exc

        grad = Function(name="grad_wavecert", grid=self.model.grid)
        residual = Receiver(
            name="residual_wavecert",
            grid=self.model.grid,
            time_range=self.geometry.time_axis,
            coordinates=self.geometry.rec_positions,
        )
        d_pred, u0 = self.solver.forward(vp=self.model.vp, save=True)[0:2]
        if hasattr(observed, "resample"):
            obs = observed.resample(self.geometry.dt).data[: d_pred.data.shape[0], :]
        else:
            obs = np.asarray(observed)[: d_pred.data.shape[0], :]
        residual.data[:] = d_pred.data[:] - obs
        fval = 0.5 * float(np.linalg.norm(residual.data.ravel()) ** 2)
        self.solver.gradient(rec=residual, u=u0, vp=self.model.vp, grad=grad)
        nbl = int(self.model.nbl)
        if nbl > 0:
            cropped = np.array(grad.data[nbl:-nbl, nbl:-nbl], dtype=np.float64, copy=True)
        else:
            cropped = np.array(grad.data[:], dtype=np.float64, copy=True)
        return fval, cropped
