from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _edge_padding_pullback(gradient: np.ndarray, shape: tuple[int, int], nbl: int) -> np.ndarray:
    """Adjoint of edge extension used by Devito ``Model.update``.

    Padded cells depend on physical boundary parameters. Cropping their
    gradient loses those contributions and fails finite-difference checks for
    directions that change the physical edges.
    """
    gradient = np.asarray(gradient, dtype=np.float64)
    expected = tuple(n + 2 * nbl for n in shape)
    if gradient.shape != expected:
        raise ValueError(f"padded gradient shape {gradient.shape} does not match {expected}")
    ix = np.clip(np.arange(expected[0]) - nbl, 0, shape[0] - 1)
    iz = np.clip(np.arange(expected[1]) - nbl, 0, shape[1] - 1)
    result = np.zeros(shape, dtype=np.float64)
    np.add.at(result, (ix[:, None], iz[None, :]), gradient)
    return result


@dataclass(frozen=True)
class DevitoAcousticConfig:
    """Optional time-domain backend configuration (m, ms, km/s).

    ``dt`` is fixed for the complete forward/adjoint/optimization workflow.
    If omitted it is selected conservatively for ``max_velocity`` (or the
    initial maximum velocity); changing velocity never silently changes dt.
    """

    spacing: tuple[float, float] = (10.0, 10.0)
    origin: tuple[float, float] = (0.0, 0.0)
    nbl: int = 40
    t0: float = 0.0
    tn: float = 1000.0
    f0: float = 0.010
    space_order: int = 4
    dt: float | None = None
    max_velocity: float | None = None
    dtype: str = "float32"


class DevitoAcousticFWI:
    """Fixed-sampling adapter around Devito's acoustic tutorial operators.

    This is an independent exact-physics baseline. The frequency-domain
    WaveCert certificate is not transferred to this time-domain solver.
    """

    def __init__(
        self,
        vp: np.ndarray,
        *,
        source_coordinates: np.ndarray,
        receiver_coordinates: np.ndarray,
        config: DevitoAcousticConfig | None = None,
    ) -> None:
        if config is None:
            config = DevitoAcousticConfig()
        self.dtype = np.dtype(config.dtype)
        if self.dtype not in {np.dtype("float32"), np.dtype("float64")}:
            raise ValueError("dtype must be float32 or float64")
        vp = np.asarray(vp, dtype=self.dtype)
        if vp.ndim != 2 or min(vp.shape) < 3 or np.any(~np.isfinite(vp)) or np.any(vp <= 0):
            raise ValueError("vp must be a finite positive 2-D velocity array")
        if (len(config.spacing) != 2 or np.any(~np.isfinite(config.spacing))
                or np.any(np.asarray(config.spacing) <= 0)):
            raise ValueError("spacing must contain two finite positive values")
        if not np.isfinite(config.t0) or not np.isfinite(config.tn) or config.tn <= config.t0:
            raise ValueError("time interval must be finite with tn > t0")
        if not np.isfinite(config.f0) or config.f0 <= 0:
            raise ValueError("source peak frequency must be finite and positive")
        if config.nbl < 0 or int(config.nbl) != config.nbl:
            raise ValueError("nbl must be a nonnegative integer")
        if config.space_order < 2 or int(config.space_order) != config.space_order or config.space_order % 2:
            raise ValueError("space_order must be a positive even integer >=2")
        if config.dt is not None and (not np.isfinite(config.dt) or config.dt <= 0):
            raise ValueError("dt must be finite and positive")
        if config.max_velocity is not None and (
            not np.isfinite(config.max_velocity) or config.max_velocity < float(np.max(vp))
        ):
            raise ValueError("max_velocity must be finite and at least the initial maximum")
        for label, coordinates in (("source", source_coordinates), ("receiver", receiver_coordinates)):
            coordinates = np.asarray(coordinates)
            if coordinates.ndim != 2 or coordinates.shape[1] != 2 or coordinates.shape[0] < 1 or np.any(~np.isfinite(coordinates)):
                raise ValueError(f"{label} coordinates must be a nonempty finite (N,2) array")
            lower = np.asarray(config.origin)
            upper = lower + (np.asarray(vp.shape) - 1) * np.asarray(config.spacing)
            if np.any(coordinates < lower) or np.any(coordinates > upper):
                raise ValueError(f"{label} coordinates must lie in the physical model domain")
        try:
            from examples.seismic import AcquisitionGeometry, Model
            from examples.seismic.acoustic import AcousticWaveSolver
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Devito seismic examples are required. Install the 'devito' extra.") from exc

        self.config = config
        self.Model = Model
        self.AcquisitionGeometry = AcquisitionGeometry
        self.AcousticWaveSolver = AcousticWaveSolver
        self.model = Model(
            vp=vp, origin=config.origin, shape=vp.shape, spacing=config.spacing,
            nbl=config.nbl, space_order=config.space_order, bcs="damp", dtype=self.dtype.type,
        )
        self._cfl_velocity_product = float(self.model.critical_dt) * float(np.max(vp))
        maximum = float(np.max(vp)) if config.max_velocity is None else config.max_velocity
        safe_dt = self._cfl_velocity_product / maximum
        self.dt = float(config.dt) if config.dt is not None else 0.95 * safe_dt
        if self.dt > safe_dt * (1.0 + 1e-7):
            raise ValueError("dt exceeds the CFL limit for the declared maximum velocity")
        self.geometry = AcquisitionGeometry(
            self.model, np.asarray(receiver_coordinates, dtype=self.dtype),
            np.asarray(source_coordinates, dtype=self.dtype), config.t0, config.tn,
            f0=config.f0, src_type="Ricker",
        )
        self.geometry.resample(self.dt)
        self.solver = AcousticWaveSolver(self.model, self.geometry, space_order=config.space_order)

    @property
    def time_values(self) -> np.ndarray:
        """The fixed sample times shared by sources, predictions and residuals."""
        return np.array(self.geometry.time_axis.time_values, dtype=np.float64, copy=True)

    def update_velocity(self, vp: np.ndarray) -> None:
        velocity = np.asarray(vp, dtype=self.dtype)
        if velocity.shape != self.model.shape or np.any(~np.isfinite(velocity)) or np.any(velocity <= 0):
            raise ValueError(f"velocity must be finite, positive and have shape {self.model.shape}")
        maximum = float(np.max(velocity))
        if self.config.max_velocity is not None and maximum > self.config.max_velocity * (1.0 + 1e-7):
            raise ValueError("velocity exceeds configured max_velocity")
        if self.dt * maximum > self._cfl_velocity_product * (1.0 + 1e-7):
            raise ValueError("velocity update violates the fixed-dt CFL limit")
        self.model.update("vp", velocity)

    def forward(self, *, save: bool = False) -> tuple[np.ndarray, Any]:
        rec, u = self.solver.forward(vp=self.model.vp, save=save, dt=self.dt)[0:2]
        return np.array(rec.data[:], copy=True), u

    def _observed_array(self, observed: Any, observed_time_values: np.ndarray | None) -> np.ndarray:
        if isinstance(observed, np.ndarray):
            obs = np.asarray(observed)
        elif hasattr(observed, "data"):
            obs = np.asarray(observed.data)
            if observed_time_values is None:
                axis = getattr(observed, "_time_range", None)
                if axis is not None:
                    observed_time_values = np.asarray(axis.time_values)
            coordinates = getattr(observed, "coordinates", None)
            if coordinates is not None and not np.allclose(
                np.asarray(coordinates.data), self.geometry.rec_positions, rtol=0, atol=1e-5,
            ):
                raise ValueError("observed receiver coordinates do not match the inversion geometry")
        else:
            raise ValueError("observed data must be a NumPy record or a Devito Receiver")
        expected = (len(self.time_values), self.geometry.nrec)
        if obs.shape != expected:
            raise ValueError(f"observed shape {obs.shape} does not match fixed sampling {expected}; no truncation is permitted")
        if not np.all(np.isfinite(obs)):
            raise ValueError("observed data must be finite")
        if observed_time_values is None:
            raise ValueError("NumPy observed records require observed_time_values; sampling cannot be inferred from shape")
        samples = np.asarray(observed_time_values, dtype=np.float64)
        if samples.shape != self.time_values.shape or not np.allclose(
            samples, self.time_values, rtol=1e-7, atol=1e-6,
        ):
            raise ValueError("observed sample times do not match the fixed solver time axis; resample explicitly before inversion")
        return obs

    def objective(self, observed: Any, *, observed_time_values: np.ndarray | None = None) -> float:
        obs = self._observed_array(observed, observed_time_values)
        predicted, _ = self.forward(save=False)
        return 0.5 * float(np.sum((predicted.astype(np.float64) - obs) ** 2))

    def objective_and_gradient(
        self, observed: Any, *, observed_time_values: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        """Return fixed-sampling objective and squared-slowness gradient.

        NumPy shot records require their actual sample-time array. Receiver
        objects carry this metadata. Resampling and truncation are never
        performed implicitly, since either changes the inverse problem.
        """
        obs = self._observed_array(observed, observed_time_values)
        try:
            from devito import Function
            from examples.seismic import Receiver
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Devito is required") from exc
        grad = Function(name="grad_wavecert", grid=self.model.grid)
        residual = Receiver(
            name="residual_wavecert", grid=self.model.grid,
            time_range=self.geometry.time_axis, coordinates=self.geometry.rec_positions,
        )
        predicted, wavefield = self.solver.forward(vp=self.model.vp, save=True, dt=self.dt)[0:2]
        residual.data[:] = predicted.data[:] - obs
        value = 0.5 * float(np.sum(np.asarray(residual.data, dtype=np.float64) ** 2))
        self.solver.gradient(rec=residual, u=wavefield, vp=self.model.vp, grad=grad, dt=self.dt)
        gradient = _edge_padding_pullback(grad.data, self.model.shape, int(self.model.nbl))
        return value, gradient

    def squared_slowness_loss(
        self, x: np.ndarray, observed: Any, observed_time_values: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        x = np.asarray(x, dtype=np.float64).reshape(self.model.shape)
        if np.any(~np.isfinite(x)) or np.any(x <= 0.0):
            raise ValueError("squared slowness must be finite and strictly positive")
        self.update_velocity(1.0 / np.sqrt(x))
        value, gradient = self.objective_and_gradient(observed, observed_time_values=observed_time_values)
        return float(value), np.asarray(gradient, dtype=np.float64).reshape(-1)

    def check_gradient(
        self, observed: Any, *, observed_time_values: np.ndarray | None = None,
        seed: int = 31, relative_steps: tuple[float, ...] = (1e-3, 3e-4, 1e-4),
    ) -> dict:
        """Compare adjoint sensitivities with central differences at fixed dt.

        Includes a full-domain direction (testing the padding pullback) and an
        interior direction. All perturbation results are retained, including
        finite-precision deterioration. The original model is restored.
        """
        if not relative_steps or any(not np.isfinite(step) or step <= 0 for step in relative_steps):
            raise ValueError("relative_steps must be nonempty, finite and positive")
        vp = np.asarray(self.model.vp.data, dtype=np.float64)
        nbl = int(self.model.nbl)
        if nbl:
            vp = vp[nbl:-nbl, nbl:-nbl]
        x0 = 1.0 / vp.copy() ** 2
        _, gradient = self.squared_slowness_loss(x0, observed, observed_time_values)
        rng = np.random.default_rng(seed)
        full = rng.normal(size=self.model.shape)
        interior = full.copy()
        width = min(4, max(1, min(self.model.shape) // 4))
        interior[:width] = interior[-width:] = 0
        interior[:, :width] = interior[:, -width:] = 0
        rows = []
        try:
            for label, direction in (("full-domain-including-edges", full), ("interior", interior)):
                direction /= np.linalg.norm(direction)
                derivative = float(np.dot(gradient, direction.ravel()))
                for relative_step in relative_steps:
                    epsilon = float(relative_step) * float(np.linalg.norm(x0))
                    self.update_velocity(1.0 / np.sqrt(x0 + epsilon * direction))
                    plus = self.objective(observed, observed_time_values=observed_time_values)
                    self.update_velocity(1.0 / np.sqrt(x0 - epsilon * direction))
                    minus = self.objective(observed, observed_time_values=observed_time_values)
                    finite_difference = (plus - minus) / (2.0 * epsilon)
                    error = abs(finite_difference - derivative) / max(abs(finite_difference), abs(derivative), 1e-12)
                    rows.append({
                        "direction": label, "relative_step": float(relative_step),
                        "epsilon": epsilon, "adjoint_derivative": derivative,
                        "finite_difference": finite_difference, "relative_error": float(error),
                    })
        finally:
            self.update_velocity(1.0 / np.sqrt(x0))
        best_errors = {
            label: min(row["relative_error"] for row in rows if row["direction"] == label)
            for label in {row["direction"] for row in rows}
        }
        return {
            "seed": seed, "fixed_dt": self.dt, "dtype": self.dtype.name,
            "checks": rows, "best_relative_error_by_direction": best_errors,
            "passed": all(np.isfinite(error) and error < 1e-2 for error in best_errors.values()),
            "scope": "Finite-difference verification of the discrete squared-slowness gradient, including edge-padding dependence; not a PDE error bound.",
        }

    def run_lbfgsb(
        self, observed: Any, *, observed_time_values: np.ndarray | None = None,
        maxiter: int = 5, vmin: float = 1.4, vmax: float = 4.0, ftol: float = 0.1,
    ) -> Any:
        """Run bounded FWI without changing the wave-equation time sampling."""
        from scipy import optimize

        if maxiter < 1 or int(maxiter) != maxiter or not np.isfinite(ftol) or ftol <= 0:
            raise ValueError("maxiter must be a positive integer and ftol finite and positive")
        if not (np.isfinite(vmin) and np.isfinite(vmax) and 0 < vmin < vmax):
            raise ValueError("velocity bounds must satisfy 0 < vmin < vmax")
        if vmax * self.dt > self._cfl_velocity_product * (1.0 + 1e-7):
            raise ValueError("vmax violates the fixed-dt CFL limit; initialize with max_velocity=vmax")
        if self.config.max_velocity is not None and vmax > self.config.max_velocity * (1.0 + 1e-7):
            raise ValueError("vmax exceeds configured max_velocity")
        self._observed_array(observed, observed_time_values)
        vp = np.asarray(self.model.vp.data, dtype=np.float64)
        nbl = int(self.model.nbl)
        if nbl > 0:
            vp = vp[nbl:-nbl, nbl:-nbl]
        x0 = 1.0 / vp.reshape(-1) ** 2
        bounds = [(1.0 / vmax**2, 1.0 / vmin**2) for _ in range(x0.size)]
        result = optimize.minimize(
            self.squared_slowness_loss, x0, args=(observed, observed_time_values),
            method="L-BFGS-B", jac=True, bounds=bounds,
            options={"ftol": ftol, "maxiter": maxiter},
        )
        # A rejected final line-search callback can leave the mutable model at
        # a different point from the optimizer's returned accepted iterate.
        self.update_velocity(1.0 / np.sqrt(np.asarray(result.x).reshape(self.model.shape)))
        return result
