from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None

Array = np.ndarray


@dataclass(frozen=True)
class FNOConfig:
    modes_z: int = 8
    modes_x: int = 8
    width: int = 28
    depth: int = 4
    padding: int = 2


@dataclass(frozen=True)
class WavefieldNormalization:
    model_mean: float
    model_std: float
    frequency_mean: float
    frequency_std: float
    output_mean_real: float
    output_std_real: float
    output_mean_imag: float
    output_std_imag: float

    @classmethod
    def from_arrays(cls, models: Array, frequencies: Array, wavefields: Array) -> WavefieldNormalization:
        return cls(
            model_mean=float(np.mean(models)),
            model_std=max(float(np.std(models)), 1e-8),
            frequency_mean=float(np.mean(frequencies)),
            frequency_std=max(float(np.std(frequencies)), 1e-8),
            output_mean_real=float(np.mean(wavefields[..., 0])),
            output_std_real=max(float(np.std(wavefields[..., 0])), 1e-8),
            output_mean_imag=float(np.mean(wavefields[..., 1])),
            output_std_imag=max(float(np.std(wavefields[..., 1])), 1e-8),
        )


if torch is not None:

    class SpectralConv2d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, modes_z: int, modes_x: int):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.modes_z = modes_z
            self.modes_x = modes_x
            scale = 1.0 / max(1, in_channels * out_channels)
            self.weight_top = nn.Parameter(
                scale
                * torch.randn(in_channels, out_channels, modes_z, modes_x, dtype=torch.cfloat)
            )
            self.weight_bottom = nn.Parameter(
                scale
                * torch.randn(in_channels, out_channels, modes_z, modes_x, dtype=torch.cfloat)
            )

        @staticmethod
        def _compl_mul(x, w):
            return torch.einsum("bixy,ioxy->boxy", x, w)

        def forward(self, x):
            b, _, nz, nx = x.shape
            x_ft = torch.fft.rfft2(x, norm="ortho")
            mx = min(self.modes_x, x_ft.shape[-1])
            mz = min(self.modes_z, nz // 2)
            out_ft = torch.zeros(
                b,
                self.out_channels,
                nz,
                x_ft.shape[-1],
                dtype=x_ft.dtype,
                device=x.device,
            )
            out_ft[:, :, :mz, :mx] = self._compl_mul(
                x_ft[:, :, :mz, :mx], self.weight_top[:, :, :mz, :mx]
            )
            out_ft[:, :, -mz:, :mx] = self._compl_mul(
                x_ft[:, :, -mz:, :mx], self.weight_bottom[:, :, :mz, :mx]
            )
            return torch.fft.irfft2(out_ft, s=(nz, nx), norm="ortho")


    class FNO2dWavefield(nn.Module):
        """Compact Fourier Neural Operator for complex Helmholtz wavefields.

        Input channels are normalized model, source mask, normalized frequency,
        depth coordinate, and horizontal coordinate. Output channels are
        normalized real and imaginary wavefield components.
        """

        def __init__(self, config: FNOConfig | None = None):
            super().__init__()
            if config is None:
                config = FNOConfig()
            self.config = config
            self.lift = nn.Conv2d(5, config.width, 1)
            self.spectral = nn.ModuleList(
                [
                    SpectralConv2d(config.width, config.width, config.modes_z, config.modes_x)
                    for _ in range(config.depth)
                ]
            )
            self.local = nn.ModuleList(
                [nn.Conv2d(config.width, config.width, 1) for _ in range(config.depth)]
            )
            self.norms = nn.ModuleList(
                [nn.GroupNorm(1, config.width) for _ in range(config.depth)]
            )
            self.project1 = nn.Conv2d(config.width, config.width * 2, 1)
            self.project2 = nn.Conv2d(config.width * 2, 2, 1)

        def forward(self, x):
            x = self.lift(x)
            pad = self.config.padding
            if pad:
                x = F.pad(x, (0, pad, 0, pad))
            for spec, local, norm in zip(self.spectral, self.local, self.norms, strict=True):
                residual = x
                x = spec(x) + local(x)
                x = F.gelu(norm(x)) + 0.1 * residual
            if pad:
                x = x[..., :-pad, :-pad]
            x = F.gelu(self.project1(x))
            return self.project2(x)

else:  # pragma: no cover

    class FNO2dWavefield:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for FNO2dWavefield")


class FNOInputEncoder:
    def __init__(self, shape: tuple[int, int], normalization: WavefieldNormalization):
        if torch is None:  # pragma: no cover
            raise ImportError("PyTorch is required")
        self.shape = shape
        self.normalization = normalization
        nz, nx = shape
        z = torch.linspace(-1.0, 1.0, nz).view(1, 1, nz, 1).expand(1, 1, nz, nx)
        x = torch.linspace(-1.0, 1.0, nx).view(1, 1, 1, nx).expand(1, 1, nz, nx)
        self.coords = torch.cat([z, x], dim=1)

    def encode_batch(self, models, source_indices, frequencies_hz):
        n = models.shape[0]
        nz, nx = self.shape
        norm = self.normalization
        m = (models - norm.model_mean) / norm.model_std
        m = m.view(n, 1, nz, nx)
        src = torch.zeros((n, 1, nz, nx), dtype=models.dtype, device=models.device)
        flat = src.view(n, -1)
        flat[torch.arange(n, device=models.device), source_indices.long()] = 1.0
        f = ((frequencies_hz - norm.frequency_mean) / norm.frequency_std).view(n, 1, 1, 1)
        f = f.expand(n, 1, nz, nx)
        coords = self.coords.to(device=models.device, dtype=models.dtype).expand(n, -1, -1, -1)
        return torch.cat([m, src, f, coords], dim=1)

    def decode_output(self, y):
        norm = self.normalization
        real = y[:, 0] * norm.output_std_real + norm.output_mean_real
        imag = y[:, 1] * norm.output_std_imag + norm.output_mean_imag
        return torch.stack([real, imag], dim=-1)

    def normalize_target(self, y):
        norm = self.normalization
        out = y.clone()
        out[..., 0] = (out[..., 0] - norm.output_mean_real) / norm.output_std_real
        out[..., 1] = (out[..., 1] - norm.output_mean_imag) / norm.output_std_imag
        return out


class TrainedFNOWavefieldSurrogate:
    """Differentiable adapter from a trained FNO checkpoint to WaveCert APIs."""

    def __init__(
        self,
        model,
        *,
        shape: tuple[int, int],
        normalization: WavefieldNormalization,
        device: str = "cpu",
    ):
        if torch is None:  # pragma: no cover
            raise ImportError("PyTorch is required")
        self.model = model.to(device)
        self.model.eval()
        self.shape = shape
        self.normalization = normalization
        self.device = device
        self.encoder = FNOInputEncoder(shape, normalization)

    def _forward_tensor(self, m_tensor, source_index: int, frequency_hz: float):
        models = m_tensor.reshape(1, *self.shape)
        src = torch.tensor([source_index], dtype=torch.long, device=m_tensor.device)
        freq = torch.tensor([frequency_hz], dtype=m_tensor.dtype, device=m_tensor.device)
        x = self.encoder.encode_batch(models, src, freq)
        y_norm = self.model(x)
        y = self.encoder.decode_output(y_norm)
        return y[0]

    def state(self, m: Array, q: Array, frequency_hz: float) -> Array:
        q = np.asarray(q)
        source_index = int(np.argmax(np.abs(q)))
        mt = torch.as_tensor(np.asarray(m), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y = self._forward_tensor(mt, source_index, float(frequency_hz))
        yc = torch.complex(y[..., 0], y[..., 1])
        return yc.reshape(-1).cpu().numpy()

    def jvp(self, m: Array, direction: Array, q: Array, frequency_hz: float) -> Array:
        q = np.asarray(q)
        source_index = int(np.argmax(np.abs(q)))
        mt = torch.as_tensor(np.asarray(m), dtype=torch.float32, device=self.device)
        vt = torch.as_tensor(np.asarray(direction), dtype=torch.float32, device=self.device)

        def fn(x):
            return self._forward_tensor(x, source_index, float(frequency_hz))

        _, tangent = torch.func.jvp(fn, (mt,), (vt,))
        tc = torch.complex(tangent[..., 0], tangent[..., 1])
        return tc.reshape(-1).detach().cpu().numpy()

    def objective_and_gradient(
        self,
        m: Array,
        q: Array,
        observed: Array,
        receiver_indices: tuple[int, ...],
        frequency_hz: float,
    ) -> tuple[float, Array, Array, Array]:
        q = np.asarray(q)
        source_index = int(np.argmax(np.abs(q)))
        mt = torch.tensor(np.asarray(m), dtype=torch.float32, device=self.device, requires_grad=True)
        y = self._forward_tensor(mt, source_index, float(frequency_hz))
        flat = y.reshape(-1, 2)
        idx = torch.as_tensor(receiver_indices, dtype=torch.long, device=self.device)
        pred_ri = flat[idx]
        obs = np.asarray(observed)
        obs_ri = torch.as_tensor(
            np.stack([obs.real, obs.imag], axis=-1), dtype=torch.float32, device=self.device
        )
        residual_ri = pred_ri - obs_ri
        objective = 0.5 * torch.sum(residual_ri**2)
        (grad,) = torch.autograd.grad(objective, mt)
        wave = torch.complex(y[..., 0], y[..., 1]).reshape(-1)
        residual = torch.complex(residual_ri[..., 0], residual_ri[..., 1])
        return (
            float(objective.detach().cpu()),
            grad.detach().cpu().numpy().astype(float),
            wave.detach().cpu().numpy(),
            residual.detach().cpu().numpy(),
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> TrainedFNOWavefieldSurrogate:
        if torch is None:  # pragma: no cover
            raise ImportError("PyTorch is required")
        payload: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
        config = FNOConfig(**payload["model_config"])
        model = FNO2dWavefield(config)
        model.load_state_dict(payload["model_state"])
        normalization = WavefieldNormalization(**payload["normalization"])
        return cls(model, shape=tuple(payload["shape"]), normalization=normalization, device=device)


def checkpoint_payload(model, config: FNOConfig, normalization: WavefieldNormalization, shape, **metadata):
    return {
        "model_state": model.state_dict(),
        "model_config": asdict(config),
        "normalization": asdict(normalization),
        "shape": list(shape),
        **metadata,
    }
