from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

Array = np.ndarray


@dataclass
class TorchWavefieldSurrogate:
    """Adapter for differentiable PyTorch wavefield surrogates.

    ``forward_fn`` must accept ``(m, q, frequency_hz)`` and return a flattened
    complex tensor, or a real tensor with final dimension 2 storing real/imag.
    This class intentionally does not prescribe an FNO architecture: official
    ``neuralop`` models can be wrapped here once input/output channel encoding
    is fixed for the chosen dataset.
    """

    forward_fn: Callable
    device: str = "cpu"
    dtype: str = "float64"

    def _require_torch(self) -> None:
        if torch is None:
            raise ImportError("Install the 'torch' extra to use TorchWavefieldSurrogate")

    def _real_dtype(self):
        return torch.float64 if self.dtype == "float64" else torch.float32

    def _to_complex(self, y):
        if torch.is_complex(y):
            return y.reshape(-1)
        if y.ndim and y.shape[-1] == 2:
            return torch.view_as_complex(y.contiguous()).reshape(-1)
        return y.to(torch.complex128 if self.dtype == "float64" else torch.complex64).reshape(-1)

    def state(self, m: Array, q: Array, frequency_hz: float) -> Array:
        self._require_torch()
        dtype = self._real_dtype()
        mt = torch.as_tensor(np.asarray(m), dtype=dtype, device=self.device)
        qt = torch.as_tensor(np.asarray(q), dtype=torch.complex128 if dtype == torch.float64 else torch.complex64, device=self.device)
        with torch.no_grad():
            y = self._to_complex(self.forward_fn(mt, qt, float(frequency_hz)))
        return y.detach().cpu().numpy()

    def jvp(self, m: Array, direction: Array, q: Array, frequency_hz: float) -> Array:
        self._require_torch()
        dtype = self._real_dtype()
        mt = torch.as_tensor(np.asarray(m), dtype=dtype, device=self.device)
        vt = torch.as_tensor(np.asarray(direction), dtype=dtype, device=self.device)
        qt = torch.as_tensor(np.asarray(q), dtype=torch.complex128 if dtype == torch.float64 else torch.complex64, device=self.device)

        def wrapped(x):
            return self._to_complex(self.forward_fn(x, qt, float(frequency_hz)))

        _, tangent = torch.func.jvp(wrapped, (mt,), (vt,))
        return tangent.detach().cpu().numpy()
