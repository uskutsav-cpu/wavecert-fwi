import numpy as np
import pytest


def test_fno_shapes_and_model_derivative_are_finite():
    torch = pytest.importorskip("torch")
    from wavecert.surrogates.fno import (
        FNO2dWavefield,
        FNOConfig,
        FNOInputEncoder,
        WavefieldNormalization,
    )

    shape = (10, 10)
    models = np.full((3, *shape), 0.22, dtype=np.float32)
    frequencies = np.array([3.0, 3.5, 4.0], dtype=np.float32)
    wavefields = np.zeros((3, *shape, 2), dtype=np.float32)
    wavefields[..., 0] = 0.1
    rng = np.random.default_rng(1)
    norm = WavefieldNormalization.from_arrays(
        models + rng.normal(scale=0.01, size=models.shape),
        frequencies,
        wavefields + 0.01,
    )
    enc = FNOInputEncoder(shape, norm)
    model = FNO2dWavefield(FNOConfig(modes_z=3, modes_x=3, width=8, depth=2, padding=1))
    m = torch.tensor(models[:1], requires_grad=True)
    s = torch.tensor([22])
    f = torch.tensor([3.0])
    x = enc.encode_batch(m, s, f)
    y = model(x)
    assert y.shape == (1, 2, *shape)
    loss = y.square().mean()
    (g,) = torch.autograd.grad(loss, m)
    assert torch.isfinite(g).all()
