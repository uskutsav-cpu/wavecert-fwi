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


def test_combined_state_tangent_and_forward_only_merit_match_adapter():
    pytest.importorskip("torch")
    from wavecert.surrogates.fno import TrainedFNOWavefieldSurrogate

    surrogate = TrainedFNOWavefieldSurrogate.load("checkpoints/fno_phase2.pt")
    n = int(np.prod(surrogate.shape))
    model = np.full(n, 0.22)
    direction = np.linspace(-0.01, 0.01, n)
    source = np.zeros(n)
    source[22] = 400.0
    receivers = (3, 5, 5, 10)
    observed = np.zeros(len(receivers), dtype=complex)
    state, tangent = surrogate.state_and_jvp(model, direction, source, 3.0)
    assert np.allclose(state, surrogate.state(model, source, 3.0), rtol=2e-5, atol=1e-6)
    epsilon = 0.005
    centered = (surrogate.state(model + epsilon * direction, source, 3.0)
                - surrogate.state(model - epsilon * direction, source, 3.0)) / (2 * epsilon)
    assert np.linalg.norm(tangent - centered) / np.linalg.norm(tangent) < 0.01
    value, _, _, _ = surrogate.objective_and_gradient(model, source, observed, receivers, 3.0)
    assert np.isclose(value, surrogate.objective(model, source, observed, receivers, 3.0), rtol=2e-6)
