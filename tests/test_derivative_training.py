import numpy as np
import pytest

from wavecert.data.wavefields import build_wavefield_dataset
from wavecert.experiments.derivative_training import make_tangent_labels
from wavecert.physics.helmholtz import Helmholtz2D


def test_fixed_tangent_labels_match_independent_state_finite_differences():
    arrays = build_wavefield_dataset(
        n_samples=2, shape=(8, 8), seed=55, n_source_positions=2, n_receivers=4,
    )
    labels = make_tangent_labels(arrays, role="train", seed=123)
    repeat = make_tangent_labels(arrays, role="train", seed=123)
    np.testing.assert_array_equal(labels.directions, repeat.directions)
    np.testing.assert_array_equal(labels.tangents, repeat.tangents)
    physics = Helmholtz2D(shape=arrays.shape, spacing=arrays.spacing, damping_width=3, damping_strength=2.0)
    for index in range(len(arrays.models)):
        model = arrays.models[index].astype(float).reshape(-1)
        direction = labels.directions[index].astype(float).reshape(-1)
        q = physics.source(int(arrays.source_indices[index]))
        frequency = float(arrays.frequencies_hz[index])
        epsilon = 1e-5
        finite_difference = (
            physics.solve_state(model + epsilon * direction, q, frequency)
            - physics.solve_state(model - epsilon * direction, q, frequency)
        ) / (2 * epsilon)
        stored = (labels.tangents[index, ..., 0] + 1j * labels.tangents[index, ..., 1]).reshape(-1)
        assert np.linalg.norm(stored - finite_difference) / np.linalg.norm(finite_difference) < 5e-5
    assert labels.metadata["offline_label_physics_cost"]["tangent_rhs"] == 2
    with pytest.raises(ValueError, match="only for train"):
        make_tangent_labels(arrays, role="calibration", seed=123)


def test_jacobian_loss_backpropagates_to_parameters_and_never_centers_tangents():
    torch = pytest.importorskip("torch")
    from wavecert.experiments.derivative_training import (
        normalized_tangent_target,
        normalized_value_and_tangent,
    )
    from wavecert.surrogates.fno import (
        FNO2dWavefield,
        FNOConfig,
        FNOInputEncoder,
        WavefieldNormalization,
    )

    torch.manual_seed(1)
    shape = (8, 8)
    normalization = WavefieldNormalization(
        model_mean=.23, model_std=.1, frequency_mean=3., frequency_std=1.,
        output_mean_real=8., output_std_real=2., output_mean_imag=-9., output_std_imag=3.,
    )
    encoder = FNOInputEncoder(shape, normalization)
    physical_tangents = torch.empty((2, *shape, 2))
    physical_tangents[..., 0] = 4.
    physical_tangents[..., 1] = 6.
    target = normalized_tangent_target(encoder, physical_tangents)
    torch.testing.assert_close(target, torch.full((2, 2, *shape), 2.))
    model = FNO2dWavefield(FNOConfig(modes_z=2, modes_x=2, width=4, depth=1, padding=1))
    models = torch.full((2, *shape), .23)
    directions = torch.randn_like(models)
    directions /= torch.linalg.vector_norm(directions.reshape(2, -1), dim=1)[:, None, None]
    sources = torch.tensor([18, 21])
    frequencies = torch.tensor([3., 4.])
    _, tangent = normalized_value_and_tangent(model, encoder, models, sources, frequencies, directions)
    loss = torch.mean((tangent - target) ** 2)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(torch.linalg.vector_norm(gradient)) for gradient in gradients) > 0.
    with torch.no_grad():
        _, validation_tangent = normalized_value_and_tangent(model, encoder, models, sources, frequencies, directions)
    torch.testing.assert_close(validation_tangent, tangent.detach())
