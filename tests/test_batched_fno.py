from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wavecert.experiments.batched_neural_controls import (  # noqa: E402
    make_neural_control_survey,
    neural_control_agreement,
)
from wavecert.physics.helmholtz import SurveyBlock  # noqa: E402
from wavecert.surrogates.fno import (  # noqa: E402
    FNO2dWavefield,
    FNOConfig,
    TrainedFNOWavefieldSurrogate,
    WavefieldNormalization,
)


def _survey():
    torch.set_num_threads(1)
    torch.manual_seed(612)
    shape = (8, 9)
    norm = WavefieldNormalization(0.22, 0.04, 3.25, 0.75, 0.2, 0.7, -0.1, 0.4)
    model = FNO2dWavefield(FNOConfig(modes_z=2, modes_x=2, width=8, depth=2, padding=1))
    surrogate = TrainedFNOWavefieldSurrogate(model, shape=shape, normalization=norm)
    rng = np.random.default_rng(910)
    m = 0.22 + rng.normal(scale=0.003, size=np.prod(shape))
    receivers = (3, 12, 12, 26, 48)
    blocks = tuple(SurveyBlock(source, frequency,
        rng.normal(size=len(receivers)) + 1j*rng.normal(size=len(receivers)), str(i))
        for i, (source, frequency) in enumerate(((10, 2.5), (23, 3.5), (51, 4.0))))
    return surrogate, m, receivers, blocks


def _source(n, index):
    q = np.zeros(n)
    q[index] = 400.0
    return q


def test_batched_survey_matches_serial_sum_and_preserves_block_order():
    surrogate, m, receivers, blocks = _survey()
    calls = []
    hook = surrogate.model.register_forward_hook(lambda *_: calls.append(1))
    value, gradient, states, residuals = surrogate.survey_objective_and_gradient(m, blocks, receivers)
    assert len(calls) == 1
    serial = [surrogate.objective_and_gradient(m, _source(m.size, b.source_index), b.observed,
                                               receivers, b.frequency_hz) for b in blocks]
    serial_gradient = np.sum([r[1] for r in serial], axis=0)
    assert len(calls) == 1 + len(blocks)
    hook.remove()
    assert value == pytest.approx(sum(r[0] for r in serial), rel=2e-6)
    assert np.linalg.norm(gradient-serial_gradient) / np.linalg.norm(serial_gradient) < 2e-5
    assert states.shape == (3, m.size)
    assert residuals.shape == (3, len(receivers))
    assert np.allclose(states, np.stack([r[2] for r in serial]), rtol=2e-5, atol=1e-6)
    assert np.allclose(residuals, np.stack([r[3] for r in serial]), rtol=2e-5, atol=1e-6)
    assert np.allclose(residuals, states[:, receivers] - np.stack([b.observed for b in blocks]), rtol=2e-6, atol=1e-6)
    assert all(parameter.grad is None for parameter in surrogate.model.parameters())


def test_shared_model_vjp_matches_sum_of_jvps_and_independent_finite_difference():
    surrogate, m, receivers, blocks = _survey()
    _, gradient, _, residuals = surrogate.survey_objective_and_gradient(m, blocks, receivers)
    direction = np.random.default_rng(718).normal(size=m.size)
    direction /= np.linalg.norm(direction)
    directional = float(np.dot(gradient, direction))
    jvp_directional = sum(np.vdot(residuals[i], surrogate.jvp(
        m, direction, _source(m.size, b.source_index), b.frequency_hz,
    )[list(receivers)]).real for i, b in enumerate(blocks))
    assert directional == pytest.approx(jvp_directional, rel=3e-5, abs=1e-5)

    def objective(model):
        return sum(surrogate.objective(model, _source(m.size, b.source_index), b.observed,
                                       receivers, b.frequency_hz) for b in blocks)

    errors = []
    for step in (2e-3, 1e-3, 5e-4):
        finite_difference = (objective(m+step*direction)-objective(m-step*direction))/(2*step)
        errors.append(abs(finite_difference-directional)/max(abs(directional), 1e-12))
    assert min(errors) < 3e-3


def test_repeated_receivers_accumulate_their_shared_model_adjoint():
    surrogate, m, _, blocks = _survey()
    observed = np.array([0.2+0.3j])
    single = (replace(blocks[0], observed=observed),)
    repeated = (replace(blocks[0], observed=np.repeat(observed, 2)),)
    one = surrogate.survey_objective_and_gradient(m, single, (12,))
    two = surrogate.survey_objective_and_gradient(m, repeated, (12, 12))
    assert two[0] == pytest.approx(2*one[0], rel=2e-6)
    assert np.allclose(two[1], 2*one[1], rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("change", ["model_shape", "nan_model", "negative_model", "complex_model",
    "empty_blocks", "source", "bool_source", "frequency", "observed_shape", "nan_observed",
    "receiver", "empty_receivers", "float32_overflow", "float32_underflow"])
def test_batched_survey_rejects_invalid_or_unrepresentable_inputs(change):
    surrogate, m, receivers, blocks = _survey()
    if change == "model_shape":
        m = m.reshape(surrogate.shape)
    elif change == "nan_model":
        m[0] = np.nan
    elif change == "negative_model":
        m[0] = -0.1
    elif change == "complex_model":
        m = m.astype(complex)
    elif change == "empty_blocks":
        blocks = ()
    elif change == "source":
        blocks = (replace(blocks[0], source_index=m.size),)
    elif change == "bool_source":
        blocks = (replace(blocks[0], source_index=True),)
    elif change == "frequency":
        blocks = (replace(blocks[0], frequency_hz=np.nan),)
    elif change == "observed_shape":
        blocks = (replace(blocks[0], observed=np.zeros((len(receivers), 1))),)
    elif change == "nan_observed":
        blocks = (replace(blocks[0], observed=np.full(len(receivers), np.nan)),)
    elif change == "receiver":
        receivers = (True,)
    elif change == "empty_receivers":
        receivers = ()
    elif change == "float32_overflow":
        m[:] = 1e100
    else:
        m[:] = 1e-100
    with pytest.raises(ValueError):
        surrogate.survey_objective_and_gradient(m, blocks, receivers)


@pytest.mark.parametrize("batch_size", [4, 12, 40])
def test_benchmark_inputs_have_distinct_acquisition_blocks_and_one_common_model(batch_size):
    model, blocks, receivers, record = make_neural_control_survey(batch_size)
    assert model.shape == (576,)
    assert len(blocks) == len({(b.source_index, b.frequency_hz) for b in blocks}) == batch_size
    assert len({b.source_index for b in blocks}) >= 2
    assert len({b.frequency_hz for b in blocks}) >= 2
    assert len(receivers) == 20
    assert all(b.observed.shape == (20,) for b in blocks)
    assert record["common_model_sha256"] != record["truth_model_sha256"]


def test_benchmark_agreement_checks_actual_gradients_and_complex_receiver_values():
    reference = {"objective": 2.0, "gradient": [0.1, 0.2],
        "receiver_state_real": [[0.4, 0.5]], "receiver_state_imag": [[0.1, 0.2]],
        "residual_real": [[0.1, 0.3]], "residual_imag": [[0.05, 0.1]]}
    assert neural_control_agreement(reference, reference)["passes_float32_tolerances"]
    changed = {**reference, "gradient": [0.2, 0.1]}
    assert not neural_control_agreement(reference, changed)["passes_float32_tolerances"]
    changed = {**reference, "receiver_state_imag": [[0.2, 0.1]]}
    assert not neural_control_agreement(reference, changed)["passes_float32_tolerances"]
    with pytest.raises(ValueError, match="non-finite"):
        neural_control_agreement(reference, {**reference, "gradient": [np.nan, 0.2]})
