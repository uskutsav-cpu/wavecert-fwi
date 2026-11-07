import importlib
from pathlib import Path

import numpy as np
import pytest

from wavecert.physics.helmholtz import Helmholtz2D, SurveyBlock
from wavecert.surrogates.pod import PODGalerkinSurrogate
from wavecert.surrogates.pod_batch import batched_pod_survey_fields


def _case():
    physics = Helmholtz2D(shape=(7, 9), spacing=.05, damping_width=2, damping_strength=2.)
    rng = np.random.default_rng(771)
    basis = np.linalg.qr(rng.normal(size=(physics.n, 12)))[0]
    model, receivers = rng.uniform(.18, .22, physics.n), (2, 5, 5, 13)
    blocks = []
    for source, frequency in ((3, 3.), (17, 2.), (9, 3.), (17, 3.)):
        observed = physics.restrict(physics.solve_state(model*1.03, physics.source(source), frequency), receivers)
        observed[2] += (.01+.02j)*np.linalg.norm(observed)  # Repeated row, different datum.
        blocks.append(SurveyBlock(source, frequency, observed, f"{source}@{frequency}"))
    return physics, basis, model, receivers, tuple(blocks)


def test_batch_fields_match_existing_scalar_objective_gradient_states_adjoints_and_matrix_rhs_counts():
    p, basis, model, receivers, blocks = _case()
    scalar, batch = PODGalerkinSurrogate(p, basis), PODGalerkinSurrogate(p, basis)
    references, adjoints = [], []
    for block in blocks:
        references.append(scalar.objective_and_gradient(model, p.source(block.source_index), block.observed,
            receivers, block.frequency_hz))
        _, _, factor = scalar._prepare(model, block.frequency_hz)
        injected = p.inject_receivers(references[-1][3], receivers)
        adjoints.append(basis @ scalar._solve(factor, basis.T @ injected, category="reduced_adjoint_rhs", adjoint=True))
    result = batched_pod_survey_fields(batch, model, blocks, receivers)
    assert result.blocks == blocks and result.receiver_indices == receivers
    assert result.block_states.shape == result.block_adjoints.shape == result.block_gradients.shape == (p.n, len(blocks))
    assert np.allclose(result.block_states, np.column_stack([r[2] for r in references]), rtol=2e-12, atol=1e-13)
    assert np.allclose(result.block_adjoints, np.column_stack(adjoints), rtol=2e-12, atol=1e-13)
    assert np.allclose(result.block_gradients, np.column_stack([r[1] for r in references]), rtol=2e-12, atol=1e-13)
    assert np.allclose(result.aggregate_gradient, sum(r[1] for r in references), rtol=2e-12, atol=1e-13)
    assert result.objective == pytest.approx(sum(r[0] for r in references), rel=2e-12)
    assert np.allclose(result.block_objectives, [r[0] for r in references], rtol=2e-12)
    assert result.costs["pod"]["reduced_forward_rhs"] == result.costs["pod"]["reduced_adjoint_rhs"] == 4
    assert result.costs["pod"]["reduced_factorizations"] == 2 and result.costs["pod"]["mass_assemblies"] == 1
    assert result.batch_dispatches["pod_forward_solve_dispatches"] == result.batch_dispatches["pod_adjoint_solve_dispatches"] == 2
    assert result.batch_dispatches["pod_state_lift_dispatches"] == 2
    assert not result.block_states.flags.writeable and not result.aggregate_gradient.flags.writeable


def test_batched_fields_preserve_content_cache_and_charge_changed_models():
    p, basis, model, receivers, blocks = _case()
    surrogate = PODGalerkinSurrogate(p, basis)
    first = batched_pod_survey_fields(surrogate, model, blocks, receivers)
    copied = batched_pod_survey_fields(surrogate, model.copy(), blocks, receivers)
    assert np.array_equal(first.block_states, copied.block_states)
    assert copied.costs["pod"]["reduced_factorizations"] == copied.costs["pod"]["mass_assemblies"] == 0
    assert copied.costs["pod"]["factor_cache_hits"] == 2
    changed = model.copy()
    changed[0] *= 1.01
    fresh = batched_pod_survey_fields(surrogate, changed, blocks, receivers)
    assert fresh.costs["pod"]["reduced_factorizations"] == 2 and fresh.costs["pod"]["mass_assemblies"] == 1
    assert not np.array_equal(first.block_states, fresh.block_states)


def test_batched_raw_gradient_matches_independent_reduced_objective_finite_difference():
    p, basis, model, receivers, blocks = _case()
    surrogate = PODGalerkinSurrogate(p, basis)
    result = batched_pod_survey_fields(surrogate, model, blocks, receivers)
    direction = -result.aggregate_gradient/np.linalg.norm(result.aggregate_gradient)

    def objective(values):
        return sum(surrogate.objective(values, p.source(b.source_index), b.observed, receivers, b.frequency_hz) for b in blocks)

    step = 1e-5
    derivative = (objective(model+step*direction)-objective(model-step*direction))/(2*step)
    assert derivative == pytest.approx(result.aggregate_gradient @ direction, rel=1e-6, abs=1e-10)


@pytest.mark.parametrize("module_name", ["wavecert.surrogates.pod", "wavecert.physics.helmholtz"])
def test_private_reference_binding_rejects_foreign_import_cache_even_with_matching_sibling_bytes(module_name, tmp_path, monkeypatch):
    import wavecert.surrogates.pod_batch as adapter

    reference = importlib.import_module(module_name)
    copied = tmp_path / Path(reference.__file__).name
    copied.write_bytes(Path(reference.__file__).read_bytes())
    monkeypatch.setattr(reference, "__file__", str(copied))
    with pytest.raises(RuntimeError, match="different checkout"):
        adapter._verify_reference_sources()


def test_private_reference_binding_rejects_changed_source_digest(monkeypatch):
    import wavecert.surrogates.pod_batch as adapter

    monkeypatch.setitem(adapter.REFERENCE_SOURCE_SHA256, "surrogates/pod.py", "0"*64)
    with pytest.raises(RuntimeError, match="frozen reference source"):
        adapter._verify_reference_sources()


@pytest.mark.parametrize("invalid", ["model_shape", "complex_model", "repeated_block", "receiver", "observation"])
def test_invalid_batched_survey_inputs_fail_explicitly(invalid):
    p, basis, model, receivers, blocks = _case()
    if invalid == "model_shape":
        model = model.reshape(p.shape)
    elif invalid == "complex_model":
        model = model.astype(complex)
    elif invalid == "repeated_block":
        blocks = blocks+(blocks[0],)
    elif invalid == "receiver":
        receivers = (True, 5, 5, 13)
    else:
        blocks = (SurveyBlock(3, 3., np.full(4, np.nan), "bad"),)
    with pytest.raises(ValueError):
        batched_pod_survey_fields(PODGalerkinSurrogate(p, basis), model, blocks, receivers)
