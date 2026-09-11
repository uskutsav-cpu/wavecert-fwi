from wavecert.physics.devito_backend import DevitoAcousticFWI


def test_devito_adapter_exposes_scipy_optimization_contract():
    assert hasattr(DevitoAcousticFWI, "objective_and_gradient")
    assert hasattr(DevitoAcousticFWI, "squared_slowness_loss")
    assert hasattr(DevitoAcousticFWI, "run_lbfgsb")
