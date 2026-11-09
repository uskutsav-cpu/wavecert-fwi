from wavecert.physics.independent_verification import verify_physics_independently


def test_sparse_wave_solver_matches_independent_spectral_solution_and_continuum_order():
    result = verify_physics_independently(sizes=(7, 15, 31))
    assert result["passed"]
    assert min(result["observed_orders"]) > 1.9
