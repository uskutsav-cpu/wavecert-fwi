from wavecert.repair.policies import BlockCertificate, selectively_verify


def test_selective_verification_repairs_largest_uncertainty_first():
    blocks = [
        BlockCertificate("a", -1.0, 2.0, lambda: -1.1),
        BlockCertificate("b", -0.8, 0.1, lambda: -0.9),
    ]
    result = selectively_verify(blocks)
    assert result.certified_descent
    assert result.repaired_blocks == ("a",)
    assert result.exact_evaluations == 1
