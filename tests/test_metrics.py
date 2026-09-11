import numpy as np

from wavecert.metrics import cosine_similarity, relative_l2


def test_metrics_sanity():
    x = np.array([1.0, 2.0, 3.0])
    assert relative_l2(x, x) == 0.0
    assert np.isclose(cosine_similarity(x, x), 1.0)
    assert np.isclose(cosine_similarity(x, -x), -1.0)
