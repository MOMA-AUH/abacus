import numpy as np
import pytest

from abacus.haplotyping import discrete_multivariate_normal_logpdf


@pytest.mark.parametrize(
    ("x", "mean", "var", "expected"),
    [
        pytest.param(
            np.array([]),
            np.array([0]),
            np.array([1]),
            np.array([]),
            id="No counts",
        ),
        pytest.param(
            np.array([[1]]),
            np.array([1]),
            np.array([0.01]),
            np.array([0]),
            id="One count",
        ),
        pytest.param(
            np.array([[1], [1], [1]]),
            np.array([1]),
            np.array([0.01]),
            np.array([0, 0, 0]),
            id="Multiple counts",
        ),
        pytest.param(
            np.array([[1, 2, 3]]),
            np.array([1, 2, 3]),
            np.array([0.01, 0.01, 0.01]),
            np.array([0]),
            id="One count, multiple dimensions",
        ),
        pytest.param(
            np.array([[1, 2, 3], [1, 2, 3]]),
            np.array([1, 2, 3]),
            np.array([0.01, 0.01, 0.01]),
            np.array([0, 0]),
            id="Multiple counts, multiple dimensions",
        ),
    ],
)
def test_discrete_multivariate_normal_logpdf(x: np.ndarray, mean: np.ndarray, var: np.ndarray, expected: np.ndarray):
    logpdf = discrete_multivariate_normal_logpdf(x, mean, var)

    assert logpdf.shape[0] == x.shape[0]
    assert np.all(logpdf <= 0)
    assert np.all(logpdf >= -np.inf)
    assert np.allclose(logpdf, expected, atol=1e-5)
