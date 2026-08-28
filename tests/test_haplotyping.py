import numpy as np
import pytest

from abacus.parameter_estimation import (
    discrete_multivariate_normal_logpdf,
    estimate_mean_flanking,
    flanking_logpdf,
)


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
            np.array([0.001, 0.001, 0.001]),
            np.array([0]),
            id="One count, multiple dimensions",
        ),
        pytest.param(
            np.array([[1, 2, 3], [1, 2, 3]]),
            np.array([1, 2, 3]),
            np.array([0.001, 0.001, 0.001]),
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


@pytest.mark.parametrize(
    ("x", "mean", "unit_var", "is_left_flanking", "expected"),
    [
        pytest.param(
            np.array([]),
            np.array([0]),
            np.array([1]),
            np.array([]),
            np.array([]),
            id="No counts",
        ),
        pytest.param(
            np.array([[1]]),
            np.array([1]),
            np.array([0.01]),
            np.array([True]),
            np.array([0]),
            id="One count",
        ),
        pytest.param(
            np.array([[1, 1, 1]]),
            np.array([1, 1, 1]),
            np.array([0.01, 0.01, 0.01]),
            np.array([True]),
            np.array([0]),
            id="One count, multiple dimensions",
        ),
        pytest.param(
            np.array([[1], [1], [1]]),
            np.array([1]),
            np.array([0.01]),
            np.array([True, True, True]),
            np.array([0, 0, 0]),
            id="Multiple counts",
        ),
        pytest.param(
            np.array([[1, 1, 1], [1, 1, 1]]),
            np.array([1, 1, 1]),
            np.array([0.01, 0.01, 0.01]),
            np.array([True, True]),
            np.array([0, 0]),
            id="Multiple counts, multiple dimensions",
        ),
    ],
)
def test_flanking_logpdf(x: np.ndarray, mean: np.ndarray, unit_var: np.ndarray, is_left_flanking: list[bool], expected: np.ndarray):
    logpdf = flanking_logpdf(x, mean, unit_var, is_left_flanking)

    assert logpdf.shape[0] == x.shape[0]
    assert np.all(logpdf <= 0)
    assert np.all(logpdf >= -np.inf)
    assert np.allclose(logpdf, expected, atol=1e-5)


@pytest.mark.parametrize(
    ("x", "gamma", "is_left_flank", "expected"),
    [
        pytest.param(
            np.array([]),
            np.array([]),
            [],
            np.array([]),
            id="Empty",
        ),
        pytest.param(
            np.array([[0, 0, 0]]),
            np.array([1]),
            [True],
            np.array([0, 0, 0]),
            id="All zeros, left flank",
        ),
        pytest.param(
            np.array([[0, 1, 1]]),
            np.array([1]),
            [True],
            np.array([0, 1, 1]),
            id="One sample, left flank",
        ),
        pytest.param(
            np.array([[1, 1, 0]]),
            np.array([1]),
            [False],
            np.array([1, 1, 0]),
            id="One sample, right flank",
        ),
        pytest.param(
            np.array([[1, 1, 0], [1, 1, 0]]),
            np.array([1, 1]),
            [False, False],
            np.array([1, 1, 0]),
            id="Two sample, right flank",
        ),
        pytest.param(
            np.array([[1, 2], [1, 2], [2, 2]]),
            np.array([0, 0.5, 0.5]),
            [True, True, True],
            np.array([1.25, 1.5]),
            id="Multiple samples, left flank",
        ),
    ],
)
def test_estimate_mean_flanking(x: np.ndarray, gamma: np.ndarray, is_left_flank: list[bool], expected: np.ndarray):
    mean_flanking = estimate_mean_flanking(x, gamma, is_left_flank)

    assert np.allclose(mean_flanking, expected, atol=1e-5)
