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
        pytest.param([], [0], [1], [], id="No counts"),
        pytest.param([[1]], [1], [0.01], [0], id="One count"),
        pytest.param([[1], [1], [1]], [1], [0.01], [0, 0, 0], id="Multiple counts"),
        pytest.param([[1, 2, 3]], [1, 2, 3], [0.001, 0.001, 0.001], [0], id="One count, multiple dimensions"),
        pytest.param([[1, 2, 3], [1, 2, 3]], [1, 2, 3], [0.001, 0.001, 0.001], [0, 0], id="Multiple counts, multiple dimensions"),
    ],
)
def test_discrete_multivariate_normal_logpdf(x: list, mean: list, var: list, expected: list):
    x, mean, var, expected = map(np.array, (x, mean, var, expected))
    logpdf = discrete_multivariate_normal_logpdf(x, mean, var)

    assert logpdf.shape[0] == x.shape[0]
    assert np.all(logpdf <= 0)
    assert np.allclose(logpdf, expected, atol=1e-5)


@pytest.mark.parametrize(
    ("x", "mean", "unit_var", "is_left_flanking", "expected"),
    [
        pytest.param([], [0], [1], [], [], id="No counts"),
        pytest.param([[1]], [1], [0.01], [True], [0], id="One count"),
        pytest.param([[1, 1, 1]], [1, 1, 1], [0.01, 0.01, 0.01], [True], [0], id="One count, multiple dimensions"),
        pytest.param([[1], [1], [1]], [1], [0.01], [True, True, True], [0, 0, 0], id="Multiple counts"),
        pytest.param([[1, 1, 1], [1, 1, 1]], [1, 1, 1], [0.01, 0.01, 0.01], [True, True], [0, 0], id="Multiple counts, multiple dimensions"),
    ],
)
def test_flanking_logpdf(x: list, mean: list, unit_var: list, is_left_flanking: list[bool], expected: list):
    x, mean, unit_var, is_left_flanking, expected = map(np.array, (x, mean, unit_var, is_left_flanking, expected))
    logpdf = flanking_logpdf(x, mean, unit_var, is_left_flanking)

    assert logpdf.shape[0] == x.shape[0]
    assert np.all(logpdf <= 0)
    assert np.allclose(logpdf, expected, atol=1e-5)


@pytest.mark.parametrize(
    ("x", "gamma", "is_left_flank", "expected"),
    [
        pytest.param([], [], [], [], id="Empty"),
        pytest.param([[0, 0, 0]], [1], [True], [0, 0, 0], id="All zeros, left flank"),
        pytest.param([[0, 1, 1]], [1], [True], [0, 1, 1], id="One sample, left flank"),
        pytest.param([[1, 1, 0]], [1], [False], [1, 1, 0], id="One sample, right flank"),
        pytest.param([[1, 1, 0], [1, 1, 0]], [1, 1], [False, False], [1, 1, 0], id="Two sample, right flank"),
        pytest.param([[1, 2], [1, 2], [2, 2]], [0, 0.5, 0.5], [True, True, True], [1.25, 1.5], id="Multiple samples, left flank"),
    ],
)
def test_estimate_mean_flanking(x: list, gamma: list, is_left_flank: list[bool], expected: list):
    x, gamma, expected = map(np.array, (x, gamma, expected))
    mean_flanking = estimate_mean_flanking(x, gamma, is_left_flank)

    assert np.allclose(mean_flanking, expected, atol=1e-5)
