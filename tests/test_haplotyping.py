import numpy as np
import pytest

from abacus.haplotyping import (
    adjust_logpdf_for_flanks,
    discrete_multivariate_normal_logpdf,
    estimate_heterozygous_parameters,
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
    ("x", "is_left_flank", "logpdf", "expected"),
    [
        pytest.param(
            np.array([]),
            [],
            np.array([]),
            np.array([]),
            id="Empty",
        ),
        pytest.param(
            np.array([[0, 0, 0]]),
            [True],
            np.array([[0, 0, 0]]),
            np.array([[0, 0, 0]]),
            id="All zeros, left flank",
        ),
        pytest.param(
            np.array([[0, 0, 0]]),
            [False],
            np.array([[0, 0, 0]]),
            np.array([[0, 0, 0]]),
            id="All zeros, right flank",
        ),
        pytest.param(
            np.array([[0, 1, 1]]),
            [True],
            np.array([[1, 1, 1]]),
            np.array([[0, 1, 1]]),
            id="Fixed left flank",
        ),
        pytest.param(
            np.array([[1, 1, 0]]),
            [False],
            np.array([[1, 1, 1]]),
            np.array([[1, 1, 0]]),
            id="Fixed right flank",
        ),
        pytest.param(
            np.array([[0, 1, 1, 1], [1, 1, 0, 0]]),
            [True, False],
            np.array([[1, 1, 1, 1], [1, 1, 1, 1]]),
            np.array([[0, 1, 1, 1], [1, 1, 0, 0]]),
            id="Multiple samples",
        ),
    ],
)
def test_adjust_logpdf_for_flanks(x: np.ndarray, is_left_flank: list[bool], logpdf: np.ndarray, expected: np.ndarray):
    adjust_logpdf_for_flanks(x, is_left_flank, logpdf)

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


@pytest.mark.parametrize(
    ("spanning_counts", "flanking_counts", "is_left_flank"),
    [
        pytest.param(
            np.array([[1, 2], [3, 4]]),
            np.array([[1, 2], [3, 4]]),
            [True, False],
            id="Simple case",
        ),
        pytest.param(
            np.array([[1, 1], [1, 1]]),
            np.array([[1, 1], [1, 1]]),
            [True, True],
            id="Identical counts",
        ),
        pytest.param(
            np.array([[2], [3], [3], [15], [16]]),
            np.array([[1], [4], [5], [9]]),
            [False, False, False, False],
            id="Case1",
        ),
    ],
)
def test_estimate_heterozygous_parameters(spanning_counts: np.ndarray, flanking_counts: np.ndarray, is_left_flank: list[bool]):
    mean_h1, mean_h2, var_h1, var_h2, pi = estimate_heterozygous_parameters(spanning_counts, flanking_counts, is_left_flank)

    # Check dimensions
    assert mean_h1.shape == mean_h2.shape == var_h1.shape == var_h2.shape
