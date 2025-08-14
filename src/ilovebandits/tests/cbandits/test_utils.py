"""Test utils functions."""

import math

import numpy as np
import pytest

from src.ilovebandits.utils import argmax, find_max_indices, find_max_numbers


# to debug to see if a function returns the expected error:
def test_divide_by_zero():
    """Test pytest library."""
    with pytest.raises(ZeroDivisionError):
        res = 10 / 0
        print(res)


def test_find_max_numbers():
    """Test find_max_numbers()."""
    numbers = [1, 3, 7, 7, 2, 5, 7]
    assert find_max_numbers(numbers) == [7, 7, 7]


def test_find_max_indices():
    """Test find_max_indices()."""
    numbers = [1, 3, 7, 7, 2, 5, 7]
    assert find_max_indices(numbers) == [2, 3, 6]


def test_argmax():
    """Test argmax()."""
    rng = np.random.default_rng()
    numbers = [float("inf"), 0, 1, 2, float("-inf")]
    idx_chosen, prob, _ = argmax(numbers, rng=rng)
    assert (idx_chosen == 0) and (prob == 1)

    numbers = [1, 3, 7, 7, 2, 5, 7]
    idx_chosen, prob, list_ties = argmax(numbers, rng=rng)
    assert list_ties == [2, 3, 6]
    assert (idx_chosen in [2, 3, 6]) and math.isclose(prob, 1 / 3, rel_tol=1e-6)
    idx_chosen, prob, _ = argmax(numbers, rng=rng)
    assert (idx_chosen in [2, 3, 6]) and math.isclose(prob, 1 / 3, rel_tol=1e-6)
    idx_chosen, prob, _ = argmax(numbers, rng=rng)
    assert (idx_chosen in [2, 3, 6]) and math.isclose(prob, 1 / 3, rel_tol=1e-6)

    numbers = [1, 3, float("inf"), float("inf"), 2, 5, float("inf")]
    idx_chosen, prob, list_ties = argmax(numbers, rng=rng)
    assert list_ties == [2, 3, 6]
    assert idx_chosen in [2, 3, 6] and math.isclose(prob, 1 / 3, rel_tol=1e-6)
    idx_chosen, prob, _ = argmax(numbers, rng=rng)
    assert idx_chosen in [2, 3, 6] and math.isclose(prob, 1 / 3, rel_tol=1e-6)
    idx_chosen, prob, _ = argmax(numbers, rng=rng)
    assert idx_chosen in [2, 3, 6] and math.isclose(prob, 1 / 3, rel_tol=1e-6)
