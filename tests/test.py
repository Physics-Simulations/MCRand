import pytest
import sys
import os

import numpy as np

sys.path.insert(0, os.getcwd())

from mcrand import sample, uniform_integration  # noqa: E402


def uniform_function(x):
    return 1


@pytest.mark.parametrize('bounds, raises', [
    ((0, 1), False),
    ((1, 1), True),
    ((1, 0), True)
], ids=['Valid', 'Equal', 'Smaller'])
def test_bounds_are_valid(bounds, raises):
    if raises:
        with pytest.raises(ValueError):
            sample(uniform_function, bounds[0], bounds[1], (5, 4))
    else:
        sample(uniform_function, bounds[0], bounds[1], (5, 4))


@pytest.mark.parametrize('shape', [
    (10, ),
    (4, 50),
    [3, 2, 1],
    np.array([5, 6])
], ids=['int', 'tuple', 'list', 'numpy array'])
def test_sample_shape(shape):
    rands = sample(uniform_function, 0, 1, shape)
    rands.shape = tuple(shape)


@pytest.mark.parametrize('limits, raises', [
    ((0, 1), False),
    ((1, 1), True),
    ((1, 0), True)
], ids=['Valid', 'Equal', 'Smaller'])
def test_integration_limits(limits, raises):
    if raises:
        with pytest.raises(ValueError):
            uniform_integration(uniform_function, limits, 10)
    else:
        uniform_integration(uniform_function, limits, 10)


@pytest.mark.parametrize('f, limits, result', [
    (uniform_function, (0, 1), 1),
    (np.exp, (0, 1), np.expm1(1)),
    (np.cos, (0, np.pi / 2), 1),
    (np.sqrt, (0, 1), 2 / 3)
], ids=['uniform', 'exp', 'cos', 'sqrt'])
def test_uniform_integration(f, limits, result):
    x, e = uniform_integration(f, limits, 10**5)
    assert np.abs(x - result) <= e
