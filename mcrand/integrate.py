import numpy as np


def uniform_sampling(f, limits, N, *args):
    """MonteCarlo integration of multidimensional functions.

    Parameters
    ----------
    f : func
        integration function.
        input parameters:
        *   x : numpy.ndarray
                the evaluation point, n-dimensional vector
        *   *args : tuple
                extra parameters
        output parameters:
        *   y : float
    limits : list of tuple
        a list containing the lower and upper limits of integration for each dimension
        as tuple with two ints (low, high).
    N : int
        number of points used
    *args : args
        extra arguments for the evaluation function

    Returns
    -------
    integral : float
        the result of the integral
    error : float
        the standard deviation
    """
    if isinstance(limits, tuple):
        limits = [limits]

    for limit in limits:
        if limit[0] >= limit[1]:
            raise ValueError('Final value is smaller than initial value.')

    rands = np.array([np.random.uniform(low, high, N) for low, high in limits]).T

    weight = np.prod([high - low for low, high in limits])

    ys = np.array([f(x, *args) for x in rands]) * weight

    return np.mean(ys), np.std(ys)
