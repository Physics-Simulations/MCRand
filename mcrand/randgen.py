import numpy as np
from scipy.optimize import minimize_scalar


def distribution(pdf, bounds, max_sample=10**5, *args):
    """Generator of random numbers following the given probability density function.

    Parameters
    ----------
    pdf : func
        the probability density function.

        - Input parameters:
            -   `x` : float
                    the evaluation point
            -   `*args` : tuple
                    extra parameters
        - Output parameters:
            -   `y` : float
    bounds : tuple of floats
        lower and upper limit
    max_sample : int
        the quantity of random numbers to return
    *args : args
        extra arguments for the pdf

    Returns
    -------
    random_numbers : numpy.ndarray
        collection of random numbers following the given pdf.

    Notes
    -----
    pdf is expected to be a probability density function therefore it must be positively defined in the range specified.
    """
    if bounds[0] >= bounds[1]:
        raise ValueError('End limit is smaller or equal than initial limit.')

    # maximum number of samples used at each iteration of the MC
    max_sample_per_iter = max_sample // 2

    # Get the maximum of the f probability function
    res = minimize_scalar(lambda x: -pdf(x, *args), bounds=bounds, method='bounded')
    if not res.success:
        raise RuntimeError('scipy.optimize resulted in error code {}: {}'.format(res.status, res.message))
    maximum = res.x

    new_randoms = np.empty(max_sample, dtype=float)

    n = 0
    while n < max_sample:
        random_numbers = np.random.uniform(bounds[0], bounds[1], max_sample_per_iter)
        dices = np.random.uniform(0, maximum, max_sample_per_iter)

        accept = random_numbers[pdf(random_numbers, *args) > dices]
        accepted = len(accept)

        if n + accepted > max_sample:
            new_randoms[n:max_sample] = accept[:max_sample - n]
            n = max_sample
        else:
            new_randoms[n:n+accepted] = accept[:]
            n += accepted

    return new_randoms


def sample(pdf, x0, xf, shape, *args):
    """Generator of random numbers following the given probability density function.

    Parameters
    ----------
    pdf : func
        the probability density function
    x0 : float
        the lower limit
    xf : float
        the upper limit
    shape : int or tuple of ints
        the shape of the generated random numbers array
    *args : args
        extra arguments for the pdf

    Returns
    -------
    random_numbers : numpy.ndarray
        collection of random numbers following the given pdf with the specified shape.

    See Also
    --------
    mcrand.distribution
    """
    numbers = distribution(pdf, (x0, xf), 10 * np.prod(shape), *args)
    return np.random.choice(numbers, shape)
