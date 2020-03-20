"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from random import choices

class RandGen(object):
	"""docstring for RandGen"""
	def __init__(self):
		pass
		
	@classmethod
	def distribution(cls, f, x0, xf, *args):
		# maximum number of samples returned

		max_sample = 10**5
		
		# maximum number of samples used at each iteration of the MC
		max_sample_per_iter = max_sample

		#Get the maximum of the f probability function

		x = f(np.linspace(x0, xf, 1000), *args)

		maximum = np.amax(x)

		new_randoms = np.empty(max_sample, dtype=float)

		n = 0
		while n < max_sample:

			random_numbers = np.random.uniform(x0, xf, max_sample_per_iter)
			dices = np.random.uniform(0, maximum, max_sample_per_iter)

			accept = random_numbers[f(random_numbers, *args) > dices]
			accepted = len(accept)

			if n + accepted > max_sample:
				new_randoms[n:max_sample] = accept[:max_sample - n]
				n = max_sample
			else:
				new_randoms[n:n+accepted] = accept[:]
				n += accepted

		return new_randoms

	@classmethod
	def sample(cls, f, x0, xf, size, *args):
		
		numbers = cls.distribution(f, x0, xf, *args)

		if isinstance(size, int):	
			return choices(numbers, k = size)

		elif isinstance(size, tuple):
			samples = choices(numbers, np.prod(size))
			return np.reshape(samples, size)

		else:
			raise NameError('Invalid size argument!')

class Integrate(object):
	"""docstring for HitMiss"""
	def __init__(self):
		pass

	@classmethod
	def UniformSampling(cls, f, x0, xf, N, *args):

		if not isinstance(x0, (list, np.ndarray)):
			raise NameError('x0 must be a list or numpy.ndarray of the initial points!')

		if not isinstance(xf, (list, np.ndarray)):
			raise NameError('x0 must be a list or numpy.ndarray of the final points!')

		if len(x0) != len(xf):
			raise ValueError('x0 and xf must have the same length!')

		x0 = np.array(x0)
		xf = np.array(xf)

		D = len(x0)

		rands = [np.random.uniform(x0[i], xf[i], N) for i in range(len(x0))]

		random_numbers_x = np.reshape(rands, (N,D))

		weight = np.prod(xf - x0)

		ys = np.array([f(x, *args) for x in random_numbers_x]) * weight

		integral = np.mean(ys)

		error = np.sqrt(np.var(ys)) / np.sqrt(N)

		return (integral, error)
		
