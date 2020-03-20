#Tests and examples for the Monte Carlo Random Number Generator library (MCRand)

import numpy as np
import matplotlib.pyplot as plt
import time

import scipy.stats as sp_stats
from MCRand import RandGen as rg

def gaussian(x, mu, sigma):
	return (1/(np.sqrt(2*np.pi*sigma**2))) * np.exp(-(x-mu)**2/(2*sigma**2))

def exponential(x):
	return np.exp(-x)

def cauchy(x, x0, gamma):
	return 1 / (np.pi * gamma * (1 + ((x-x0)/(gamma))**2))

def rayleigh(x, sigma):
	return (x*np.exp(-(x**2)/(2*sigma**2))) / (sigma**2)

def maxwell_boltzmann(x, sigma):
	return (np.sqrt(2/np.pi))*(x**2*np.exp(-(x**2)/(2*sigma**2))) / (sigma**3)

def symmetric_maxwell_boltzmann(x, sigma):
	return 0.5*(np.sqrt(2/np.pi))*(x**2*np.exp(-(x**2)/(2*sigma**2))) / (sigma**3)

def invented(x, sigma):
	return (x**2*np.exp(-(x**2)/(2*sigma**2))) / (2.506628*sigma**2)

current_milli_time = lambda: time.time() * 1000


#####################################################################
#																	#
#								TESTS								#
#																	#
#####################################################################

#Gaussian distribution
def gaussian_test():

	print('---------- GAUSSIAN TEST ----------\n')

	x0 = -5
	xf = 5
	N = 50000
	sigma = 1
	mu = 0

	print('sigma=%.2f, mu=%.2f\n' % (sigma, mu))

	t0 = current_milli_time()
	numpy_rand = np.random.normal(mu, sigma, N)
	tf = current_milli_time()

	print('NumPy gaussian random generator mean: ', np.mean(numpy_rand))
	print('NumPy std error:', np.std(numpy_rand)/np.sqrt(N) * 100, '%')
	print('Elapsed time: %.4f ms' % (tf-t0, ))


	t0 = current_milli_time()	
	rand = rg.sample(gaussian, x0, xf, N, mu, sigma)
	tf = current_milli_time()

	print('\nMC gaussian random generator mean: ', np.mean(rand))
	print('MCRand std error:', np.std(rand)/np.sqrt(N) * 100, '%')
	print('Elapsed time: %.4f ms' % (tf-t0, ))

	x = np.linspace(x0, xf, N)

	plt.figure(figsize=(9,6))

	plt.hist(numpy_rand, bins=30, density=True, color=(0,0,1,0.8), label='NumPy sample')
	plt.hist(rand, bins=30, density=True, color=(0,1,0,0.5), label='MCRand sample')
	plt.plot(x, gaussian(x, mu, sigma), color='r', label=r'Gaussian PDF $\mu=%.2f$, $\sigma=%.2f$' % (mu,sigma))
	plt.text(-5.3, 0.45, r'PDF(x)=$\frac{1}{\sqrt{2\pi\sigma^2}}\cdot e^{-\frac{(x-\mu)^2}{2\sigma^2}}$', size=15)

	#(1/(np.sqrt(2*np.pi*sigma**2))) * np.exp(-(x-mu)**2/(2*sigma**2))

	plt.title('Gaussian distribution test')
	plt.xlabel('Sample number')
	plt.ylabel('Probability')

	plt.ylim(0, 0.5)

	plt.legend()
	plt.show()

#Exponential distribution
def exponential_test():
	
	print('\n\n---------- EXPONENTIAL DISTRIBUTION TEST ----------\n')

	x0 = 0
	xf = 10
	N = 10**5

	
	t0 = current_milli_time()
	numpy_rand = np.random.exponential(size=N)
	tf = current_milli_time()

	print('Numpy exponential random generator mean: ', np.mean(numpy_rand))
	print('NumPy std error:', np.std(numpy_rand)/np.sqrt(N) * 100, '%')
	print('Elapsed time: %.4f ms' % (tf-t0, ))
	

	t0 = current_milli_time()	
	rand = rg.sample(exponential, x0, xf, N)
	tf = current_milli_time()

	print('\nMC exponential random generator mean: ', np.mean(rand))
	print('MCRand std error:', np.std(rand)/np.sqrt(N) * 100, '%')
	print('Elapsed time: %.4f ms' % (tf-t0, ))

	x = np.linspace(x0, xf, N)

	plt.hist(numpy_rand, bins=30, density=True, color=(0,0,1,0.8), label='NumPy sample')
	plt.hist(rand, bins=30, density=True, color=(0,1,0,0.5), label='MCRand sample')
	plt.plot(x, exponential(x), color='r', label='Exponential PDF')
	plt.text(2, 0.95, r'$PDF(x)=e^{-x}$', size=15)

	plt.title('Exponential distribution test')
	plt.xlabel('Sample number')
	plt.ylabel('Probability')

	plt.legend()
	plt.show()

#Cauchy distribution
def cauchy_test():

	print('\n\n---------- CAUCHY DISTRIBUTION TEST ----------\n')

	x0 = -10
	xf = 10
	N = 10**5

	x0_cauchy = 0
	gamma = 1


	t0 = current_milli_time()
	s = np.random.standard_cauchy(size=N)
	numpy_cauchy = s[(s>-10) & (s<10)]  # truncate distribution so it plots well
	tf = current_milli_time()

	print('NumPy Cauchy random generator mean: ', np.mean(numpy_cauchy))
	print('NumPy std error:', np.std(numpy_cauchy)/np.sqrt(N) * 100, '%')
	print('Elapsed time: %.4f ms' % (tf-t0, ))


	t0 = current_milli_time()	
	rand = rg.sample(cauchy, x0, xf, N, x0_cauchy, gamma)
	tf = current_milli_time()

	print('\nMC Cauchy random generator mean: ', np.mean(rand))
	print('MCRand std error:', np.std(rand)/np.sqrt(N) * 100, '%')
	print('Elapsed time: %.4f ms' % (tf-t0, ))

	x = np.linspace(x0, xf, N)

	plt.figure(figsize=(9,6))

	plt.hist(numpy_cauchy, bins=50, density=True, color=(0,0,1,0.8), label='NumPy sample')
	plt.hist(rand, bins=50, density=True, color=(0,1,0,0.5), label='MCRand sample')
	plt.plot(x, cauchy(x, x0_cauchy, gamma), color='r', label=r'Cauchy PDF $\gamma=%.2f$, $x_0=%.2f$' % (gamma, x0_cauchy))
	plt.text(-10, 0.34, r'PDF(x)=$\frac{1}{\pi\gamma[1+(\frac{x-x_0}{\gamma})^2]}$', size=15)

	plt.title('Cauchy distribution test')
	plt.xlabel('Sample number')
	plt.ylabel('Probability')

	plt.xlim(-11,11)
	plt.ylim(0, 0.38)

	plt.legend(loc='upper right')
	plt.show()

#Rayleigh distribution
def rayleigh_test():
	
	print('\n\n---------- RAYLEIGH DISTRIBUTION TEST ----------\n')

	x0 = 0
	xf = 4
	sigma = 1
	N = 10**5

	
	t0 = current_milli_time()
	numpy_rand = np.random.rayleigh(scale=sigma,size=N)
	tf = current_milli_time()

	print('Numpy rayleigh random generator mean: ', np.mean(numpy_rand))
	print('NumPy std error:', np.std(numpy_rand)/np.sqrt(N) * 100, '%')
	print('Elapsed time: %.4f ms' % (tf-t0, ))
	

	t0 = current_milli_time()	
	rand = rg.sample(rayleigh, x0, xf, N, sigma)
	tf = current_milli_time()

	print('\nMC rayleigh random generator mean: ', np.mean(rand))
	print('MCRand std error:', np.std(rand)/np.sqrt(N) * 100, '%')
	print('Elapsed time: %.4f ms' % (tf-t0, ))

	x = np.linspace(x0, xf, N)

	plt.hist(numpy_rand, bins=30, density=True, color=(0,0,1,0.8), label='NumPy sample')
	plt.hist(rand, bins=30, density=True, color=(0,1,0,0.5), label='MCRand sample')
	plt.plot(x, rayleigh(x, sigma), color='r', label=r'Rayleigh PDF $\sigma=%.2f$' % sigma)
	plt.text(2.5, 0.3, r'$PDF(x)=\frac{x\cdot\exp(-\frac{x^2}{2\sigma^2})}{\sigma^2}$', size=15)

	plt.title('Rayleigh distribution test')
	plt.xlabel('Sample number')
	plt.ylabel('Probability')

	plt.legend()
	plt.show()

#Maxwell-Boltzmann distribution
def maxwell_boltzmann_test():
	
	print('\n\n---------- MAXWELL-BOLTZMANN DISTRIBUTION TEST ----------\n')

	x0 = 0
	xf = 10
	sigma = 2
	N = 10**5	

	t0 = current_milli_time()	
	rand = rg.sample(maxwell_boltzmann, x0, xf, N, sigma)
	tf = current_milli_time()

	print('\nMC Maxwell-Boltzmann random generator mean: ', np.mean(rand))
	print('Elapsed time: %.4f ms' % (tf-t0, ))

	x = np.linspace(x0, xf, N)

	plt.hist(rand, bins=30, density=True, color=(0,1,0,0.5), label='MCRand sample')
	plt.plot(x, maxwell_boltzmann(x, sigma), color='r', label=r'Maxwell-Boltzmann PDF $\sigma=%.2f$' % sigma)
	plt.text(5, 0.15, r'$PDF(x)=\sqrt{\frac{2}{\pi}}\cdot\frac{x^2\exp(-\frac{x^2}{2\sigma^2})}{\sigma^3}$', size=15)

	plt.title('Maxwell-Boltmann distribution test')
	plt.xlabel('Sample number')
	plt.ylabel('Probability')

	plt.legend()
	plt.show()

#Symmetric Maxwell-Boltzmann distribution
def symmetric_maxwell_boltzmann_test():
	
	print('\n\n---------- SYMMETRIC MAXWELL-BOLTZMANN DISTRIBUTION TEST ----------\n')

	x0 = -10
	xf = 10
	sigma = 2
	N = 10**5	

	t0 = current_milli_time()	
	rand = rg.sample(symmetric_maxwell_boltzmann, x0, xf, N, sigma)
	tf = current_milli_time()

	print('\nMC Symmetric Maxwell-Boltzmann random generator mean: ', np.mean(rand))
	print('Elapsed time: %.4f ms' % (tf-t0, ))

	x = np.linspace(x0, xf, N)

	plt.figure(figsize=(12,6))

	plt.hist(rand, bins=40, density=True, color=(0,1,0,0.5), label='MCRand sample')
	plt.plot(x, symmetric_maxwell_boltzmann(x, sigma), color='r', label=r'Maxwell-Boltzmann PDF $\sigma=%.2f$' % sigma)
	plt.text(-10.5, 0.135, r'$PDF(x)=\sqrt{\frac{1}{2\pi}}\cdot\frac{x^2\exp(-\frac{x^2}{2\sigma^2})}{\sigma^3}$', size=15)

	plt.title('Symmetric Maxwell-Boltmann distribution test')
	plt.xlabel('Sample number')
	plt.ylabel('Probability')

	plt.legend()
	plt.show()

#Invented distribution
def invented_test():
	
	print('\n\n---------- MODIFIED RAYLEIGH DISTRIBUTION TEST ----------\n')

	x0 = -4
	xf = 4
	sigma = 1
	N = 10**5	

	t0 = current_milli_time()	
	rand = rg.sample(invented, x0, xf, N, sigma)
	tf = current_milli_time()

	print('\nMC invented random generator mean: ', np.mean(rand))
	print('Elapsed time: %.4f ms' % (tf-t0, ))

	x = np.linspace(x0, xf, N)

	plt.figure(figsize=(12,5))

	plt.hist(rand, bins=40, density=True, color=(0,1,0,0.5), label='MCRand sample')
	plt.plot(x, invented(x, sigma), color='r', label=r'Modified Rayleigh PDF $\sigma=%.2f$' % sigma)
	plt.text(-4.5, 0.3, r'$PDF(x)=\frac{x^2\exp(-\frac{x^2}{2\sigma^2})}{2.506628\cdot\sigma^2}$', size=15)

	plt.title('Modified Rayleigh distribution test')
	plt.xlabel('Sample number')
	plt.ylabel('Probability')

	plt.xlim(-5,5)
	plt.ylim(0, 0.35)

	plt.legend()
	plt.show()


#####################################################################
#																	#
#						ACTIVATE FUNCTIONS							#
#																	#
#####################################################################

gaussian_test()
#exponential_test()
#cauchy_test()
#rayleigh_test()
#maxwell_boltzmann_test()
#symmetric_maxwell_boltzmann_test()
#invented_test()

