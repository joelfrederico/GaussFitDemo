#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import collections as col
import scipy as sp
import scipy.optimize as spopt
import mytools as mt

plt.close('all')

def _gaussvar(x,amp,mu,variance,bg=0):
	return amp*np.exp(-(x-mu)**2/(2*variance))+bg

def line(x,b):
	return b*np.ones(len(x))

def iter(factor):
	# Generate a random-ish line from 0 to 1
	num_pts = 1000
	x = np.linspace(1,num_pts,num_pts)
	y = 30 + np.random.randn(num_pts)*4

	# Expected error in counts is the sigma of the
	# randomly generated points
	sigma = 4 * np.ones(num_pts)

	# Factor scales error, which should cause
	# the pcov matrix to change
	sigma = sigma*factor
	
	# Initial guess close to 30
	p0 = (25)

	# Do actual curve fit
	func = line
	popt,pcov = spopt.curve_fit(func,x,y,sigma=sigma,p0=p0)

	output = col.namedtuple('iterout',['popt','pcov','x','y','sigma'])
	out = output(popt,pcov,x,y,sigma)
	return out

num_samples       = 1000
variances_regular = np.ones(num_samples)
# variances_large   = np.ones(num_samples)
means_regular     = np.ones(num_samples)
# means_large       = np.ones(num_samples)
for i in np.linspace(1,num_samples,num_samples)-1:
	out                  = iter(1)
	means_regular[i]     = out.popt[0]
	variances_regular[i] = out.pcov[0,0]

	# out                = iter(1e10)
	# means_large[i]     = out.popt[0]
	# variances_large[i] = out.pcov[0,0]

# plt.hist(variances_regular,bins=20)
# plt.hist(variances_large,bins=20)

print 'Average mean is {}.'.format(np.mean(means_regular))
print 'Average variance is {}.'.format(np.mean(variances_regular))

plt.hist(means_regular)

print np.std(means_regular)

plt.show()
