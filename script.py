#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import collections as col
import scipy as sp
import scipy.optimize as spopt

plt.close('all')

def _gaussvar(x,amp,mu,variance,bg=0):
	return amp*np.exp(-(x-mu)**2/(2*variance))+bg

def iter(factor):
	# Generate random numbers with gaussian
	# distribution, mu=0, sigma = 3
	x=np.random.randn(10000)*3
	
	# Bin the counts
	bins = 50
	h,edge = np.histogram(x,bins=bins)

	# Find the midpoints of the bins
	mids = edge + (edge[1]-edge[0])/2
	mids = mids[:-1]

	# Expected error in counts is sqrt(counts)
	sigma = np.sqrt(h)*1

	# Error of zero counts isn't zero, but less than one.
	# Use 0.5 as a guess.
	sigma[sigma==0] = 0.5

	# Factor scales error, which should cause
	# the pcov matrix to change
	sigma = sigma*factor
	
	# Fit the histogram to a gaussian
	# popt,pcov,red_chisq = mt.gaussfit(mids,h,sigma_y=sigma,plot=True,variance_bool=True)


	# Find initial guesses
	y=h
	x=mids
	amp = max(y)
	mu  = sum(x*y)/sum(y)
	variance = sum(x**2 * y)/sum(y)
	bg  = 0
	p0 = np.array((amp,mu,variance,bg))
	

	# Do actual curve fit
	func = _gaussvar
	popt,pcov = spopt.curve_fit(func,x,y,sigma=sigma,p0=p0)

	output = col.namedtuple('iterout',['popt','pcov'])
	out = output(popt,pcov)
	return out

num_samples       = 1000
variances_regular = np.ones(num_samples)
variances_large   = np.ones(num_samples)
means_regular     = np.ones(num_samples)
means_large       = np.ones(num_samples)
for i in np.linspace(1,num_samples,num_samples)-1:
	out                  = iter(1)
	means_regular[i]     = out.popt[2]
	variances_regular[i] = out.pcov[2,2]

	out                = iter(1e10)
	means_large[i]     = out.popt[2]
	variances_large[i] = out.pcov[2,2]

plt.hist(variances_regular,bins=20)
plt.hist(variances_large,bins=20)

plt.show()
