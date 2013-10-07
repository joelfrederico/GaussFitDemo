#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import mytools as mt
import collections as col
import scipy as sp
import scipy.optimize as spopt

plt.close('all')

def gauss(x,amp,mu,sigma):
	return amp*np.exp(-(x-mu)**2/(2*sigma**2))

def iter():
	x=np.random.randn(10000)*3
	
	bins = 50
	
	# plt.hist(x,bins=bins)
	
	h,edge = np.histogram(x,bins=bins)
	# sigma = 1e10
	# sigma = None
	sigma = np.sqrt(h)
	sigma[sigma==0] = 1e10
	
	mids = edge + (edge[1]-edge[0])/2
	mids = mids[:-1]
	
	# plt.plot(mids,h,'o-')
	# print gauss
	# print mids
	# print h
	# print sigma
	
	# popt,pcov = spopt.curve_fit(gauss,mids,h,sigma=sigma)
	popt,pcov = mt.gaussfit(mids,h,sigma_y=sigma)

	# print popt
	# print pcov
	vari = np.sqrt(pcov[1,1])
	
	# x = np.linspace(-15,15,200)
	# y = gauss(x,popt[0],popt[1],popt[2])
	
	# plt.plot(mids,h,'o-',x,y)
	# plt.figure()
	# plt.plot(x,y)
	
	# print 'Mean is {} +/- {}.'.format(popt[1],vari)
	# print 'One std dev range is {} to {}.'.format(popt[1]-vari,popt[1]+vari)
	print '--------'
	print popt[1]
	print vari

	output = col.namedtuple('iterout',['popt','pcov'])
	out = output(popt,pcov)
	return out

num_samples = 1000
variances = np.ones(num_samples)
means = np.ones(num_samples)
for i in np.linspace(1,num_samples,num_samples)-1:
	out = iter()
	means[i] = out.popt[1]
	variances[i] = out.pcov[1,1]
	# mt.hist2d(out.x.flatten(),out.y.flatten(),bins=70)

print 'Std Dev. is {}'.format(np.std(means))
print 'Average std dev is {}.'.format(np.mean(np.sqrt(variances)))
print 'Std dev of std devs is {}.'.format(np.std(np.sqrt(variances)))

# plt.hist(variances,bins=20)
h,edge = mt.hist(means,bins=15)
plt.figure()
mt.gaussfit(edge,h)
plt.show()

