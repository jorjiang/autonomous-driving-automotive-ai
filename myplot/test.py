# -*- coding: utf-8 -*-
"""
bivariate normal plotting test
@author: Joseph Corbett-Davies
"""

from matplotlib import mlab, pyplot as plt, cm
import covariance_ellipse as cove
import numpy as np

 # Make a rotated convariance matrix
obst_cov = np.array([[2e-1,0],[0,1e-1]])
theta = -np.pi/4
R = lambda theta: np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
cov = np.dot(np.dot(R(theta), obst_cov), R(theta).T)
mu = np.array([2, 2]) # mean obstacle position

stds = np.sqrt(np.diag(cov))
    
plt.close('all')

twosig = 10*stds.max()
x = np.linspace(-twosig,twosig,100) + mu[0]
y = np.linspace(-twosig,twosig,100) + mu[1]
X, Y = np.meshgrid(x,y)
Z = mlab.bivariate_normal(X, Y, stds[0], stds[1], 
                                      mu[0], mu[1],
                                     cov[1,0])

plt.figure()
plt.imshow(-Z, interpolation='bilinear', origin='lower',
                cmap=cm.gray, extent=(-twosig+mu[0], twosig+mu[0],
                                       -twosig+mu[1], twosig+mu[1]))
#plt.contour(X,Y,Z, levels=[.1,.2,.3,.4,.5,.6])

cove.plot_cov_ellipse(cov, mu, color='r', alpha=0.1)
plt.ylim([-10,10])
plt.xlim([-10,10])