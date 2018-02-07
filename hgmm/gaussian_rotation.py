# -*- coding: utf-8 -*-
"""
Gaussian rotation test
Created on Thu May  5 08:48:36 2016

@author: Joseph Corbett-Davies
"""

from __future__ import division
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import covariance_ellipse as cvre
import scipy.stats as stats
plt.close('all')


def get_rot_mat_from_vects (x, y):
    ''' Find an n-dimensional rotation matrix between two n-length vectors x and y '''
    assert max(y.shape) == max(x.shape)
    n = max(x.shape)
    x = np.reshape(x, (n,1))
    y = np.reshape(y, (n,1))
    
    u = x / np.linalg.norm(x) # (n,1)
    v = y - np.dot(u.T,y) * u # (n,1)
    v = v / np.linalg.norm(v) # (n,1)
    cost = np.dot(x.T, y) / np.linalg.norm(x) / np.linalg.norm(y) #(1,1)
    sint = np.sqrt(1 - cost**2)
    cost = cost[0,0]
    sint = sint[0,0]
    print cost, sint
    u_v = np.concatenate((u, v), axis=1)
    print u_v
    R = np.eye(n) - np.dot(u,u.T) - np.dot(v,v.T) + \
            np.dot(np.dot(u_v, np.array([[cost, -sint],[sint, cost]])), u_v.T)
    # R = eye(length(x))-uu'-vv' + [u v]* [cost -sint;sint cost] *[u v]';
    
    # Test result
    assert np.allclose(np.dot(R,R.T), np.eye(n))
    assert np.allclose(y, np.dot(R,x))
    
    return R
    
    
n = 10
cov = np.random.rand(n,n)
cov = np.dot(np.dot(cov.T, np.diag(np.abs(np.random.rand(n)))), cov)

T = np.linalg.cholesky(cov)

e_split = np.random.rand(n,1)
e_split = e_split / np.linalg.norm(e_split) # normalise
print e_split

e_desired = np.zeros((n,1))
e_desired[0,0] = 1

print rot_mat_nd(e_split, e_desired)