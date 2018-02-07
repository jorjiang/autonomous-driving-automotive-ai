# -*- coding: utf-8 -*-
"""
Gaussian splitting test
Created on Thu May  5 08:48:36 2016

@author: Joseph Corbett-Davies
"""

from __future__ import division
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import gmm
import scipy.stats as stats
plt.close('all')


cov_reduction = 0.3
N = 3

opt_spread, opt_weights, isd = None, None, None

def precompute(N, cov_reduction, n):
    ''' Precompute split parameters for splitting into N n-dimensional mixands, with
    covariance reduction cov_reduction'''
    global opt_spread, opt_weights, isd
    global J11
    J11 = stats.multivariate_normal(mean=np.zeros(n), cov=2*np.eye(n)).pdf(np.zeros(n))
    opt_spread, opt_weights, isd = find_optimal_spread(N, cov_reduction, n)
    print 'Desired number of splits =', N, ', desired cov reduction =', cov_reduction 
    print 'Precomputed split params: delta =', opt_spread, ', weights =', opt_weights

def split_gaussian(mixand, e_split):
    ''' Split gaussian along axis e_split, using a transformation of precomputed 
    split result '''
    
    if opt_weights is None:
        raise Exception('No precomputed split available')    
    
    n = max(mixand.mu.shape)
    w_prev = mixand.w
    x_d = mixand.x_d
    
    # Find transformation matrices
    T = np.linalg.cholesky(mixand.P)
    e_1 = np.concatenate( ([1.], np.zeros(n-1)) ) # rotate to x-axis splitting axis
    R = get_rot_mat_from_vects(e_split, e_1)
    
    # Get precomputed split result
    mu_star = np.zeros((n,N))
    for i in range(N):
        mu_star[0,i] = (i - (N-1)/2)*opt_spread  
    cov_star = np.eye(n)
    cov_star[0,0] = cov_reduction
    
    
        
    # Transform back
    return [gmm.Mixand( opt_weights[i]*w_prev, 
            np.dot(np.dot(T,R.T), mu_star[:,i,np.newaxis]) + mixand.mu,
            np.dot(np.dot(np.dot(np.dot(T,R.T), cov_star), R), T.T), x_d.copy() ) for i in range(N)]
    
    
def get_rot_mat_from_vects (x, y):
    ''' Find an n-dimensional rotation matrix between two n-length vectors x and y '''
    assert max(y.shape) == max(x.shape)
    n = max(x.shape)

    if np.allclose(x, y):
        return np.eye(n)
        
    x = np.reshape(x, (n,1)) #/ np.linalg.norm(x)
    y = np.reshape(y, (n,1)) #/ np.linalg.norm(y)
    
    u = x / np.linalg.norm(x) # (n,1)
    v = y - np.dot(u.T,y) * u # (n,1)
    v = v / np.linalg.norm(v) # (n,1)
    cost = np.dot(x.T, y) / np.linalg.norm(x) / np.linalg.norm(y) #(1,1)
    sint = np.sqrt(1 - cost**2)
    cost = cost[0,0]
    sint = sint[0,0]
    u_v = np.concatenate((u, v), axis=1)
    R = np.eye(n) - np.dot(u,u.T) - np.dot(v,v.T) + \
            np.dot(np.dot(u_v, np.array([[cost, -sint],[sint, cost]])), u_v.T)
    #R = eye(length(x))-uu'-vv' + [u v]* [cost -sint;sint cost] *[u v]';
    
    # Test result
    try:
        assert np.allclose(np.dot(R,R.T), np.eye(n)), 'Not orthogonal'
        assert np.allclose(y*np.linalg.norm(x)/np.linalg.norm(y), np.dot(R,x)), 'Doesn\'t rotate correctly'
        assert np.allclose(np.linalg.det(R), 1), 'Not det=1'
    except Exception:
        print 'x:', x.flatten(),'y:', y.flatten()
        print R
        print np.dot(R, R.T)
        raise Exception
    
    return R

def find_optimal_spread(N, cov_reduction, n):
    """ Use numerical optimization to find best mean_spread value for a given number
    of gaussians N and target covariance reduction cov_reduction. """
    
    def loss(mean_spread):
        return find_optimal_weights(N, cov_reduction, mean_spread, n)[1]
    
    opt_spread = optimize.brute(loss, ranges=[(0,1)], Ns=200)
    opt_w, isd = find_optimal_weights(N, cov_reduction, opt_spread, n)
    return opt_spread, opt_w, isd

def find_optimal_weights(N, cov_reduction, mean_spread, n):
    """ Solve quadratic program to find optimal weights for N gaussians to miminize
    integral squared difference bewteen the mixture and a zero-mean unity gaussian, 
    given target covariance reduction cov_reduction and distance between means 
    mean_spread. Also optionally returns the value of the integral squared 
    difference for the optimal weights."""
    
    cov_i = np.eye(n)
    cov_i[0,0] = cov_reduction
    
    means = np.zeros((n,N))
    for i in range(N):
        means[0,i] = (i - (N-1)/2)*mean_spread
    
    H = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            H[i,j] = stats.multivariate_normal(means[:,j], 2*cov_i).pdf(means[:,i])
    
    f = np.zeros((N,1))
    for i in range(N):
        f[i,0] = stats.multivariate_normal(means[:,i], np.eye(n)+cov_i).pdf(np.zeros(n))
        
    c = -f.T    
    b = np.zeros((N,1))
    A = np.eye(N)
        
    # J_ISD = J_11 - 2*f.T*w + w^T*H*w
    def loss(w, sign=1.):
        return sign * (np.dot(w.T, np.dot(H, w)) + 2*np.dot(c, w) + J11)

    def jac(w, sign=1.):
        return sign * (2*np.dot(w.T, H) + 2*c)
    
    one_array = np.ones((N,1))
    cons = ({'type':'eq',
             'fun': lambda w: np.dot(one_array.T, w) - 1.,
             'jac': lambda w: one_array.T },)
    '''{'type':'ineq',
             'fun':lambda w: (np.dot(A,w) - b)[:,0],
             'jac':lambda w: A}'''
    
    bounds = [(0,1) for _ in range(N)]    
    
    opt = {'disp': False}
    
    res = optimize.minimize(loss, x0=np.ones(N)/N, jac=jac, bounds=bounds, 
                            constraints=cons, method='SLSQP', options=opt)
    w = res.x

    # Check results
    assert res.success, "quadratic program optimization failed"
    assert np.allclose(np.sum(w), 1.0), "Weights don't add to one" 
    assert np.all(w >= 0) and np.all(w <= 1), "Weights are't 0 < w < 1, " + str(w)
    assert loss(w).shape[0] == 1, "Weird things are happening" 
    
    return w, loss(w)[0]
