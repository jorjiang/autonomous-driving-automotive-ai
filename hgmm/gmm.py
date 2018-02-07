# -*- coding: utf-8 -*-
"""
Implementation of Frank Havlak's hybrid Gaussian mixture models, from his 2013 
journal paper.
 
Created on Sun Apr 24 07:27:13 2016

@author: Joseph Corbett-Davies
"""

from __future__ import division
import numpy as np
from myplot import covariance_ellipse as cvre, covariance_heatmap as cvrh
import itertools
from matplotlib import pyplot as plt, cm
import warnings

class Mixand(object):
    def __init__(self, w, mu, P, x_d):
        '''
        Parameters
        -----------
        w : float [0,1]
            weight of the mixand
        mu : np.array with shape (n_x,1)
            mean of mixand gaussian
        P : np.array with shape (n_x,n_x)
            covariance of gaussian
        x_d : discrete hypthesis object
            
        
        
        
        '''
        assert mu.shape[0] == P.shape[0]
        
        self.w = w
        self.mu = np.reshape(mu, (mu.shape[0],1))
        self.P = P
        
        # x_d object must implement copy() and __eq__(), 
        # a dictionary would work. See VehicleDiscreteState for another example
        self.x_d = x_d 
                
    @property
    def mu(self):
        return self._mu        
        
    @mu.setter    
    def mu(self, value):
        if hasattr(Mixand, '_mu'):
            assert value.shape == self.mu.shape
        self._mu = value
        
    def copy(self):
        return Mixand(self.w, self.mu, self.P, self.x_d)
                
        
class GaussianMixture(object):
    def __init__(self, mixands):
        self.mixands = set(mixands)
        assert np.allclose(np.sum(self.weights), 1), 'weights don\'t add to one'
        
        self._im = None #plotting image

        
    def reduce_N(self, m):
        ''' reduce mixture to m mixands, using kld upper bound metric and moment
        preserving merge'''
        while self.N > m:
            best = (np.inf, None, None)
            for i, j in itertools.combinations(self.mixands, 2):
                if i.x_d == j.x_d: # ensure the discrete states are the same
                    ub = kld_upper_bound(i, j)
                    if ub < best[0]:
                        best = (ub, i, j)
            _, i, j = best
            
            self.mixands.remove(i)
            self.mixands.remove(j)
            
            self.mixands.add(moment_preserving_merge(i, j))
            if not np.allclose(np.sum(self.weights), 1):
                warnings.warn("after reduction weights don't sum to one")
        
    @property    
    def N(self):
        return len(self.mixands)

        
    @property    
    def weights(self):
        return np.array([m.w for m in self.mixands])
        
    def __str__(self):
        ret = 'GMM: \n'
        for m in self.mixands:
            ret += 'w: {}, mu: {}, P {}\n'.format(m.w, m.mu, m.P)
        return ret
        
    def copy(self):
        ''' Return a shallow copy of the mixture'''
        return GaussianMixture({m.copy() for m in self.mixands})
        
    def plot_ellipse(self, **kwargs):
        ''' plot first two dimensions of the mixture'''
        ellipses = []
        for m in self.mixands:
            p = cvre.plot_cov_ellipse(m.P[:2,:2], m.mu[:2], alpha=m.w, **kwargs)
            ellipses.append(p)
        return ellipses
        
            
    def plot_heatmap(self, nstd=10, nsteps=200, cmap=cm.gray_r):
        ''' plot first two dimensions of the mixture'''
        xmin, xmax, ymin, ymax = np.inf, None, np.inf, None
        for m in self.mixands:
            # find image extents
            xmin = min(xmin, m.mu.flatten()[0] - nstd*np.sqrt(np.diag(m.P)[0]))
            xmax = max(xmax, m.mu.flatten()[0] + nstd*np.sqrt(np.diag(m.P)[0]))
            ymin = min(ymin, m.mu.flatten()[1] - nstd*np.sqrt(np.diag(m.P)[1]))
            ymax = max(ymax, m.mu.flatten()[1] + nstd*np.sqrt(np.diag(m.P)[1]))

        Z = 0
        lim = (xmin, xmax, ymin, ymax)
        for m in self.mixands:
            Z += m.w*cvrh.get_cov_heatmap(m.P[:2,:2], m.mu[:2].flatten(), lim, nsteps)
            
        if self._im is not None: # remove previously plotted image
            self._im.remove()
            del self._im
            
        self._im = plt.imshow(Z, vmin=0., vmax=1., interpolation='none', origin='lower',
               extent=lim, cmap=cmap)
    
    def plot(self, **kwargs):
        self.plot_heatmap(**kwargs)

def moment_preserving_merge(mixand_i, mixand_j):
    ''' Find a single gaussian mixand that preserves the 0th, 1st, and 2nd moments of
    the gaussian mixture of mixand_i and mixand_j'''
    
    w_ij = mixand_i.w + mixand_j.w
    
    w_i_ij = mixand_i.w/w_ij
    w_j_ij = mixand_j.w/w_ij
    mu_ij = w_i_ij*mixand_i.mu + w_j_ij*mixand_j.mu
    
    mu_diff = mixand_i.mu - mixand_j.mu
    P_ij = w_i_ij*mixand_i.P + w_j_ij*mixand_j.P + \
                w_i_ij*w_j_ij*np.dot(mu_diff, mu_diff.T)
    
    return Mixand(w_ij, mu_ij, P_ij, mixand_i.x_d)
    

def kld_upper_bound(mixand_i, mixand_j):
    ''' find the upper bound on the kl discrimination between the mixture given by 
    {mixand_i, mixand_j} and the moment-preserving merge of the two gaussians.
    '''
    merged = moment_preserving_merge(mixand_i, mixand_j)
    
    return 0.5*( merged.w*np.log(np.linalg.det(merged.P)) - \
            mixand_i.w*np.log(np.linalg.det(mixand_i.P)) - \
            mixand_j.w*np.log(np.linalg.det(mixand_j.P)) )


