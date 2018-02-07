# -*- coding: utf-8 -*-
"""
bivariate normal plotting test
@author: Joseph Corbett-Davies
"""

from matplotlib import mlab, pyplot as plt, cm
import covariance_ellipse as cove
import numpy as np


def plot_cov_heatmap(cov, mu, nstd=4, nsteps=100, ax=None, 
                     cmap=cm.gray_r, scale=1.0, **kwargs):
    ''' Plot heatmap of bivariate normal distribution '''
    stds = np.sqrt(np.diag(cov))
    lim = nstd*stds.max()
    x = np.linspace(-lim,lim,nsteps) + mu[0]
    y = np.linspace(-lim,lim,nsteps) + mu[1]
    X, Y = np.meshgrid(x,y)
    Z = get_cov_heatmap(cov, mu, (-lim+mu[0], lim+mu[0], -lim+mu[1], lim+mu[1]), nsteps, scale)
        
    if ax is None:
        ax = plt.gca()
    
    return plt.imshow(Z, interpolation='bilinear', origin='lower', 
                      vmin=0.0, vmax=1.0,
               extent=(-lim+mu[0], lim+mu[0],
                       -lim+mu[1], lim+mu[1]), cmap=cmap, **kwargs)



def get_cov_heatmap(cov, mu, lim, nsteps=100, scale=1.0):
    stds = np.sqrt(np.diag(cov))
    x = np.linspace(lim[0],lim[1],nsteps)
    y = np.linspace(lim[2],lim[3],nsteps)
    X, Y = np.meshgrid(x,y)
    return scale*mlab.bivariate_normal(X, Y, stds[0], stds[1], 
                                      mu[0], mu[1],
                                     cov[1,0])


if __name__ == '__main__':
    plt.close('all')
     # Make a rotated convariance matrix
    obst_cov = np.array([[2e-1,0],[0,1e-1]])
    theta = -np.pi/4
    R = lambda theta: np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
    cov = np.dot(np.dot(R(theta), obst_cov), R(theta).T)
    mu = np.array([2, 2]) # mean obstacle position
    
    scale = 1.0    
    
    fig = plt.figure()
    img = plot_cov_heatmap(cov, mu, sign=-1, cmap=cm.RdBu, scale=scale)
    fig.colorbar(img)
    xlim = plt.xlim()
    ylim = plt.ylim()
    
    obst_cov = np.array([[4e-1,0],[0,2e-1]])
    theta = -np.pi/4
    R = lambda theta: np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
    cov = np.dot(np.dot(R(theta), obst_cov), R(theta).T)
    mu = np.array([2, 2]) # mean obstacle position
    
    fig = plt.figure()
    img = plot_cov_heatmap(cov, mu, sign=-1, cmap=cm.RdBu, scale=scale)
    fig.colorbar(img)
    plt.xlim(xlim)
    plt.ylim(ylim)