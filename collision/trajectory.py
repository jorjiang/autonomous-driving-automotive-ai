# -*- coding: utf-8 -*-
"""
Collision check a trajectory

Created on Mon Jun 13 13:23:39 2016

@author: Joseph Corbett-Davies
"""

import numpy as np
from dynamics.roadmap import Roadmap
from dynamics.controller import Vehicle4d
import collision
from shapely import geometry, affinity
from myplot import covariance_ellipse as cove, covariance_heatmap as covh
 
import matplotlib.pyplot as plt
from matplotlib import colorbar as cb, cm



def coll_prob(car, obst, car_pos, car_orientation, obst_mu, obst_cov, gamma_mu, 
              gamma_sd, fast=False):
    ''' Transform obstacle into car-centric coordinates to use with total_coll_prob 
    function. '''
    
    
    R = lambda theta: np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
    
    rot_mat = R(-car_orientation)
    
    # Put into car-centric frame
    rel_obst_mu = np.dot(rot_mat, obst_mu - car_pos) # shift and rotate about origin
    rel_obst_cov = np.dot(rot_mat, np.dot(obst_cov, rot_mat.T))
    rel_gamma_mu = gamma_mu - car_orientation

    if fast:
        return collision.fast_total_coll_prob(car, obst, 
                                     rel_obst_mu, rel_obst_cov, 
                                     rel_gamma_mu, gamma_sd)
    else:
        return collision.total_coll_prob(car, obst, 
                                     rel_obst_mu, rel_obst_cov, 
                                     rel_gamma_mu, gamma_sd)
    
    






def go():
    ''' Test simple roadmap following example '''    
    roadmap = Roadmap( {(4,0): [(8,4)],
                 (8,4): [(4,8)],
                 (4,8): [(0,4), (8,12)],
                 (0,4): [(4,0)],
                 (8,12): [(4,16)],
                 (4,16): [(0, 12)],
                 (0,12): [(4,8)]} )
    
    policy = {(4,8): (8,12)} # dict of intersection nodes with corresponding choice

    plt.close('all')
    fig = plt.figure()
    ax2 = fig.add_axes([0.8, 0.05, 0.05, 0.85]) 
    
    import matplotlib.animation as manim
    FFMpegWriter = manim.writers['ffmpeg']
    writer = FFMpegWriter(fps=10, metadata={'title': 'Collision Prob Test'} )       
    

    ax1 = fig.add_axes([0.05, 0.05, 0.75, 0.85])   
    plt.title('Collision probability')
    roadmap.plot()
    plt.autoscale()
    plt.axis('square')
    
        
    # Obstacle and car shape, mean, covariance setup
    obstacle = geometry.box(0,0,1.5,1)
    obstacle = affinity.translate(obstacle, -0.5, -0.5) # shift so back axle at origin
    
    
    # Make a rotated convariance matrix
    obst_cov = np.array([[4e-1,0],[0,2e-1]])
    theta = -np.pi/4
    R = lambda theta: np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
    obst_cov = np.dot(np.dot(R(theta), obst_cov), R(theta).T)
    obst_mu = np.array([5.5, 12.]) # mean obstacle position
    
    gamma_mu = -np.pi/4 # mean obstacle orientation
    gamma_sd = 0.5 # obstacle orientation std dev        
    
    
    
    # obstacle display polygon    
    disp_obst = affinity.rotate(obstacle, gamma_mu, origin=(0,0), use_radians=True)
    disp_obst = affinity.translate(disp_obst, obst_mu[0], obst_mu[1])
    #cove.plot_cov_ellipse(obst_cov, obst_mu, alpha=1.0, zorder=1)
    covh.plot_cov_heatmap(obst_cov, obst_mu, alpha=1.0, zorder=-5, cmap=cm.gray_r)

    plt.gca().add_patch(plt.Polygon(np.array(disp_obst.boundary), fc='w', lw=2, alpha=0.4, zorder=5))    

        
    
    # Set up vehicle instance
    vehicle_z0 = np.array([5,3,np.pi/4,0.0]).T
    v = Vehicle4d(vehicle_z0, geometry.box(-0.5,-0.5,1.0,0.5), roadmap, policy,
                  plot_lookahead=False, cmap=cm.hot)
    
    
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = cb.ColorbarBase(ax2, cmap=cm.hot)
    
    
    # Simulate
    coll_probs = []
    
    #with writer.saving(fig, 'Trajectory Collision Test.mp4', 200):

    for i in range(200):    
    
        p_coll = coll_prob(v._p, obstacle, 
                                    v._z[0:2], v._z[2], 
                                    obst_mu, obst_cov, 
                                    gamma_mu, gamma_sd)
        coll_probs.append(p_coll)
        
        v.plot(plt.gca(), prob=p_coll)
    
    
        #plt.title(str(i))
        #writer.grab_frame()
        print i
        plt.pause(0.001)
        
        v.move(0.2)
    
    plt.figure()
    plt.plot(coll_probs)
    

if __name__ == "__main__":
    go()
    
    
    
