# -*- coding: utf-8 -*-
"""
Recreating Jason Hardy's efficient collision probability algorithm

Created on Thu Jun  9 09:32:12 2016

@author: Joseph Corbett-Davies
"""

from __future__ import division
from shapely import affinity, ops, speedups, validation
speedups.enable()
import shapely.geometry as geom
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from scipy import special
from myplot.covariance_ellipse import plot_cov_ellipse
import numpy as np
import math
from minkowski import minkowski_sum
import contextlib
import sys

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout
            
obstacle_bounds = []
obstacle_orientations = []


def obstacle_bound(shape, orientations):
    ''' return the convex hull of a shape rotated about (0,0) by the angles in the
     orientations sequence. '''
    with nostdout():
        shapes = []
        for angle in orientations:
            shapes.append(affinity.rotate(shape, angle, origin=(0,0), use_radians=True))
    
        union = ops.cascaded_union(shapes)
        bound = union.convex_hull.convex_hull # apparently sometimes one convex_hull isn't enough
        # TODO: check for convexity before retrying convex hull operation
        # sometimes convex-hull returns valid non-convex polygons well fuck
        if not bound.is_valid: # in case it randomly shits out
            # For debug:        
            
    
            # clean shape
            bound = bound.buffer(0.0).convex_hull.convex_hull
            
            if not bound.is_valid:
                raise Exception('why doesnt this work')
            
        obstacle_bounds.append(bound)
        obstacle_orientations.append(orientations)        
            
        return bound
    
def min_bounding_box(p):
    ''' return the minimum-area bounding box (any orientation) of a convex polygon p.'''
    
    def rotate_points(pts, theta):
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        return np.dot(R, pts.T).T
        
    def bbox_area(pts):
        min_pts = pts.min(axis=0)
        max_pts = pts.max(axis=0)
        return np.product(max_pts - min_pts)
    
    edges = np.diff(np.array(p.boundary), axis=0)
    angles = np.arctan2(edges[:,1], edges[:,0]) # polar angles of each edge in p

    zero_pt = (0,0)
    pts = np.array(p.boundary)
    
    '''Minimum-area bounding box of cvx poly must be aligned with an edge.
    So rotate polygon by negative polar angle, fit axis-aligned bbox, repeat and 
    choose box with minimum area'''
    min_area = np.inf
    for angle in angles: 
        pts_rot = rotate_points(pts, -angle)
        area = bbox_area(pts_rot)
        if area < min_area:
            min_area = bbox_area(pts_rot)
            min_angle = angle
            
    p_rot = affinity.rotate(p, -min_angle, origin=zero_pt, use_radians=True)
    return affinity.rotate(p_rot.envelope, min_angle, origin=zero_pt, use_radians=True), min_angle

        
def combined_body(car, obst, orientations):
    ''' Find combined body of car with (x,y) centre coords car_cent and obstacle with 
    coords obst_cent. i.e if a measured/predicted obst_cent is within the CB then it 
    is in collision with the car.
    obst_cent is tuple (not point object) of the center coordinates rel. to obst'''
    
    # Convex hull of copies of obst rotated about obst_cent
    bound = obstacle_bound(obst, orientations)
    
    bound_flip = affinity.scale(bound, xfact=-1, yfact=-1, origin=(0,0)) 
    
    return minkowski_sum(car, bound_flip)    
    
    


def cov_transform(p, obst_cov):
    ''' Returns the convex polygon p transformed by W, where W*cov*W^T = I'''
    W = np.linalg.inv( np.linalg.cholesky(obst_cov) )
    affine_params = np.concatenate((W.flatten(), np.zeros(2)))  
    return affinity.affine_transform(p, affine_params), W
    
    
def cov_trans_and_rotate(p, obst_cov):
    ''' do cov transform as above, fit minimum bounding box, then rotate so bbox is
    axis-aligned '''
    p_w, W = cov_transform(p, obst_cov)
    bbox, bbox_angle = min_bounding_box(p_w)
       
    p_rw = affinity.rotate(p_w, -bbox_angle, origin=(0,0), use_radians=True)
    bbox_r = affinity.rotate(bbox, -bbox_angle, origin=(0,0), use_radians=True)
    
    theta = -bbox_angle
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]]) 
    return p_rw, bbox_r, np.dot(R,W)

    
def rect_coll_prob_ub(car, obst, obst_mu, obst_cov, obst_angles):
    ''' collision probability for transformed cb bounding rectangle
    Find an upper bound approx collision probabilty for a known car and obstacle
    geometry and a normally distributed obstacle position, for some given obstacle 
    orientation range.
    
    - car and obst 
        are shapely polygons with centre (e.g. back axle) at (0,0) and
        with orientation of 0.
    - obst_mu and obst_cov
        are the mean and covariance of the x,y position of the 
        obstacle centre relative to the car centre.
    - obst_angles 
        is a list of angles (usually just max/min) describing the 
        orientation interval
    
    '''
        
    cb = combined_body(car, obst, obst_angles)
    
    cb_rw, bbox_r, RW = cov_trans_and_rotate(cb, obst_cov)
    
    cdf = lambda z: 0.5*(1 + math.erf(z/np.sqrt(2)))
    
    minx, miny, maxx, maxy = bbox_r.bounds
    mu_rw = np.dot(RW, np.reshape(obst_mu, (2,1)) ).flatten()
    
    return ( cdf(maxx-mu_rw[0]) - cdf(minx-mu_rw[0]) ) * ( 
                 cdf(maxy-mu_rw[1]) - cdf(miny-mu_rw[1]) ) , mu_rw, cb_rw, bbox_r

def max_extent(p):
    ''' Get distance to furthest point in polygon p '''
    verts = np.array(p.exterior)[:-1]
    return np.max(np.linalg.norm(verts, axis=0))
    
def circ_coll_prob(car, obst, obst_mu, obst_cov):   
    ''' estimate collision probability between car and obst using circular 
    over-approximation '''
     
    radius = max_extent(car) + max_extent(obst) # radius of circular CB
    
    W = np.linalg.inv( np.linalg.cholesky(obst_cov) ) # diagonalize covariance

    U,S,V = np.linalg.svd(W)
        
    RW = np.dot(U,W)
    
    extents = S*radius # [major axis length, minor axis length] # of cb_rw
    # bounding box of cb_rw
    minx, miny, maxx, maxy = -extents[0], -extents[1], extents[0], extents[1]
    #bbox_r = geom.box(minx, miny, maxx, maxy)

    # Calculate probability
    cdf = lambda z: 0.5*(1 + math.erf(z/np.sqrt(2)))
    
    mu_rw = np.dot(RW, np.reshape(obst_mu, (2,1)) ).flatten()
    
    return ( cdf(maxx-mu_rw[0]) - cdf(minx-mu_rw[0]) ) * ( 
                 cdf(maxy-mu_rw[1]) - cdf(miny-mu_rw[1]) ) #, mu_rw, bbox_r
    
    

def total_coll_prob(car, obst, 
                    obst_mu, obst_cov, 
                    gamma_mu, gamma_sd,
                    delta=0.99, n_intervals=5):
    
    """
    Calculates collision probability between car at x,y,theta = (0,0,0) and some
    uncertain obstacle. Returns an upper-bound collision probability using combined 
    body and circular over-approximation.

    Parameters
    ----------
    
        car, obst
            Shapely convex polygons of the car and obstacle, with the object origin 
            at (0,0) and at a zero angle orientation
        obst_mu, obst_cov
            mean (2,) and covariance (2,2) of the obstacle position
        gamma_mu, gamma_sd
            mean and standard deviation of obstacle orientation
        delta
            (optional) the proportion of probability to be explained by 'accurate' 
            probability calculations (the remainder is calculated using circular 
            over-approximation).
        n_intervals
            the number of discrete obstacle orientation ranges used for the 'accurate'
            probability calculations.

    Returns
    -------
        collision probability upper bound
    """
    
    cdf = lambda z: 0.5*(1 + math.erf(z/np.sqrt(2)))
    probit = lambda p: -np.sqrt(2)*special.erfcinv(2*p)

    # gamma confidence interval to consider 
    # (pick min and max such that p(gamma between min & max) = delta)
    min_gamma = probit((1-delta)/2)*gamma_sd + gamma_mu   
    max_gamma = (gamma_mu - min_gamma) + gamma_mu      
    #print 'gamma range:', min_gamma, max_gamma

    gamma_ranges = np.linspace(min_gamma, max_gamma, n_intervals+1)
    p_gamma = np.empty(n_intervals) # p that gamma is in range i
    p_rect = np.empty(n_intervals)  # p of collision | gamma range i
    p_coll = np.empty(n_intervals)  # p of collision + gamma range i
    for i in range(n_intervals):
        g_min = gamma_ranges[i]
        g_max = gamma_ranges[i+1]
        p_gamma[i] = cdf((g_max-gamma_mu)/gamma_sd) - cdf((g_min-gamma_mu)/gamma_sd)
        p_rect[i] = rect_coll_prob_ub(car, obst, obst_mu, obst_cov, (g_min, g_max))[0]
        p_coll[i] = p_gamma[i] * p_rect[i]
    # Approximate the circular bound probability with p = 1
    p_circ = 1 #circ_coll_prob(car, obst, obst_mu, obst_cov)[0]
    #print 'p_circ:', p_circ
    
    assert np.allclose(p_gamma.sum(), delta)
    
    return p_coll.sum() + (1-delta)*p_circ
   
   
def fast_total_coll_prob(car, obst, 
                         obst_mu, obst_cov, 
                         gamma_mu, gamma_sd,
                         delta=0.99, n_intervals=5, circ_thresh=0.05):  
    '''
    Check collision first using circular over-approximation, if the collision 
    probability calculated using the approximation is above *circ_thresh*, then 
    use total_coll_prob function for a better approximation.
    '''
    p_coll_circ = circ_coll_prob(car, obst, obst_mu, obst_cov) 

    if p_coll_circ > circ_thresh:
        return total_coll_prob(car, obst, 
                               obst_mu, obst_cov, 
                               gamma_mu, gamma_sd, delta, n_intervals)
    else:
        return p_coll_circ
            
    
if __name__ == "__main__":
    # Test it
    plt.close('all')
    
    obstacle = geom.box(0,0,2,1)
    obstacle = affinity.translate(obstacle, -0.5, -0.5) # shift so back axle at origin
    
    car = geom.box(0,0,2,1)
    car = affinity.translate(car, -0.5, -0.5) # shift so back axle at origin
    
        
    plt.figure()
 
    # Make a rotated convariance matrix
    obst_cov = np.array([[2e-1,0],[0,1e-1]])
    theta = -np.pi/4
    R = lambda theta: np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
    obst_cov = np.dot(np.dot(R(theta), obst_cov), R(theta).T)
    
    
    obst_mu = np.array([2, 2]) # mean obstacle position
    gamma_mu = -np.pi/4 # mean obstacle orientation
    gamma_sd = 0.1 # obstacle std dev
    

    prob_tot = total_coll_prob(car, obstacle, obst_mu, obst_cov, gamma_mu, gamma_sd)
    print 'prob total:', prob_tot    
    
    circ = geom.Point(0,0).buffer(max_extent(car)+max_extent(obstacle))
    
    disp_obst = affinity.translate(
        affinity.rotate(obstacle, gamma_mu, origin=(0,0), use_radians=True), 
        obst_mu[0], obst_mu[1])   
    
    plt.gca().add_patch(ptch.Polygon(np.array(car.exterior), alpha=0.5))
    plt.gca().add_patch(ptch.Polygon(np.array(disp_obst.exterior), alpha=0.5, color='r'))
    plt.gca().add_patch(ptch.Polygon(np.array(circ.exterior), alpha=0.1, color='g'))
    plot_cov_ellipse(obst_cov, obst_mu, fill=False, ls='--', alpha=0.1)

    plt.plot(obst_mu[0], obst_mu[1], 'k+')
    plt.plot(0, 0, 'k+')
    
    plt.autoscale()
    plt.axis('square')