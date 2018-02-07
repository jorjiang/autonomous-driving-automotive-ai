# -*- coding: utf-8 -*-
"""
Code using shapely to find the minkowski sum of 2 convex polygons

Created on Thu Jun  9 10:25:56 2016

@author: Joseph Corbett-Davies
"""
from shapely import speedups, validation
speedups.enable()
import shapely.geometry as geom

import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import numpy as np

def minkowski_sum(poly1, poly2):
    ''' minkowski sum of two convex polygons. 
    Any random polygon that gets input here should be:
        oriented with geom.polygon.orient
        made convex with *.convex_hull
        cleaned with *.buffer(0)
        
    Some of this might be unnecesary but who knows.'''
    #assert poly1.is_valid, validation.explain_validity(poly1)
    #assert poly2.is_valid, validation.explain_validity(poly2)
    
    # Ensure correct edge ordering
    poly1 = geom.polygon.orient(poly1)
    poly2 = geom.polygon.orient(poly2)
    
    sum_edges = []
    sum_angles = []

    min_angle_points = []
    for p in [poly1, poly2]:
        try:
            boundary_pts = np.array(p.boundary)
        except TypeError:
            print 'In minkowski sum: p.boundary didn\'t work, so trying p.exterior'
            plt.figure()
            plt.gca().add_patch(plt.Polygon(np.array(p.exterior)))
            plt.gca().add_patch(plt.Polygon(np.array(p.convex_hull.exterior), alpha=0.5))

            plt.autoscale()
            plt.pause(2)
            plt.close()
            raise Exception
        edges = np.diff(boundary_pts, axis=0)
        angles = np.arctan2(edges[:,1], edges[:,0]) # polar angles
        angles[angles<0] = angles[angles<0] + 2*np.pi # wrap [0, 2pi]


        # save the point where the smallest-angle edge originates so we can shift final 
        # shape correctly
        min_angle_points.append(boundary_pts[np.argmin(angles),:])
        
        min_idx = angles.argmin() # does first edge have the smallest polar angle?
        if min_idx != 0: # shuffle until edges are sorted by angle
            edges = np.roll(edges, -min_idx, axis=0)
            angles = np.roll(angles, -min_idx)
        sum_edges.append(edges)
        sum_angles.append(angles)
    
    sum_edges = np.concatenate(sum_edges) # stick into a big numpy array
    sum_angles = np.concatenate(sum_angles)
    
    idx = np.argsort(sum_angles)
    
    sum_edges = sum_edges[idx,:] # sorted edges of the minkowski sum
    sum_angles = sum_angles[idx] # sorted angles

    # find exterior points on the resulting minkowski sum polygon
    # do this hack to get the absolute coord of the first point
    first_point = min_angle_points[0] + min_angle_points[1]    
    
    sum_ext = np.cumsum(np.concatenate((first_point[np.newaxis,:],sum_edges)), axis=0)
    sum_ext[-1,:] = sum_ext[0,:] # ensure start and end points match
    
    poly = geom.asPolygon(sum_ext)
    #assert poly.is_valid
    return poly
    
def minkowski_sum_2(poly1, poly2):
    ''' Slower minkowski sum algorithm (convex hull operation is a lot slower), but 
    uses blazing fast numpy sums and doesn't require particular polygon orientation'''
    
    assert poly1.is_valid, validation.explain_validity(poly1)
    assert poly2.is_valid, validation.explain_validity(poly2)

    ext1 = np.array(poly1.boundary)[:-1,:]
    ext2 = np.array(poly2.boundary)[:-1,:]
    n_ext2 = ext2.shape[0]    
    n_ext1 = ext1.shape[0]
    
    ext_sum = np.repeat(ext1, n_ext2, axis=0) + np.tile(ext2, (n_ext1, 1))    
    
    return geom.asPolygon(ext_sum).convex_hull



if __name__ == "__main__":
    # Test some stuff
    p1 = geom.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    p1 = geom.Point(0,0).buffer(2.0).simplify(0.5)
    p2 = geom.Polygon([(0, 0), (0, 1), (0.1, 0)])
    p2 = geom.Point(0,0).buffer(2.0).simplify(1)

    p3 = minkowski_sum(p1, p2) #+ p2 #minkowski sum    
    
    plt.close('all')

    plt.figure()
    patch1 = ptch.Polygon(np.array(p1.exterior), color='g', alpha=0.5)
    patch2 = ptch.Polygon(np.array(p2.exterior), alpha=0.5)
    patch3 = ptch.Polygon(np.array(p3.exterior), color='r')
    
    patch4 = ptch.Polygon(np.array(minkowski_sum_2(p1,p2).exterior), color='b', alpha=0.5)


    plt.gca().add_patch(patch1)
    plt.gca().add_patch(patch2)
    plt.autoscale()
    plt.axis('square')
    plt.title('A and B')
    
    plt.figure()
    plt.gca().add_patch(patch3)
    plt.gca().add_patch(patch4)

    plt.title('Minkowski sum A+B (blue is mink sum 2, red is mink sum 1)')
    plt.show()
    plt.axis('square')
    plt.autoscale()

