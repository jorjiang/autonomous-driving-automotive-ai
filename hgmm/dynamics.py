# -*- coding: utf-8 -*-
"""
Bicycle dynamics and road following behaviour for a road vehicle

Created on Mon May  9 12:32:32 2016

@author: Joseph Corbett-Davies
"""

from __future__ import division
import numpy as np
n_states = 4


def bicycle_plant(x_k, u_k, v_k, dt=1., l=5.):
    ''' find x(k+1) for state x(k)=[x,y,v,theta] and control input u(k) = [throttle, 
    steering angle]. Control input has noise v(k)'''
    res = np.array([[x_k[0] + dt*np.cos(x_k[3])*x_k[2]],
                     [x_k[1] + dt*np.sin(x_k[3])*x_k[2]],
                      [x_k[2] + dt*(u_k[0] + v_k[0])],
                       [x_k[3] + dt*l*x_k[2]*(u_k[1] + v_k[1])]], dtype=np.float64)
    return np.reshape(res, (n_states,1))
    

def path_following_controller(x_k, v0=10, kv=0.01, th0=np.pi/4, kth=0.1):
    ''' controller to keep speed at v0 and follow some path '''
    if x_k[0] < 20:
        th0 = np.pi/6
    elif x_k[0] < 40:
        th0 = -np.pi/6
    elif x_k[0] < 60:
        th0 = np.pi/6
    return np.array([[ kv*(v0 - x_k[2])    ],
                     [ kth*(th0 - x_k[3])  ]], dtype=np.float64)
    
def bicycle_following(x_k, v_k):
    ''' closed-loop dynamics for bicycle with controller u().'''
    return bicycle_plant(x_k, path_following_controller(x_k), v_k)
    
def UNGM(x_k, v_k, k):
    return 0.3*x_k + x_k/(1+x_k**2) + np.cos(1.2*k)