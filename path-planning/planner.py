#!/usr/bin/env python
###############################################################################
# Duckietown - Project Unicorn ETH
# Author: Marta Tintore
# Plan optimal path from start to goal point, depending on intersection type.
###############################################################################
__all__ = [
    'path_generate',
]

import numpy as np
from scipy.special import comb
import helpers
from src.gym_duckietown import graphics

def bernstein_poly(i, n, t):
    """ The Bernstein polynomial of n, i as a function of t. """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, n_steps=20):
    """ Given a set of control points, return the bezier curve defined
    by the control points. Points should be a list of lists, or list of tuples
    such as [ [1,1], [2,3], [4,5], ..[Xn, Yn] ].
    @param[in]  n_steps         number of time steps. """
    n_points = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])
    t = np.linspace(0.0, 1.0, n_steps)
    polynomial_array = np.array([bernstein_poly(i, n_points-1, t) for i in range(0,n_points)])
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)
    return xvals, yvals

def path_generate_4way(env, trajectory, n_steps=20):
    ''' Generate path given driving direction using bezier curves
    based on (hard coded) optimal path points.
    @param[in]  direction       l=-1, r=1, s=0. '''
    
    i, j = helpers.get_4way_coord(env)
    #cps = env._get_curve(i,j)
    
    cps = env._get_curve(i,j)[trajectory, :, :]
    
    pts = [graphics.bezier_point(cps, i / (n_steps - 1)) for i in range(0, n_steps)]
    
    pts_2d = [[item[0], item[2]] for item in pts]
        
    return np.asarray(pts_2d)
    #else:
        #raise ValueError("Invalid path direction !")
    
    '''assert len(xs) == len(ys)
    xs, ys = np.insert(xs, len(xs), np.linspace(-0.16,-1,20)), np.insert(ys, len(ys), -0.1225*np.ones((20,)))
    path = np.zeros((len(xs),2))
    path[:,0] = np.asarray(np.flipud(xs))
    path[:,1] = np.asarray(np.flipud(ys))
    #print("path: ", path)
    return path'''