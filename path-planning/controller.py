#!/usr/bin/env python
###############################################################################
# Duckietown - Project intnav ETH
# Author: Marta Tintore
# Pure pursuit controller implementation.
###############################################################################
__all__ = [
    'Controller',
]

import numpy as np
from scipy.spatial.distance import cdist

class Controller(object):
    def __init__(self,direction,path,wheel_distance,
                     adm_error=0.005, la_dis=0.25, min_r=0.2, vel=0.1, n_hist=4):
        '''@param[in]  path            [[x,y],...] target path, numpy array.
        @param[in]  wheel_distance  differential drive vehicle baseline [m].
        @param[in]  adm_error       admissible error (perpendicular distance
                         of future point from the path) [m].
        @param[in]  la_dis          look ahead distance [m].
        @param[in]  min_r           minimal navigation radius [m].
        @param[in]  t_step          time interval to calculate future point [s].
        @param[in]    min_r           minimal turning radius [m].
        @param[in]  vel             output velocity as PP merely controls
                         the steering angle [m/s].'''
        self.direction = direction
        self.path = path
        self.wheel_distance = wheel_distance
        self.adm_error = adm_error
        self.la_dis = la_dis
        self.min_r=min_r
        self.vel=vel
        self.x = 0
        self.y = 0
        self.a = 0 #With x axis (driving direction of the car)
        self.theta_hist = np.zeros(n_hist,)
        self.right_angle = -np.pi/2 
        self.n_hist = n_hist

    def pure_pursuit(self,pose):
        ''' Pure pursuit implementation determining control commands (v, tau)
        based on the current pose (x,y,thetha) and an target path.
        @param[in]  pose            (x,y,thetha) current pose [m,rad].'''
        assert len(pose) == 3
        self.x, self.y, self.theta = pose
        self.theta_hist = np.roll(self.theta_hist,1)
        self.theta_hist[0]= self.theta
        #Check if Right angle is reached
        exit = False
        if(self.direction=='R'):
            exit = True
        for i in range(self.n_hist):
            if(self.theta_hist[i]>self.right_angle):
                exit=False
        if(exit): 
            return 0,0
        # Predict future point location.
        actual = np.zeros((1,2))
        #future = np.zeros((1,2))
        actual[0,0]=self.x
        actual[0,1]=self.y
        dist_all = cdist(self.path,actual,'euclidean')
        idx_shortest = np.where(dist_all==np.min(dist_all))
        idx_shortest = idx_shortest[0]
        projected_pt = self.path[idx_shortest[0],:]
        distance = 0
        while(distance < self.la_dis and idx_shortest<len(self.path)-1):
            distance += np.linalg.norm(self.path[idx_shortest+1,:]-self.path[idx_shortest,:])
            idx_shortest += 1
        idx_next = idx_shortest
        goal = self.path[idx_next,:]
        # From goal point --> vehicle action (velocity & steering vector).
        sv = (goal[0,0]-actual[0,0],
              goal[0,1]-actual[0,1]) #Steering_vector
        # New orientation for the car.
        ori = np.arctan2(sv[1],sv[0])
        # Compute omega (pure pursuit geometry).
        l = np.linalg.norm(sv)
        al = (np.pi/2) - (ori - self.theta)
        # Complementary of current orientation and desired orientation
        xL = l*np.sin(al)
        yL = l*np.cos(al)
        r = (xL**2)/(2*yL) + (yL/2)
        r = np.sign(r)*max(abs(r), self.min_r)
        if(self.direction=='R' and self.y<(-4.65*self.x-0.8)):
            self.vel = 0.065
        if(self.direction=='R' and self.y>(-4.65*self.x-0.8)):
            r=-self.min_r
            self.vel = 0.275
        tau = self.vel/r
        return (self.vel-0.5*tau*self.wheel_distance),(self.vel+0.5*tau*self.wheel_distance)