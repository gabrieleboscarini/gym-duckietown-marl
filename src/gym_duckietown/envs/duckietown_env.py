# coding=utf-8
import numpy as np
from gymnasium import spaces
import gymnasium as gym

from ..simulator import Simulator
from .. import logger
from path_planning.controller import Controller 
from scipy.spatial.distance import cdist


class PurePursuitEnv(Simulator):
    
    def __init__(self,**kwargs):
        Simulator.__init__(self, **kwargs)
        
        logger.info("using PurePursuitEnv")
        
        self.action_space = spaces.Box(low=np.array([0.1, 0.1]),
                                       high=np.array([1.5, 1.0]),
                                       dtype=np.float32)
        
        self.observation_space = spaces.Box(low=np.array([0, 0,-3.141,0,0,0,0]),
                                       high=np.array([3, 3,3.141,1.20,3,3,2]),
                                       dtype=np.float32)
        
        _,self.path,self.direction = self.compute_trajectory("N2L", 20)
        
        self.controller = Controller(direction='l', path=self.path, wheel_distance=0.102)
        
        self.obs = None  # Initialize state
        
        self.cte = 0 #cross track error
        
        self.distance_covered = 0
        
        self.reset(seed=self.seed_value)
        
        
    def _done_pose(self, pos):
        done = False
        coords = self.get_grid_coords(pos)
        tile = self._get_tile(*coords)
        
        if self.direction == "N2L":
            
            if tile["coords"] == (4,2):
                
                done = True
                
        return done
    
            
    def reward_function(self, actual_position, prev_position):
        
        v_ref = 0.3
        
        dist_all = cdist(self.path,actual_position,'euclidean').flatten()
        self.cte = np.min(dist_all)

        step_distance = np.linalg.norm(actual_position - prev_position)
        self.distance_covered += step_distance
        
        reward = -( +1.0 * self.cte**2 + 1.0*(v_ref - self.speed)**2) #+ 0.2*self.distance_covered
        
        return reward
        
        
    def _compute_done_reward(self, actual_position, prev_position):
        
        # If the agent is not in a valid pose (on drivable tiles)
        if not self._valid_pose(self.cur_pos, self.cur_angle):
            msg = "Stopping the simulator because we are at an invalid pose."
            # logger.info(msg)
            reward = -1
            done_code = "invalid-pose"
            done = True
            
        elif self.cte > 0.3:
            msg = "Stopping the simulator because cte exceeded the treshold"
            reward = -1
            done_code = "invalid-pose"
            done = True
            
        #if the agent reach the target tile    
        elif self._done_pose(self.cur_pos):
            msg = "Stopping the simulator because we arrived at target"
            reward = 0
            done_code = "done-pose"
            done = True
            
        # If the maximum time step count is reached
        elif self.step_count >= self.max_steps:
            msg = "Stopping the simulator because we reached max_steps = %s" % self.max_steps
            # logger.info(msg)
            done = True
            reward = 0
            done_code = "max-steps-reached"
        else:
            done = False
            reward = self.reward_function(actual_position, prev_position)
            msg = ""
            done_code = "in-progress"
            
        return reward, done, msg, done_code
        
        
        
    def reset(self, seed=None):
        
        _, info = super().reset(seed=seed)
        
        self.distance_covered = 0
        
        self.cte = 0
        
        self.goal = np.array([[self.cur_pos[0], self.cur_pos[2]]])
        
        self.controller.reset()
        
        self.obs = np.array([self.cur_pos[0],
                        self.cur_pos[2],
                        self.cur_angle,
                        self.speed,
                        self.goal[0,0],  #prior lookahead x
                        self.goal[0,1],  #prior lookahead y
                        self.controller.vel
                        ])
        
        return self.obs, info
        
    def step(self,action):
        
        # Unpack the action into look-ahead distance and velocity
        la_dis, vel = action
        
        # Update the controller parameters
        #self.controller.la_dis = la_dis
        #self.controller.vel = vel
        self.controller.update_parameters(la_dis, vel)
        
        # Get the current pose of the Duckiebot
        pose = self.cur_pos[0], self.cur_pos[2], self.cur_angle # Returns (x, y, theta)
        
        prev_position = np.zeros((1,2))
        prev_position[0,0]=self.cur_pos[0]
        prev_position[0,1]=self.cur_pos[2]
        
        
        # Compute wheel velocities using pure pursuit
        v_left, v_right, current_goal = self.controller.pure_pursuit(pose)

        # Convert wheel velocities to gym-duckietown's action format
        vels = np.array([v_left, v_right])
        vels = np.clip(vels, -1, 1)
        
        for _ in range(self.frame_skip):
            self.update_physics(vels)
            
        actual_position = np.zeros((1,2))
        actual_position[0,0]=self.cur_pos[0]
        actual_position[0,1]=self.cur_pos[2]
            
        self.obs = np.array([self.cur_pos[0],
                        self.cur_pos[2],
                        self.cur_angle,
                        self.speed,
                        self.goal[0,0], #prior lookahead x
                        self.goal[0,1], #prior lookahead y
                        self.controller.vel
                        ])
        
        misc = self.get_agent_info()
        
        #update goal
        self.goal = current_goal

        #d = self._compute_done_reward()
        
        reward, done, msg, done_code = self._compute_done_reward(actual_position,prev_position)
        misc["Simulator"]["msg"] = msg
        
        return self.obs, reward, done, _, misc
    
        
class DuckietownEnv(Simulator):
    """
    Wrapper to control the simulator using velocity and steering angle
    instead of differential drive motor velocities
    """

    def __init__(self, gain=1.0, trim=0.0, radius=0.0318, k=27.0, limit=1.0, **kwargs):
        Simulator.__init__(self, **kwargs)
        logger.info("using DuckietownEnv")

        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit

    def step(self, action):
        vel, angle = action

        # Distance between the wheels
        baseline = self.unwrapped.wheel_dist

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * baseline) / self.radius
        omega_l = (vel - 0.5 * angle * baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])

        obs, reward, done,truncated, info = Simulator.step(self, vels)
        mine = {}
        mine["k"] = self.k
        mine["gain"] = self.gain
        mine["train"] = self.trim
        mine["radius"] = self.radius
        mine["omega_r"] = omega_r
        mine["omega_l"] = omega_l
        info["DuckietownEnv"] = mine
        return obs, reward, done, truncated,info


class DuckietownLF(DuckietownEnv):
    """
    Environment for the Duckietown lane following task with
    and without obstacles (LF and LFV tasks)
    """

    def __init__(self, **kwargs):
        DuckietownEnv.__init__(self, **kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = DuckietownEnv.step(self, action)
        return obs, reward, done,truncated, info


class DuckietownNav(DuckietownEnv):
    """
    Environment for the Duckietown navigation task (NAV)
    """

    def __init__(self, **kwargs):
        self.goal_tile = None
        DuckietownEnv.__init__(self, **kwargs)

    def reset(self, segment=False):
        DuckietownNav.reset(self)

        # Find the tile the agent starts on
        start_tile_pos = self.get_grid_coords(self.cur_pos)
        start_tile = self._get_tile(*start_tile_pos)

        # Select a random goal tile to navigate to
        assert len(self.drivable_tiles) > 1
        while True:
            tile_idx = self.np_random.randint(0, len(self.drivable_tiles))
            self.goal_tile = self.drivable_tiles[tile_idx]
            if self.goal_tile is not start_tile:
                break

    def step(self, action):
        obs, reward, done, truncated, info = DuckietownNav.step(self, action)

        info["goal_tile"] = self.goal_tile

        # TODO: add term to reward based on distance to goal?

        cur_tile_coords = self.get_grid_coords(self.cur_pos)
        cur_tile = self._get_tile(*cur_tile_coords)

        if cur_tile is self.goal_tile:
            done = True
            reward = 1000

        return obs, reward, done, truncated, info
