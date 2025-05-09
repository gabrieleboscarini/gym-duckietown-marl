import ast
import argparse
import logging
import os
import sys
import numpy as np

from pyglet.window import Window


# Duckietown Specific
from src.gym_duckietown.envs.duckietown_env import PurePursuitEnv
from learning.reinforcement.pytorch.td3 import TD3

def _enjoy():
    # Launch the env with our helper function
    env = PurePursuitEnv(
            seed=123,  # random seed
            map_name="4way_multi",
            max_steps=500001,  # we don't want the gym to reset itself
            domain_rand=False,
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=4,  # start close to straight
            full_transparency=True,
            distortion=False,
    )
    
    print("Initialized environment")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    low_action = env.action_space.low

    # Initialize policy
    policy = TD3(state_dim, action_dim, max_action, low_action)
    policy.load(filename="learning/reinforcement/pytorch/runs/models/seed0/td3")

    obs, _ = env.reset()
    done = False

    while True:
        while not done:
            action = policy.select_action(np.array(obs))
            #action = env.action_space.sample()
            env_action = low_action + (action + 1.0) * 0.5 * (max_action - low_action)
            # Perform action
            obs, reward, done, _, _ = env.step(env_action)
            env.render("top_down")
            
        done = False
        obs, _ = env.reset()

if __name__ == "__main__":
    _enjoy()