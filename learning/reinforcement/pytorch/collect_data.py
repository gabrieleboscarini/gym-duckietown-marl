import ast
import argparse
import logging
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from pyglet.window import Window


# Duckietown Specific
from src.gym_duckietown.envs.duckietown_env import multibot_env, curriculumNav, ego_multibot_env
from learning.reinforcement.pytorch.td3 import TD3

def _enjoy():
    # Launch the env with our helper function
    env = ego_multibot_env(
            seed=0,  # random seed
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
    policy.load(filename="learning/reinforcement/pytorch/runs/models/seed11/td3")

    NUM_EPISODES = 10
    all_velocities = []
    all_ctes = []

    for ep in range(NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        velocities = []
        ctes = []
        
        while not done:
            action = policy.select_action(np.array(obs))
            env_action = low_action + (action + 1.0) * 0.5 * (max_action - low_action)
            obs, reward, done, _, _ = env.step(env_action)

            # ----- Extract velocity and CTE -----
            velocity = obs[2]  # replace with correct index
            cte = obs[0]       # replace with correct index
            velocities.append(velocity)
            ctes.append(cte)

            #env.render("top_down")

        all_velocities.append(velocities)
        all_ctes.append(ctes)
        print(f"Episode {ep + 1} finished — Steps: {len(velocities)}")
    
    # Pad with NaNs to equal length
    max_len = max(len(x) for x in all_velocities)
    
    def pad(seq): return np.array([*seq, *[np.nan]*(max_len - len(seq))])
    
    vel_array = np.array([pad(seq) for seq in all_velocities])
    cte_array = np.array([pad(seq) for seq in all_ctes])
    
    # Compute statistics
    mean_velocity = np.nanmean(vel_array, axis=0)
    std_velocity = np.nanstd(vel_array, axis=0)
    
    mean_cte = np.nanmean(cte_array, axis=0)
    std_cte = np.nanstd(cte_array, axis=0)
    
    max_timesteps = 300
    mean_velocity = mean_velocity[:max_timesteps]
    std_velocity = std_velocity[:max_timesteps]
    mean_cte = mean_cte[:max_timesteps]
    std_cte = std_cte[:max_timesteps]
    timesteps = np.arange(max_timesteps)

    

    #timesteps = np.arange(len(mean_velocity))

    # ----- Plot Mean Velocity with Std and Target Line -----
    plt.figure(figsize=(10, 4))
    plt.plot(timesteps, mean_velocity, label="Mean Velocity", color='blue')
    plt.fill_between(timesteps,
                    mean_velocity - std_velocity,
                    mean_velocity + std_velocity,
                    color='blue', alpha=0.3, label="±1 Std Dev")

    # Add horizontal target line
    plt.axhline(y=0.2, color='green', linestyle='--', linewidth=2, label="Target Velocity (0.2)")

    plt.xlabel("Timestep")
    plt.ylabel("Velocity")
    plt.title("Mean Velocity over 10 Episodes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mean_velocity.png")
    plt.show()

    # ----- Plot Mean CTE with Std -----
    plt.figure(figsize=(10, 4))
    plt.plot(timesteps, mean_cte, label="Mean CTE", color='red')
    plt.fill_between(timesteps,
                    mean_cte - std_cte,
                    mean_cte + std_cte,
                    color='red', alpha=0.3, label="±1 Std Dev")
    plt.ylim(0, 1.0)  # or any range you want
    plt.xlabel("Timestep")
    plt.ylabel("CTE")
    plt.title("Mean CTE over 10 Episodes")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("mean_cte.png")
    plt.show()
    
if __name__ == "__main__":
    _enjoy()