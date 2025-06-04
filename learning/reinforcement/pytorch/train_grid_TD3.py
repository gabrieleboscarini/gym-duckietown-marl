###############################################################################
# Duckietown - UNIPD
# Author: Gabriele Boscarini
# TD3 Training.
###############################################################################

import ast
import argparse
import logging
import sys

import os
import numpy as np

# Duckietown Specific
from learning.reinforcement.pytorch.td3 import TD3
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper, ResizeWrapper
from learning.reinforcement.pytorch.utils import seed, evaluate_policy, ReplayBuffer
from src.gym_duckietown.envs.duckietown_env import GridPurePursuitEnv
from pyglet.window import Window 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def _train(args):
    
    # Base path is the current directory (pytorch)
    base_path = os.path.dirname(__file__)

    # Define the 'runs' path
    runs_path = os.path.join(base_path, 'runs')

    # Define the paths for 'results' and 'models' inside 'runs'
    results_path = os.path.join(runs_path, 'results')
    models_path = os.path.join(runs_path, 'models')

    # Define the 'seed0' paths
    results_seed0_path = os.path.join(results_path, 'seed0')
    models_seed0_path = os.path.join(models_path, 'seed0')

    # Create directories individually
    os.makedirs(runs_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(results_seed0_path, exist_ok=True)
    os.makedirs(models_seed0_path, exist_ok=True)
    
    '''results_path = "learning/reinforcement/pytorch/runs/results/seed" + str(args.seed)
    models_path = "learning/reinforcement/pytorch/runs/models/seed" + str(args.seed)'''
    '''
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(models_path):
        os.makedirs(models_path)'''

    # Launch the env
    env = GridPurePursuitEnv(
            seed=args.seed,  # random seed
            map_name="4way_multi",
            max_steps=500001,  # we don't want the gym to reset itself
            domain_rand=False,
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=4,  # start close to straight
            full_transparency=True,
            distortion=True,
    )
    
    print("Initialized environment")
    
    # Set seeds
    seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    low_action = env.action_space.low

    # Initialize policy
    policy = TD3(state_dim, action_dim, max_action, low_action)
    replay_buffer = ReplayBuffer(args.replay_buffer_max_size)
    print("Initialized TD3")
    
    print(env.obs)
    

    # Evaluate untrained policy
    evaluations = [evaluate_policy(env, policy)]

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    episode_reward = None
    env_counter = 0
    reward = 0
    episode_timesteps = 0
    
    vel_grid = np.linspace(0.2, 0.4, 10)
    
    print("Starting training")
    while total_timesteps < args.max_timesteps:

        #print("timestep: {} | reward: {}".format(total_timesteps, reward))

        if done:
            if total_timesteps != 0:
                print(
                    ("Total T: %d Episode Num: %d Episode T: %d Reward: %f")
                    % (total_timesteps, episode_num, episode_timesteps, episode_reward)
                )
                policy.train(replay_buffer)

                # Evaluate episode
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval %= args.eval_freq
                    evaluations.append(evaluate_policy(env, policy))
                    print("rewards at time {}: {}".format(total_timesteps, evaluations[-1]))

                    if args.save_models:
                        filename = os.path.join(models_seed0_path, "td3")
                        policy.save(filename= filename)
                    np.savez(results_path + "/rewards.npz", evaluations)

            # Reset environment
            env_counter += 1
            obs, _ = env.reset(seed=args.seed)
            done = False
            episode_reward = 0
            episode_num += 1
            episode_timesteps = 0

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
            env_action = action
        else:
            action = policy.select_action(np.array(obs))
            if args.expl_noise != 0:
                action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(
                    -1, 1
                )
                
            env_action = low_action + (action + 1.0) * 0.5 * (max_action - low_action)    
             
        # Perform action
        env.update_velocity_target(np.random.choice(vel_grid))
        
        new_obs, reward, done,_, _ = env.step(env_action)

        if episode_timesteps >= args.env_timesteps:
            done = True

        done_bool = 0 if episode_timesteps + 1 == args.env_timesteps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add(obs, new_obs, action, reward, done_bool)

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
        
        '''
        if (total_timesteps % 500) == 0:
            policy.save("ddpg", args.model_dir)
        '''

    print("Training done, about to save..")
    filename = os.path.join(models_seed0_path, "td3")
    policy.save(filename=filename)
    print("Finished saving..should return now!")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument(
        "--start_timesteps", default=1e4, type=int
    )  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true", default=True)  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=64, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument(
        "--policy_noise", default=0.2, type=float
    )  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--env_timesteps", default=1000, type=int)  # Frequency of delayed policy updates
    parser.add_argument(
        "--replay_buffer_max_size", default=50000, type=int
    )  # Maximum number of steps to keep in the replay buffer
    parser.add_argument("--model-dir", type=str, default="learning/reinforcement/pytorch/models/")
        
    _train(parser.parse_args())