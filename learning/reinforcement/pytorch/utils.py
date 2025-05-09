import random

import gymnasium as gym
import numpy as np
import torch
import time
from pyglet.window import Window



def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer
class ReplayBuffer(object):
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, state, next_state, action, reward, done):
        if len(self.storage) < self.max_size:
            self.storage.append((state, next_state, action, reward, done))
        else:
            # Remove random element in the memory beforea adding a new one
            self.storage.pop(random.randrange(len(self.storage)))
            self.storage.append((state, next_state, action, reward, done))

    def sample(self, batch_size=100, flat=True):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, next_states, actions, rewards, dones = [], [], [], [], []

        for i in ind:
            state, next_state, action, reward, done = self.storage[i]

            if flat:
                states.append(np.array(state, copy=False).flatten())
                next_states.append(np.array(next_state, copy=False).flatten())
            else:
                states.append(np.array(state, copy=False))
                next_states.append(np.array(next_state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            dones.append(np.array(done, copy=False))

        # state_sample, action_sample, next_state_sample, reward_sample, done_sample
        return {
            "state": np.stack(states),
            "next_state": np.stack(next_states),
            "action": np.stack(actions),
            "reward": np.stack(rewards).reshape(-1, 1),
            "done": np.stack(dones).reshape(-1, 1),
        }


def evaluate_policy(env, policy, eval_episodes=10, max_timesteps=500):
    print("---------------Evaluations---------------")
    max_action = env.action_space.high
    low_action = env.action_space.low
    
    avg_reward = 0.0
    for ep in range(eval_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        while not done and step < max_timesteps:
            action = policy.select_action(np.array(obs))
            env_action = low_action + (action + 1.0) * 0.5 * (max_action - low_action)
            obs, reward, done, _, _ = env.step(env_action)
            avg_reward += reward
            step += 1
            #env.render()
        print(f"Episode {ep+1}: avg reward: {avg_reward:.2f}")

    avg_reward /= eval_episodes
    print(f"\nAverage reward over {eval_episodes} episodes: {avg_reward:.2f}")
    print("---------------------------------------------")

    return avg_reward

'''# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(seed=seed), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward
'''