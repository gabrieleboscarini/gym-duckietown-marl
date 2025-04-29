import numpy as np
from src.gym_duckietown.envs.duckietown_env import PurePursuitEnv
from pyglet.window import Window

# Launch the env
env = PurePursuitEnv(
            seed=123,  # random seed
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

# Number of steps to take before reset
n_steps_before_reset = 100

# Reset environment at start
obs = env.reset()
print("After initial reset:")
# Print the variables of interest
print(f"My variable: {obs}")

# Run the environment for a few random steps
for step in range(n_steps_before_reset):
    # Sample a random action
    action = env.action_space.sample()
    
    # Step through the environment
    observation, reward, done, _, info = env.step(action)
    
    # Optionally, print at each step
    print(f"Step {step + 1}:")
    print(f"My variable: {observation}")
    
    # You could also check if `done` is True and manually reset, but for now, you said after a fixed number of steps

# After the steps, manually reset the environment
obs = env.reset()

print("After manual reset:")
# Print the variables of interest again
print(f"My variable: {obs}")

env.close()