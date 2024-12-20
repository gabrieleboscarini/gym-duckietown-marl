import helpers
from pyglet.window import Window
from controller import Controller
from planner import path_generate_4way
import gymnasium as gym
import numpy as np

def _navigate():
    
    env = helpers.launch_env()
    print("Environment initialized.")
    
    path = path_generate_4way(env=env,trajectory=2, n_steps=20)
    
    controller = Controller(direction='l', path=path, wheel_distance=0.102)
    
    done = False
    
    while not done:
        # Get the current pose of the Duckiebot
        pose = env.cur_pos[0], env.cur_pos[2], env.cur_angle # Returns (x, y, theta)

        # Compute wheel velocities using pure pursuit
        v_left, v_right = controller.pure_pursuit(pose)

        # Convert wheel velocities to gym-duckietown's action format
        #action = [v_left + v_right, v_right - v_left]
        action = [v_left, v_right]

        # Step the environment
        obs, reward, done, _, info = env.step(action)

        # Render the environment
        env.render("top_down")
    
    env.close()
    
    

    '''for _ in range(1000):  # You can change the number of iterations as needed
    # Render the environment
        env.render(mode="top_down")
    
    env.close()'''

    
    
    
    
    
    

if __name__ == "__main__":
    _navigate()

'''
direction = 1  # For example, right turn
n_steps = 20  # Number of steps in the Bezier curve
path = path_generate(direction=direction, n_steps=n_steps)



path = np.array([
        [1, 1],   # Starting point near bottom-left
        [2, 2],   # Moving towards center
        [3, 1],   # Curving across the map
        [4, 3]    # Ending near top-right
    ])

 
# Render the map
img = env.render('top_down')
    

i, j = env.unwrapped.get_grid_coords(env.cur_pos)
tile = env.unwrapped._get_tile(i, j)
curves = env.unwrapped._get_curve(i, j)

print(curves)

# Generate the path based on the planner
direction = 1  # For example, right turn
n_steps = 20  # Number of steps in the Bezier curve
path = path_generate(direction=direction, n_steps=n_steps)

# Create a controller with a predefined path
controller = Controller(direction='R', path=path, wheel_distance=0.1)

done = False
while not done:
    # Get the current pose of the Duckiebot
    pose = env.cur_pos  # Returns (x, y, theta)

    # Compute wheel velocities using pure pursuit
    v_left, v_right = controller.pure_pursuit(pose)

    # Convert wheel velocities to gym-duckietown's action format
    action = [v_left + v_right, v_right - v_left]

    # Step the environment
    obs, reward, done, _, info = env.step(action)

    # Render the environment
    env.render()
    
env.close()


# Display key attributes
print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)
print("Map Name:", env.map_name)
print("Current Position:", env.cur_pos)
print("Current Orientation:", env.cur_angle)

# Step through the environment
action = [0.5, 0.0]  # Move forward
obs, reward, done, _, info = env.step(action)
print("Observation:", obs)
print("Reward:", reward)
print("Done:", done)
print("Info:", info)

env.close()

# Loop for rendering
for _ in range(1000):  # You can change the number of iterations as needed
    # Render the environment
    env.render(mode="top_down")
    
env.close() '''


