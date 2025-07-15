### AUTHOR: Gabriele Boscarini

import helpers
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = helpers.launch_env()
print("Environment initialized.")

path = env.compute_trajectory("N2R", 20)[1]

# Extract x and y coordinates, ignoring z
x, y = path[:, 0], path[:, 1]


plt.figure(figsize=(5, 5))

dx = np.diff(x)
dy = np.diff(y)
plt.quiver(x[:-1], y[:-1], dx, dy, angles='xy', scale_units='xy', scale=1, color='red')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('2D Projection of 3D Points')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

