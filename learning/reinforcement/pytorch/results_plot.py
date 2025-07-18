import numpy as np
import matplotlib.pyplot as plt

'''data = np.load("learning/reinforcement/pytorch/runs/results/seed0/rewards.npy")
#rewards = data["arr_0"]
#print(rewards.shape)

plt.plot(data)
plt.xlabel("Evaluation step")
plt.ylabel("Average reward")
plt.title("Policy Evaluation Over Time")
plt.grid(True)
plt.show()'''

# Load rewards
rewards = np.load("learning/reinforcement/pytorch/runs/results/seed17/rewards.npy")

# Load collisions
collisions = np.load("learning/reinforcement/pytorch/runs/results/seed17/collisions.npy")

# === Plot Average Reward ===
window = 10
reward_moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')

plt.figure()
plt.plot(rewards, label="Per Eval (avg over episodes)", alpha=0.4)
plt.plot(reward_moving_avg, label=f"Moving Avg (window={window})", linewidth=2)
plt.xlabel("Evaluation step")
plt.ylabel("Average reward")
plt.title("Policy Evaluation Over Time")
plt.grid(True)
plt.legend()
plt.show()

# === Plot Collision Count ===
collision_moving_avg = np.convolve(collisions, np.ones(window)/window, mode='valid')

plt.figure()
plt.plot(collisions, label="Collisions per Evaluation", alpha=0.4, color='r')
plt.plot(collision_moving_avg, label=f"Moving Avg (window={window})", linewidth=2, color='darkred')
plt.xlabel("Evaluation step")
plt.ylabel("Collision Count")
plt.title("Collisions Over Time")
plt.grid(True)
plt.legend()
plt.show()