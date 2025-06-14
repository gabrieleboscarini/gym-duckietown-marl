import numpy as np
import matplotlib.pyplot as plt

data = np.load("learning/reinforcement/pytorch/runs/results/rewards.npz")
rewards = data["arr_0"]
print(rewards.shape)

plt.plot(rewards)
plt.xlabel("Evaluation step")
plt.ylabel("Average reward")
plt.title("Policy Evaluation Over Time")
plt.grid(True)
plt.show()