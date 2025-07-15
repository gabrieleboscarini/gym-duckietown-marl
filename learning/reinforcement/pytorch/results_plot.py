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

data = np.load("learning/reinforcement/pytorch/runs/results/seed11/rewards.npy")

window = 10  # moving average over 5 evaluation points (you can try 10 or 20 too)
moving_avg = np.convolve(data, np.ones(window)/window, mode='valid')

plt.plot(data, label="Per Eval (avg over 10 runs)", alpha=0.4)
plt.plot(moving_avg, label=f"Moving Avg (window={window})", linewidth=2)
plt.xlabel("Evaluation step")
plt.ylabel("Average reward")
plt.title("Policy Evaluation Over Time")
plt.grid(True)
plt.legend()
plt.show()