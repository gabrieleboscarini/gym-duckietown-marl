import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob 


# Folder containing all seed subfolders

script_dir = os.path.dirname(os.path.abspath(__file__))
base_results = os.path.join(script_dir, "runs", "results")
# Collect per-seed reward arrays
rewards_dict = {}
for path in sorted(glob.glob(os.path.join(base_results, "seed*"))):
    seed = os.path.basename(path)
    reward_path = os.path.join(path, "rewards.npy")
    if os.path.exists(reward_path):
        rewards_dict[seed] = np.load(reward_path)

# Save all rewards into one .npz file
np.savez("rewards.npz", **rewards_dict)
# Load your data
#data = np.load("my_results.npz")  
data = np.load("rewards.npz") # shape [epochs, seeds]

'''# Convert to long-form DataFrame
df = pd.DataFrame([
    {'epoch': epoch, 'reward': reward, 'seed': seed_name}
    for seed_name in data.files
    for epoch, reward in enumerate(data[seed_name])
])

plt.figure(figsize=(10, 6))
sns.set_theme(style="darkgrid", font_scale=1.5)
#sns.lineplot(data=df, x="epoch", y="reward", hue="seed", ci=None, alpha=0.3)  # per-seed lines
sns.lineplot(data=df, x="epoch", y="reward", errorbar="sd", color="black", label="Mean ± SD")  # average

plt.title("Average Reward per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()'''

all_seeds = sorted(data.files)

# Define two groups
curriculum_seeds = all_seeds[:5]
no_curriculum_seeds = all_seeds[-5:]

# Convert to long-form DataFrame with a new column: group
df = pd.DataFrame([
    {'epoch': epoch, 'reward': reward, 'seed': seed_name,
     'group': 'no_curriculum' if seed_name in no_curriculum_seeds else 'curriculum'}
    for seed_name in no_curriculum_seeds + curriculum_seeds
    for epoch, reward in enumerate(data[seed_name])
])

# Plot
plt.figure(figsize=(10, 6))
sns.set_theme(style="darkgrid", font_scale=1.5)

# Plot average per group
sns.lineplot(data=df, x="epoch", y="reward", hue="group", errorbar="sd")

plt.title("Average Reward per Epoch: Curriculum vs No Curriculum")
plt.xlabel("Epoch")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.show()