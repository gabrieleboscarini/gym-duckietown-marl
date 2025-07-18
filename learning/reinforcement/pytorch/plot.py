import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob 


# Base directory containing "seed*" folders
script_dir = os.path.dirname(os.path.abspath(__file__))
base_results = os.path.join(script_dir, "runs", "results")

# Load rewards for each seed
rewards_dict = {}
for path in sorted(glob.glob(os.path.join(base_results, "seed*"))):
    seed = os.path.basename(path)
    reward_path = os.path.join(path, "rewards.npy")
    if os.path.exists(reward_path):
        rewards_dict[seed] = np.load(reward_path)

# Save to a .npz file for later use
np.savez("rewards.npz", **rewards_dict)

# Load rewards
data = np.load("rewards.npz")

# ✅ SELECT RANGE OF SEEDS HERE
selected_seeds = [f"seed{i}" for i in range(12, 17)]
available_seeds = [s for s in selected_seeds if s in data.files]

# Prepare DataFrame
df = pd.DataFrame([
    {'eval_step': epoch, 'reward': reward, 'seed': seed_name}
    for seed_name in available_seeds
    for epoch, reward in enumerate(data[seed_name])
    if epoch >= 2
])

# Compute mean and std per evaluation step
grouped = df.groupby("eval_step")["reward"].agg(["mean", "std"]).reset_index()

# Apply moving average
window = 5
grouped["mean_smooth"] = grouped["mean"].rolling(window=window, center=True, min_periods=1).mean()
grouped["std_smooth"] = grouped["std"].rolling(window=window, center=True, min_periods=1).mean()

# Set theme and color palette
sns.set_theme(style="whitegrid", font_scale=1.5)
cmap = plt.get_cmap("viridis")
main_color = cmap(0.6)  # Pick a vibrant color

# Plot
plt.figure(figsize=(12, 7))

# Mean line
plt.plot(grouped["eval_step"], grouped["mean_smooth"],
         color=main_color, linewidth=2.5, label="Mean (Smoothed)")

# ± SD shaded area
plt.fill_between(
    grouped["eval_step"],
    grouped["mean_smooth"] - grouped["std_smooth"],
    grouped["mean_smooth"] + grouped["std_smooth"],
    color=main_color, alpha=0.3, label="± SD"
)

# Optional: add dashed lines for SD bounds
plt.plot(grouped["eval_step"], grouped["mean_smooth"] - grouped["std_smooth"],
         color=main_color, linestyle="--", linewidth=1, alpha=0.7)
plt.plot(grouped["eval_step"], grouped["mean_smooth"] + grouped["std_smooth"],
         color=main_color, linestyle="--", linewidth=1, alpha=0.7)

# Labels and title
plt.title("Smoothed Average Reward per Evaluation Step", fontsize=18)
plt.xlabel("Evaluation Step", fontsize=14)
plt.ylabel("Reward", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.legend()
plt.show()