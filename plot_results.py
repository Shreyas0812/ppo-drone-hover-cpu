import csv
import matplotlib.pyplot as plt
import os

LOG_FILE = "training_log.csv"

if not os.path.exists(LOG_FILE):
    print(f"No training log found at {LOG_FILE}. Run train.py first.")
    exit(1)

with open(LOG_FILE, newline="") as f:
    rows = list(csv.DictReader(f))

iterations   = [int(r["iteration"])      for r in rows]
mean_reward  = [float(r["mean_reward"])  for r in rows]
explained_var= [float(r["explained_var"])for r in rows]
policy_loss  = [float(r["policy_loss"])  for r in rows]
value_loss   = [float(r["value_loss"])   for r in rows]
entropy      = [float(r["entropy"])      for r in rows]
clip_frac    = [float(r["clip_frac"])    for r in rows]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("PPO Drone Hover — Training Curves", fontsize=14, fontweight="bold")

def plot(ax, y, title, ylabel, color):
    ax.plot(iterations, y, color=color, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

plot(axes[0, 0], mean_reward,   "Mean Reward",       "Reward",        "#2196F3")
plot(axes[0, 1], explained_var, "Explained Variance","Variance Ratio","#4CAF50")
plot(axes[0, 2], entropy,       "Policy Entropy",    "Entropy",       "#FF9800")
plot(axes[1, 0], policy_loss,   "Policy Loss",       "Loss",          "#F44336")
plot(axes[1, 1], value_loss,    "Value Loss",        "Loss",          "#9C27B0")
plot(axes[1, 2], clip_frac,     "Clip Fraction",     "Fraction",      "#795548")

plt.tight_layout()
os.makedirs("results", exist_ok=True)
out_path = "results/training_curves.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.show()
