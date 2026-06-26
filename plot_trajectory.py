import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from gym_pybullet_drones.envs import HoverAviary
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from actor_critic import ActorCritic

POLICY_PATH = os.path.join("scripts", "policy.pt")
CHECKPOINT_PATH = os.path.join("scripts", "checkpoint.pt")
OUT_PATH = os.path.join("results", "eval_trajectory.png")
STEPS = 300  # ~10s at 30Hz

env = HoverAviary(gui=False, record=False)
obs_dim = env.observation_space.shape[1]
action_dim = env.action_space.shape[1]

policy = ActorCritic(obs_dim, action_dim)
if os.path.exists(POLICY_PATH):
    policy.load_state_dict(torch.load(POLICY_PATH))
else:
    checkpoint = torch.load(CHECKPOINT_PATH)
    policy.load_state_dict(checkpoint["policy"])
    print(f"policy.pt not found, loaded checkpoint from iteration {checkpoint['iteration'] + 1}")
policy.eval()

trajectory = {"x": [], "y": [], "z": [], "reward": []}

obs, _ = env.reset()
for _ in range(STEPS):
    with torch.no_grad():
        action = policy.actor(torch.as_tensor(obs, dtype=torch.float32))
    obs, reward, done, _, _ = env.step(action.clamp(-1.0, 1.0))
    trajectory["x"].append(float(obs[0, 0]))
    trajectory["y"].append(float(obs[0, 1]))
    trajectory["z"].append(float(obs[0, 2]))
    trajectory["reward"].append(float(reward))
    if float(obs[0, 2]) < 0.05 or done:
        obs, _ = env.reset()

env.close()

t = np.arange(len(trajectory["z"])) / 30.0

fig = plt.figure(figsize=(12, 8))
fig.suptitle("PPO Drone Hover — Trained Policy Evaluation", fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig)

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, trajectory["z"], color="#2196F3", linewidth=1.5, label="Altitude (m)")
ax1.axhline(1.0, color="#F44336", linestyle="--", linewidth=1, label="Target (1.0 m)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Altitude (m)")
ax1.set_title("Drone Altitude — rises to target and holds")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(trajectory["x"], trajectory["y"], color="#4CAF50", linewidth=1, alpha=0.7)
ax2.scatter([trajectory["x"][0]], [trajectory["y"][0]], color="green", zorder=5, label="Start")
ax2.scatter([trajectory["x"][-1]], [trajectory["y"][-1]], color="red", zorder=5, label="End")
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.set_title("Horizontal Drift (top-down view)")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_aspect("equal")

ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(t, trajectory["reward"], color="#FF9800", linewidth=1, alpha=0.8)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Reward")
ax3.set_title("Per-step Reward")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved to {OUT_PATH}")
plt.show()
