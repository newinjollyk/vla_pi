import gymnasium as gym
import gymnasium_robotics
import numpy as np
import os
import time

from PIL import Image

gym.register_envs(gymnasium_robotics)

# Create environment
env = gym.make(
    "FetchPickAndPlace-v4",
    render_mode="rgb_array"
)

obs, info = env.reset()

# Storage
actions_list = []
observations_list = []

# Dataset folder
save_dir = "/home/newin/Projects/vla_pi/dataset/episode_001"
os.makedirs(save_dir, exist_ok=True)

print("Recording trajectory...")

for step in range(200):

    action = np.zeros(env.action_space.shape)

    # Simple scripted motion
    if step < 50:
        action[0] = 1.0

    elif step < 100:
        action[2] = -1.0

    elif step < 150:
        action[3] = -1.0

    else:
        action[2] = 1.0

    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)

    # Render RGB frame
    frame = env.render()

    # Save image
    image = Image.fromarray(frame)
    image.save(f"{save_dir}/frame_{step:04d}.png")

    # Store action and observation
    actions_list.append(action.copy())
    observations_list.append(obs["observation"].copy())

    if terminated or truncated:
        print("Episode ended early")
        break

# Save arrays
np.save(f"{save_dir}/actions.npy", np.array(actions_list))
np.save(f"{save_dir}/observations.npy", np.array(observations_list))

env.close()

print("Trajectory saved successfully.")