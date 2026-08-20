import gymnasium as gym
import gymnasium_robotics
import numpy as np
import time

gym.register_envs(gymnasium_robotics)

env = gym.make(
    "FrankaKitchen-v1",
    render_mode="human"
)

obs, info = env.reset()

joint_id = 4

for step in range(2000):

    # Reset action every timestep
    action = np.zeros(env.action_space.shape)

    # Apply small pulse only periodically
    if step < 300:
        action[joint_id] = 0.02

    obs, reward, terminated, truncated, info = env.step(action)

    time.sleep(0.01)

    if terminated or truncated:
        obs, info = env.reset()

env.close()