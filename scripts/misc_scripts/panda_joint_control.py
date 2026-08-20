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

# Small controlled action
action = np.zeros(env.action_space.shape)

# Slowly move one joint
action[0] = 0.2

for _ in range(2000):
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.01)

    if terminated or truncated:
        obs, info = env.reset()

env.close()