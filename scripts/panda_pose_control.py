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

# Define target joint action
target_action = np.array([
    0.3,   # joint 1
    -0.2,  # joint 2
    0.1,   # joint 3
    -0.3,  # joint 4
    0.2,   # joint 5
    0.1,   # joint 6
    0.0,   # joint 7
    0.0,   # gripper
    0.0
])

for _ in range(2000):
    obs, reward, terminated, truncated, info = env.step(target_action)

    time.sleep(0.01)

    if terminated or truncated:
        obs, info = env.reset()

env.close()