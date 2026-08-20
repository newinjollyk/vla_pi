import gymnasium as gym
import gymnasium_robotics
import numpy as np
import matplotlib.pyplot as plt
import time

gym.register_envs(gymnasium_robotics)

# Load your modified environment
env = gym.make(
    "FetchPickAndPlace-v4",
    render_mode="human"   # opens live MuJoCo window
)

obs, info = env.reset()

print("Environment loaded")
print("Action space:", env.action_space)

# Small movement actions
# [dx, dy, dz, gripper]

for step in range(2000):

    action = np.array([
        0.01,   # move x
        0.0,   # move y
        0.0,   # move z
        0.0    # gripper
    ])

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    time.sleep(0.01)

env.close()