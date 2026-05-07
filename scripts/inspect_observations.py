import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)

env = gym.make(
    "FetchPickAndPlace-v4",
    render_mode="human"
)

obs, info = env.reset()

print("\nObservation Keys:")
print(obs.keys())

print("\nObservation Shape:")
print(obs["observation"].shape)
print(obs["observation"])

print("\nAchieved Goal:")
print(obs["achieved_goal"])

print("\nDesired Goal:")
print(obs["desired_goal"])

env.close()