import gymnasium as gym
import gymnasium_robotics
import numpy as np
import time

gym.register_envs(gymnasium_robotics)

env = gym.make(
    "FetchPickAndPlace-v4",
    render_mode="human"
)

obs, info = env.reset()

for step in range(500):

    # cube position
    cube_pos = obs["achieved_goal"]

    # target position
    target_pos = obs["desired_goal"]

    print("\nCube Position:", cube_pos)
    print("Target Position:", target_pos)

    action = np.zeros(env.action_space.shape)

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    time.sleep(0.05)

    if terminated or truncated:
        break

env.close()