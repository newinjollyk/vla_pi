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

print("Starting scripted trajectory...")

for step in range(500):

    action = np.zeros(env.action_space.shape)

    # Phase 1: move forward
    if step < 80:
        action[0] = 1.0

    # Phase 2: move sideways
    elif step < 160:
        action[1] = 1.0

    # Phase 3: move downward
    elif step < 240:
        action[2] = -1.0

    # Phase 4: close gripper
    elif step < 320:
        action[3] = -1.0

    # Phase 5: move upward
    elif step < 400:
        action[2] = 1.0

    # Phase 6: open gripper
    else:
        action[3] = 1.0

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    time.sleep(0.005)

    if terminated or truncated:
        print("Environment terminated/truncated")
        break

env.close()

print("Trajectory complete.")