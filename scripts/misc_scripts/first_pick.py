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

print("Started")


# =========================================================
# Move above cube using REAL positions
# =========================================================

for _ in range(200):

    cube_pos = env.unwrapped.data.body("object0").xpos.copy()

    grip_pos = env.unwrapped.data.site("robot0:grip").xpos.copy()

    # target slightly ABOVE cube
    target_pos = cube_pos + np.array([0.0, 0.0, 0.10])

    direction = target_pos - grip_pos

    direction = np.clip(direction * 5.0, -1.0, 1.0)

    action = np.array([
        direction[0],
        direction[1],
        direction[2],
        0.0
    ])

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    time.sleep(0.01)

print("Reached above cube")


# =========================================================
# Open gripper
# =========================================================

for _ in range(60):

    action = np.array([
        0.0,
        0.0,
        0.0,
        1.0
    ])

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    time.sleep(0.01)

print("Opened gripper")


# =========================================================
# Lower vertically
# =========================================================

for _ in range(50):

    action = np.array([
        0.0,
        0.0,
        -0.3,
        0.0
    ])

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    time.sleep(0.01)

print("Lowered")


# =========================================================
# Close gripper
# =========================================================

for _ in range(60):

    action = np.array([
        0.0,
        0.0,
        0.0,
        -1.0
    ])

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    time.sleep(0.01)

print("Closed gripper")


# =========================================================
# Lift cube upward
# =========================================================

target_height = 0.65

while True:

    grip_pos = env.unwrapped.data.site("robot0:grip").xpos.copy()

    current_height = grip_pos[2]

    # stop lifting when target height reached
    if current_height >= target_height:
        break

    action = np.array([
        0.0,
        0.0,
        0.25,
        -1.0
    ])

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    time.sleep(0.01)

print("Lifted cube safely")

# =========================================================
# Move ABOVE tray center
# =========================================================

for _ in range(180):

    grip_pos = env.unwrapped.data.site("robot0:grip").xpos.copy()
    tray_pos = env.unwrapped.data.body("tray").xpos.copy()

    direction = tray_pos - grip_pos

    direction = np.clip(direction * 5.0, -1.0, 1.0)

    action = np.array([
        direction[0],
        direction[1],
        0.0,      # keep height while moving
        -1.0
    ])

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    time.sleep(0.01)

print("Reached tray center")


# =========================================================
# Lower cube into tray
# =========================================================

for _ in range(60):

    action = np.array([
        0.0,
        0.0,
        -0.25,
        -1.0
    ])

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    time.sleep(0.01)

print("Lowered cube")


# =========================================================
# Open gripper to release cube
# =========================================================

for _ in range(60):

    action = np.array([
        0.0,
        0.0,
        0.0,
        1.0
    ])

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    time.sleep(0.01)

print("Released cube")


# =========================================================
# Move arm upward after release
# =========================================================

for _ in range(80):

    action = np.array([
        0.0,
        0.0,
        0.4,
        1.0
    ])

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    time.sleep(0.01)

print("Task completed")

env.close()