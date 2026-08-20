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
# STABLE CONTROLLER FUNCTION
# Task 2.2 Improvements:
# - smooth trajectories
# - stop threshold
# - pose clamping
# - reduced oscillation
# =========================================================

def move_to_target(target_pos, gripper_cmd=0.0):

    while True:

        grip_pos = env.unwrapped.data.site("robot0:grip").xpos.copy()

        # ---------------------------------------------
        # Clamp invalid target poses
        # ---------------------------------------------

        target_pos[2] = np.clip(target_pos[2], 0.42, 0.75)

        # ---------------------------------------------
        # Direction vector
        # ---------------------------------------------

        direction = target_pos - grip_pos

        distance = np.linalg.norm(direction)

        # ---------------------------------------------
        # Stop when close enough
        # Prevent oscillation
        # ---------------------------------------------

        if distance < 0.01:
            break

        # ---------------------------------------------
        # Smooth controller
        # Smaller gain = smoother motion
        # ---------------------------------------------

        action_xyz = direction * 2.0

        # ---------------------------------------------
        # Clamp actions
        # ---------------------------------------------

        action_xyz = np.clip(action_xyz, -1.0, 1.0)

        action = np.array([
            action_xyz[0],
            action_xyz[1],
            action_xyz[2],
            gripper_cmd
        ])

        obs, reward, terminated, truncated, info = env.step(action)

        env.render()

        time.sleep(0.01)


# =========================================================
# GET POSITIONS
# =========================================================

cube_pos = env.unwrapped.data.body("object0").xpos.copy()

tray_pos = env.unwrapped.data.body("tray").xpos.copy()


# =========================================================
# 1. MOVE ABOVE CUBE
# =========================================================

target_above_cube = cube_pos + np.array([0.0, 0.0, 0.10])

move_to_target(
    target_above_cube,
    gripper_cmd=0.0
)

print("Reached above cube")


# =========================================================
# 2. OPEN GRIPPER
# =========================================================

for _ in range(40):

    action = np.array([0.0, 0.0, 0.0, 1.0])

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    time.sleep(0.01)

print("Opened gripper")


# =========================================================
# 3. LOWER TO CUBE
# =========================================================

target_grasp = cube_pos + np.array([0.0, 0.0, 0.01])

move_to_target(
    target_grasp,
    gripper_cmd=0.0
)

print("Reached grasp pose")


# =========================================================
# 4. CLOSE GRIPPER
# =========================================================

for _ in range(50):

    action = np.array([0.0, 0.0, 0.0, -1.0])

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    time.sleep(0.01)

print("Closed gripper")


# =========================================================
# 5. LIFT CUBE
# =========================================================

lift_target = cube_pos + np.array([0.0, 0.0, 0.25])

move_to_target(
    lift_target,
    gripper_cmd=-1.0
)

print("Lifted cube")


# =========================================================
# 6. MOVE ABOVE TRAY
# =========================================================

tray_hover = tray_pos + np.array([0.0, 0.0, 0.20])

move_to_target(
    tray_hover,
    gripper_cmd=-1.0
)

print("Reached tray")


# =========================================================
# 7. LOWER INTO TRAY
# =========================================================

tray_drop = tray_pos + np.array([0.0, 0.0, 0.08])

move_to_target(
    tray_drop,
    gripper_cmd=-1.0
)

print("Lowered cube")


# =========================================================
# 8. RELEASE CUBE
# =========================================================

for _ in range(50):

    action = np.array([0.0, 0.0, 0.0, 1.0])

    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    time.sleep(0.01)

print("Released cube")


# =========================================================
# 9. MOVE ARM UPWARD SAFELY
# =========================================================

safe_pose = tray_pos + np.array([0.0, 0.0, 0.30])

move_to_target(
    safe_pose,
    gripper_cmd=1.0
)

print("Task completed")

env.close()