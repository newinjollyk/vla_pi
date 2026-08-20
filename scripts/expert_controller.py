import gymnasium as gym
import gymnasium_robotics
import numpy as np
import time
import mujoco
from data_recorder import DataRecorder  # Import the DataRecorder class from the data_recorder.py file

gym.register_envs(gymnasium_robotics)

env = gym.make(
    "FetchPickAndPlace-v4",
    render_mode="human"
)

obs, info = env.reset()

print("Started")

TASK = "Pick the blue cube and place it in the tray."

recorder = DataRecorder()

recorder.episode_index = 101  # Set the starting episode index to 1

# =========================================================
# RANDOMIZE OBJECTS
# =========================================================
def randomize_objects():

    tray_center = np.array([1.3, 0.75])

    min_tray_distance = 0.24

    min_cube_distance = 0.12

    while True:

        # -------------------------------------------------
        # RED cube
        # -------------------------------------------------

        red_x = np.random.uniform(1.15, 1.45)
        red_y = np.random.uniform(0.55, 0.95)

        # -------------------------------------------------
        # BLUE cube
        # -------------------------------------------------

        blue_x = np.random.uniform(1.15, 1.45)
        blue_y = np.random.uniform(0.55, 0.95)

        red_pos = np.array([red_x, red_y])

        blue_pos = np.array([blue_x, blue_y])

        # -------------------------------------------------
        # Distance from tray
        # -------------------------------------------------

        red_tray_dist = np.linalg.norm(
            red_pos - tray_center
        )

        blue_tray_dist = np.linalg.norm(
            blue_pos - tray_center
        )

        # cubes too close to tray
        if (
            red_tray_dist < min_tray_distance or
            blue_tray_dist < min_tray_distance
        ):
            continue

        # -------------------------------------------------
        # Distance between cubes
        # -------------------------------------------------

        cube_dist = np.linalg.norm(
            red_pos - blue_pos
        )

        if cube_dist < min_cube_distance:
            continue

        break


    # =====================================================
    # SET RED cube
    # =====================================================

    red_joint = env.unwrapped.data.joint(
        "object0:joint"
    ).qpos.copy()

    red_joint[:3] = [
        red_x,
        red_y,
        0.425
    ]

    env.unwrapped.data.joint(
        "object0:joint"
    ).qpos[:] = red_joint


    # =====================================================
    # SET BLUE cube
    # =====================================================

    blue_joint = env.unwrapped.data.joint(
        "blueobj:joint"
    ).qpos.copy()

    blue_joint[:3] = [
        blue_x,
        blue_y,
        0.425
    ]

    env.unwrapped.data.joint(
        "blueobj:joint"
    ).qpos[:] = blue_joint


    # =====================================================
    # UPDATE PHYSICS
    # =====================================================

    import mujoco

    mujoco.mj_forward(
        env.unwrapped.model,
        env.unwrapped.data
    )

    print("Objects randomized safely")
randomize_objects()
# =========================================================
# BASIC STEP FUNCTION
# =========================================================
'''
def step_action(action, delay=0.01):

    global obs
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    time.sleep(delay)
'''
def step_action(action, delay=0.01):

    global obs

    # ------------------------------------------
    # Record CURRENT observation + action
    # BEFORE applying the action
    # ------------------------------------------

    state = obs["observation"].copy()

    recorder.record_frame(
        env,
        state,
        action,
        time.time(),
        0.0,
        False
    )

    # ------------------------------------------
    # Apply action
    # ------------------------------------------

    obs, reward, terminated, truncated, info = env.step(action)

    time.sleep(delay)

    return reward, terminated, truncated

# =========================================================
# OPEN GRIPPER
# =========================================================

def open_gripper(steps=3):

    for _ in range(steps):

        action = np.array([0.0, 0.0, 0.0, 0.02])
        step_action(action)


# =========================================================
# CLOSE GRIPPER
# =========================================================

def close_gripper(steps=40):

    for _ in range(steps):

        action = np.array([0.0, 0.0, 0.0, -1.0])
        step_action(action)


# =========================================================
# MOVE TO TARGET
# =========================================================

def move_to_target(
    target_pos,
    gripper_cmd=0.0,
    threshold=0.01
):

    while True:

        grip_pos = env.unwrapped.data.site(
            "robot0:grip"
        ).xpos.copy()

        target_pos[2] = np.clip(
            target_pos[2],
            0.42,
            0.75
        )
        direction = target_pos - grip_pos
        distance = np.linalg.norm(direction)
        if distance < threshold:
            break

        action_xyz = direction * 2.0
        action_xyz = np.clip(
            action_xyz,
            -1.0,
            1.0
        )

        action = np.array([
            action_xyz[0],
            action_xyz[1],
            action_xyz[2],
            gripper_cmd
        ])

        step_action(action)

# =========================================================
# PICK OBJECT
# =========================================================

def pick_object(object_name):
    
    cube_pos = env.unwrapped.data.body(
        object_name
    ).xpos.copy()

    hover_pos = cube_pos + np.array([0.0, 0.0, 0.10])
    move_to_target(hover_pos, 0.0)
    open_gripper()

    left = env.unwrapped.data.joint(
        "robot0:l_gripper_finger_joint"
    ).qpos

    right = env.unwrapped.data.joint(
        "robot0:r_gripper_finger_joint"
    ).qpos

    print("Left finger:", left)
    print("Right finger:", right)

    grasp_pos = cube_pos + np.array([0.0, 0.0, 0.01])

    move_to_target(grasp_pos, 0.0)

    grasp_pos = cube_pos + np.array([0.0, 0.0, 0.01])
    move_to_target(grasp_pos, 0.0)
    close_gripper()
    lift_pos = cube_pos + np.array([0.0, 0.0, 0.25])
    move_to_target(lift_pos, -1.0)
    print(f"Picked {object_name}")

# =========================================================
# PLACE IN TRAY
# =========================================================

def place_in_tray():

    tray_pos = env.unwrapped.data.body(
        "tray"
    ).xpos.copy()

    tray_hover = tray_pos + np.array([0.0, 0.0, 0.20])
    move_to_target(tray_hover, -1.0)
    tray_drop = tray_pos + np.array([0.0, 0.0, 0.08])
    move_to_target(tray_drop, -1.0)
    open_gripper()
    safe_pos = tray_pos + np.array([0.0, 0.0, 0.30])
    move_to_target(safe_pos, 1.0)
    print("Placed object in tray")


# =========================================================
# EXECUTE TASK
# =========================================================

NUM_EPISODES = 100

for episode in range(NUM_EPISODES):

    print(f"\n===== Episode {episode + 1}/{NUM_EPISODES} =====")
    obs, info = env.reset()
    randomize_objects()
    recorder.start_episode(TASK)

    try:

        pick_object("blueobj")  # Pick the blue cube
        #pick_object("object0")  # Pick the red cube

        place_in_tray()

        recorder.finish_episode(success=True)

        print("Episode successful")

    except Exception as e:

        print(f"Episode failed: {e}")
        recorder.delete_episode()

        # Skip this episode and continue

        continue

env.close()
