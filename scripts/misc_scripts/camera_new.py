import os
import cv2
import time
import gymnasium as gym
import gymnasium_robotics

# Register environments
gym.register_envs(gymnasium_robotics)

# Create environment
env = gym.make(
    "FetchPickAndPlace-v4",
    render_mode="rgb_array"
)

obs, info = env.reset()

# -----------------------------------
# SAVE PATHS
# -----------------------------------

BASE_PATH = "/home/newin/Projects/vla_pi/dataset/sample"

TOP_PATH = os.path.join(BASE_PATH, "top_cam")
GRIP_PATH = os.path.join(BASE_PATH, "gripper_cam")

os.makedirs(TOP_PATH, exist_ok=True)
os.makedirs(GRIP_PATH, exist_ok=True)

# -----------------------------------
# PRINT AVAILABLE CAMERAS
# -----------------------------------

print("\nAvailable cameras:")

for cam_id in range(env.unwrapped.model.ncam):
    print(cam_id, env.unwrapped.model.camera(cam_id).name)

# -----------------------------------
# GET VIEWER
# -----------------------------------

viewer = env.unwrapped.mujoco_renderer._get_viewer("rgb_array")

print("\nCapturing images...")

# -----------------------------------
# CAPTURE LOOP
# -----------------------------------

for i in range(5):

    # Small random motion
    action = env.action_space.sample() * 0.05

    obs, reward, terminated, truncated, info = env.step(action)

    # =====================================
    # TOP CAMERA
    # =====================================

    viewer.cam.type = 2
    viewer.cam.fixedcamid = 4

    top_img = viewer.render(render_mode="rgb_array")

    # =====================================
    # GRIPPER CAMERA
    # =====================================

    viewer.cam.type = 2
    viewer.cam.fixedcamid = 2

    grip_img = viewer.render(render_mode="rgb_array")

    # =====================================
    # RGB -> BGR
    # =====================================

    top_img_bgr = cv2.cvtColor(top_img, cv2.COLOR_RGB2BGR)
    grip_img_bgr = cv2.cvtColor(grip_img, cv2.COLOR_RGB2BGR)

    # =====================================
    # SAVE IMAGES
    # =====================================

    top_file = os.path.join(TOP_PATH, f"top_{i}.png")
    grip_file = os.path.join(GRIP_PATH, f"gripper_{i}.png")

    cv2.imwrite(top_file, top_img_bgr)
    cv2.imwrite(grip_file, grip_img_bgr)

    print(f"Saved image pair {i}")

    time.sleep(0.2)

env.close()

print("\nDone.")