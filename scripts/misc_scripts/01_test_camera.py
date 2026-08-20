import gymnasium as gym
import gymnasium_robotics

import matplotlib.pyplot as plt

# Register robotics envs
gym.register_envs(gymnasium_robotics)

# Create environment using custom XML
env = gym.make(
    "FetchPickAndPlace-v4",
    model_path="/home/newin/Projects/vla_pi/assets/pick_and_place_custom.xml",
    render_mode="rgb_array"
)

# Reset
obs, info = env.reset()

# Access renderer
renderer = env.unwrapped.mujoco_renderer

# Render top camera
top_img = renderer.render(
    render_mode="rgb_array",
    camera_name="top_cam"
)

# Render gripper camera
gripper_img = renderer.render(
    render_mode="rgb_array",
    camera_name="gripper_camera_rgb"
)

# Display
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(top_img)
plt.title("Top Camera")

plt.subplot(1, 2, 2)
plt.imshow(gripper_img)
plt.title("Gripper Camera")

plt.show()

env.close()