import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt
import os

gym.register_envs(gymnasium_robotics)

env = gym.make(
    "FetchPickAndPlace-v4",
    render_mode="rgb_array"
)

obs, info = env.reset()

renderer = env.unwrapped.mujoco_renderer

os.makedirs("test_top", exist_ok=True)
os.makedirs("test_wrist", exist_ok=True)

for i in range(10):

    action = env.action_space.sample() * 0.05

    obs, reward, terminated, truncated, info = env.step(action)

    # Top
    renderer.camera_id = 4
    top = renderer.render("rgb_array")
    plt.imsave(f"test_top/{i:03d}.png", top)

    # Wrist
    renderer.camera_id = 2
    wrist = renderer.render("rgb_array")
    plt.imsave(f"test_wrist/{i:03d}.png", wrist)

print("Finished")

env.close()