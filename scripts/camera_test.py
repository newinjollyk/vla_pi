import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt

gym.register_envs(gymnasium_robotics)

env = gym.make(
    "FetchPickAndPlace-v4",
    render_mode="rgb_array"
)

obs, info = env.reset()

# Capture image frame
frame = env.render()

print("Frame shape:", frame.shape)

# Save image
plt.imsave("fetch_frame.png", frame)

print("Image saved as fetch_frame.png")

env.close()