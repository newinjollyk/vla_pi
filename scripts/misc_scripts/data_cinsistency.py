import numpy as np
import os

episode = "dataset/episode_0000"

states = np.load(os.path.join(episode, "states.npy"))
actions = np.load(os.path.join(episode, "actions.npy"))
timestamps = np.load(os.path.join(episode, "timestamps.npy"))

print("States:", states.shape)
print("Actions:", actions.shape)
print("Timestamps:", timestamps.shape)

print("Top images:", len(os.listdir(os.path.join(episode, "top"))))
print("Wrist images:", len(os.listdir(os.path.join(episode, "wrist"))))