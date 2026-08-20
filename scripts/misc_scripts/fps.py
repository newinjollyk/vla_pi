import numpy as np

path = "/home/newin/Projects/vla_pi/dataset/episode_0000/timestamps.npy"

timestamps = np.load(path)

dt = np.diff(timestamps)

print("Number of timestamps:", len(timestamps))
print("Mean timestep:", dt.mean())
print("Median timestep:", np.median(dt))
print("Estimated FPS:", 1.0 / np.median(dt))