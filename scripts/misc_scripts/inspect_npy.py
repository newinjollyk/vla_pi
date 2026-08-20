import numpy as np

# Load arrays
actions = np.load("/home/newin/Projects/vla_pi/dataset/episode_001/actions.npy")
observations = np.load("/home/newin/Projects/vla_pi/dataset/episode_001/observations.npy")

print("\nActions Shape:")
print(actions.shape)

print("\nObservations Shape:")
print(observations.shape)

print("\nFirst Action:")
print(actions[0])

print("\nFirst Observation:")
print(observations[0])

print("\nLast Action:")
print(actions[-1])

print("\nLast Observation:")
print(observations[-1])