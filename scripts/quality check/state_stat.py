from datasets import load_from_disk
import numpy as np

ds = load_from_disk("/home/newin/Projects/vla_pi/lerobot_vla_dataset")

states = np.array(ds["observation.state"], dtype=np.float32)
actions = np.array(ds["action"], dtype=np.float32)

print("States shape :", states.shape)
print("Actions shape:", actions.shape)
print()

print("State mean (first 5 dims):")
print(np.round(states.mean(axis=0)[:5], 4))
print()

print("State std (first 5 dims):")
print(np.round(states.std(axis=0)[:5], 4))
print()

print("Action mean:")
print(np.round(actions.mean(axis=0), 4))
print()

print("Action std:")
print(np.round(actions.std(axis=0), 4))