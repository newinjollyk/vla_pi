from datasets import load_from_disk
from collections import Counter, defaultdict
import numpy as np

ds = load_from_disk("/home/newin/Projects/vla_pi/lerobot_vla_dataset")

# Count frames per episode
ep_lengths = defaultdict(int)
ep_task = {}

for sample in ds:
    ep = sample["episode_index"]
    ep_lengths[ep] += 1
    ep_task[ep] = sample["task_id"]

lengths = np.array(list(ep_lengths.values()))

print("Total episodes:", len(ep_lengths))
print("Total frames  :", len(ds))
print()

print("Episode length statistics")
print("Min   :", lengths.min())
print("Max   :", lengths.max())
print("Mean  :", round(lengths.mean(), 2))
print("Median:", int(np.median(lengths)))
print()

tasks = Counter(ep_task.values())

print("Episodes per task")
for task, count in tasks.items():
    print(f"{task}: {count}")