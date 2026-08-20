from datasets import load_from_disk
import json
import os

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATASET_PATH = "/home/newin/Projects/vla_pi/lerobot_vla_dataset_224"

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
ds = load_from_disk(DATASET_PATH)

print("Dataset summary")
print(ds)
print()

# --------------------------------------------------
# Basic counts
# --------------------------------------------------
print("Total samples :", len(ds))
print("Total episodes:", len(set(ds["episode_index"])))
print()

# --------------------------------------------------
# Column names
# --------------------------------------------------
print("Columns:")
for c in ds.column_names:
    print(" -", c)
print()

# --------------------------------------------------
# Image sizes
# --------------------------------------------------
top_size = ds[0]["observation.images.top"].size
wrist_size = ds[0]["observation.images.wrist"].size

print("Image sizes")
print("Top  :", top_size)
print("Wrist:", wrist_size)
print()

# --------------------------------------------------
# State and action dimensions
# --------------------------------------------------
print("State dimension :", len(ds[0]["observation.state"]))
print("Action dimension:", len(ds[0]["action"]))
print()

# --------------------------------------------------
# Unique instructions
# --------------------------------------------------
instruction_counts = {}

for instr in ds["instruction"]:
    instruction_counts[instr] = instruction_counts.get(instr, 0) + 1

print("Unique instructions:", len(instruction_counts))
print()

for instr, count in sorted(instruction_counts.items()):
    print(f"{instr} : {count} samples")

print()

# --------------------------------------------------
# Split information
# --------------------------------------------------
split_path = os.path.join(DATASET_PATH, "split_info.json")

if os.path.exists(split_path):
    with open(split_path, "r") as f:
        split_info = json.load(f)

    print("Split summary")
    print("Train episodes:", len(split_info["train_episodes"]))
    print("Val episodes  :", len(split_info["val_episodes"]))
    print("Test episodes :", len(split_info["test_episodes"]))
    print()

# --------------------------------------------------
# Normalization statistics
# --------------------------------------------------
stats_path = os.path.join(DATASET_PATH, "normalization_stats.json")

if os.path.exists(stats_path):
    with open(stats_path, "r") as f:
        stats = json.load(f)

    print("Normalization stats")
    print("State mean dim :", len(stats["state_mean"]))
    print("State std dim  :", len(stats["state_std"]))
    print("Action mean dim:", len(stats["action_mean"]))
    print("Action std dim :", len(stats["action_std"]))
    print()

# --------------------------------------------------
# Final check
# --------------------------------------------------
print("Final check")
print("Top image is 224x224   :", top_size == (224, 224))
print("Wrist image is 224x224 :", wrist_size == (224, 224))
print("State dim is 25        :", len(ds[0]["observation.state"]) == 25)
print("Action dim is 4        :", len(ds[0]["action"]) == 4)
print("Instructions are 10    :", len(instruction_counts) == 10)