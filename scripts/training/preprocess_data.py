from datasets import load_from_disk
from PIL import Image
import numpy as np
import json
import os
import random
from collections import defaultdict

# =========================================================
# Configuration
# =========================================================

INPUT_DATASET = "/home/newin/Projects/vla_pi/lerobot_vla_dataset"
OUTPUT_DATASET = "/home/newin/Projects/vla_pi/lerobot_vla_dataset_224"

IMAGE_SIZE = 224
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

os.makedirs(OUTPUT_DATASET, exist_ok=True)

# =========================================================
# Load dataset
# =========================================================

print("Loading dataset...")
ds = load_from_disk(INPUT_DATASET)

print(ds)
print(f"Total samples: {len(ds)}")

# =========================================================
# Resize top camera images
# =========================================================

def resize_images(example):
    top = example["observation.images.top"]
    wrist = example["observation.images.wrist"]

    top = top.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    wrist = wrist.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

    example["observation.images.top"] = top
    example["observation.images.wrist"] = wrist

    return example

print(f"Resizing top and wrist images to {IMAGE_SIZE}x{IMAGE_SIZE}...")

ds = ds.map(
    resize_images,
    desc="Resizing top and wrist images"
)

# =========================================================
# Compute normalization statistics
# =========================================================

print("Computing normalization statistics...")

states = np.stack(ds["observation.state"]).astype(np.float32)
actions = np.stack(ds["action"]).astype(np.float32)

stats = {
    "state_mean": states.mean(axis=0).tolist(),
    "state_std": states.std(axis=0).tolist(),
    "action_mean": actions.mean(axis=0).tolist(),
    "action_std": actions.std(axis=0).tolist(),
}

stats_path = os.path.join(OUTPUT_DATASET, "normalization_stats.json")

with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)

print(f"Saved: {stats_path}")

# =========================================================
# Build episode -> task mapping
# =========================================================

print("Building episode mapping...")

episode_to_task = {}

for ex in ds:
    ep = int(ex["episode_index"])
    task = ex["instruction"]

    if ep not in episode_to_task:
        episode_to_task[ep] = task

print(f"Total episodes: {len(episode_to_task)}")

# =========================================================
# Group episodes by exact instruction phrase
# =========================================================

phrase_groups = defaultdict(list)

for ep, task in episode_to_task.items():
    phrase_groups[task].append(ep)

print("\nPhrase distribution:")
for phrase, eps in phrase_groups.items():
    print(f"{phrase} : {len(eps)} episodes")

# =========================================================
# Stratified split by phrase
# 80% train, 10% val, 10% test
# =========================================================

train_eps = []
val_eps = []
test_eps = []

print("\nCreating balanced split...")

for phrase, eps in phrase_groups.items():

    eps = sorted(eps)
    random.shuffle(eps)

    n = len(eps)

    n_train = int(round(n * 0.8))
    n_val = int(round(n * 0.1))
    n_test = n - n_train - n_val

    train = eps[:n_train]
    val = eps[n_train:n_train + n_val]
    test = eps[n_train + n_val:]

    train_eps.extend(train)
    val_eps.extend(val)
    test_eps.extend(test)

    print(f"\n{phrase}")
    print(f"  Train: {len(train)}")
    print(f"  Val  : {len(val)}")
    print(f"  Test : {len(test)}")

# Sort for readability
train_eps = sorted(train_eps)
val_eps = sorted(val_eps)
test_eps = sorted(test_eps)

# Sanity checks
assert len(set(train_eps) & set(val_eps)) == 0
assert len(set(train_eps) & set(test_eps)) == 0
assert len(set(val_eps) & set(test_eps)) == 0

split_info = {
    "seed": SEED,
    "image_size": IMAGE_SIZE,
    "train_episodes": train_eps,
    "val_episodes": val_eps,
    "test_episodes": test_eps,
}

split_path = os.path.join(OUTPUT_DATASET, "split_info.json")

with open(split_path, "w") as f:
    json.dump(split_info, f, indent=2)

print(f"\nSaved: {split_path}")

print("\nFinal split:")
print(f"Train episodes: {len(train_eps)}")
print(f"Val episodes  : {len(val_eps)}")
print(f"Test episodes : {len(test_eps)}")

# =========================================================
# Save processed dataset
# =========================================================

print("\nSaving processed dataset...")
ds.save_to_disk(OUTPUT_DATASET)

print("\nDone!")
print(f"Processed dataset saved to:")
print(f"  {OUTPUT_DATASET}")

print("\nFiles created:")
print("  normalization_stats.json")
print("  split_info.json")
print("  dataset_info.json")
print("  data-xxxxx.arrow shards")