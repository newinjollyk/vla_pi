import json
from pathlib import Path


# ============================================================
# Paths
# ============================================================

NATIVE_DATASET = Path(
    "/home/newin/Projects/vla_pi/lerobot_dataset_native"
)

ORIGINAL_DATASET = Path(
    "/home/newin/Projects/vla_pi/lerobot_vla_dataset_224"
)

SPLIT_INFO = ORIGINAL_DATASET / "split_info.json"

OUTPUT_DIR = (
    Path("/home/newin/Projects/vla_pi")
    / "native_splits"
)


# ============================================================
# Load original split information
# ============================================================

print("=" * 70)
print("Preparing native LeRobot dataset splits")
print("=" * 70)

if not NATIVE_DATASET.exists():
    raise FileNotFoundError(
        f"Native dataset not found:\n{NATIVE_DATASET}"
    )

if not SPLIT_INFO.exists():
    raise FileNotFoundError(
        f"Split information not found:\n{SPLIT_INFO}"
    )


with open(SPLIT_INFO, "r") as f:
    split_info = json.load(f)


print("Loaded:")
print(SPLIT_INFO)
print()


# ============================================================
# Inspect split structure
# ============================================================

print("Split information:")
print(json.dumps(split_info, indent=2))
print()


# ============================================================
# Extract episode lists
# ============================================================

train_episodes = split_info["train_episodes"]
val_episodes = split_info["val_episodes"]
test_episodes = split_info["test_episodes"]


# ============================================================
# Validation
# ============================================================

train_episodes = sorted(
    int(x) for x in train_episodes
)

val_episodes = sorted(
    int(x) for x in val_episodes
)

test_episodes = sorted(
    int(x) for x in test_episodes
)


all_episodes = (
    train_episodes
    + val_episodes
    + test_episodes
)


if len(train_episodes) != 159:
    raise ValueError(
        f"Expected 159 training episodes, "
        f"got {len(train_episodes)}"
    )

if len(val_episodes) != 20:
    raise ValueError(
        f"Expected 20 validation episodes, "
        f"got {len(val_episodes)}"
    )

if len(test_episodes) != 20:
    raise ValueError(
        f"Expected 20 test episodes, "
        f"got {len(test_episodes)}"
    )


if len(set(all_episodes)) != 199:
    raise ValueError(
        "Episode overlap detected between train/val/test."
    )


print("Split validation passed.")
print()
print(f"Train episodes: {len(train_episodes)}")
print(f"Val episodes  : {len(val_episodes)}")
print(f"Test episodes : {len(test_episodes)}")
print()


# ============================================================
# Save episode lists
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


splits = {
    "train": train_episodes,
    "val": val_episodes,
    "test": test_episodes,
}


for name, episodes in splits.items():

    output_file = (
        OUTPUT_DIR / f"{name}_episodes.json"
    )

    with open(output_file, "w") as f:

        json.dump(
            {
                "dataset": str(NATIVE_DATASET),
                "split": name,
                "num_episodes": len(episodes),
                "episodes": episodes,
            },
            f,
            indent=2,
        )

    print(
        f"Saved {name} split:"
    )

    print(
        f"  {output_file}"
    )


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 70)
print("Native split preparation complete")
print("=" * 70)

print()
print("Train:", len(train_episodes))
print("Val  :", len(val_episodes))
print("Test :", len(test_episodes))

print()
print("The original native dataset was NOT modified.")