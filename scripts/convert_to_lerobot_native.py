import os
import shutil
import numpy as np

from datasets import load_from_disk
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ============================================================
# Paths
# ============================================================

SOURCE_DATASET = (
    "/home/newin/Projects/vla_pi/lerobot_vla_dataset_224"
)

OUTPUT_DATASET = (
    "/home/newin/Projects/vla_pi/lerobot_dataset_native"
)


# ============================================================
# Dataset configuration
# ============================================================

REPO_ID = "local/vla_red_blue"

FPS = 30

USE_VIDEOS = True


# ============================================================
# Load processed Hugging Face dataset
# ============================================================

print("=" * 70)
print("Loading processed dataset")
print("=" * 70)

ds = load_from_disk(SOURCE_DATASET)

print(ds)
print()
print("Total frames:", len(ds))
print()


# ============================================================
# Basic validation
# ============================================================

required_columns = [
    "episode_index",
    "frame_index",
    "observation.images.top",
    "observation.images.wrist",
    "observation.state",
    "action",
    "instruction",
]

print("Checking required columns...")

for column in required_columns:
    if column not in ds.column_names:
        raise ValueError(
            f"Required column missing from dataset: {column}"
        )

print("All required columns found.")
print()


# ============================================================
# Check output directory
# ============================================================

if os.path.exists(OUTPUT_DATASET):
    raise FileExistsError(
        f"\nOutput directory already exists:\n"
        f"{OUTPUT_DATASET}\n\n"
        "Delete or rename it before running the conversion.\n"
        "The script will NOT overwrite an existing dataset."
    )


# ============================================================
# Determine image shape
# ============================================================

sample = ds[0]

top_image = np.asarray(
    sample["observation.images.top"]
)

wrist_image = np.asarray(
    sample["observation.images.wrist"]
)

print("Image validation")
print("----------------")

print("Top image shape  :", top_image.shape)
print("Wrist image shape:", wrist_image.shape)

if top_image.shape != (224, 224, 3):
    raise ValueError(
        f"Expected top image shape (224, 224, 3), "
        f"got {top_image.shape}"
    )

if wrist_image.shape != (224, 224, 3):
    raise ValueError(
        f"Expected wrist image shape (224, 224, 3), "
        f"got {wrist_image.shape}"
    )

print("Image sizes are correct.")
print()


# ============================================================
# Check state/action dimensions
# ============================================================

state = np.asarray(
    sample["observation.state"],
    dtype=np.float32
)

action = np.asarray(
    sample["action"],
    dtype=np.float32
)

print("State shape :", state.shape)
print("Action shape:", action.shape)

if state.shape != (25,):
    raise ValueError(
        f"Expected state shape (25,), got {state.shape}"
    )

if action.shape != (4,):
    raise ValueError(
        f"Expected action shape (4,), got {action.shape}"
    )

print("State/action dimensions are correct.")
print()


# ============================================================
# LeRobot feature definition
# ============================================================

features = {
    "observation.images.top": {
        "dtype": "video",
        "shape": (3, 224, 224),
        "names": ["channels", "height", "width"],
    },

    "observation.images.wrist": {
        "dtype": "video",
        "shape": (3, 224, 224),
        "names": ["channels", "height", "width"],
    },

    "observation.state": {
        "dtype": "float32",
        "shape": (25,),
        "names": None,
    },

    "action": {
        "dtype": "float32",
        "shape": (4,),
        "names": None,
    },
}


# ============================================================
# Create native LeRobot dataset
# ============================================================

print("=" * 70)
print("Creating native LeRobot dataset")
print("=" * 70)

print("Output:", OUTPUT_DATASET)
print("FPS   :", FPS)
print("Videos:", USE_VIDEOS)
print()

dataset = LeRobotDataset.create(
    repo_id=REPO_ID,
    fps=FPS,
    features=features,
    root=OUTPUT_DATASET,
    robot_type="custom_vla_robot",
    use_videos=USE_VIDEOS,
)


# ============================================================
# Find episodes
# ============================================================

episode_indices = sorted(
    set(int(x) for x in ds["episode_index"])
)

print("Total episodes:", len(episode_indices))
print()


# ============================================================
# Convert episodes
# ============================================================

for episode_number, episode_index in enumerate(
    episode_indices,
    start=1,
):

    print(
        f"[{episode_number}/{len(episode_indices)}] "
        f"Converting episode {episode_index}"
    )

    episode_rows = [
        i
        for i, ep in enumerate(ds["episode_index"])
        if int(ep) == episode_index
    ]

    # Sort by frame index
    episode_rows.sort(
        key=lambda i: int(ds[i]["frame_index"])
    )

    for row_index in episode_rows:

        sample = ds[row_index]

        # ----------------------------------------------------
        # Images
        # ----------------------------------------------------

        top = np.asarray(
            sample["observation.images.top"]
        )

        wrist = np.asarray(
            sample["observation.images.wrist"]
        )

        # Convert grayscale to RGB if necessary
        if top.ndim == 2:
            top = np.stack(
                [top, top, top],
                axis=-1,
            )

        if wrist.ndim == 2:
            wrist = np.stack(
                [wrist, wrist, wrist],
                axis=-1,
            )

        # LeRobot image representation:
        # H x W x C
        if top.shape != (224, 224, 3):
            raise ValueError(
                f"Unexpected top image shape at "
                f"episode {episode_index}, "
                f"frame {sample['frame_index']}: "
                f"{top.shape}"
            )

        if wrist.shape != (224, 224, 3):
            raise ValueError(
                f"Unexpected wrist image shape at "
                f"episode {episode_index}, "
                f"frame {sample['frame_index']}: "
                f"{wrist.shape}"
            )

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        state = np.asarray(
            sample["observation.state"],
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Action
        # ----------------------------------------------------

        action = np.asarray(
            sample["action"],
            dtype=np.float32,
        )

        # ----------------------------------------------------
        # Instruction → LeRobot task
        # ----------------------------------------------------

        task = str(
            sample["instruction"]
        )

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        # Try to preserve the original timestamp.
        #
        # If the processed dataset does not contain a
        # timestamp field, derive it from frame index and FPS.
        if "timestamp" in ds.column_names:
            timestamp = float(
                sample["timestamp"]
            )
        else:
            timestamp = (
                int(sample["frame_index"]) / FPS
            )

        # ----------------------------------------------------
        # Build LeRobot frame
        # ----------------------------------------------------

        frame = {
            "observation.images.top": top,
            "observation.images.wrist": wrist,
            "observation.state": state,
            "action": action,
            "task": task,
        }

        # ----------------------------------------------------
        # Add frame
        # ----------------------------------------------------

        dataset.add_frame(frame)

    # --------------------------------------------------------
    # Save episode
    # --------------------------------------------------------

    dataset.save_episode()

    print(
        f"  Saved episode {episode_index} "
        f"({len(episode_rows)} frames)"
    )


# ============================================================
# Final output
# ============================================================

print()
print("=" * 70)
print("Conversion complete")
print("=" * 70)

print()
print("Native LeRobot dataset:")
print(OUTPUT_DATASET)

print()
print("Episodes converted:", len(episode_indices))
print("Frames converted  :", len(ds))
print()
print("Next step: verify the dataset using LeRobotDataset.")
dataset.finalize()