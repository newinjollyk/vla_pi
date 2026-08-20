import os
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ============================================================
# CONFIG
# ============================================================

RAW_DATASET = Path(
    "/home/newin/Projects/vla_pi/Dataset/dataset_2"
)

OUTPUT_DATASET = Path(
    "/home/newin/Projects/vla_pi/Dataset/lerobot_dataset_native_v2"
)

REPO_ID = "local/fetch_pick_place_v2"

FPS = 30

IMAGE_SIZE = 224


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):
    """
    Load image as RGB uint8 and resize to 224x224.
    Returns HWC uint8.
    """

    img = cv2.imread(str(path))

    if img is None:
        raise RuntimeError(f"Could not read image: {path}")

    # OpenCV BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Direct resize to model resolution
    img = cv2.resize(
        img,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    return img.astype(np.uint8)


# ============================================================
# FIND EPISODES
# ============================================================

def find_episodes():

    episodes = []

    for path in RAW_DATASET.glob("episode_*"):

        if not path.is_dir():
            continue

        try:
            index = int(path.name.split("_")[1])
            episodes.append((index, path))
        except Exception:
            continue

    episodes.sort(key=lambda x: x[0])

    return episodes


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RAW → NATIVE LEROBOT DATASET")
    print("=" * 70)

    print(f"Raw dataset : {RAW_DATASET}")
    print(f"Output      : {OUTPUT_DATASET}")
    print(f"Image size  : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"FPS         : {FPS}")
    print("=" * 70)

    if not RAW_DATASET.exists():
        raise FileNotFoundError(
            f"Raw dataset does not exist:\n{RAW_DATASET}"
        )

    if OUTPUT_DATASET.exists():

        raise FileExistsError(
            f"\nOutput dataset already exists:\n"
            f"{OUTPUT_DATASET}\n\n"
            f"Delete it or choose another output directory."
        )

    episodes = find_episodes()

    if not episodes:
        raise RuntimeError("No episode_* directories found.")

    print(f"Found {len(episodes)} episodes.")

    # ========================================================
    # CHECK FIRST EPISODE
    # ========================================================

    first_episode = episodes[0][1]

    states = np.load(
        first_episode / "states.npy"
    )

    actions = np.load(
        first_episode / "actions.npy"
    )

    top_files = sorted(
        (first_episode / "top").glob("*.png")
    )

    wrist_files = sorted(
        (first_episode / "wrist").glob("*.png")
    )

    print("\nFirst episode:")
    print(f"  states : {states.shape}")
    print(f"  actions: {actions.shape}")
    print(f"  top    : {len(top_files)}")
    print(f"  wrist  : {len(wrist_files)}")

    if not (
        len(states)
        == len(actions)
        == len(top_files)
        == len(wrist_files)
    ):
        raise RuntimeError(
            "Frame count mismatch in first episode."
        )

    # ========================================================
    # FEATURES
    # ========================================================

    features = {

        "observation.images.top": {
            "dtype": "video",
            "shape": (3, IMAGE_SIZE, IMAGE_SIZE),
            "names": ["channels", "height", "width"],
        },

        "observation.images.wrist": {
            "dtype": "video",
            "shape": (3, IMAGE_SIZE, IMAGE_SIZE),
            "names": ["channels", "height", "width"],
        },

        "observation.state": {
            "dtype": "float32",
            "shape": (25,),
            "names": ["state"],
        },

        "action": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["action"],
        },
    }

    # ========================================================
    # CREATE LEROBOT DATASET
    # ========================================================

    print("\nCreating LeRobot dataset...")

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        root=OUTPUT_DATASET,
        fps=FPS,
        features=features,

        use_videos=True,

        # Keep conversion reasonably fast.
        image_writer_processes=0,
        image_writer_threads=4,

        video_backend="torchcodec",
    )

    # ========================================================
    # CONVERT EPISODES
    # ========================================================

    for episode_number, episode_dir in episodes:

        print("\n" + "-" * 70)
        print(
            f"Converting episode {episode_number}: "
            f"{episode_dir.name}"
        )
        print("-" * 70)

        # ----------------------------------------------------
        # Load arrays
        # ----------------------------------------------------

        states = np.load(
            episode_dir / "states.npy"
        ).astype(np.float32)

        actions = np.load(
            episode_dir / "actions.npy"
        ).astype(np.float32)

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata_file = (
            episode_dir / "metadata.json"
        )

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        task = metadata.get(
            "task",
            "Pick the red cube and place it in the tray."
        )

        # ----------------------------------------------------
        # Images
        # ----------------------------------------------------

        top_files = sorted(
            (episode_dir / "top").glob("*.png")
        )

        wrist_files = sorted(
            (episode_dir / "wrist").glob("*.png")
        )

        n = len(states)

        if not (
            len(actions)
            == len(top_files)
            == len(wrist_files)
            == n
        ):
            raise RuntimeError(
                f"Frame mismatch in {episode_dir.name}: "
                f"states={len(states)}, "
                f"actions={len(actions)}, "
                f"top={len(top_files)}, "
                f"wrist={len(wrist_files)}"
            )

        # ----------------------------------------------------
        # Add frames
        # ----------------------------------------------------

        for i in range(n):

            top_img = load_image(
                top_files[i]
            )

            wrist_img = load_image(
                wrist_files[i]
            )

            frame = {

                "observation.images.top":
                    top_img,

                "observation.images.wrist":
                    wrist_img,

                "observation.state":
                    states[i],

                "action":
                    actions[i],

                "task":
                    task,
            }

            dataset.add_frame(frame)

        # ----------------------------------------------------
        # Save episode
        # ----------------------------------------------------

        dataset.save_episode(
            parallel_encoding=True
        )

        print(
            f"Saved episode {episode_number} "
            f"({n} frames)"
        )

    # ========================================================
    # FINALIZE
    # ========================================================

    print("\nFinalizing dataset...")

    dataset.finalize()

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)

    print(
        f"Native LeRobot dataset:\n"
        f"{OUTPUT_DATASET}"
    )

    print(
        f"\nEpisodes converted: {len(episodes)}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()