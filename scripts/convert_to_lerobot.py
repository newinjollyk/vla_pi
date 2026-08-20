from pathlib import Path
import json
import numpy as np
from datasets import Dataset, Features, Sequence, Value, Image

RAW_ROOT = Path("/home/newin/Projects/vla_pi/dataset")
OUT_DIR = Path("/home/newin/Projects/vla_pi/lerobot_vla_dataset")

RED_INSTRUCTIONS = [
    "pick up the red cube and place it in the target area",
    "grasp the red cube and put it in the tray",
    "move the red block to the goal position",
    "take the red cube and place it in the container",
    "pick and place the red cube into the tray",
]

BLUE_INSTRUCTIONS = [
    "pick up the blue cube and place it in the target area",
    "grasp the blue cube and put it in the tray",
    "move the blue block to the goal position",
    "take the blue cube and place it in the container",
    "pick and place the blue cube into the tray",
]

records = []

episode_dirs = sorted(RAW_ROOT.glob("episode_*"))

print(f"Found {len(episode_dirs)} episodes")

for ep_idx, ep in enumerate(episode_dirs):

    meta_path = ep / "metadata.json"

    if not meta_path.exists():
        print(f"Skipping {ep.name}: missing metadata.json")
        continue

    with open(meta_path, "r") as f:
        meta = json.load(f)

    task_id = meta.get("task", "unknown_task")

    # Choose one of 5 instructions based on episode number
    instruction_idx = ep_idx % 5

    if "red" in task_id:
        instruction = RED_INSTRUCTIONS[instruction_idx]
    elif "blue" in task_id:
        instruction = BLUE_INSTRUCTIONS[instruction_idx]
    else:
        instruction = task_id.replace("_", " ")

    states = np.load(ep / "states.npy")
    actions = np.load(ep / "actions.npy")

    top_images = sorted((ep / "top").glob("*.png"))
    wrist_images = sorted((ep / "wrist").glob("*.png"))

    T = len(states)

    if not (len(actions) == len(top_images) == len(wrist_images) == T):
        print(f"Skipping {ep.name}: mismatch")
        continue

    for t in range(T):
        records.append({
            "episode_index": ep_idx,
            "frame_index": t,
            "observation.images.top": str(top_images[t]),
            "observation.images.wrist": str(wrist_images[t]),
            "observation.state": states[t].astype(np.float32).tolist(),
            "action": actions[t].astype(np.float32).tolist(),
            "task_id": task_id,
            "instruction": instruction,
        })

    print(f"Added {ep.name} -> {instruction}")

features = Features({
    "episode_index": Value("int32"),
    "frame_index": Value("int32"),
    "observation.images.top": Image(),
    "observation.images.wrist": Image(),
    "observation.state": Sequence(Value("float32")),
    "action": Sequence(Value("float32")),
    "task_id": Value("string"),
    "instruction": Value("string"),
})

dataset = Dataset.from_list(records, features=features)

OUT_DIR.mkdir(parents=True, exist_ok=True)
dataset.save_to_disk(str(OUT_DIR))

print()
print("VLA dataset conversion complete!")
print(f"Saved to: {OUT_DIR}")
print(dataset)