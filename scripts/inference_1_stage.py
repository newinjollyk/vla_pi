import os
import time

import cv2
import gymnasium as gym
import gymnasium_robotics
import mujoco
import numpy as np
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors


# ============================================================
# Configuration
# ============================================================

CHECKPOINT = (
    "/home/newin/Projects/vla_pi/"
    "outputs/train/vla_red_blue/"
    "checkpoints/001000/pretrained_model"
)

TASK = "Pick the blue cube and place it in the tray."

DEVICE = "cuda"


# ============================================================
# Check checkpoint
# ============================================================

if not os.path.isdir(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )

print("=" * 70)
print("SmolVLA → MuJoCo inference test")
print("=" * 70)

print("Checkpoint:")
print(CHECKPOINT)

print("Task:")
print(TASK)

print("Device:")
print(DEVICE)

print("=" * 70)


# ============================================================
# Load MuJoCo environment
# ============================================================

gym.register_envs(gymnasium_robotics)

env = gym.make(
    "FetchPickAndPlace-v4",
    render_mode="human",
)

obs, info = env.reset()

print("\nMuJoCo environment loaded.")


# ============================================================
# Get renderer
# ============================================================

renderer = env.unwrapped.mujoco_renderer


# ============================================================
# Capture images
# ============================================================

def capture_images():

    # -------------------------------
    # Top camera
    # -------------------------------

    renderer.camera_id = 4

    top_img = renderer.render("rgb_array")

    # -------------------------------
    # Wrist camera
    # -------------------------------

    renderer.camera_id = 2

    wrist_img = renderer.render("rgb_array")

    return top_img, wrist_img


# ============================================================
# Capture initial observation
# ============================================================

top_img, wrist_img = capture_images()

print("\nRaw image information:")
print("Top  :", top_img.shape)
print("Wrist:", wrist_img.shape)


# ============================================================
# Resize to dataset resolution
# ============================================================

top_img = cv2.resize(
    top_img,
    (224, 224),
    interpolation=cv2.INTER_AREA,
)

wrist_img = cv2.resize(
    wrist_img,
    (224, 224),
    interpolation=cv2.INTER_AREA,
)


# ============================================================
# Convert HWC uint8 → CHW float32
# ============================================================

top_tensor = (
    torch.from_numpy(top_img)
    .permute(2, 0, 1)
    .float()
    / 255.0
)

wrist_tensor = (
    torch.from_numpy(wrist_img)
    .permute(2, 0, 1)
    .float()
    / 255.0
)


# ============================================================
# Robot state
# ============================================================

state = np.asarray(
    obs["observation"],
    dtype=np.float32,
)

state_tensor = torch.from_numpy(state)


print("\nState:")
print("Shape:", state_tensor.shape)
print("Values:", state)


# ============================================================
# Build raw LeRobot policy batch
# ============================================================

batch = {

    "observation.images.top":
        top_tensor.unsqueeze(0),

    "observation.images.wrist":
        wrist_tensor.unsqueeze(0),

    "observation.state":
        state_tensor.unsqueeze(0),

    "task":
        [TASK],
}


print("\nRaw policy batch:")
for key, value in batch.items():
    if isinstance(value, torch.Tensor):
        print(
            f"  {key}: "
            f"shape={tuple(value.shape)} "
            f"dtype={value.dtype}"
        )
    else:
        print(
            f"  {key}: {value}"
        )


# ============================================================
# Load trained SmolVLA
# ============================================================

print("\nLoading trained SmolVLA...")

policy = SmolVLAPolicy.from_pretrained(
    CHECKPOINT,
    local_files_only=True,
)

policy.eval()
policy.to(DEVICE)

print("Policy loaded.")


# ============================================================
# Load saved preprocessing pipeline
# ============================================================

print("\nLoading LeRobot preprocessing pipeline...")

preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path=CHECKPOINT,
)

print("Preprocessing pipeline loaded.")


# ============================================================
# Move tensors to GPU
# ============================================================

for key in batch:

    if isinstance(batch[key], torch.Tensor):

        batch[key] = batch[key].to(
            DEVICE,
            non_blocking=True,
        )


# ============================================================
# Preprocess
# ============================================================

print("\nRunning preprocessing...")

processed_batch = preprocessor(batch)


print("\nProcessed batch:")

for key, value in processed_batch.items():

    if isinstance(value, torch.Tensor):

        print(
            f"  {key}: "
            f"shape={tuple(value.shape)} "
            f"dtype={value.dtype} "
            f"device={value.device}"
        )

    else:

        print(
            f"  {key}: {value}"
        )


# ============================================================
# Predict action
# ============================================================

print("\nRunning SmolVLA inference...")

with torch.no_grad():

    start = time.perf_counter()

    action = policy.select_action(
        processed_batch
    )

    elapsed = time.perf_counter() - start


# ============================================================
# Postprocess action
# ============================================================

action = postprocessor(action)


# ============================================================
# Display result
# ============================================================

print("\n" + "=" * 70)
print("Inference result")
print("=" * 70)

print(
    f"Inference time: {elapsed:.4f} seconds"
)

print(
    "Action shape:",
    tuple(action.shape)
)

print(
    "Predicted action:",
    action.detach()
    .cpu()
    .numpy()
)

print("=" * 70)

print("\nIMPORTANT:")
print("The action was NOT sent to MuJoCo.")
print("This was a policy-only inference test.")

env.close()