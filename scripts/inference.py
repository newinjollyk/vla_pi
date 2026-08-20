import os
import time

import cv2
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = (
    "/home/newin/Projects/vla_pi/"
    "outputs/train/vla_red_blue_5k/"
    "checkpoints/005000/pretrained_model"
)

TASK = "Pick the blue cube and place it in the tray."

DEVICE = "cuda"

# Safety limit for first real rollout
MAX_STEPS = 300

# MuJoCo cameras used during dataset recording
TOP_CAMERA_ID = 4
WRIST_CAMERA_ID = 2

# Dataset/model image resolution
IMAGE_SIZE = 224

# MuJoCo runs at approximately 30 Hz
CONTROL_HZ = 30


# ============================================================
# CHECKPOINT
# ============================================================

if not os.path.isdir(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )


# ============================================================
# ENVIRONMENT
# ============================================================

gym.register_envs(gymnasium_robotics)

env = gym.make(
    "FetchPickAndPlace-v4",
    render_mode="human",
)

print("=" * 70)
print("SmolVLA MuJoCo Inference Node")
print("=" * 70)

print("Checkpoint:", CHECKPOINT)
print("Task      :", TASK)
print("Device    :", DEVICE)
print("Max steps :", MAX_STEPS)
print("=" * 70)


# ============================================================
# LOAD POLICY
# ============================================================

print("\nLoading SmolVLA...")

policy = SmolVLAPolicy.from_pretrained(
    CHECKPOINT,
    local_files_only=True,
)

policy.to(DEVICE)
policy.eval()

print("Policy loaded.")


# ============================================================
# LOAD PROCESSORS
# ============================================================

print("\nLoading preprocessing pipeline...")

preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path=CHECKPOINT,
)

print("Processors loaded.")


# ============================================================
# RESET POLICY
# ============================================================

policy.reset()

# Reset processor state as well if supported
if hasattr(preprocessor, "reset"):
    preprocessor.reset()

if hasattr(postprocessor, "reset"):
    postprocessor.reset()


# ============================================================
# MUJOCO RENDERER
# ============================================================

renderer = env.unwrapped.mujoco_renderer


# ============================================================
# CAPTURE CAMERA
# ============================================================

def capture_camera(camera_id):

    renderer.camera_id = camera_id

    image = renderer.render(
        "rgb_array"
    )

    return image


# ============================================================
# BUILD POLICY OBSERVATION
# ============================================================

def get_policy_observation(obs):

    # --------------------------------------------------------
    # Capture images
    # --------------------------------------------------------

    top_img = capture_camera(
        TOP_CAMERA_ID
    )

    wrist_img = capture_camera(
        WRIST_CAMERA_ID
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    top_img = cv2.resize(
        top_img,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    wrist_img = cv2.resize(
        wrist_img,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    # --------------------------------------------------------
    # HWC uint8 → CHW float32 [0,1]
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Robot state
    # --------------------------------------------------------

    state = np.asarray(
        obs["observation"],
        dtype=np.float32,
    )

    state_tensor = torch.from_numpy(
        state
    )

    # --------------------------------------------------------
    # Build raw LeRobot observation
    # --------------------------------------------------------

    batch = {

        "observation.images.top":
            top_tensor.unsqueeze(0),

        "observation.images.wrist":
            wrist_tensor.unsqueeze(0),

        "observation.state":
            state_tensor.unsqueeze(0),

        "task": [TASK],
    }

    # --------------------------------------------------------
    # Move tensors to GPU
    # --------------------------------------------------------

    for key in batch:

        if isinstance(
            batch[key],
            torch.Tensor,
        ):

            batch[key] = batch[key].to(
                DEVICE,
                non_blocking=True,
            )

    return batch


# ============================================================
# MAIN ROLLOUT
# ============================================================

try:

    obs, info = env.reset()

    # Reset policy action queue at beginning of episode
    policy.reset()

    print("\nEnvironment reset.")
    print("Starting autonomous control...")
    print()
    print("Task:")
    print(TASK)
    print()

    for step in range(MAX_STEPS):

        loop_start = time.perf_counter()

        # ----------------------------------------------------
        # Build observation
        # ----------------------------------------------------

        raw_batch = get_policy_observation(
            obs
        )

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        batch = preprocessor(
            raw_batch
        )

        # ----------------------------------------------------
        # SmolVLA inference
        # ----------------------------------------------------

        with torch.inference_mode():

            action = policy.select_action(
                batch
            )

        # ----------------------------------------------------
        # Postprocess / denormalize
        # ----------------------------------------------------

        action = postprocessor(
            action
        )

        # ----------------------------------------------------
        # Convert to NumPy
        # ----------------------------------------------------

        action = (
            action
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Safety clamp
        # ----------------------------------------------------

        action = np.clip(
            action,
            -1.0,
            1.0,
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # PRINT ACTION
        # ----------------------------------------------------

        if step % 10 == 0:

            print(
                f"Step {step:03d} | "
                f"Action: {action}"
            )

        # ----------------------------------------------------
        # SEND ACTION TO MUJOCO
        # ----------------------------------------------------

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        # ----------------------------------------------------
        # Render
        # ----------------------------------------------------

        env.render()

        # ----------------------------------------------------
        # Episode termination
        # ----------------------------------------------------

        if terminated or truncated:

            print(
                f"\nEnvironment finished at step {step}."
            )

            print(
                f"Reward: {reward}"
            )

            break

        # ----------------------------------------------------
        # Maintain approximately 30 Hz
        # ----------------------------------------------------

        elapsed = (
            time.perf_counter()
            - loop_start
        )

        remaining = (
            1.0 / CONTROL_HZ
            - elapsed
        )

        if remaining > 0:

            time.sleep(
                remaining
            )


except KeyboardInterrupt:

    print(
        "\n\nInference interrupted by user."
    )


finally:

    print("\nClosing MuJoCo...")

    env.close()

    print("Inference node stopped.")