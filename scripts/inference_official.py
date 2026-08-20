import os
import time

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT = (
    "/home/newin/Projects/vla_pi/"
    "outputs/train/vla_red_blue_5k_v2/"
    "checkpoints/005000/pretrained_model"
)

TASK = "Pick the red cube and place it in the tray."

DEVICE = "cuda"

MAX_STEPS = 600

TOP_CAMERA_ID = 4
WRIST_CAMERA_ID = 2


# ============================================================
# CHECKPOINT
# ============================================================

if not os.path.isdir(CHECKPOINT):
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )


print("=" * 70)
print("SmolVLA CLEAN OFFICIAL INFERENCE")
print("=" * 70)

print(f"Checkpoint : {CHECKPOINT}")
print(f"Task       : {TASK}")
print(f"Device     : {DEVICE}")
print(f"Max steps  : {MAX_STEPS}")

print("=" * 70)


# ============================================================
# ENVIRONMENT
# ============================================================

gym.register_envs(gymnasium_robotics)

env = gym.make(
    "FetchPickAndPlace-v4",
    render_mode="human",
    max_episode_steps=1000,
)

print("\nMuJoCo environment created.")
print("Action space:", env.action_space)


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

print("SmolVLA loaded.")


# ============================================================
# LOAD SAVED PREPROCESSOR / POSTPROCESSOR
# ============================================================

print("\nLoading saved preprocessing pipeline...")

preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path=CHECKPOINT,
)

print("Preprocessing pipeline loaded.")


# ============================================================
# CAMERA CAPTURE
# ============================================================

renderer = env.unwrapped.mujoco_renderer


def capture_camera(camera_id):
    """
    Capture raw RGB image from MuJoCo.

    IMPORTANT:
    No resizing.
    No /255.
    No normalization.

    The saved LeRobot preprocessing pipeline
    handles the model-side processing.
    """

    renderer.camera_id = camera_id

    image = renderer.render(
        "rgb_array"
    )

    # MuJoCo/OpenGL can return an array with negative strides.
    # Make it contiguous before giving it to LeRobot.
    return np.ascontiguousarray(image)


# ============================================================
# BUILD RAW POLICY OBSERVATION
# ============================================================

def build_observation(obs):

    top_image = capture_camera(
        TOP_CAMERA_ID
    )

    wrist_image = capture_camera(
        WRIST_CAMERA_ID
    )

    state = np.asarray(
        obs["observation"],
        dtype=np.float32,
    )

    observation = {
        "observation.images.top": top_image,
        "observation.images.wrist": wrist_image,
        "observation.state": state,
        "task": TASK,
    }

    return observation


# ============================================================
# DIAGNOSTICS
# ============================================================

def print_diagnostics(step, action):

    grip_pos = env.unwrapped.data.site(
        "robot0:grip"
    ).xpos.copy()

    cube_pos = env.unwrapped.data.body(
        "object0"
    ).xpos.copy()

    distance = np.linalg.norm(
        grip_pos - cube_pos
    )

    left = env.unwrapped.data.joint(
        "robot0:l_gripper_finger_joint"
    ).qpos[0]

    right = env.unwrapped.data.joint(
        "robot0:r_gripper_finger_joint"
    ).qpos[0]

    print()
    print(
        f"--- Diagnostic step {step} ---"
    )

    print(
        f"Action:   {action}"
    )

    print(
        f"Gripper:  {grip_pos}"
    )

    print(
        f"Cube:     {cube_pos}"
    )

    print(
        f"Distance: {distance:.4f} m"
    )

    print(
        f"Fingers:  L={left:.4f}, "
        f"R={right:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

try:

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    obs, info = env.reset()

    # VERY IMPORTANT:
    # clear SmolVLA's internal action queue
    policy.reset()

    print()
    print("=" * 70)
    print("ENVIRONMENT RESET")
    print("=" * 70)

    print()
    print("Task:")
    print(TASK)

    print()
    print(
        "Using official SmolVLA select_action()"
    )

    print(
        "with saved checkpoint preprocessing."
    )

    print()
    print("Starting autonomous control...")
    print()

    # --------------------------------------------------------
    # CONTROL LOOP
    # --------------------------------------------------------

    success = False
    last_reward = 0.0

    for step in range(MAX_STEPS):

        # ====================================================
        # CAPTURE RAW OBSERVATION
        # ====================================================

        capture_start = time.time()

        raw_observation = build_observation(
            obs
        )

        capture_time = (
            time.time() - capture_start
        )

        # ====================================================
        # OFFICIAL PREPROCESSOR
        # ====================================================

        preprocess_start = time.time()

        batch = preprocessor(
            raw_observation
        )

        preprocess_time = (
            time.time() - preprocess_start
        )

        # ====================================================
        # OFFICIAL SMOLVLA POLICY
        # ====================================================

        inference_start = time.time()

        with torch.inference_mode():

            action = policy.select_action(
                batch
            )

        inference_time = (
            time.time() - inference_start
        )

        # ====================================================
        # OFFICIAL POSTPROCESSOR
        # ====================================================

        postprocess_start = time.time()

        action = postprocessor(
            action
        )

        postprocess_time = (
            time.time() - postprocess_start
        )

        # ====================================================
        # CONVERT ACTION TO NUMPY
        # ====================================================

        if isinstance(
            action,
            torch.Tensor
        ):

            action = (
                action
                .detach()
                .cpu()
                .numpy()
            )

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        # Remove batch dimension if present
        if action.ndim == 2:
            action = action[0]

        # Remove singleton dimensions if present
        action = np.squeeze(action)

        # ----------------------------------------------------
        # FINAL SAFETY CHECK
        # ----------------------------------------------------

        if action.shape != (4,):

            raise RuntimeError(
                f"Expected action shape (4,), "
                f"got {action.shape}"
            )

        # Do NOT normalize/unnormalize here.
        # The official postprocessor already did it.

        action = np.clip(
            action,
            -1.0,
            1.0,
        ).astype(
            np.float32
        )

        # ====================================================
        # PRINT
        # ====================================================

        print(
            f"Step {step:03d} | "
            f"action = {action} | "
            f"capture={capture_time:.3f}s | "
            f"prep={preprocess_time:.3f}s | "
            f"inference={inference_time:.3f}s | "
            f"post={postprocess_time:.3f}s"
        )

        # ====================================================
        # MUJOCO STEP
        # ====================================================

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        last_reward = reward

        # ====================================================
        # DIAGNOSTICS
        # ====================================================

        if step % 10 == 0:

            print_diagnostics(
                step,
                action
            )

        # ====================================================
        # SUCCESS
        # ====================================================

        success = bool(
            info.get(
                "is_success",
                False
            )
        )

        if success:

            print()
            print("=" * 70)
            print("TASK SUCCESS!")
            print("=" * 70)

            print(
                f"Environment step: {step + 1}"
            )

            print(
                f"Reward: {reward}"
            )

            break

        # ====================================================
        # ENVIRONMENT TERMINATION
        # ====================================================

        if terminated or truncated:

            print()
            print(
                "Environment terminated."
            )

            break

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 70)
    print("ENVIRONMENT FINISHED")
    print("=" * 70)

    print(
        f"Environment step: {step + 1}"
    )

    print(
        f"Reward: {last_reward}"
    )

    if success:

        print(
            "RESULT: SUCCESS"
        )

    else:

        print(
            "RESULT: NOT SUCCESSFUL"
        )

    print("=" * 70)


except KeyboardInterrupt:

    print()
    print("=" * 70)
    print("Inference interrupted.")
    print("=" * 70)


finally:

    print()
    print("Closing MuJoCo...")

    env.close()

    print("Inference node stopped.")