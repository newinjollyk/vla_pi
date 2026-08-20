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
    "/home/newin/Projects/vla_pi/outputs/train/vla_red_blue_5k_v2/checkpoints/005000/pretrained_model"
)

TASK = "Pick the red cube and place it in the tray."

DEVICE = "cuda"

# ------------------------------------------------------------
# Control configuration
# ------------------------------------------------------------

# SmolVLA predicts 50 actions.
MODEL_CHUNK_SIZE = 50

# We execute only 20, then get a new observation
# and predict another 50.
REPLAN_AFTER = 50

# Maximum MuJoCo environment steps.
MAX_STEPS = 600

# MuJoCo action frequency
CONTROL_HZ = 30

# Dataset image size
IMAGE_SIZE = 224

# Cameras used when creating the dataset
TOP_CAMERA_ID = 4
WRIST_CAMERA_ID = 2


# ============================================================
# CHECKPOINT
# ============================================================

if not os.path.isdir(CHECKPOINT):
    raise FileNotFoundError(
        f"\nCheckpoint not found:\n{CHECKPOINT}\n"
    )


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("=" * 70)
print("SmolVLA 20-Action Receding-Horizon Inference")
print("=" * 70)

print(f"Checkpoint       : {CHECKPOINT}")
print(f"Task             : {TASK}")
print(f"Device           : {DEVICE}")
print(f"Model chunk      : {MODEL_CHUNK_SIZE} actions")
print(f"Execute/replan   : {REPLAN_AFTER} actions")
print(
    f"Replan interval  : "
    f"{REPLAN_AFTER / CONTROL_HZ:.2f} seconds"
)
print(f"Max steps        : {MAX_STEPS}")
print("=" * 70)


# ============================================================
# CREATE MUJOCO ENVIRONMENT
# ============================================================

gym.register_envs(gymnasium_robotics)

env = gym.make(
    "FetchPickAndPlace-v4",
    render_mode="human",
    max_episode_steps=1000,
)

print("\nMuJoCo environment created.")

print(
    "Action space:",
    env.action_space
)


# ============================================================
# LOAD SMOLVLA
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
# LOAD PREPROCESSOR / POSTPROCESSOR
# ============================================================

print("\nLoading preprocessing pipeline...")

preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path=CHECKPOINT,
)

print("Preprocessing pipeline loaded.")


# ============================================================
# MUJOCO RENDERER
# ============================================================

renderer = env.unwrapped.mujoco_renderer


# ============================================================
# CAMERA CAPTURE
# ============================================================

def capture_camera(camera_id):

    renderer.camera_id = camera_id

    image = renderer.render(
        "rgb_array"
    )

    return image


# ============================================================
# BUILD CURRENT POLICY OBSERVATION
# ============================================================

def build_policy_observation(obs):

    # --------------------------------------------------------
    # Capture TOP image
    # --------------------------------------------------------

    top_img = capture_camera(
        TOP_CAMERA_ID
    )

    # --------------------------------------------------------
    # Capture WRIST image
    # --------------------------------------------------------

    wrist_img = capture_camera(
        WRIST_CAMERA_ID
    )

    # --------------------------------------------------------
    # Resize to dataset resolution
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
    # HWC → CHW
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
    # 25-D robot state
    # --------------------------------------------------------

    state = np.asarray(
        obs["observation"],
        dtype=np.float32,
    )

    state_tensor = torch.from_numpy(
        state
    )

    # --------------------------------------------------------
    # Build LeRobot observation
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Move tensors to CUDA
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
# GET ONE ACTION FROM SMOLVLA
# ============================================================

def get_action_chunk(batch):

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    processed = preprocessor(batch)

    # --------------------------------------------------------
    # Predict FULL action chunk
    # --------------------------------------------------------

    with torch.inference_mode():

        actions = policy.predict_action_chunk(
            processed
        )

    # --------------------------------------------------------
    # Postprocess / unnormalize
    # --------------------------------------------------------

    actions = postprocessor(actions)

    # --------------------------------------------------------
    # Tensor -> NumPy
    # --------------------------------------------------------

    actions = (
        actions
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Remove batch dimension
    #
    # Expected input:
    #   (1, 50, 4)
    #
    # Output:
    #   (50, 4)
    # --------------------------------------------------------

    if actions.ndim == 3:
        actions = actions[0]

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if actions.ndim != 2:
        raise RuntimeError(
            f"Expected action chunk with shape "
            f"(N, 4), got {actions.shape}"
        )

    if actions.shape[-1] != 4:
        raise RuntimeError(
            f"Expected action dimension 4, "
            f"got {actions.shape}"
        )

    actions = np.clip(
        actions,
        -1.0,
        1.0,
    ).astype(np.float32)

    return actions


# ============================================================
# MAIN CLOSED-LOOP INFERENCE
# ============================================================

try:

    # --------------------------------------------------------
    # RESET ENVIRONMENT
    # --------------------------------------------------------

    obs, info = env.reset()

    policy.reset()

    print("\n" + "=" * 70)
    print("Environment reset")
    print("=" * 70)

    print("\nTask:")
    print(TASK)

    print(
        f"\nExecuting {REPLAN_AFTER} actions "
        f"before each new prediction."
    )

    print(
        f"Each prediction contains "
        f"{MODEL_CHUNK_SIZE} actions."
    )

    print("\nStarting autonomous control...\n")

    # --------------------------------------------------------
    # CONTROL LOOP
    # --------------------------------------------------------

    step = 0
    success = False
    terminated = False
    truncated = False

    while step < MAX_STEPS:

        print("\n" + "-" * 70)
        print(
            f"NEW PREDICTION at environment step {step}"
        )
        print("-" * 70)

        # ====================================================
        # CAPTURE FRESH OBSERVATION
        # ====================================================

        t0 = time.time()

        batch = build_policy_observation(obs)

        observation_time = time.time() - t0

        print(
            f"Observation capture: "
            f"{observation_time:.3f}s"
        )

        # ====================================================
        # PREDICT NEW ACTION CHUNK
        # ====================================================

        t0 = time.time()

        actions = get_action_chunk(batch)

        prediction_time = time.time() - t0

        # ----------------------------------------------------
        # Make sure actions are numpy
        # ----------------------------------------------------

        actions = np.asarray(
            actions,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Normalize shape
        #
        # Expected:
        #     (50, 4)
        # ----------------------------------------------------

        if actions.ndim == 1:

            # Single action
            if actions.shape[0] == 4:

                actions = actions.reshape(
                    1, 4
                )

            else:

                raise RuntimeError(
                    f"Unexpected action shape: "
                    f"{actions.shape}"
                )

        elif actions.ndim == 3:

            # Possible shape: (1, 50, 4)

            actions = actions.squeeze(0)

        if actions.ndim != 2:

            raise RuntimeError(
                f"Unexpected action shape after "
                f"processing: {actions.shape}"
            )

        if actions.shape[-1] != 4:

            raise RuntimeError(
                f"Expected action dimension 4, "
                f"got {actions.shape}"
            )

        print(
            f"New prediction time: "
            f"{prediction_time:.3f}s"
        )

        print(
            f"Predicted action chunk: "
            f"{actions.shape}"
        )

        print(
            f"First action: "
            f"{actions[0]}"
        )

        # ====================================================
        # EXECUTE ONLY FIRST 20 ACTIONS
        # ====================================================

        horizon = min(
            REPLAN_AFTER,
            len(actions),
            MAX_STEPS - step
        )

        print(
            f"Executing {horizon} actions "
            f"from this prediction."
        )

        for i in range(horizon):

            action = np.clip(
                actions[i],
                -1.0,
                1.0
            ).astype(
                np.float32
            )

            # ------------------------------------------------
            # APPLY ACTION
            # ------------------------------------------------

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            print(
                f"Step {step:03d} | "
                f"chunk {i+1:02d}/{horizon} | "
                f"action = {action}"
            )

            # ========================================================
            # GRIPPER / CUBE DIAGNOSTIC
            # ========================================================

            if step % 20 == 0:

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

                print(
                    f"  GRIPPER: {grip_pos}"
                )

                print(
                    f"  CUBE:    {cube_pos}"
                )

                print(
                    f"  DISTANCE: {distance:.4f} m"
                )

                print(
                    f"  FINGERS: L={left:.4f}, R={right:.4f}"
                )

            step += 1

            # ------------------------------------------------
            # CHECK SUCCESS
            # ------------------------------------------------

            success = bool(
                info.get(
                    "is_success",
                    False
                )
            )

            if success:

                print(
                    "\n"
                    + "=" * 70
                )
                print(
                    "TASK SUCCESS!"
                )
                print(
                    f"Environment step: {step}"
                )
                print(
                    f"Reward: {reward}"
                )
                print(
                    "=" * 70
                )

                break

            # ------------------------------------------------
            # CHECK ENVIRONMENT TERMINATION
            # ------------------------------------------------

            if terminated or truncated:

                print(
                    "\nEnvironment terminated."
                )

                break

        # ====================================================
        # END CONDITIONS
        # ====================================================

        if success:

            break

        if terminated or truncated:

            break

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n" + "=" * 70)
    print("ENVIRONMENT FINISHED")
    print("=" * 70)

    print(
        f"Environment step: {step}"
    )

    print(
        f"Reward: {reward}"
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
    print("Inference interrupted by user.")
    print("=" * 70)


finally:

    print("\nClosing MuJoCo...")

    env.close()

    print("Inference node stopped.")