
"""
lerobot-train \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --dataset.root=/home/newin/Projects/vla_pi/lerobot_vla_dataset_224 \
  --batch_size=4 \
  --steps=1000 \
  --output_dir=/home/newin/Projects/vla_pi/outputs/train/vla_red_blue \
  --job_name=vla_red_blue \
  --policy.device=cuda
"""

import subprocess
import shutil


# ============================================================
# SmolVLA Fine-Tuning Configuration
# ============================================================

DATASET_PATH = (
    "/home/newin/Projects/vla_pi/Dataset/lerobot_dataset_native_v2"
)

OUTPUT_DIR = (
    "/home/newin/Projects/vla_pi/outputs/train/vla_red_blue_5k_v2"
)

MODEL = "lerobot/smolvla_base"

BATCH_SIZE = 4

# First use 1 step to verify everything works.
# After successful test, change this to 1000.
STEPS = 5000

JOB_NAME = "vla_red_blue"


# ============================================================
# Check LeRobot Training Command
# ============================================================

lerobot_train = shutil.which("lerobot-train")

if lerobot_train is None:
    raise RuntimeError(
        "lerobot-train was not found.\n"
        "Make sure the vlaenv virtual environment is activated."
    )


# ============================================================
# Build Training Command
# ============================================================

command = [
    lerobot_train,

    # SmolVLA policy
    "--policy.type=smolvla",

    # Pretrained SmolVLA base model
    f"--policy.pretrained_path={MODEL}",

    # Required by this LeRobot version
    "--policy.repo_id=local/vla_red_blue",

    # Local native LeRobot dataset
    "--dataset.repo_id=local/fetch_pick_place",
    f"--dataset.root={DATASET_PATH}",

    "--policy.push_to_hub=false",

    # Training parameters
    f"--batch_size={BATCH_SIZE}",
    f"--steps={STEPS}",

    # Output
    f"--output_dir={OUTPUT_DIR}",
    f"--job_name={JOB_NAME}",

    # GPU
    "--policy.device=cuda",
]


# ============================================================
# Display Configuration
# ============================================================

print("=" * 60)
print("SmolVLA Fine-Tuning")
print("=" * 60)

print(f"Model         : {MODEL}")
print(f"Dataset       : {DATASET_PATH}")
print(f"Dataset size  : 199 episodes / 61369 frames")
print(f"Batch size    : {BATCH_SIZE}")
print(f"Training steps: {STEPS}")
print("Device        : CUDA")
print(f"Output        : {OUTPUT_DIR}")

print("=" * 60)
print("Starting training...")
print("=" * 60)


# ============================================================
# Start LeRobot Training
# ============================================================

result = subprocess.run(command)


# ============================================================
# Training Result
# ============================================================

if result.returncode == 0:

    print()
    print("=" * 60)
    print("Training finished successfully.")
    print("=" * 60)

    print("Results saved to:")
    print(OUTPUT_DIR)

else:

    print()
    print("=" * 60)
    print("Training stopped with an error.")
    print(f"Exit code: {result.returncode}")
    print("=" * 60)