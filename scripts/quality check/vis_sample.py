from datasets import load_from_disk
import matplotlib.pyplot as plt
import random

ds = load_from_disk("/home/newin/Projects/vla_pi/lerobot_vla_dataset")

indices = random.sample(range(len(ds)), 4)

fig, axes = plt.subplots(4, 2, figsize=(8, 16))

for row, idx in enumerate(indices):
    sample = ds[idx]

    axes[row, 0].imshow(sample["observation.images.top"])
    axes[row, 0].set_title(
        f"TOP | Ep {sample['episode_index']} | {sample['instruction'][:30]}..."
    )
    axes[row, 0].axis("off")

    axes[row, 1].imshow(sample["observation.images.wrist"])
    axes[row, 1].set_title(
        f"WRIST | Frame {sample['frame_index']}"
    )
    axes[row, 1].axis("off")

plt.tight_layout()
plt.show()