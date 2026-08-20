from lerobot.datasets.lerobot_dataset import LeRobotDataset

DATASET_ROOT = "/home/newin/Projects/vla_pi/lerobot_vla_dataset_224"

print("Loading with LeRobotDataset...")
print()

dataset = LeRobotDataset(
    repo_id="local/vla_red_blue",
    root=DATASET_ROOT,
)

print("LeRobot dataset loaded successfully!")
print()
print("Episodes:", dataset.meta.total_episodes)
print("Frames:", dataset.meta.total_frames)
print("Features:")
for name in dataset.meta.features:
    print(" -", name)