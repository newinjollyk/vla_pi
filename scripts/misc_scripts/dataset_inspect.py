from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("lerobot/stanford_kuka_multimodal_dataset")

print("Dataset length:", len(dataset))

sample = dataset[0]

# ← Paste the new code HERE

print("\n================ SAMPLE DETAILS ================\n")

print("Image shape:")
print(sample["observation.images.image"].shape)

print("\nState shape:")
print(sample["observation.state"].shape)

print("\nAction shape:")
print(sample["action"].shape)

print("\nRobot state:")
print(sample["observation.state"])

print("\nAction:")
print(sample["action"])

print("\nTask:")
print(sample["task"])

print("\n===============================================\n")

print("\nKeys:")
print(sample.keys())

print("\nContents:")
for key, value in sample.items():
    print(f"{key}: {type(value)}")