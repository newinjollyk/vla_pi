import os
import json
import numpy as np

DATASET = "/home/newin/Projects/vla_pi/dataset"

EPISODES_TO_CHECK = 3

print("=" * 70)
print("Checking state/action alignment")
print("=" * 70)

for ep in range(EPISODES_TO_CHECK):

    episode_dir = os.path.join(
        DATASET,
        f"episode_{ep:04d}"
    )

    states = np.load(
        os.path.join(episode_dir, "states.npy")
    )

    actions = np.load(
        os.path.join(episode_dir, "actions.npy")
    )

    print(f"\nEpisode {ep}")
    print("States :", states.shape)
    print("Actions:", actions.shape)

    print("\nFirst 5 frames:")
    print("-" * 70)

    for i in range(min(5, len(actions))):

        print(f"\nFrame {i}")

        print("State:")
        print(states[i][:5])

        print("Action:")
        print(actions[i])

        if i + 1 < len(states):

            movement = states[i + 1][:3] - states[i][:3]

            print("Next-state movement:")
            print(movement)

    print("\n" + "=" * 70)

print("\nDone.")
