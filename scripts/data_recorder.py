import os
import json
import cv2
import numpy as np
import shutil

class DataRecorder:

    def __init__(
        self,
        dataset_dir="dataset_2"
    ):

        self.dataset_dir = dataset_dir

        self.episode_index = 0
        self.frame_index = 0

        self.renderer = None

        os.makedirs(
            self.dataset_dir,
            exist_ok=True
        )

    # =====================================================
    # START NEW EPISODE
    # =====================================================

    def start_episode(
        self,
        task
    ):

        self.frame_index = 0

        self.episode_dir = os.path.join(
            self.dataset_dir,
            f"episode_{self.episode_index:04d}"
        )

        self.top_dir = os.path.join(
            self.episode_dir,
            "top"
        )

        self.wrist_dir = os.path.join(
            self.episode_dir,
            "wrist"
        )

        os.makedirs(
            self.top_dir,
            exist_ok=True
        )

        os.makedirs(
            self.wrist_dir,
            exist_ok=True
        )

        self.states = []
        self.actions = []
        self.timestamps = []
        self.rewards = []
        self.done = []

        self.metadata = {
            "episode": self.episode_index,
            "task": task,
            "environment": "FetchPickAndPlace-v4",
            "success": False,
            "num_frames": 0,
            "duration": 0.0
        }
        print(f"Started Episode {self.episode_index}")

    # =====================================================
    # RECORD ONE FRAME
    # =====================================================

    def record_frame(
        self,
        env,
        state,
        action,
        timestamp,
        reward,
        done
    ):

        # ------------------------------------------
        # Get renderer (only once)
        # ------------------------------------------

        if self.renderer is None:

            self.renderer = env.unwrapped.mujoco_renderer

        # ------------------------------------------
        # Capture Top Camera (Camera ID = 4)
        # ------------------------------------------

        self.renderer.camera_id = 4

        top_img = self.renderer.render(
            "rgb_array"
        )

        # ------------------------------------------
        # Capture Wrist Camera (Camera ID = 2)
        # ------------------------------------------

        self.renderer.camera_id = 2

        wrist_img = self.renderer.render(
            "rgb_array"
        )

        # ------------------------------------------
        # Convert RGB → BGR
        # ------------------------------------------

        top_img = cv2.cvtColor(
            top_img,
            cv2.COLOR_RGB2BGR
        )

        wrist_img = cv2.cvtColor(
            wrist_img,
            cv2.COLOR_RGB2BGR
        )

        # ------------------------------------------
        # Image filenames
        # ------------------------------------------

        top_file = os.path.join(
            self.top_dir,
            f"{self.frame_index:06d}.png"
        )

        wrist_file = os.path.join(
            self.wrist_dir,
            f"{self.frame_index:06d}.png"
        )

        # ------------------------------------------
        # Save images
        # ------------------------------------------

        cv2.imwrite(
            top_file,
            top_img
        )

        cv2.imwrite(
            wrist_file,
            wrist_img
        )

        # ------------------------------------------
        # Save robot data
        # ------------------------------------------

        self.states.append(
            np.array(state)
        )

        self.actions.append(
            np.array(action)
        )

        self.timestamps.append(
            timestamp
        )

        self.rewards.append(
            reward
        )

        self.done.append(
            done
        )

        self.frame_index += 1

    # =====================================================
    # FINISH EPISODE
    # =====================================================

    def finish_episode(self, success=True):

        np.save(
            os.path.join(
                self.episode_dir,
                "states.npy"
            ),
            np.array(self.states)
        )

        np.save(
            os.path.join(
                self.episode_dir,
                "actions.npy"
            ),
            np.array(self.actions)
        )

        np.save(
            os.path.join(
                self.episode_dir,
                "timestamps.npy"
            ),
            np.array(self.timestamps)
        )

        np.save(
            os.path.join(
                self.episode_dir,
                "rewards.npy"
            ),
            np.array(self.rewards)
        )

        np.save(
            os.path.join(
                self.episode_dir,
                "done.npy"
            ),
            np.array(self.done)
        )

        # ------------------------------------------
        # Update metadata
        # ------------------------------------------

        self.metadata["success"] = success
        self.metadata["num_frames"] = self.frame_index

        if len(self.timestamps) > 1:
            self.metadata["duration"] = (
                self.timestamps[-1] - self.timestamps[0]
            )
        else:
            self.metadata["duration"] = 0.0

        with open(
            os.path.join(
                self.episode_dir,
                "metadata.json"
            ),
            "w"
        ) as file:

            json.dump(
                self.metadata,
                file,
                indent=4
            )

        print(
            f"Episode {self.episode_index} saved."
        )

        self.episode_index += 1

    # =====================================================
    # DELETE FAILED EPISODE
    # =====================================================

    def delete_episode(self):

        if os.path.exists(self.episode_dir):

            shutil.rmtree(
                self.episode_dir
            )

            print(
                f"Episode {self.episode_index} deleted."
            )
        self.episode_index += 1