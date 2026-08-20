import os
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class DataRecorder:

    def __init__(
        self,
        dataset_dir="/home/newin/Projects/vla_pi/lerobot_dataset_native",
        repo_id="local/fetch_pick_place",
        fps=30,
        image_height=224,
        image_width=224,
    ):

        self.dataset_dir = dataset_dir
        self.repo_id = repo_id
        self.fps = fps

        self.image_height = image_height
        self.image_width = image_width

        # -----------------------------------------------------
        # LeRobot feature definition
        # -----------------------------------------------------

        features = {

            "observation.images.top": {
                "dtype": "image",
                "shape": (
                    3,
                    image_height,
                    image_width,
                ),
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            },

            "observation.images.wrist": {
                "dtype": "image",
                "shape": (
                    3,
                    image_height,
                    image_width,
                ),
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            },

            "observation.state": {
                "dtype": "float32",
                "shape": (25,),
                "names": None,
            },

            "action": {
                "dtype": "float32",
                "shape": (4,),
                "names": None,
            },
        }

        # -----------------------------------------------------
        # Create native LeRobot dataset
        # -----------------------------------------------------

        if os.path.exists(dataset_dir):
            raise FileExistsError(
                f"\nDataset directory already exists:\n"
                f"{dataset_dir}\n\n"
                "Refusing to overwrite it.\n"
                "Delete it or choose another directory."
            )

        print("=" * 70)
        print("Creating native LeRobot dataset")
        print("=" * 70)

        print("Dataset:", dataset_dir)
        print("FPS    :", fps)
        print("Images :", f"{image_width}x{image_height}")
        print()

        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=dataset_dir,
            robot_type="fetch_pick_place",
            use_videos=True,
        )

        self.episode_index = 0
        self.frame_index = 0

        self.current_task = None

        print("Native LeRobot dataset created.")
        print()

    # =========================================================
    # START EPISODE
    # =========================================================

    def start_episode(self, task):

        self.frame_index = 0
        self.current_task = task

        print()
        print(
            f"Started Episode {self.episode_index}"
        )

        print(
            f"Task: {task}"
        )

    # =========================================================
    # RECORD FRAME
    # =========================================================

    def record_frame(
        self,
        env,
        state,
        action,
        timestamp,
        reward,
        done,
    ):

        # -----------------------------------------------------
        # Get renderer
        # -----------------------------------------------------

        if not hasattr(self, "renderer"):

            self.renderer = (
                env.unwrapped.mujoco_renderer
            )

        # -----------------------------------------------------
        # TOP CAMERA
        # -----------------------------------------------------

        self.renderer.camera_id = 4

        top_img = self.renderer.render(
            "rgb_array"
        )

        # -----------------------------------------------------
        # WRIST CAMERA
        # -----------------------------------------------------

        self.renderer.camera_id = 2

        wrist_img = self.renderer.render(
            "rgb_array"
        )

        # -----------------------------------------------------
        # Validate images
        # -----------------------------------------------------

        top_img = np.asarray(
            top_img,
            dtype=np.uint8,
        )

        wrist_img = np.asarray(
            wrist_img,
            dtype=np.uint8,
        )

        expected_shape = (
            self.image_height,
            self.image_width,
            3,
        )

        if top_img.shape != expected_shape:

            raise ValueError(
                f"Top image has wrong shape: "
                f"{top_img.shape}, "
                f"expected {expected_shape}"
            )

        if wrist_img.shape != expected_shape:

            raise ValueError(
                f"Wrist image has wrong shape: "
                f"{wrist_img.shape}, "
                f"expected {expected_shape}"
            )

        # -----------------------------------------------------
        # State
        # -----------------------------------------------------

        state = np.asarray(
            state,
            dtype=np.float32,
        )

        if state.shape != (25,):

            raise ValueError(
                f"State has wrong shape: "
                f"{state.shape}, expected (25,)"
            )

        # -----------------------------------------------------
        # Action
        # -----------------------------------------------------

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        if action.shape != (4,):

            raise ValueError(
                f"Action has wrong shape: "
                f"{action.shape}, expected (4,)"
            )

        # -----------------------------------------------------
        # Build LeRobot frame
        # -----------------------------------------------------

        frame = {

            "observation.images.top":
                top_img,

            "observation.images.wrist":
                wrist_img,

            "observation.state":
                state,

            "action":
                action,

            # LeRobot treats task specially.
            "task":
                self.current_task,
        }

        # -----------------------------------------------------
        # Add frame
        # -----------------------------------------------------

        self.dataset.add_frame(
            frame
        )

        self.frame_index += 1

    # =========================================================
    # FINISH EPISODE
    # =========================================================

    def finish_episode(
        self,
        success=True,
    ):

        self.dataset.save_episode()

        print(
            f"Episode {self.episode_index} saved."
        )

        print(
            f"Frames: {self.frame_index}"
        )

        print(
            f"Success: {success}"
        )

        self.episode_index += 1

    # =========================================================
    # DELETE FAILED EPISODE
    # =========================================================

    def delete_episode(self):

        print(
            f"Episode {self.episode_index} "
            f"failed."
        )

        print(
            "Discarding current episode."
        )

        # The current episode has not been
        # saved with save_episode(), so the
        # incomplete episode is discarded.

        self.frame_index = 0

        self.episode_index += 1