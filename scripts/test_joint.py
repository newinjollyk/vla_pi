import mujoco
import mujoco.viewer
import numpy as np
import time

# Load Panda XML model
model = mujoco.MjModel.from_xml_path(
    "/home/newin/Projects/vla_pi/vlaenv/lib/python3.10/site-packages/gymnasium_robotics/envs/assets/kitchen_franka/kitchen_assets/kitchen_env_model.xml"
)

data = mujoco.MjData(model)

# Choose actuator/joint
joint_id = 0

with mujoco.viewer.launch_passive(model, data) as viewer:

    start = time.time()

    while viewer.is_running() and time.time() - start < 30:

        # Reset controls
        data.ctrl[:] = 0

        # Move ONE joint actuator
        data.ctrl[joint_id] = 0.2

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(0.01)