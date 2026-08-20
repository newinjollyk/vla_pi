import numpy as np
from pathlib import Path
ep=Path('/home/newin/Projects/vla_pi/dataset/episode_0000')
print('states :', np.load(ep/'states.npy').shape)
print('actions :', np.load(ep/'actions.npy').shape)
print('timestamps:', np.load(ep/'timestamps.npy').shape)
print('rewards :', np.load(ep/'rewards.npy').shape)
print('done :', np.load(ep/'done.npy').shape)