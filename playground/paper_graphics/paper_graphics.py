import matplotlib.pyplot as plt
import random
import procgen_tools.maze as maze
from procgen_tools.maze import EnvState, create_venv, render_inner_grid
import numpy as np
import torch as t
import procgen_tools.patch_utils as pu
from procgen_tools.imports import hook, default_layer
import os

AX_SIZE = 3.5

"""
What's the role of axis size in matplotlib above? I thought axes were like layers in gimp.
"""

cheese_channel = 55
venv = create_venv(1,0,1)
state = maze.EnvState(venv.env.callmethod('get_state')[0])

obs = t.tensor(venv.reset(), dtype=t.float32)

with hook.set_hook_should_get_custom_data():
    hook.network(obs)

#fig, ax = plt.subplots()
fig, axd = plt.subplot_mosaic(
    ['L1', 'M', 'M', 'R1'],
    figsize=(AX_SIZE * 4, AX_SIZE),
    tight_layout=True,
)

activ = hook.get_value_by_label(default_layer)[0][cheese_channel]
axd['L1'].imshow(activ)
axd['L1'].imshow(activ, cmap='RdBu', vmin=-1, vmax=1)

plt.show()