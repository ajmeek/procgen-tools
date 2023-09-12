import matplotlib.pyplot as plt
import random
import procgen_tools.maze as maze
from procgen_tools.maze import EnvState, create_venv, render_inner_grid
import numpy as np
import torch as t
import procgen_tools.patch_utils as pu
from procgen_tools.imports import *
import os
import procgen_tools.visualization as viz
import procgen_tools.patch_utils as patch_utils

AX_SIZE = 3.5
patch_coords = (5,6)
seed = 0
"""
What's the role of axis size in matplotlib above? I thought axes were like layers in gimp.
"""

cheese_channel = 55

def fig_1():
    venv = create_venv(1,seed,1)
    state = maze.EnvState(venv.env.callmethod('get_state')[0])

    obs = t.tensor(venv.reset(), dtype=t.float32)

    with hook.set_hook_should_get_custom_data():
        hook.network(obs)

    #fig, ax = plt.subplots()
    fig1, axd1 = plt.subplot_mosaic(
        [['orig_mpp', 'orig_act', 'patch_act', 'patch_mpp']],
        figsize=(AX_SIZE * 4, AX_SIZE),
        tight_layout=True,
    )

    activ = hook.get_value_by_label(default_layer)[0][cheese_channel]
    axd1['orig_act'].imshow(activ)
    axd1['orig_act'].imshow(activ, cmap='RdBu', vmin=-1, vmax=1)

    axd1['orig_act'].set_xticks([])
    axd1['orig_act'].set_yticks([])

    #orig_mpp = viz.visualize_venv(venv, render_padding=False)
    vf = viz.vector_field(venv, policy)

    img = viz.plot_vf_mpp(vf, ax=axd1['orig_mpp'], save_img=False)
    axd1['orig_mpp'].imshow(img)

    patch_coords = (3, 12)
    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer,channel=55, value=5.6, coord=patch_coords)
    #patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer,channel=55, value=5.6, coord=(17,17)) # to test that this is on a 16x16 grid - yes it is

    # creating a new venv unnecessary. the reason why I see the original cheese's activations too is becaue I'm not zeroing out
    # the original activations. I can do that, but it looks worse. Best to just have the cheese graphic. Showing this to Alex
    # might make him change his mind.
    # venv = create_venv(1,seed,1)
    # state = maze.EnvState(venv.env.callmethod('get_state')[0])
    #
    # obs = t.tensor(venv.reset(), dtype=t.float32)
    #
    # # with hook.set_hook_should_get_custom_data():
    # #     hook.network(obs)

    # alright the below works to visualize the activations, finally. But, it still

    with hook.use_patches(patches):
        hook.network(obs) # trying before asking Uli
        patched_vfield = viz.vector_field(venv, hook.network)
        #hook.network(obs) # trying before asking Uli
    img = viz.plot_vf_mpp(patched_vfield, ax=axd1['patch_mpp'], save_img=False)
    axd1['patch_mpp'].imshow(img)

    # TODO - I think the activations for seed = 0 get vertically flipped somehow in the display
    # actually, it's that the coordinates for the below function use the full grid, not the inner grid.
    # and, the axes count from the standard origin. not flipped.
    #maze.move_cheese_in_state(state, patch_coords)
    maze.move_cheese_in_state(state, (18, 18))
    venv.env.callmethod('set_state', [state.state_bytes])
    obs = t.tensor(venv.reset(), dtype=t.float32)

    # with hook.set_hook_should_get_custom_data():
    #     hook.network(obs)

    # with hook.use_patches(patches):
    #     hook.network(obs)

    activ = hook.get_value_by_label(default_layer)[0][cheese_channel]
    axd1['patch_act'].imshow(activ)
    axd1['patch_act'].imshow(activ, cmap='RdBu', vmin=-1, vmax=1)

    axd1['patch_act'].set_xticks([])
    axd1['patch_act'].set_yticks([])

    #plt.show()
    #print(os.getcwd())
    plt.savefig('playground/paper_graphics/visualizations/fig_1.svg', bbox_inches="tight", format='svg')

#fig_1()

# ---------------------------------------------------- fig 2 ----------------------------------------------------
def fig_2():
    #move cheese in state uses full grid. use padding manually to get right grid coords
    #cheese_a_pos = ()
    cheese_b_pos = (18, 18) #top right
    cheese_c_pos = (6, 18) #bottom right

    fig2, axd2 = plt.subplot_mosaic(
        [['reg_venv', 'cheese_a', 'cheese_b', 'cheese_c']],
        figsize=(AX_SIZE * 4, AX_SIZE),
        tight_layout=True,
    )

    venv = create_venv(1,seed,1)
    state = maze.EnvState(venv.env.callmethod('get_state')[0])

    img = viz.visualize_venv(venv, ax=axd2['reg_venv'], render_padding=False)
    axd2['reg_venv'].imshow(img)

    # want to show activations from first cheese
    # maze.move_cheese_in_state(state, cheese_a_pos)
    # venv.env.callmethod('set_state', [state.state_bytes])
    obs = t.tensor(venv.reset(), dtype=t.float32)

    with hook.set_hook_should_get_custom_data():
        hook.network(obs)

    activ = hook.get_value_by_label(default_layer)[0][cheese_channel]
    axd2['cheese_a'].imshow(activ)
    axd2['cheese_a'].imshow(activ, cmap='RdBu', vmin=-1, vmax=1)

    maze.move_cheese_in_state(state, cheese_b_pos)
    venv.env.callmethod('set_state', [state.state_bytes])
    obs = t.tensor(venv.reset(), dtype=t.float32)

    with hook.set_hook_should_get_custom_data():
        hook.network(obs)

    #img = viz.visualize_venv(venv, render_padding=False, show_plot=True) #checking cheese coords, it is top right
    activ = hook.get_value_by_label(default_layer)[0][cheese_channel]
    axd2['cheese_b'].imshow(activ)
    axd2['cheese_b'].imshow(activ, cmap='RdBu', vmin=-1, vmax=1)

    maze.move_cheese_in_state(state, cheese_c_pos)
    venv.env.callmethod('set_state', [state.state_bytes])
    obs = t.tensor(venv.reset(), dtype=t.float32)

    with hook.set_hook_should_get_custom_data():
        hook.network(obs)

    #img = viz.visualize_venv(venv, render_padding=False, show_plot=True) #checking cheese coords, it is bottom right
    activ = hook.get_value_by_label(default_layer)[0][cheese_channel]
    axd2['cheese_c'].imshow(activ)
    axd2['cheese_c'].imshow(activ, cmap='RdBu', vmin=-1, vmax=1)


    #plt.show()
    plt.savefig('playground/paper_graphics/visualizations/fig_2.svg', bbox_inches="tight", format='svg')

fig_2()

# ---------------------------------------------------- fig 3 ----------------------------------------------------

# want to ask Mrinank what he means by 'ghost cheese' really, and how they may substantially differ from fig 4.
# computing it will be quite easy though

# ---------------------------------------------------- fig 4 ----------------------------------------------------
def fig_4():
    success_a_pos = (5,6)
    success_b_pos = (12,12)
    success_c_pos = (0,0) #(8,9) - too close to original mpp location
    fail = (0,12) #(2,3) - fail, but so is everything

    fig4, axd4 = plt.subplot_mosaic(
        [['success_a', 'success_b', 'success_c', 'fail']],
        figsize=(AX_SIZE * 4, AX_SIZE),
        tight_layout=True,
    )

    venv = create_venv(1,seed,1)
    state = maze.EnvState(venv.env.callmethod('get_state')[0])

    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer,channel=55, value=5, coord=success_a_pos)
    with hook.use_patches(patches):
        patched_vfield = viz.vector_field(venv, hook.network)
    img = viz.plot_vf_mpp(patched_vfield, ax=axd4['success_a'], save_img=False)

    #viz.visualize_venv(venv, show_plot=True, render_padding=False)

    padding = maze.get_padding(maze.get_inner_grid_from_seed(seed))
    viz.plot_pixel_dot(axd4['success_a'], success_a_pos[0], success_a_pos[1], hidden_padding=padding)
    #viz.plot_red_dot(venv, axd4['success_a'], success_a_pos[0], success_a_pos[1])
    axd4['success_a'].imshow(img)

    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer,channel=55, value=5, coord=success_b_pos)
    with hook.use_patches(patches):
        patched_vfield = viz.vector_field(venv, hook.network)
    img = viz.plot_vf_mpp(patched_vfield, ax=axd4['success_b'], save_img=False)

    padding = maze.get_padding(maze.get_inner_grid_from_seed(seed))
    viz.plot_pixel_dot(axd4['success_b'], success_b_pos[0], success_b_pos[1], hidden_padding=padding)
    axd4['success_b'].imshow(img)

    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer,channel=55, value=5, coord=success_c_pos)
    with hook.use_patches(patches):
        patched_vfield = viz.vector_field(venv, hook.network)
    img = viz.plot_vf_mpp(patched_vfield, ax=axd4['success_c'], save_img=False)

    padding = maze.get_padding(maze.get_inner_grid_from_seed(seed))
    viz.plot_pixel_dot(axd4['success_c'], success_c_pos[0], success_c_pos[1], hidden_padding=padding)
    axd4['success_c'].imshow(img)

    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer,channel=55, value=5, coord=fail)
    with hook.use_patches(patches):
        patched_vfield = viz.vector_field(venv, hook.network)
    img = viz.plot_vf_mpp(patched_vfield, ax=axd4['fail'], save_img=False)

    padding = maze.get_padding(maze.get_inner_grid_from_seed(seed))
    viz.plot_pixel_dot(axd4['fail'], fail[0], fail[1], hidden_padding=padding)
    axd4['fail'].imshow(img)

    #plt.show()
    plt.savefig('playground/paper_graphics/visualizations/fig_4.svg', bbox_inches="tight", format='svg')
#fig_4()