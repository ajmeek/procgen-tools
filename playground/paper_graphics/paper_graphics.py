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

# ------------------------------------------------ util functions------------------------------------------------

def linear_mapping(coords):
    """
    This function takes in 16x16 coordinates and outputs 25x25 coordinates.
    This assumes that the far corners (0,0) and (15,15) map to (0,0) and (24, 24).

    Keep in mind that when displaying these new coordinates, care must be taken.
    For most of the seeds we'll be using (aka 13x13 inner grid), we need to make sure that the coordinates lie
    within that. So patching activations in the 16x16 grid that maps outside of this inner core is not a good idea.
    """

    # Define the linear transformation coefficients
    A = 25 / 16  # Scale factor in x-direction
    E = 25 / 16  # Scale factor in y-direction

    # Apply the linear transformation
    u = A * coords[0]
    v = E * coords[1]


    # Round the mapped coordinates to the nearest integers
    u_rounded = np.round(u).astype(int)
    v_rounded = np.round(v).astype(int)

    if not 6 <= u_rounded <= 18:
        print(f"u value of {u_rounded} and x value of {coords[0]} is not in the inner grid")
    if not 6 <= v_rounded <= 18:
        print(f"v value of {v_rounded} and y value of {coords[1]} is not in the inner grid")

    return u_rounded, v_rounded


# test the above
#print(linear_mapping((0,0)))
#print(linear_mapping((15,15)))
#print(linear_mapping((8,8)))
#print(linear_mapping((5,10)))

def inner_grid_coords_from_full_grid_coords(coords):
    """
    This function takes in coords for a 25x25 grid and outputs coords for a 13x13 grid.
    Will need modification if we move to different seeds.
    """
    if not 6 <= coords[0] <= 18:
        raise ValueError(f"x value of {coords[0]} is not in the inner grid")
    if not 6 <= coords[1] <= 18:
        raise ValueError(f"y value of {coords[1]} is not in the inner grid")
    return coords[0] - 6, coords[1] - 6

def return_seeds_with_inner_grid_size(size = (13,13)):
    """
    How many possible seeds are there? Let me check openAI docs to try and find this.

    didn't see it. Trial and error below
    alright, tried seeds up to 10 mil. All pass.
    So at what point do we stop? This would be important for calculating statistics.
    """


    seeds = []
    for seed in range(10000):
        venv = create_venv(1,seed,1)
        inner_grid = maze.get_inner_grid_from_seed(seed)
        if inner_grid.shape == size:
            seeds.append(seed)
        if len(seeds) >= 1000:
            return seeds
    return seeds

def mass_display(seeds):
    """
    Displays 16 seeds at a time from the list of seeds.
    """
    fig, axd = plt.subplot_mosaic(
        [
            ['1', '2', '3', '4'],
            ['5', '6', '7', '8'],
            ['9', '10', '11', '12'],
            ['13', '14', '15', '16'],
        ],
        figsize=(AX_SIZE * 4, AX_SIZE * 3),
        tight_layout=True,
    )

    #now grab 16 seeds at a time in a for loop
    for i in range(0, 1000, 16):
        for j in range(16):
            seed = seeds[i+j]
            venv = create_venv(1,seed,1)
            state = maze.EnvState(venv.env.callmethod('get_state')[0])
            img = viz.visualize_venv(venv, ax=axd[str(j+1)], render_padding=False)

            axd[str(j+1)].imshow(img)
            axd[str(j+1)].set_xticks([])
            axd[str(j+1)].set_yticks([])
            axd.set_title(f'Seed {seed}', fontsize=18)

        plt.show()



# ---------------------------------------------------- fig 1 ----------------------------------------------------

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

    axd2['cheese_a'].set_xticks([])
    axd2['cheese_a'].set_yticks([])

    maze.move_cheese_in_state(state, cheese_b_pos)
    venv.env.callmethod('set_state', [state.state_bytes])
    obs = t.tensor(venv.reset(), dtype=t.float32)

    with hook.set_hook_should_get_custom_data():
        hook.network(obs)

    #img = viz.visualize_venv(venv, render_padding=False, show_plot=True) #checking cheese coords, it is top right
    activ = hook.get_value_by_label(default_layer)[0][cheese_channel]
    axd2['cheese_b'].imshow(activ)
    axd2['cheese_b'].imshow(activ, cmap='RdBu', vmin=-1, vmax=1)

    axd2['cheese_b'].set_xticks([])
    axd2['cheese_b'].set_yticks([])

    maze.move_cheese_in_state(state, cheese_c_pos)
    venv.env.callmethod('set_state', [state.state_bytes])
    obs = t.tensor(venv.reset(), dtype=t.float32)

    with hook.set_hook_should_get_custom_data():
        hook.network(obs)

    #img = viz.visualize_venv(venv, render_padding=False, show_plot=True) #checking cheese coords, it is bottom right
    activ = hook.get_value_by_label(default_layer)[0][cheese_channel]
    axd2['cheese_c'].imshow(activ)
    axd2['cheese_c'].imshow(activ, cmap='RdBu', vmin=-1, vmax=1)

    axd2['cheese_c'].set_xticks([])
    axd2['cheese_c'].set_yticks([])


    #plt.show()
    plt.savefig('playground/paper_graphics/visualizations/fig_2.svg', bbox_inches="tight", format='svg')

#fig_2()

# ---------------------------------------------------- fig 3 ----------------------------------------------------

# want to ask Mrinank what he means by 'ghost cheese' really, and how they may substantially differ from fig 4.
# computing it will be quite easy though

# ---------------------------------------------------- fig 4 ----------------------------------------------------
def fig_4():
    #patching works on a 16x16 grid. Putting the dot near there on a 25x25 grid requires some massaging.

    # Mrinank wants this done with a linear mapping from 16x16 to 25x25.

    #These are done with the 16x16 grid. Check the resulting 25x25 coordinates afterwards too.
    success_a_pos = (5, 5) # top left
    success_b_pos = (8, 4) # bottom right
    success_c_pos = (5, 11) # bottom left
    fail = (10, 4) #

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

    u, v = linear_mapping(success_a_pos)
    u, v = inner_grid_coords_from_full_grid_coords((u,v))
    padding = maze.get_padding(maze.get_inner_grid_from_seed(seed))
    viz.plot_pixel_dot(axd4['success_a'], u, v, hidden_padding=padding)
    #viz.plot_red_dot(venv, axd4['success_a'], success_a_pos[0], success_a_pos[1])
    axd4['success_a'].imshow(img)

    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer,channel=55, value=5, coord=success_b_pos)
    with hook.use_patches(patches):
        patched_vfield = viz.vector_field(venv, hook.network)
    img = viz.plot_vf_mpp(patched_vfield, ax=axd4['success_b'], save_img=False)

    u, v = linear_mapping(success_b_pos)
    u, v = inner_grid_coords_from_full_grid_coords((u,v))
    padding = maze.get_padding(maze.get_inner_grid_from_seed(seed))
    viz.plot_pixel_dot(axd4['success_b'], u, v, hidden_padding=padding)
    axd4['success_b'].imshow(img)

    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer,channel=55, value=5, coord=success_c_pos)
    with hook.use_patches(patches):
        patched_vfield = viz.vector_field(venv, hook.network)
    img = viz.plot_vf_mpp(patched_vfield, ax=axd4['success_c'], save_img=False)

    u, v = linear_mapping(success_c_pos)
    u, v = inner_grid_coords_from_full_grid_coords((u,v))
    padding = maze.get_padding(maze.get_inner_grid_from_seed(seed))
    viz.plot_pixel_dot(axd4['success_c'], u, v, hidden_padding=padding)
    axd4['success_c'].imshow(img)

    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer,channel=55, value=5, coord=fail)
    with hook.use_patches(patches):
        patched_vfield = viz.vector_field(venv, hook.network)
    img = viz.plot_vf_mpp(patched_vfield, ax=axd4['fail'], save_img=False)

    u, v = linear_mapping(fail)
    u, v = inner_grid_coords_from_full_grid_coords((u,v))
    padding = maze.get_padding(maze.get_inner_grid_from_seed(seed))
    viz.plot_pixel_dot(axd4['fail'], u, v, hidden_padding=padding)
    axd4['fail'].imshow(img)

    #plt.show()
    plt.savefig('playground/paper_graphics/visualizations/fig_4.svg', bbox_inches="tight", format='svg')
#fig_4()


# ---------------------------------------------------- fig 5 ----------------------------------------------------
# This figure is from Mrinank's slack request. He wants a 1x4 example of 2 times where the policy routes to the goal,
# and 2 times where the policy routes to the historic cheese location. Flick through different seeds to find these.
# What util functions would be helpful here?
# I want a function to flick through different seeds and display many of them at a time.
# I want a function that gives me all seeds of a certain size inner grid.

# # # find largest seed starting at seed = 100k with a step size of 1k
# # for seed in range(100000, 10000000, 1000):
# #     try:
# #         venv = create_venv(1,seed,1)
# #         # inner_grid = maze.get_inner_grid_from_seed(seed)
# #         # if inner_grid.shape == (13,13):
# #         #     print(seed)
# #         #     break
# #         print("seed passed! : ", seed)
# #     except:
# #         print("seed broke! : ", seed)
# # # seed = 100000
# # # venv = create_venv(1,seed,1)
#
# # seeds with inner grid 13x13
# # seeds = return_seeds_with_inner_grid_size()
# #
# # #save seeds to a file
# # with open('playground/paper_graphics/visualizations/seeds_13x13.txt', 'w') as f:
# #     for item in seeds:
# #         f.write("%s\n" % item)
#
#
# #mass_display(seeds)
#
# fig, axd = plt.subplot_mosaic(
#     [
#         ['1', '2', '3', '4'],
#         ['5', '6', '7', '8'],
#         ['9', '10', '11', '12'],
#         ['13', '14', '15', '16'],
#     ],
#     figsize=(AX_SIZE * 4, AX_SIZE * 4),
#     tight_layout=True,
# )
# # venv = create_venv(1, 166, 1)
# # state = maze.EnvState(venv.env.callmethod('get_state')[0])
# # img = viz.visualize_venv(venv, render_padding=False)
#
# seeds = [344, 346, 385, 389, 392, 414, 433, 435, 449, 455, 470, 516, 517, 543, 555, 559]
# for i, seed in enumerate(seeds):
#     venv = create_venv(1,seed,1)
#     state = maze.EnvState(venv.env.callmethod('get_state')[0])
#
#     vf = viz.vector_field(venv, policy)
#     print(str(i))
#     print(f"{i}")
#     img = viz.plot_vf_mpp(vf, ax=axd[str(i+1)], save_img=False)
#
#     axd[str(i+1)].imshow(img)
#     axd[str(i+1)].set_xticks([])
#     axd[str(i+1)].set_yticks([])
#     axd[str(i+1)].set_title(f'Seed {seed}', fontsize=18)
#
# #plt.show()
# plt.savefig('playground/paper_graphics/visualizations/13x13_group_3.svg', bbox_inches="tight", format='svg')


# alright, after all the above flicking through a ton of seeds, I found two good ones in each category.
# one - mouse fails to go to cheese and instead goes to historic goal location - seeds 2 and 543
# two - mouse properly generalizes and goes to the cheese - seeds 128 and 167

def fig_5():
    """

    """

    success_a = 2
    success_b = 543
    fail_a = 128
    fail_b = 167

    fig5, axd5 = plt.subplot_mosaic(
        [['success_a', 'success_b', 'fail_a', 'fail_b']],
        figsize=(AX_SIZE * 4, AX_SIZE),
        tight_layout=True,
    )

    venv = create_venv(1,success_a,1)
    vf = viz.vector_field(venv, policy)

    img = viz.plot_vf_mpp(vf, ax=axd5['success_a'], save_img=False)
    axd5['success_a'].imshow(img)


    venv = create_venv(1,success_b,1)
    vf = viz.vector_field(venv, policy)

    img = viz.plot_vf_mpp(vf, ax=axd5['success_b'], save_img=False)
    axd5['success_b'].imshow(img)


    venv = create_venv(1,fail_a,1)
    vf = viz.vector_field(venv, policy)

    img = viz.plot_vf_mpp(vf, ax=axd5['fail_a'], save_img=False)
    axd5['fail_a'].imshow(img)


    venv = create_venv(1,fail_b,1)
    vf = viz.vector_field(venv, policy)

    img = viz.plot_vf_mpp(vf, ax=axd5['fail_b'], save_img=False)
    axd5['fail_b'].imshow(img)

    plt.savefig('playground/paper_graphics/visualizations/fig_5.svg', bbox_inches="tight", format='svg')

fig_5()