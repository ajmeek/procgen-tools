import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib as mpl
from colorspacious import cspace_converter
import random
import procgen_tools.vfield_stats as vfield_stats
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

def combined_px_patch(layer_name : str, resampling_seed: int, channels : List[int], cheese_loc : Tuple[int, int] = None):
    """
    Get a combined patch which randomly replaces channel activations with other activations from different levels.

    Instead change to not be random. Use modified version of patch_utils.get_random_patch.
    """
    # patches = [patch_utils.get_specific_patch(layer_name=layer_name, hook=hook, channel=channel, cheese_loc=cheese_loc, resampling_seed=resampling_seed) for channel in channels]
    patches = [patch_utils.get_random_patch(layer_name=layer_name, hook=hook, channel=channel, cheese_loc=cheese_loc) for channel in channels]
    print(patches)
    combined_patch = patch_utils.compose_patches(*patches)
    return combined_patch # NOTE we're resampling from a fixed maze for all target forward passes

def resample_activations(original_seed, channels, resampling_seed):
    """
    This function is to resample the activations from one seed to another.

    original_seed is the seed whose activations we want to change.
    channels is a list of channels from the new maze that we wish to combine and overwrite channel 55 with.
    resampling_seed is the seed whose activations we want to sample from.

    Returns an image of the new vector field.
    """

    venv_orig = patch_utils.get_cheese_venv_pair(seed=original_seed)
    cheese_loc = maze.get_cheese_pos_from_seed(resampling_seed, flip_y=False)
    patches = combined_px_patch(layer_name=default_layer, resampling_seed=resampling_seed, channels=channels, cheese_loc=cheese_loc)

    with hook.use_patches(patches):
        patched_vfield = viz.vector_field(venv_orig, hook.network)
    img = viz.plot_vf_mpp(patched_vfield, save_img=False)

    patch_utils.compare_patched_vfields_mpp(venv_orig, patches, hook, default_layer)#, show_plot=True, save_img=False)
    pass

# sanity check
# list of cheese channels

# WAITING FOR ULI

# cheese_channels = [7, 8, 42, 44, 55, 77, 82, 88, 89, 99, 113]
# resample_activations(0, cheese_channels, 435) # - no errors! at least neutral sign :)


def plot_heatmap(seed, prob_type, ax):
    cheese_channels = [7, 8, 42, 44, 55, 77, 82, 88, 89, 99, 113]
    effective_channels = [8, 55, 77, 82, 88, 89, 113]

    dfs = []
    DATA_DIR = "experiments/statistics/data/retargeting"
    # Find every CSV file
    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(DATA_DIR, file))
            dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)


    # creating heatmap part a
    prob_type = prob_type
    venv = maze.create_venv(num=1, start_level=seed, num_levels=1)
    seed_data: pd.DataFrame = data[data["seed"] == seed]
    label = prob_type
    if prob_type == "all":
        label = str(cheese_channels)
    elif prob_type == "effective":
        label = str(effective_channels)
    elif prob_type == "55":
        label = "[55]"

    prob_data = seed_data[seed_data["intervention"] == label]
    # Check if removed_cheese is true for prob_data's first row
    if prob_data["removed_cheese"].iloc[0]:
        venv = maze.remove_all_cheese(venv)

    heatmap = prob_data.pivot(index="row", columns="col", values="probability")

    viz.show_grid_heatmap(
        venv=venv,
        heatmap=heatmap.values,
        ax = ax,
        ax_size=AX_SIZE,
        mode="human",
        size=0.5,
        alpha=0.9,
    )

def heatmap_data_by_seed_and_prob_type(seed, prob_type):
    cheese_channels = [7, 8, 42, 44, 55, 77, 82, 88, 89, 99, 113]
    effective_channels = [8, 55, 77, 82, 88, 89, 113]

    dfs = []
    DATA_DIR = "experiments/statistics/data/retargeting"
    # Find every CSV file
    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(DATA_DIR, file))
            dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    prob_type = prob_type
    seed_data: pd.DataFrame = data[data["seed"] == seed]
    label = prob_type
    if prob_type == "all":
        label = str(cheese_channels)
    elif prob_type == "effective":
        label = str(effective_channels)
    elif prob_type == "55":
        label = "[55]"

    prob_data = seed_data[seed_data["intervention"] == label]

    return prob_data



def plot_colorbar():
    gradient = np.linspace(1, 0, 256)
    gradient = np.vstack((gradient, gradient))
    # Create figure and adjust figure height to number of colormaps
    nrows = 2

    figh = 0.25 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows - 1, figsize=(6.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.25 / figh,
                        left=0.2, right=0.8)
    axs.set_title(f'Activations', fontsize=14)

    # for ax, name in zip(axs, cmap_list):
    axs.imshow(gradient, aspect='auto', cmap=mpl.colormaps['coolwarm'])

    axs.set_xticks([])
    axs.set_yticks([])
    plt.xticks([0, 128, 256])
    axs.set_xticklabels(['-1', '0', '1'])

    # Save colormap list for later.
    #cmaps[category] = cmap_list
    plt.savefig('playground/paper_graphics/visualizations/colorbar.pdf', bbox_inches="tight", format='pdf')


#plot_colorbar()

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
        figsize=(AX_SIZE * 4, AX_SIZE*1.5),
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
    patch_img = axd1['patch_act'].imshow(activ)
    axd1['patch_act'].imshow(activ, cmap='RdBu', vmin=-1, vmax=1)

    axd1['patch_act'].set_xticks([])
    axd1['patch_act'].set_yticks([])


    # NOTE just make the colorbar look good and then I can adjust it in Illustrator later

    #plt.colorbar()
    # cbar_orig_act = plt.colorbar(axd1['orig_act'], ax=axd1['orig_act'])
    # cbar_orig_act.set_label('Label for Original Activation')

    # Add a colorbar for the 'patch_act' image
    # cbar_patch_act = plt.colorbar(patch_img, ax=axd1['patch_act'])
    # cbar_patch_act.set_label('Label for Patched Activation')

    # Create a custom colormap from red to dark blue
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        'custom_cmap', [(1, 0, 0), (0, 0, 0.5)]
    )

    # Create a new subplot for the colorbar in the center
    cbar_ax = fig1.add_axes([0.2, 0.1, 0.4, 0.01])  # Adjust the [left, bottom, width, height] values as needed

    # Add a colorbar for the entire mosaic plot
    cbar = plt.colorbar(patch_img, cax=cbar_ax, orientation='horizontal', cmap=custom_cmap)
    cbar.set_label('Label for Activation')

    # Set the tick locations and labels on the colorbar
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['-1', '0', '1'])

    #matplotlib.pyplot.p

    #plt.show()
    #print(os.getcwd())
    plt.savefig('playground/paper_graphics/visualizations/fig_1.pdf', bbox_inches="tight", format='pdf')

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

# just found and retrieved the actual cheese graphic from OpenAI repo. Now can make it blue / red in photoshop as
# Mrinank suggested for this figure.

# keep track of all seeds that we are resampling from and into. This'll be important information for reproducibility.

# plan for this figure:
# part a should be an agent moving to the cheese. Not the same seed as fig 1 since the MPP there is away from the cheese.
# part b should be from activations sampled from another 13x13 seed with the cheese at the same location.
# part c should be from activations sampled from another 13x13 seed with the cheese at a different location, with successful retargeting
# part d should be from activations sampled from another 13x13 seed with the cheese at a different location, with succesful
#   retargeting in a different location, towards the bottom right.

# I want to keep these all 13x13 images as well. That way we can show in an appendix where the cheese was coming from.
# This may not be a major point, since we'll want to give examples of the different views, but for now I'll do it like so.

# Glad I saved those mass displays of the 13x13 seeds! Found some good ones.
# part a - seed 433
# part b - find seed with cheese same location as 433
# part c - seed 435
# part d - seed 516

# To resample, I just need to replace the activations from one with another. easy peasy.

# venv = create_venv(1, 433, 1)
# state = maze.EnvState(venv.env.callmethod('get_state')[0])
# inner_grid = state.inner_grid()
# print(maze.get_cheese_pos(inner_grid)) # this is (4,5). zero indexed from standard origin, y column on the left for some reason.

# so, I want to find a seed with the cheese at (4,5). I'll just flick through the seeds I have and find one.
# for seed in range(10000):
#     venv = create_venv(1,seed,1)
#     state = maze.EnvState(venv.env.callmethod('get_state')[0])
#     inner_grid = state.inner_grid()
#     if maze.get_cheese_pos(inner_grid) == (4,5) and inner_grid[0].size > 10:
#         print(seed, inner_grid, type(inner_grid))
#         viz.visualize_venv(venv, show_plot=True, render_padding=False)
#         break

# okay, good seed found at 142

"""
Thoughts on how to get a proper resampling. Check out their resampling code in reverse_c55.py.
How they're currently doing it is to take activations from all the cheese samples and combine them together, then
replace the activations of the original maze.
So for ex, take all the cheese channels of some alternate maze, combine them, then overwrite channel 55 with those values.

Their code there should be fine, but the random resample from patch_utils needs to be modified so that it only
samples from the specific seed I want. I could alternatively ask for any seed with the cheese at some loc, but I think
it's better for reproducibility if I just specifiy a seed though.

So simply replacing their maze func does not work. Try using it but specifying a specific cheese location, then
collecting the seed from their metadata.
"""

def fig_3():
    fig3, axd3 = plt.subplot_mosaic(
        [['original', 'same_loc', 'dif_loc_historic', 'dif_loc_bottom_right']],
        figsize=(AX_SIZE * 4, AX_SIZE),
        tight_layout=True,
    )

    # part a

    venv_a = create_venv(1,433,1)
    vf = viz.vector_field(venv_a, policy)
    img = viz.plot_vf_mpp(vf, ax=axd3['original'], save_img=False)
    axd3['original'].imshow(img)

    original_activ = hook.get_value_by_label(default_layer)[0][cheese_channel]

    # part b

    # part c

    # venv_c = create_venv(1,516,1)
    # obs = t.tensor(venv_c.reset(), dtype=t.float32)
    #
    # with hook.set_hook_should_get_custom_data():
    #     hook.network(obs)
    # c_activ = hook.get_value_by_label(default_layer)[0][cheese_channel]
    #
    # np.save("playground/paper_graphics/visualizations/fig_3_516_activ.npy", c_activ)
    c_activ = np.load("playground/paper_graphics/visualizations/fig_3_516_activ.npy")

    # patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer,channel=55, value=5, coord=success_a_pos)
    # with hook.use_patches(patches):
    #     patched_vfield = viz.vector_field(venv, hook.network)
    # img = viz.plot_vf_mpp(patched_vfield, ax=axd4['success_a'], save_img=False)

    # obs = t.tensor(venv_a.reset(), dtype=t.float32)
    #
    # with hook.set_hook_should_get_custom_data():
    #     hook.network(obs)
    patches = patch_utils.get_channel_whole_patch_replace(layer_name=default_layer,channel=55, activations=c_activ)
    with hook.use_patches(patches):
        patched_vfield = viz.vector_field(venv_a, hook.network)
    img = viz.plot_vf_mpp(patched_vfield, ax=axd3['dif_loc_historic'], save_img=False)
    axd3['dif_loc_historic'].imshow(img)

    plt.show()
    print()

#fig_3()


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

#fig_5()

# ---------------------------------------------------- fig X1 ----------------------------------------------------
# going through the recording of Mrinank's request one at a time.
# "Some locations are easier to steer to than others"
# Mrinank wants a heatmap here. Alex just gave heatmap analyses. Use those.
# merging paper into paper_graphics
"""
The idea for this figure is that not everything is going to be easily retargetable. 
So we want to show a heatmap of retargetability. I'll do two different seeds showing the easiness of retargeting
from just channel 55 versus all cheese channels. 

So I'm not sure exactly how I'll set this up. Let me do two different configurations.
config 1 : 0/55, 0/all, 48/55, 48/all
config 2: 48/cheese, 48/55, 48/all, 48/none
"""

def fig_x1a():


    figx1, axdx1 = plt.subplot_mosaic(
        [['seed_0_channel_55', 'seed_0_channel_all', 'seed_48_channel_55', 'seed_48_channel_all']],
        figsize=(AX_SIZE * 4, AX_SIZE*1.5), #increase y to fit titles
        tight_layout=True,
    )

    # loading the data

    #give title to the axes
    axdx1['seed_0_channel_55'].set_title("Seed 0, Channel 55", fontsize=14)#, font="Times New Roman")
    plot_heatmap(0, "55", axdx1['seed_0_channel_55'])

    axdx1['seed_0_channel_all'].set_title("Seed 0, All Cheese Channels", fontsize=14)#, font="Times New Roman")
    plot_heatmap(0, "all", axdx1['seed_0_channel_all'])

    axdx1['seed_48_channel_55'].set_title("Seed 48, Channel 55", fontsize=14)#, font="Times New Roman")
    plot_heatmap(48, "55", axdx1['seed_48_channel_55'])

    axdx1['seed_48_channel_all'].set_title("Seed 48, All Cheese Channels", fontsize=14)#, font="Times New Roman")
    plot_heatmap(48, "all", axdx1['seed_48_channel_all'])
    #plt.show()
    plt.savefig('playground/paper_graphics/visualizations/fig_x1a.svg', bbox_inches="tight", format='svg')


#fig_x1a()

def fig_x1b():


    figx1, axdx1 = plt.subplot_mosaic(
        [['seed_48_channel_cheese', 'seed_48_channel_55', 'seed_48_channel_all', 'seed_48_channel_none']],
        figsize=(AX_SIZE * 4, AX_SIZE*1.5), #increase y to fit titles
        tight_layout=True,
    )

    # loading the data

    #give title to the axes
    axdx1['seed_48_channel_cheese'].set_title("Seed 48, Cheese", fontsize=14)#, font="Times New Roman")
    plot_heatmap(48, "cheese", axdx1['seed_48_channel_cheese'])

    axdx1['seed_48_channel_55'].set_title("Seed 0, Channel 55", fontsize=14)#, font="Times New Roman")
    plot_heatmap(48, "55", axdx1['seed_48_channel_55'])

    axdx1['seed_48_channel_all'].set_title("Seed 48, Channel All", fontsize=14)#, font="Times New Roman")
    plot_heatmap(48, "all", axdx1['seed_48_channel_all'])

    axdx1['seed_48_channel_none'].set_title("Seed 48, Base", fontsize=14)#, font="Times New Roman")
    plot_heatmap(48, "normal", axdx1['seed_48_channel_none'])

    #plt.show()
    plt.savefig('playground/paper_graphics/visualizations/fig_x1b.svg', bbox_inches="tight", format='svg')


#fig_x1b()


# ---------------------------------------------------- fig x2-4 ----------------------------------------------------
"""
These three figures will require me to do some analyses as well. 
To quantify success, I'll just average over all the heatmaps. For the different variables, I need to loop over different
seeds and then quantify each. 

For x2, I can do this as by just going through different seeds. I'll want X seeds from each size, etc.
    NOTE - x2 calculated using all cheese channels
For x3, this may require me to ask Alex. How did he generate the heatmap data? What value did he put in for that?
    We want to quantify how the activation value differs. How can I rerun this so that I can try diff ones?
For x4, this is moderate. How can I tackle adding more channels? I can do this as a line plot, plotting versus
    just 55, then effective channels, then all channels. But would Mrinank & Alex want more granularity here?
    Perhaps we should have all of the cheese channels ranked in effectiveness, then add them one by one.
"""

def fig_x2_4():
    #first, get the data for x2.
    # actually he's only got data for 100 diff seeds there. I need to bin them appropriately.

    # interesting ... only odd sizes are generated. didn't notice that before.
    # so that makes the bins easier. I'll just do 3x3, 5x5, 7x7, 9x9, 11x11, 13x13, 15x15, 17x17, 19x19, 21x21, 23x23, 25x25
    # which is 12 total. that's not going to be too cluttered on the graph.

    # intervention_methods = ["cheese", "effective", "all", "normal", "55"]
    # seeds = {f'{i}x{i}': [] for i in range(3, 26, 2)}
    # data_by_intervention = {i: {k: {f'{j}x{j}': None for j in range(3, 26)} for k in ['probability', 'ratio']} for i in intervention_methods}
    # heatmap_avg_per_size_all = {f'{i}x{i}': None for i in range(3, 26, 2)}
    # heatmap_avg_per_size_effective = {f'{i}x{i}': None for i in range(3, 26, 2)}
    # heatmap_avg_per_size_55 = {f'{i}x{i}': None for i in range(3, 26, 2)}
    # ratio_avg_per_size_all = {f'{i}x{i}': None for i in range(3, 26, 2)}
    # ratio_avg_per_size_effective = {f'{i}x{i}': None for i in range(3, 26, 2)}
    # ratio_avg_per_size_55 = {f'{i}x{i}': None for i in range(3, 26, 2)}
    #
    # #print(heatmap_avg_per_size)
    # for i in range(100):
    #     seed = i
    #     venv = create_venv(1,seed,1)
    #     state = maze.EnvState(venv.env.callmethod('get_state')[0])
    #     inner_grid = state.inner_grid()
    #     size = inner_grid.shape[0]
    #     seeds[f'{size}x{size}'].append(seed)
    #
    # for key, value in seeds.items():
    #     #get data for that seed and all cheese channels
    #
    #     if len(value) != 0:
    #         per_key_avg_all = 0
    #         per_key_ratio_all = 0
    #
    #         per_key_avg_effective = 0
    #         per_key_ratio_effective = 0
    #
    #         per_key_avg_55 = 0
    #         per_key_ratio_55 = 0
    #
    #         for i in value:
    #             # for j in intervention_methods:
    #             data = heatmap_data_by_seed_and_prob_type(i, "all")
    #             # sum = data['probability'].sum()
    #             # sum /= data['probability'].size
    #             per_key_avg_all += data['probability'].mean()
    #
    #             # sum = data['ratio'].sum()
    #             # sum /= data['ratio'].size
    #             per_key_ratio_all += data['ratio'].mean()
    #
    #             data = heatmap_data_by_seed_and_prob_type(i, "effective")
    #             per_key_avg_effective += data['probability'].mean()
    #             per_key_ratio_effective += data['ratio'].mean()
    #
    #             data = heatmap_data_by_seed_and_prob_type(i, "55")
    #             per_key_avg_55 += data['probability'].mean()
    #             per_key_ratio_55 += data['ratio'].mean()
    #
    #         #         heatmap_avg_per_size[j][key] = per_key_avg# / len(value)
    #         # for j in intervention_methods:
    #         #     for key in heatmap_avg_per_size[j].keys():
    #         heatmap_avg_per_size_all[key] = per_key_avg_all / len(value)
    #         ratio_avg_per_size_all[key] = per_key_ratio_all / len(value)
    #
    #         heatmap_avg_per_size_effective[key] = per_key_avg_effective / len(value)
    #         ratio_avg_per_size_effective[key] = per_key_ratio_effective / len(value)
    #
    #         heatmap_avg_per_size_55[key] = per_key_avg_55 / len(value)
    #         ratio_avg_per_size_55[key] = per_key_ratio_55 / len(value)
    #
    # # after this divide heatmap_avg_per_size by len of # of seeds in "seeds" to get the average
    #
    # print(heatmap_avg_per_size_all)
    # print(ratio_avg_per_size_all)
    # print(heatmap_avg_per_size_effective)
    # print(ratio_avg_per_size_effective)
    # print(heatmap_avg_per_size_55)
    # print(ratio_avg_per_size_55)

    # data hardcoded. starting with 3x3 up to 25x25.
    heatmap_avg_per_size_all = [0.799889956501738, 0.7775709501219863, 0.7797912188517916, 0.7588451193811928, 0.7406733268789705, 0.6959504505843259, 0.6425305479030984, 0.6545479388988915, 0.5824753527436064, 0.5832421962896295, 0.6167109448569562, 0.5299525163283456]
    heatmap_avg_per_size_effective = [0.7608124070512822, 0.7041304093124999, 0.76901413325, 0.7024797808854166, 0.7087437639591837, 0.7041753128385418, 0.6503412165784831, 0.6596821581796875, 0.6037907865088384, 0.6189248616309523, 0.6371125867599068, 0.5756655757924108]
    heatmap_avg_per_size_55 = [0.700428774165744, 0.7370594442260178, 0.7010945262579837, 0.6721659938343914, 0.6751070216559201, 0.647880930832588, 0.5939934102651221, 0.6109084206556188, 0.540930443222974, 0.5470496808198526, 0.5846496184297417, 0.5023088997834109]
    ratio_avg_per_size_effective = [1.8489007321794872, 3.4623849059999996, 1.7517762465, 2.7921403650173606, 1.8093523609387756, 2.8955814034765623, 2.458575891675485, 2.8985679961171877, 2.566675839248737, 2.5696256181845234, 3.013872167534965, 1.8687405686830356]

    x_values = np.linspace(0, 1, len(heatmap_avg_per_size_all))
    #x_labels = ['3x3', '5x5', '7x7', '9x9', '11x11', '13x13', '15x15', '17x17', '19x19', '21x21', '23x23', '25x25']
    x_labels = ['3', '5', '7', '9', '11', '13', '15', '17', '19', '21', '23', '25']


    #initial box plots
    fig, ax = plt.subplots()#1, 2, figsize=(AX_SIZE * 2, AX_SIZE), tight_layout=True)
    #ax.boxplot([heatmap_avg_per_size_all.values(), heatmap_avg_per_size_effective.values(), heatmap_avg_per_size_55.values()], positions = [1, 2, 3], widths = 0.6)
    ax.plot(x_values, heatmap_avg_per_size_all, marker='o', markersize=6, color="black")
    ax.plot(x_values, heatmap_avg_per_size_effective, marker='x', markersize=6, color="red")
    ax.plot(x_values, heatmap_avg_per_size_55, marker='+', markersize=6, color="green")

    ax.set_xticks(x_values)
    ax.set_xticklabels(x_labels)

    plt.savefig('playground/paper_graphics/visualizations/heat_map_avg_bad_idea.png', bbox_inches="tight", format='png')

    fig, ax = plt.subplots()
    ax.plot(x_values, ratio_avg_per_size_effective, marker='x', markersize=6, color="red")
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_labels)
    plt.savefig('playground/paper_graphics/visualizations/ratio_avg_also_not_great.png', bbox_inches="tight", format='png')



    #ax.set_xticklabels(['All Cheese Channels', 'Effective Channels', 'Channel 55'])

    # gathering more data for x3 will take a while. going to start on the cheese vector figure for now


    # gathering data for x4

    # I want data for mazes of all sizes and all magnitudes, split by different channels.
    # intervention_methods = ["cheese", "effective", "all", "normal", "55"]
    # seeds = {f'{i}x{i}': [] for i in range(3, 26, 2)}
    #
    # avg_by_55 = 0.0 #number, count
    # avg_by_effective = 0.0
    # avg_by_all = 0.0
    # for i in range(100):
    #     # for j in intervention_methods:
    #     data = heatmap_data_by_seed_and_prob_type(i, "55")
    #     avg_by_55 += data['probability'].mean()
    #
    #     data = heatmap_data_by_seed_and_prob_type(i, "effective")
    #     avg_by_effective += data['probability'].mean()
    #
    #     data = heatmap_data_by_seed_and_prob_type(i, "all")
    #     # sum = data['probability'].sum()
    #     # sum /= data['probability'].size
    #     avg_by_all += data['probability'].mean()
    #
    # #because 100 seeds total, then divide
    # avg_by_55 /= 100
    # avg_by_effective /= 100
    # avg_by_all /= 100
    #
    # print(avg_by_55, avg_by_effective, avg_by_all)
    #plt.show()

#fig_x2_4()

# ---------------------------------------------------- fig cheese_vector ----------------------------------------------------

"""
Idea for this and the top right figure is to have these two side by side, each taking up half a page.
I want the barplots comparing the blue original v. the orange patched side. Find the code for making those in vfield_stats.py

So to run the original barplot, run a jupyter notebook in cheese_vector and then use vfield_stats.
However, that's not the one that's giving probabilities conditional on the decision square....
Read it very carefully. It may be what we want. 
"""

def cheese_vector_fig():
    # Generate some sample data for the box plots
    data1_var1 = np.random.normal(0, 1, 100)
    data1_var2 = np.random.normal(2, 1, 100)

    data2_var1 = np.random.normal(1, 1, 100)
    data2_var2 = np.random.normal(3, 1, 100)

    data3_var1 = np.random.normal(2, 1, 100)
    data3_var2 = np.random.normal(4, 1, 100)

    data4_var1 = np.random.normal(3, 1, 100)
    data4_var2 = np.random.normal(5, 1, 100)

    import matplotlib.font_manager
    #
    # font_list = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    #
    # for font_path in font_list:
    #     font_properties = matplotlib.font_manager.FontProperties(fname=font_path)
    #     print(f'Font: {font_properties.get_name()}, File: {font_path}')

    #plt.rcParams["font.family"] = "Times New Roman" - need to install a package
    # Create a figure and axis for the mosaic plot
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))

    # getting actual data
    vfields = [
        pickle.load(open(f, "rb"))
        for f in glob("experiments/statistics/data/vfields/cheese/seed-*.pkl")
    ]
    probs_original_plus_1, probs_patched_plus_1 = vfield_stats.get_probs_original_and_patched(
        vfields, coeff=1.0
    )
    probs_original_minus_1, probs_patched_minus_1 = vfield_stats.get_probs_original_and_patched(
        vfields, coeff=-1.0
    )

    probs_original_plus_1, probs_patched_plus_1 = probs_original_plus_1[:, 0], probs_patched_plus_1[:, 0]
    probs_original_minus_1, probs_patched_minus_1 = probs_original_minus_1[:, 0], probs_patched_minus_1[:, 0]


    #alright, try to get one from the same seed. Let's do subtractio seed zero.
    vfields = [
        pickle.load(open(f, "rb"))
        for f in glob("experiments/statistics/data/vfields/cheese/seed-0*.pkl")
    ]
    probs_original_same, probs_patched_same = vfield_stats.get_probs_original_and_patched(
        vfields, coeff=-1.0
    )
    probs_original_same, probs_patched_same = probs_original_same[:, 0], probs_patched_same[:, 0]


    # now seeing which seeds of the first 100 are short ones - aka 3x3 5x5 or 7x7
    # seeds = {f'{i}x{i}': [] for i in range(3, 26, 2)}
    #
    # for i in range(100):
    #     seed = i
    #     venv = create_venv(1,seed,1)
    #     state = maze.EnvState(venv.env.callmethod('get_state')[0])
    #     inner_grid = state.inner_grid()
    #     size = inner_grid.shape[0]
    #     seeds[f'{size}x{size}'].append(seed)
    #
    # small_seeds = []
    # small_seeds.extend(seeds['3x3'])
    # small_seeds.extend(seeds['5x5'])
    # small_seeds.extend(seeds['7x7'])
    # print(small_seeds)
    small_seeds = [1, 10, 11, 15, 18, 21, 23, 29, 31, 41, 77, 89, 93, 3, 7, 19, 30, 38, 39, 44, 59, 62, 69, 26, 34, 54, 72]
    vfields = []
    for s in small_seeds:
        for f in glob(f"experiments/statistics/data/vfields/cheese/seed-{s}*.pkl"):
            vfields.append(pickle.load(open(f, "rb")))
    probs_small_seed_orig, probs_small_seed_patched = vfield_stats.get_probs_original_and_patched(
        vfields, coeff=-1.0
    )

    probs_small_seed_orig, probs_small_seed_patched = probs_small_seed_orig[:, 0], probs_small_seed_patched[:, 0]

    # # Plot each box plot
    # ax[0].boxplot([data1_var1, data1_var2], labels=['Var1', 'Var2'])
    # ax[0].set_title('Boxplot 1')
    #
    # ax[1].boxplot([data2_var1, data2_var2], labels=['Var1', 'Var2'])
    # ax[1].set_title('Boxplot 2')
    # ax[1].set_yticklabels([])
    #
    # ax[2].boxplot([data3_var1, data3_var2], labels=['Var1', 'Var2'])
    # ax[2].set_title('Boxplot 3')
    # ax[2].set_yticklabels([])
    #
    # ax[3].boxplot([data4_var1, data4_var2], labels=['Var1', 'Var2'], patch_artist=True)
    # ax[3].set_title('Boxplot 4')
    # ax[3].set_yticklabels([])

    # Plot each box plot with markers, mean, and percentile lines
    # for i, (data_var1, data_var2) in enumerate([(data1_var1, data1_var2), (data2_var1, data2_var2),
    #                                              (data3_var1, data3_var2), (data4_var1, data4_var2)]):
    #
    #     mean = np.mean(data_var1), np.mean(data_var2)
    #     percentiles = np.percentile(data_var1, [25, 75]), np.percentile(data_var2, [25, 75])
    #
    #     # Plot markers for mean
    #     ax[i].plot([1, 2], mean, 'kD', markersize=10, label='Mean')
    #
    #     # Plot lines for the 25th and 75th percentiles
    #     for j in range(2):
    #         ax[i].plot([1, 1], [mean[j], percentiles[j][0]], color='black', linestyle='-', linewidth=2)
    #         ax[i].plot([2, 2], [mean[j], percentiles[j][1]], color='black', linestyle='-', linewidth=2)
    #
    #     ax[i].set_xticks([1, 2])
    #     ax[i].set_xticklabels(['Var1', 'Var2'])
    #     ax[i].set_title(f'Boxplot {i+1}')

    # Plot each box plot with markers and different box colors
    for i, (data_var1, data_var2) in enumerate([(probs_original_plus_1, probs_patched_plus_1),
                                                (probs_original_minus_1, probs_patched_minus_1),
                                                (probs_original_same, probs_patched_same),
                                                (probs_small_seed_orig, probs_small_seed_patched)]):

        # mean = np.mean(data_var1), np.mean(data_var2)

        # Plot the boxes with different colors
        bp = ax[i].boxplot([data_var1, data_var2], labels=['Var1', 'Var2'], patch_artist=True)

        # Set colors for the boxes
        for count, patch in enumerate(bp['boxes']):
            #print(patch)
            if count == 0:
                patch.set_facecolor('lightblue')  # Left part of the box
            elif count == 1:
                patch.set_facecolor('orange')
            patch.set_edgecolor('black')

        # TODO - these markers aren't at the actual mean. Try changing the boxplot data struct directly
        # Plot markers for mean without lines
        # ax[i].plot([1, 2], mean, 'kD', markersize=10, label='Mean', linestyle='None')

        # if i == 1:
        #     ax[i].set_title(f'Boxplot {i+1}')
        # elif i == 2:
        #     ax[i].set_title(f'Boxplot {i+1}')
        # elif i == 3:
        #     ax[i].set_title(f'Boxplot {i+1}')
        # else i == 4:
        #     ax[i].set_title(f'Boxplot {i+1}')

    ax[0].set_title("Coeff = 1.0")
    ax[1].set_title("Coeff = -1.0")
    ax[2].set_title("Coeff = -1.0, Same Seed")
    ax[3].set_title("Coeff = -1.0, Small Seeds")

    # Hide x-tick labels and ticks for all boxplots except the first one
    for i in range(0, len(ax)):
        ax[i].set_xticklabels([])
        ax[i].tick_params(axis='x', length=0)

    # Add markers for each variable
    # for i in range(4):
    #     ax[i].plot([1, 2], [np.mean([data1_var1, data2_var1, data3_var1, data4_var1][i]),
    #                          np.mean([data1_var2, data2_var2, data3_var2, data4_var2][i])],
    #                 'kD', markersize=10, label='Mean')

    # ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2,
    #               borderaxespad=0., title='Variables', prop={'size': 10})
    # ax[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, title='Variables', fontsize=10)
    # Create a legend for the markers
    # legend_elements = [plt.Line2D([0], [0], marker='D', color='w', label='Mean',
    #                               markersize=10, markerfacecolor='black')]

    orig_patch = mpatches.Patch(color='lightblue', label='The original probabilities')
    patch_patch = mpatches.Patch(color='orange', label='The patched probabilities')

    # Display the legend below the subplots
    fig.legend(handles=[orig_patch, patch_patch], loc='lower center', ncol=2, fontsize=10)



    # Display the plot
    #plt.show()
    plt.savefig('playground/paper_graphics/visualizations/cheese_vector.pdf', bbox_inches="tight", format='pdf')


#cheese_vector_fig()

# ---------------------------------------------------- fig top right vector figure ----------------------------------------------------

"""
Basically the same as for the cheese vector figure.
"""

def top_right_vector_figure():
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))

    # getting actual data
    vfields = [
        pickle.load(open(f, "rb"))
        for f in glob("experiments/statistics/data/vfields/top_right/seed-*.pkl")
    ]
    probs_original_plus_1, probs_patched_plus_1 = vfield_stats.get_probs_original_and_patched(
        vfields, coeff=1.0
    )
    probs_original_minus_1, probs_patched_minus_1 = vfield_stats.get_probs_original_and_patched(
        vfields, coeff=-1.0
    )

    probs_original_plus_1, probs_patched_plus_1 = probs_original_plus_1[:, 1], probs_patched_plus_1[:, 1]
    probs_original_minus_1, probs_patched_minus_1 = probs_original_minus_1[:, 1], probs_patched_minus_1[:, 1]


    #alright, try to get one from the same seed. Let's do subtractio seed zero.
    vfields = [
        pickle.load(open(f, "rb"))
        for f in glob("experiments/statistics/data/vfields/top_right/seed-0*.pkl")
    ]
    probs_original_same, probs_patched_same = vfield_stats.get_probs_original_and_patched(
        vfields, coeff=-1.0
    )
    probs_original_same, probs_patched_same = probs_original_same[:, 1], probs_patched_same[:, 1]


    small_seeds = [1, 10, 11, 15, 18, 21, 23, 29, 31, 41, 77, 89, 93, 3, 7, 19, 30, 38, 39, 44, 59, 62, 69, 26, 34, 54, 72]
    vfields = []
    for s in small_seeds:
        for f in glob(f"experiments/statistics/data/vfields/top_right/seed-{s}*.pkl"):
            vfields.append(pickle.load(open(f, "rb")))
    probs_small_seed_orig, probs_small_seed_patched = vfield_stats.get_probs_original_and_patched(
        vfields, coeff=-1.0
    )

    probs_small_seed_orig, probs_small_seed_patched = probs_small_seed_orig[:, 1], probs_small_seed_patched[:, 1]

    # Plot each box plot with markers and different box colors
    for i, (data_var1, data_var2) in enumerate([(probs_original_plus_1, probs_patched_plus_1),
                                                (probs_original_minus_1, probs_patched_minus_1),
                                                (probs_original_same, probs_patched_same),
                                                (probs_small_seed_orig, probs_small_seed_patched)]):

        # mean = np.mean(data_var1), np.mean(data_var2)

        # Plot the boxes with different colors
        bp = ax[i].boxplot([data_var1, data_var2], labels=['Var1', 'Var2'], patch_artist=True)

        # Set colors for the boxes
        for count, patch in enumerate(bp['boxes']):
            # print(patch)
            if count == 0:
                patch.set_facecolor('lightblue')  # Left part of the box
            elif count == 1:
                patch.set_facecolor('orange')
            patch.set_edgecolor('black')

        # TODO - these markers aren't at the actual mean. Try changing the boxplot data struct directly
        # Plot markers for mean without lines
        # ax[i].plot([1, 2], mean, 'kD', markersize=10, label='Mean', linestyle='None')

        # if i == 1:
        #     ax[i].set_title(f'Boxplot {i+1}')
        # elif i == 2:
        #     ax[i].set_title(f'Boxplot {i+1}')
        # elif i == 3:
        #     ax[i].set_title(f'Boxplot {i+1}')
        # else i == 4:
        #     ax[i].set_title(f'Boxplot {i+1}')

    ax[0].set_title("Coeff = 1.0")
    ax[1].set_title("Coeff = -1.0")
    ax[2].set_title("Coeff = -1.0, Same Seed")
    ax[3].set_title("Coeff = -1.0, Small Seeds")

    # Hide x-tick labels and ticks for all boxplots except the first one
    for i in range(0, len(ax)):
        ax[i].set_xticklabels([])
        ax[i].tick_params(axis='x', length=0)

    orig_patch = mpatches.Patch(color='lightblue', label='The original probabilities')
    patch_patch = mpatches.Patch(color='orange', label='The patched probabilities')

    # Display the legend below the subplots
    fig.legend(handles=[orig_patch, patch_patch], loc='lower center', ncol=2, fontsize=10)

    # Display the plot
    #plt.show()
    plt.savefig('playground/paper_graphics/visualizations/top_right_vector.pdf', bbox_inches="tight", format='pdf')


#top_right_vector_figure()