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
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1 import ImageGrid


# Load the font properties
font_prop = fm.FontProperties(fname='/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf')
plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.serif'] = ['Times New Roman']

AX_SIZE = 3.5
patch_coords = (5,6)
seed = 0

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


def first_3_seeds_of_each_size():
    """
    Returns a list of the first 3 seeds of each size.
    """
    seeds = {'3x3': [], '5x5': [], '7x7': [], '9x9': [], '11x11': [], '13x13': [], '15x15': [], '17x17': [], '19x19': [], '21x21': [], '23x23': [], '25x25': []}
    for seed in range(10000):
        inner_grid = maze.get_inner_grid_from_seed(seed)
        size = inner_grid.shape[0]
        if size == 3 and len(seeds['3x3']) < 3:
            seeds['3x3'].append(seed)
        elif size == 5 and len(seeds['5x5']) < 3:
            seeds['5x5'].append(seed)
        elif size == 7 and len(seeds['7x7']) < 3:
            seeds['7x7'].append(seed)
        elif size == 9 and len(seeds['9x9']) < 3:
            seeds['9x9'].append(seed)
        elif size == 11 and len(seeds['11x11']) < 3:
            seeds['11x11'].append(seed)
        elif size == 13 and len(seeds['13x13']) < 3:
            seeds['13x13'].append(seed)
        elif size == 15 and len(seeds['15x15']) < 3:
            seeds['15x15'].append(seed)
        elif size == 17 and len(seeds['17x17']) < 3:
            seeds['17x17'].append(seed)
        elif size == 19 and len(seeds['19x19']) < 3:
            seeds['19x19'].append(seed)
        elif size == 21 and len(seeds['21x21']) < 3:
            seeds['21x21'].append(seed)
        elif size == 23 and len(seeds['23x23']) < 3:
            seeds['23x23'].append(seed)
        elif size == 25 and len(seeds['25x25']) < 3:
            seeds['25x25'].append(seed)
        else:
            no_return = False
            for size in seeds:
                if len(seeds[size]) < 3:
                    no_return = True
            if not no_return:
                return seeds
    raise ValueError("Didn't find enough seeds for each size with range 10000")


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
    for i in range(0, 100, 16):
        for j in range(16):
            seed = seeds[i+j]
            venv = create_venv(1,seed,1)
            state = maze.EnvState(venv.env.callmethod('get_state')[0])
            img = viz.visualize_venv(venv, ax=axd[str(j+1)], render_padding=False)

            axd[str(j+1)].imshow(img)
            axd[str(j+1)].set_xticks([])
            axd[str(j+1)].set_yticks([])
            #axd.set_title(f'Seed {seed}', fontsize=18)

        plt.savefig(f'playground/paper_graphics/visualizations/mass_display_{seed}.png', bbox_inches="tight", format='png')

#seeds = [i for i in range(100)]
#mass_display(seeds)

def random_channel_patch(seed: int, layer_name: str, channel: int):
    """ Replace the given channel's activations with values from a randomly sampled observation. This invokes patch_utils.get_random_patch from patch_utils. If channel=-1, then all channels are replaced. """
    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    patches = patch_utils.get_random_patch(layer_name=layer_name, hook=hook, channel=channel)
    fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=True, ax_size=AX_SIZE)
    plt.show()



# Causal scrub 55
# We want to replace the channel 55 activations with the activations from a randomly generated maze with cheese at the same location
def random_combined_px_patch(layer_name: str, channels: List[int], cheese_loc: Tuple[int, int] = None):
    """ Get a combined patch which randomly replaces channel activations with other activations from different levels. """
    patches = [patch_utils.get_random_patch(layer_name=layer_name, hook=hook, channel=channel, cheese_loc=cheese_loc)
               for channel in channels]
    combined_patch = patch_utils.compose_patches(*patches)
    return combined_patch  # NOTE we're resampling from a fixed maze for all target forward passes


def resample_activations(seed: int, channels: List[int], cheese_loc: Tuple[int, int] = None):
    """ Resample activations for default_layer with the given channels.

    Args:
        seed (int): The seed for the maze
        channels (List[int]): The channels to resample
        different_location (bool, optional): If True, then the resampling location is randomly sampled. Otherwise, it is the cheese location. Defaults to False.
    """
    render_padding = False
    padding = maze.get_padding(maze.get_inner_grid_from_seed(seed))

    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    if cheese_loc is not None:
        resampling_loc = cheese_loc
    else:
        resampling_loc = maze.get_cheese_pos_from_seed(seed, flip_y=False)

    patches = random_combined_px_patch(layer_name=default_layer, channels=channels, cheese_loc=resampling_loc)

    # actually just have it return the patches and then I'll display it as normal.
    return patches

    # patches = random_combined_px_patch(layer_name=default_layer, channels=[55])
    # fig, axs, info = patch_utils.compare_patched_vfields(venv, patches, hook, render_padding=render_padding)
    # channel_description = f'channels {channels}' if len(channels) > 1 else f'channel {channels[0]}'
    # fig.suptitle(f'Resampling {channel_description} on seed {seed}', fontsize=20)

    # Have this be the ghost cheese instead
    # viz.plot_dots(axs[1:], resampling_loc, is_grid=True, flip_y=False,
    #                         hidden_padding=0 if render_padding else padding)
    #plt.show()

#resample_activations(0, [55], (12, 13)) #this is cheese location from outer grid
#okay this finally works. Switch it so that it only displays the mpp of the patched maze, and displays on a given axis

# cheese_channels = [7, 8, 42, 44, 55, 77, 82, 88, 89, 99, 113]
# resample_activations(0, cheese_channels, 435) # - no errors! at least neutral sign :)


def plot_heatmap(seed, prob_type, ax, magnitude = None):
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

def seeds_by_cheese_loc(cheese_loc: Tuple[int, int]):
    """
    Here return a list of 100 different seeds that have cheese at the given cheese loc.

    I do so based on outer grid. So before accepting the seed, make sure that the given cheese loc
    is within the inner grid of the seed.
    """
    list_of_seeds = []
    seed = 0
    while(len(list_of_seeds) < 100):
        # venv = create_venv(1,167,1)
        # state = maze.EnvState(venv.env.callmethod('get_state')[0])

        # get the size of the inner grid
        # inner_grid = state.inner_grid()
        # size = inner_grid.shape[0]
        # padding = (25 - size) // 2
        # if cheese_loc[0] >= padding and cheese_loc[0] <= padding + size and cheese_loc[1] >= padding and cheese_loc[1] <= padding + size:

        # actually error checking nbd. cheese will only ever be in the inner grid anyways, so if it matches then it'll be it.
        #seed = 167
        grid = maze.get_full_grid_from_seed(seed)
        this_maze_loc = maze.get_cheese_pos(grid)
        if maze.get_cheese_pos(grid) == cheese_loc:
            list_of_seeds.append(seed)
        seed += 1

    return list_of_seeds

#cheese location that's roughly in the center would be from seed 167 (d from fig 5).
# venv = create_venv(1, 167, 1)
# state = maze.EnvState(venv.env.callmethod('get_state')[0])
# inner_grid = state.inner_grid()
# print(maze.get_cheese_pos(inner_grid)) #this needs to be the outer grid because it needs to be objective between mazes with different sizes.
#
# grid = maze.get_full_grid_from_seed(167)
# print(maze.get_cheese_pos(grid))
#
# list_of_seeds = seeds_by_cheese_loc((14, 10)) #seed 167 cheese loc in outer grid
# print(list_of_seeds)


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
        figsize=(AX_SIZE * 4.25, AX_SIZE*1.5),
        tight_layout=True,
    )
    #fig1, axd1 = plt.subplots(1,4)

    activ = hook.get_value_by_label(default_layer)[0][cheese_channel]
    axd1['orig_act'].imshow(activ)
    axd1['orig_act'].imshow(activ, cmap='bwr', vmin=-1, vmax=1) #test of reversing colorscheme
    # alright, changed color scheme from RdBu to bwr. Ref: https://matplotlib.org/stable/users/explain/colors/colormaps.html

    axd1['orig_act'].set_xticks([])
    axd1['orig_act'].set_yticks([])

    #orig_mpp = viz.visualize_venv(venv, render_padding=False)
    vf = viz.vector_field(venv, policy)

    img = viz.plot_vf_mpp(vf, ax=axd1['orig_mpp'], save_img=False)
    axd1['orig_mpp'].imshow(img)

    patch_coords = (3, 12)
    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer,channel=55, value=5.6, coord=patch_coords)
    #patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer,channel=55, value=5.6, coord=(17,17)) # to test that this is on a 16x16 grid - yes it is


    with hook.use_patches(patches):
        hook.network(obs) # trying before asking Uli
        patched_vfield = viz.vector_field(venv, hook.network)
        #hook.network(obs) # trying before asking Uli
    img = viz.plot_vf_mpp(patched_vfield, ax=axd1['patch_mpp'], save_img=False)
    axd1['patch_mpp'].imshow(img)

    maze.move_cheese_in_state(state, (18, 18))
    venv.env.callmethod('set_state', [state.state_bytes])
    obs = t.tensor(venv.reset(), dtype=t.float32)


    activ = hook.get_value_by_label(default_layer)[0][cheese_channel]
    patch_img = axd1['patch_act'].imshow(activ)
    axd1['patch_act'].imshow(activ, cmap='bwr', vmin=-1, vmax=1)

    axd1['patch_act'].set_xticks([])
    axd1['patch_act'].set_yticks([])

    font_dict = {'fontname': 'Times New Roman', 'fontsize': 24}

    axd1['orig_mpp'].set_title('(a): Original Behavior', pad=8, **font_dict)#, fontproperties=font_prop)
    axd1['orig_act'].set_title('(b): Original Activations', pad=8, **font_dict)
    axd1['patch_act'].set_title('(c): Patched Activations', pad=8, **font_dict)
    axd1['patch_mpp'].set_title('(d): Patched MPP', pad=8, **font_dict)


    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cax = fig1.add_axes([0.275, 0.075, 0.45, 0.05])  # distance from left, distance from bottom, width, height
    cbar = fig1.colorbar(cax=cax, mappable=mpl.cm.ScalarMappable(norm=norm, cmap='bwr'),
                  orientation='horizontal')  # , location='bottom', shrink=0.5)

    # Define custom ticks and labels
    custom_ticks = [-1, 0, 1]  # Custom ticks positions
    custom_tick_labels = ['-1', '0', '1']  # Custom tick labels

    # Set the ticks and their labels
    cbar.set_ticks(custom_ticks)
    cbar.set_ticklabels(custom_tick_labels)
    cbar.ax.tick_params(labelsize=20)
    plt.savefig('playground/paper_graphics/visualizations/fig_1.pdf', bbox_inches="tight", format='pdf')
    #fig1.subplots_adjust(right=0.95)

    # fig1, axd1 = plt.subplots(1,4,figsize=(AX_SIZE * 4.25, AX_SIZE))#, constrained_layout=True)#), tight_layout=True,)
    #
    # activ = hook.get_value_by_label(default_layer)[0][cheese_channel]
    # axd1[1].imshow(activ)
    # axd1[1].imshow(activ, cmap='bwr', vmin=-1, vmax=1) #test of reversing colorscheme
    # # alright, changed color scheme from RdBu to bwr. Ref: https://matplotlib.org/stable/users/explain/colors/colormaps.html
    #
    # axd1[1].set_xticks([])
    # axd1[1].set_yticks([])
    #
    # #orig_mpp = viz.visualize_venv(venv, render_padding=False)
    # vf = viz.vector_field(venv, policy)
    #
    # img = viz.plot_vf_mpp(vf, ax=axd1[0], save_img=False)
    # axd1[0].imshow(img)
    #
    # patch_coords = (3, 12)
    # patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer,channel=55, value=5.6, coord=patch_coords)
    # #patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer,channel=55, value=5.6, coord=(17,17)) # to test that this is on a 16x16 grid - yes it is
    #
    #
    # with hook.use_patches(patches):
    #     hook.network(obs) # trying before asking Uli
    #     patched_vfield = viz.vector_field(venv, hook.network)
    #     #hook.network(obs) # trying before asking Uli
    # img = viz.plot_vf_mpp(patched_vfield, ax=axd1[3], save_img=False)
    # axd1[3].imshow(img)
    #
    # maze.move_cheese_in_state(state, (18, 18))
    # venv.env.callmethod('set_state', [state.state_bytes])
    # obs = t.tensor(venv.reset(), dtype=t.float32)
    #
    #
    # activ = hook.get_value_by_label(default_layer)[0][cheese_channel]
    # patch_img = axd1[2].imshow(activ)
    # axd1[2].imshow(activ, cmap='bwr', vmin=-1, vmax=1)
    #
    # axd1[2].set_xticks([])
    # axd1[2].set_yticks([])
    #
    # font_dict = {'fontname': 'Times New Roman', 'fontsize': 24}
    #
    # axd1[0].set_title('Original MPP', **font_dict)#, fontproperties=font_prop)
    # axd1[1].set_title('Original Activations', **font_dict)
    # axd1[2].set_title('Patched Activations', **font_dict)
    # axd1[3].set_title('Patched MPP', **font_dict)
    #
    # norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    #
    # fig1.subplots_adjust(right=0.9, hspace=0.01)
    # cax = fig1.add_axes([0.925, 0.135, 0.025, 0.725])
    #
    # # divider = make_axes_locatable(axd1)
    # # cax = divider.append_axes("right", size="5%", pad=0.05)
    #
    # cbar = fig1.colorbar(cax=cax, mappable=mpl.cm.ScalarMappable(norm=norm, cmap='bwr'), orientation='vertical')#, location='right')#, location='bottom', shrink=0.5)
    # # cax = fig1.add_axes([0.275, 0.05, 0.45, 0.05]) #distance from left, distance from bottom, width, height
    # # fig1.colorbar(cax=cax, mappable=mpl.cm.ScalarMappable(norm=norm, cmap='bwr'), orientation='horizontal')#, location='bottom', shrink=0.5)
    # #plt.savefig('playground/paper_graphics/visualizations/fig_1.pdf', bbox_inches="tight", format='pdf')
    # # Define custom ticks and labels
    # custom_ticks = [-1, 0, 1]  # Custom ticks positions
    # custom_tick_labels = ['-1', '0', '1']  # Custom tick labels
    #
    # # Set the ticks and their labels
    # cbar.set_ticks(custom_ticks)
    # cbar.set_ticklabels(custom_tick_labels)
    # cbar.ax.tick_params(labelsize=20)

    #plt.show()

#fig_1()


# ------------------------------------------------ fig 1 with subfigures ---------------------------------------

def fig_1_subfigures():
    venv = create_venv(1,seed,1)
    state = maze.EnvState(venv.env.callmethod('get_state')[0])

    obs = t.tensor(venv.reset(), dtype=t.float32)

    with hook.set_hook_should_get_custom_data():
        hook.network(obs)

    # #fig, ax = plt.subplots()
    # fig1, axd1 = plt.subplot_mosaic(
    #     [['orig_mpp', 'orig_act', 'patch_act', 'patch_mpp']],
    #     figsize=(AX_SIZE * 4, AX_SIZE*1.5),
    #     tight_layout=True,
    # )
    #
    # activ = hook.get_value_by_label(default_layer)[0][cheese_channel]
    # axd1['orig_act'].imshow(activ)
    # axd1['orig_act'].imshow(activ, cmap='bwr', vmin=-1, vmax=1) #test of reversing colorscheme
    # # alright, changed color scheme from RdBu to bwr. Ref: https://matplotlib.org/stable/users/explain/colors/colormaps.html
    #
    # axd1['orig_act'].set_xticks([])
    # axd1['orig_act'].set_yticks([])
    #
    # #orig_mpp = viz.visualize_venv(venv, render_padding=False)
    # vf = viz.vector_field(venv, policy)
    #
    # img = viz.plot_vf_mpp(vf, ax=axd1['orig_mpp'], save_img=False)
    # axd1['orig_mpp'].imshow(img)
    #
    # patch_coords = (3, 12)
    # patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer,channel=55, value=5.6, coord=patch_coords)
    #
    # with hook.use_patches(patches):
    #     hook.network(obs) # trying before asking Uli
    #     patched_vfield = viz.vector_field(venv, hook.network)
    # img = viz.plot_vf_mpp(patched_vfield, ax=axd1['patch_mpp'], save_img=False)
    # axd1['patch_mpp'].imshow(img)
    #
    # # TODO - I think the activations for seed = 0 get vertically flipped somehow in the display
    # maze.move_cheese_in_state(state, (18, 18))
    # venv.env.callmethod('set_state', [state.state_bytes])
    # obs = t.tensor(venv.reset(), dtype=t.float32)
    #
    # activ = hook.get_value_by_label(default_layer)[0][cheese_channel]
    # patch_img = axd1['patch_act'].imshow(activ)
    # axd1['patch_act'].imshow(activ, cmap='bwr', vmin=-1, vmax=1)
    #
    # axd1['patch_act'].set_xticks([])
    # axd1['patch_act'].set_yticks([])
    #
    # axd1['orig_mpp'].set_title('Original MPP', fontsize=18, fontproperties=font_prop)
    # axd1['orig_act'].set_title('Original Activations', fontsize=18, fontproperties=font_prop)
    # axd1['patch_act'].set_title('Patched Activations', fontsize=18, fontproperties=font_prop)
    # axd1['patch_mpp'].set_title('Patched MPP', fontsize=18, fontproperties=font_prop)
    # plt.savefig('playground/paper_graphics/visualizations/fig_1.pdf', bbox_inches="tight", format='pdf')

    fig = plt.figure(figsize=(AX_SIZE * 4, AX_SIZE*2))#, layout='constrained')
    subfigs = fig.subfigures(1, 2, wspace=0.5)

    subfigs[0].suptitle('Most Probable Path', fontsize=18, fontproperties=font_prop)
    subfigs[1].suptitle('Activations', fontsize=18, fontproperties=font_prop)
    #subfigs[2].suptitle('Patched Activations', fontsize=18, fontproperties=font_prop)
    #subfigs[3].suptitle('Patched MPP', fontsize=18, fontproperties=font_prop)

    #Original MPP
    vf = viz.vector_field(venv, policy)
    ax0 = subfigs[0].subplots(1,2)
    img = viz.plot_vf_mpp(vf, ax=ax0[0], save_img=False)
    ax0[0].imshow(img)


    #Original Activations
    activ = hook.get_value_by_label(default_layer)[0][cheese_channel]
    ax1 = subfigs[1].subplots(1, 2)
    ax1[0].imshow(activ, cmap='bwr', vmin=-1, vmax=1)
    ax1[0].set_xticks([])
    ax1[0].set_yticks([])
    ax1[0].set_title('Original Activations', fontsize=18, fontproperties=font_prop)

    #Patched MPP
    patch_coords = (3, 12)
    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer,channel=55, value=5.6, coord=patch_coords)

    with hook.use_patches(patches):
        hook.network(obs) # trying before asking Uli
        patched_vfield = viz.vector_field(venv, hook.network)
    img = viz.plot_vf_mpp(patched_vfield, ax=ax0[1], save_img=False)
    ax0[1].imshow(img)

    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cax = fig.add_axes([0.25, 0.05, 0.5, 0.05]) #
    fig.colorbar(cax=cax, mappable=mpl.cm.ScalarMappable(norm=norm, cmap='bwr'), orientation='horizontal')#, location='bottom', shrink=0.5)

    plt.show()

#fig_1_subfigures()

# ---------------------------------------------------- fig 2 ----------------------------------------------------
def fig_2():

    # all cheese channels - 7, 8, 42, 44, 55, 77, 82, 88, 89, 99, 113
    cheese_channels = [7, 8, 42, 44, 55, 77, 82, 88, 89, 99, 113]
    for channel in cheese_channels:

        #move cheese in state uses full grid. use padding manually to get right grid coords
        #cheese_a_pos = ()
        cheese_b_pos = (18, 18) #top right
        cheese_c_pos = (6, 18) #bottom right

        fig2, axd2 = plt.subplot_mosaic(
            [['reg_venv', 'cheese_a', 'cheese_b', 'cheese_c']],
            figsize=(AX_SIZE * 4, AX_SIZE*1.5),
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

        activ = hook.get_value_by_label(default_layer)[0][channel]
        axd2['cheese_a'].imshow(activ)
        axd2['cheese_a'].imshow(activ, cmap='bwr', vmin=-1, vmax=1)

        axd2['cheese_a'].set_xticks([])
        axd2['cheese_a'].set_yticks([])

        maze.move_cheese_in_state(state, cheese_b_pos)
        venv.env.callmethod('set_state', [state.state_bytes])
        obs = t.tensor(venv.reset(), dtype=t.float32)

        with hook.set_hook_should_get_custom_data():
            hook.network(obs)

        #img = viz.visualize_venv(venv, render_padding=False, show_plot=True) #checking cheese coords, it is top right
        activ = hook.get_value_by_label(default_layer)[0][channel]
        axd2['cheese_b'].imshow(activ)
        axd2['cheese_b'].imshow(activ, cmap='bwr', vmin=-1, vmax=1)

        axd2['cheese_b'].set_xticks([])
        axd2['cheese_b'].set_yticks([])

        maze.move_cheese_in_state(state, cheese_c_pos)
        venv.env.callmethod('set_state', [state.state_bytes])
        obs = t.tensor(venv.reset(), dtype=t.float32)

        with hook.set_hook_should_get_custom_data():
            hook.network(obs)

        #img = viz.visualize_venv(venv, render_padding=False, show_plot=True) #checking cheese coords, it is bottom right
        activ = hook.get_value_by_label(default_layer)[0][channel]
        axd2['cheese_c'].imshow(activ)
        axd2['cheese_c'].imshow(activ, cmap='bwr', vmin=-1, vmax=1)

        axd2['cheese_c'].set_xticks([])
        axd2['cheese_c'].set_yticks([])

        axd2['reg_venv'].set_title('(a): Maze with \n cheese locations', fontsize=24)
        axd2['cheese_a'].set_title('(b): Cheese at \n location 1', fontsize=24)
        axd2['cheese_b'].set_title('(c): Cheese at \n location 2', fontsize=24)
        axd2['cheese_c'].set_title('(d): Cheese at \n location 3', fontsize=24)

        # norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        # cax = fig2.add_axes([0.275, 0.05, 0.45, 0.05]) #distance from left, distance from bottom, width, height
        # fig2.colorbar(cax=cax, mappable=mpl.cm.ScalarMappable(norm=norm, cmap='bwr'), orientation='horizontal')#, location='bottom', shrink=0.5)

        # norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        # cax = fig2.add_axes([0.275, 0.075, 0.45, 0.05])  # distance from left, distance from bottom, width, height
        # # cbar = fig2.colorbar(cax=cax, mappable=mpl.cm.ScalarMappable(norm=norm, cmap='bwr'),
        # #                     orientation='horizontal')  # , location='bottom', shrink=0.5)
        #
        # cbar = fig2.colorbar(cax=cax, mappable=mpl.cm.ScalarMappable(norm=norm, cmap='bwr'), orientation='horizontal')
        #
        # # Define custom ticks and labels
        # custom_ticks = [-1, 0, 1]  # Custom ticks positions
        # custom_tick_labels = ['-1', '0', '1']  # Custom tick labels
        #
        # # Set the ticks and their labels
        # cbar.set_ticks(custom_ticks)
        # cbar.set_ticklabels(custom_tick_labels)
        # cbar.ax.tick_params(labelsize=20)

        # manually add cbar in illustrator

        #plt.show()
        plt.savefig(f'playground/paper_graphics/visualizations/fig_2_channel_{channel}.pdf', bbox_inches="tight", format='pdf')
        #break

#fig_2()

def fig_2_test():

    # matplotlib can't rescale the 268x268x3 maze image to a 16x16 ndarray or vice versa.
    # so this sucks.
    # but use it to generate the cbar to attach tomorrow

    # all cheese channels - 7, 8, 42, 44, 55, 77, 82, 88, 89, 99, 113
    cheese_channels = [7, 8, 42, 44, 55, 77, 82, 88, 89, 99, 113]
    for channel in cheese_channels:

        #move cheese in state uses full grid. use padding manually to get right grid coords
        #cheese_a_pos = ()
        cheese_b_pos = (18, 18) #top right
        cheese_c_pos = (6, 18) #bottom right

        # fig2, axd2 = plt.subplot_mosaic(
        #     [['reg_venv', 'cheese_a', 'cheese_b', 'cheese_c', 'colorbar']],
        #     figsize=(AX_SIZE * 4, AX_SIZE*1.5),
        #     tight_layout=True,
        # )

        # Set up figure and image grid
        fig = plt.figure(figsize=(AX_SIZE * 4, AX_SIZE*1.5))

        grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                         nrows_ncols=(1, 4),
                         axes_pad=0.15,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="7%",
                         cbar_pad=0.15,
                         )

        venv = create_venv(1,seed,1)
        state = maze.EnvState(venv.env.callmethod('get_state')[0])

        for count, ax in enumerate(grid):

            if count == 0:
                img = viz.visualize_venv(venv, ax=ax, render_padding=False)
                im = ax.imshow(img)
                ax.set_title('(a): Maze with \n cheese locations', fontsize=24)

            elif count == 1:
                obs = t.tensor(venv.reset(), dtype=t.float32)

                with hook.set_hook_should_get_custom_data():
                    hook.network(obs)

                activ = hook.get_value_by_label(default_layer)[0][channel]
                ax.imshow(activ)
                im = ax.imshow(activ, cmap='bwr', vmin=-1, vmax=1)

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('(b): Cheese at \n location 1', fontsize=24)

            elif count == 2:
                maze.move_cheese_in_state(state, cheese_b_pos)
                venv.env.callmethod('set_state', [state.state_bytes])
                obs = t.tensor(venv.reset(), dtype=t.float32)

                with hook.set_hook_should_get_custom_data():
                    hook.network(obs)

                # img = viz.visualize_venv(venv, render_padding=False, show_plot=True) #checking cheese coords, it is top right
                activ = hook.get_value_by_label(default_layer)[0][channel]
                ax.imshow(activ)
                im = ax.imshow(activ, cmap='bwr', vmin=-1, vmax=1)

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('(c): Cheese at \n location 2', fontsize=24)

            elif count == 3:
                maze.move_cheese_in_state(state, cheese_c_pos)
                venv.env.callmethod('set_state', [state.state_bytes])
                obs = t.tensor(venv.reset(), dtype=t.float32)

                with hook.set_hook_should_get_custom_data():
                    hook.network(obs)

                # img = viz.visualize_venv(venv, render_padding=False, show_plot=True) #checking cheese coords, it is bottom right
                activ = hook.get_value_by_label(default_layer)[0][channel]
                ax.imshow(activ)
                im = ax.imshow(activ, cmap='bwr', vmin=-1, vmax=1)

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title('(d): Cheese at \n location 3', fontsize=24)

        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        #norm = mpl.colors.SymLogNorm(linthresh=0.03, linscale=1,#0.03, vmin=-6.0, vmax=6.0, base=6)

        # colors = [
        #     (1, 0, 0, 0),
        #     (1, 0, 0, 0.7), #alpha by default 0.7 in Alex's code
        # ]  # Transparent to non-transparent red

        #make it grey to red, transparency doesn't work well on a cbar
        colors = [(0.5, 0.5, 0.5), (1, 0, 0)]

        cmap_name = "custom_div_cmap"
        cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

        grid.cbar_axes[0].colorbar(mappable=mpl.cm.ScalarMappable(norm=norm, cmap=cm))
        #grid.cbar_axes[0].colorbar(mappable=mpl.cm.ScalarMappable(norm=norm, cmap='bwr'))

        #grid.cbar_axes[0].set_yticklabels(['-1', '0', '1'], fontsize=20)

        # Define custom ticks and labels
        custom_ticks = [0, 0.5, 1]  # Custom ticks positions
        custom_tick_labels = ['0', '0.5', '1']  # Custom tick labels

        # Set the ticks and their labels
        grid.cbar_axes[0].set_yticks(custom_ticks)
        grid.cbar_axes[0].set_yticklabels(custom_tick_labels)
        grid.cbar_axes[0].tick_params(labelsize=20)

        # venv = create_venv(1,seed,1)
        # state = maze.EnvState(venv.env.callmethod('get_state')[0])

        # img = viz.visualize_venv(venv, ax=axd2['reg_venv'], render_padding=False)
        # axd2['reg_venv'].imshow(img)

        # want to show activations from first cheese
        # maze.move_cheese_in_state(state, cheese_a_pos)
        # venv.env.callmethod('set_state', [state.state_bytes])
        # obs = t.tensor(venv.reset(), dtype=t.float32)
        #
        # with hook.set_hook_should_get_custom_data():
        #     hook.network(obs)
        #
        # activ = hook.get_value_by_label(default_layer)[0][channel]
        # axd2['cheese_a'].imshow(activ)
        # axd2['cheese_a'].imshow(activ, cmap='bwr', vmin=-1, vmax=1)
        #
        # axd2['cheese_a'].set_xticks([])
        # axd2['cheese_a'].set_yticks([])
        #
        # maze.move_cheese_in_state(state, cheese_b_pos)
        # venv.env.callmethod('set_state', [state.state_bytes])
        # obs = t.tensor(venv.reset(), dtype=t.float32)
        #
        # with hook.set_hook_should_get_custom_data():
        #     hook.network(obs)
        #
        # #img = viz.visualize_venv(venv, render_padding=False, show_plot=True) #checking cheese coords, it is top right
        # activ = hook.get_value_by_label(default_layer)[0][channel]
        # axd2['cheese_b'].imshow(activ)
        # axd2['cheese_b'].imshow(activ, cmap='bwr', vmin=-1, vmax=1)
        #
        # axd2['cheese_b'].set_xticks([])
        # axd2['cheese_b'].set_yticks([])
        #
        # maze.move_cheese_in_state(state, cheese_c_pos)
        # venv.env.callmethod('set_state', [state.state_bytes])
        # obs = t.tensor(venv.reset(), dtype=t.float32)
        #
        # with hook.set_hook_should_get_custom_data():
        #     hook.network(obs)
        #
        # #img = viz.visualize_venv(venv, render_padding=False, show_plot=True) #checking cheese coords, it is bottom right
        # activ = hook.get_value_by_label(default_layer)[0][channel]
        # axd2['cheese_c'].imshow(activ)
        # axd2['cheese_c'].imshow(activ, cmap='bwr', vmin=-1, vmax=1)
        #
        # axd2['cheese_c'].set_xticks([])
        # axd2['cheese_c'].set_yticks([])

        # axd2['reg_venv'].set_title('(a): Maze with \n cheese locations', fontsize=24)
        # axd2['cheese_a'].set_title('(b): Cheese at \n location 1', fontsize=24)
        # axd2['cheese_b'].set_title('(c): Cheese at \n location 2', fontsize=24)
        # axd2['cheese_c'].set_title('(d): Cheese at \n location 3', fontsize=24)
        #
        # norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        # cax = fig2.add_axes([0.275, 0.05, 0.45, 0.05]) #distance from left, distance from bottom, width, height
        # fig2.colorbar(cax=cax, mappable=mpl.cm.ScalarMappable(norm=norm, cmap='bwr'), orientation='horizontal')#, location='bottom', shrink=0.5)

        # norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        # #cax = fig2.add_axes([0.275, 0.075, 0.45, 0.05])  # distance from left, distance from bottom, width, height
        # #cbar = fig2.colorbar(cax=cax, mappable=mpl.cm.ScalarMappable(norm=norm, cmap='bwr'),
        # #                     orientation='horizontal')  # , location='bottom', shrink=0.5)
        #
        # cbar = fig2.colorbar(ax=axd2['colorbar'], mappable=mpl.cm.ScalarMappable(norm=norm, cmap='bwr'))
        #
        # # Define custom ticks and labels
        # custom_ticks = [-1, 0, 1]  # Custom ticks positions
        # custom_tick_labels = ['-1', '0', '1']  # Custom tick labels
        #
        # # Set the ticks and their labels
        # cbar.set_ticks(custom_ticks)
        # cbar.set_ticklabels(custom_tick_labels)
        # cbar.ax.tick_params(labelsize=20)

        #plt.show()
        plt.savefig(f'playground/paper_graphics/visualizations/grab_colorbar_heatmap.pdf', bbox_inches="tight", format='pdf')
        break
#fig_2_test()

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

"""
19/9

Alright just had some more thoughts. I may not need to do an entire forward pass here. Wouldn't it be enough to use the
patched vector field to show the MPP?

"""

def fig_3():

    # a list of 13x13 mazes
    # some of these cause my mpp function to go haywire. unsure why, but removed them from the list
    seeds = [0, 2, 16, 51, 74, 84, 85, 99, 107, 108, 132, 169, 183, 189, 192, 195,
             204, 207, 291, 304, 314, 322, 335, 337, 338, 344, 346, 385, 389, 392, 414, 433, 435, 449, 455,
             470, 516, 517, 543, 555, 559]
    seeds = [i for i in range(200)]
    #for seed in seeds:
    seed = 1
    #try:
    if os.path.exists(f'playground/paper_graphics/visualizations/fig_3_by_seed/{seed}_fig_3.pdf'):
        print(f'Already have {seed}')
        #continue # in the for loop
    if os.path.exists(f'playground/paper_graphics/visualizations/fig_3_by_seed/{seed}_fig_3.png'):
        print(f'Already have {seed}')
        #continue
    print("current seed: ", seed)

    seeds_by_size = first_3_seeds_of_each_size()
    # for key, value in seeds_by_size.items():
    #     for i in value:
    #         fig3, axd3 = plt.subplot_mosaic(
    #             [['original', 'same_loc', 'dif_loc_historic', 'dif_loc_bottom_right']],
    #             figsize=(AX_SIZE * 4, AX_SIZE),
    #             tight_layout=True,
    #         )
    #
    #         # seed = 304
    #
    #         venv = create_venv(1, i, 1)
    #         state = maze.EnvState(venv.env.callmethod('get_state')[0])
    #
    #         vf = viz.vector_field(venv, policy)
    #         img = viz.plot_vf_mpp(vf, ax=axd3['original'], save_img=False)
    #         axd3['original'].imshow(img)
    #         axd3['original'].set_title('Original', fontsize=24)
    #
    #         # part b
    #         # move to cheese loc in same location
    #         # resample across all channels
    #         cheese_channels = [7, 8, 42, 44, 55, 77, 82, 88, 89, 99, 113]
    #
    #         patches = resample_activations(i, cheese_channels)
    #
    #         venv = create_venv(1, i, 1)
    #         obs = t.tensor(venv.reset(), dtype=t.float32)
    #
    #         with hook.use_patches(patches):
    #             hook.network(obs)
    #             patched_vfield = viz.vector_field(venv, hook.network)
    #         img = viz.plot_vf_mpp(patched_vfield, ax=axd3['same_loc'], save_img=False)
    #         axd3['same_loc'].imshow(img)
    #         axd3['same_loc'].set_title('Same Location', fontsize=24)
    #
    #         # part c
    #         # different location in the top right
    #
    #         # size of inner grid
    #         inner_grid = maze.get_inner_grid_from_seed(i)
    #         size = inner_grid.shape[0]
    #         padding = maze.get_padding(maze.get_inner_grid_from_seed(i))
    #
    #         # top right
    #         top_right = (size + padding - 1, size + padding - 1)
    #         # if key == '3x3' and i == 11:
    #         #     top_right = (13, 13)
    #         patches = resample_activations(i, cheese_channels, cheese_loc=top_right)
    #
    #         venv = create_venv(1, i, 1)
    #         obs = t.tensor(venv.reset(), dtype=t.float32)
    #
    #         with hook.use_patches(patches):
    #             hook.network(obs)
    #             patched_vfield = viz.vector_field(venv, hook.network)
    #         print(key, value, i)
    #         img = viz.plot_vf_mpp(patched_vfield, ax=axd3['dif_loc_historic'], save_img=False)
    #         axd3['dif_loc_historic'].imshow(img)
    #         axd3['dif_loc_historic'].set_title('Historic Location', fontsize=24)
    #
    #         # part d
    #         # different location in the bottom right
    #
    #         bottom_right = (padding - 1, size + padding - 1)
    #
    #         patches = resample_activations(i, cheese_channels, cheese_loc=bottom_right)
    #         padding = maze.get_padding(maze.get_inner_grid_from_seed(i))
    #         # viz.plot_pixel_dot(axd3['dif_loc_bottom_right'], 12, 0, hidden_padding=padding)
    #
    #         venv = create_venv(1, i, 1)
    #         obs = t.tensor(venv.reset(), dtype=t.float32)
    #
    #         with hook.use_patches(patches):
    #             hook.network(obs)
    #             patched_vfield = viz.vector_field(venv, hook.network)
    #         img = viz.plot_vf_mpp(patched_vfield, ax=axd3['dif_loc_bottom_right'], save_img=False)
    #         axd3['dif_loc_bottom_right'].imshow(img)
    #         axd3['dif_loc_bottom_right'].set_title('Bottom Right Location', fontsize=24)
    #
    #         # add title
    #         # plt.suptitle(f'Seed {seed}')
    #         # plt.show()
    #         plt.savefig(f'playground/paper_graphics/visualizations/fig_3_by_size/{key}_fig_3_{i}.pdf', bbox_inches="tight", format='pdf')


    fig3, axd3 = plt.subplot_mosaic(
        [['original', 'same_loc', 'dif_loc_historic', 'dif_loc_bottom_right']],
        figsize=(AX_SIZE * 4, AX_SIZE),
        tight_layout=True,
    )

    seed = 51

    venv = create_venv(1,seed,1)
    state = maze.EnvState(venv.env.callmethod('get_state')[0])

    vf = viz.vector_field(venv, policy)
    img = viz.plot_vf_mpp(vf, ax=axd3['original'], save_img=False)
    axd3['original'].imshow(img)
    axd3['original'].set_title('(a): Original Maze', pad=8, fontsize=24)

    # part b
    # move to cheese loc in same location
    # resample across all channels
    cheese_channels = [7, 8, 42, 44, 55, 77, 82, 88, 89, 99, 113]

    patches = resample_activations(seed, cheese_channels)

    venv = create_venv(1,seed,1)
    obs = t.tensor(venv.reset(), dtype=t.float32)

    with hook.use_patches(patches):
        hook.network(obs)
        patched_vfield = viz.vector_field(venv, hook.network)
    img = viz.plot_vf_mpp(patched_vfield, ax=axd3['same_loc'], save_img=False)
    axd3['same_loc'].imshow(img)
    axd3['same_loc'].set_title('(b): Resampling from\nsame location', pad=8, fontsize=24)


    # part c
    # different location in the top right

    #size of inner grid
    inner_grid = maze.get_inner_grid_from_seed(seed)
    size = inner_grid.shape[0]
    padding = maze.get_padding(maze.get_inner_grid_from_seed(seed))


    #top right
    top_right = (size + padding, size + padding)
    patches = resample_activations(seed, cheese_channels, cheese_loc=top_right)

    venv = create_venv(1,seed,1)
    obs = t.tensor(venv.reset(), dtype=t.float32)

    with hook.use_patches(patches):
        hook.network(obs)
        patched_vfield = viz.vector_field(venv, hook.network)
    img = viz.plot_vf_mpp(patched_vfield, ax=axd3['dif_loc_historic'], save_img=False)
    axd3['dif_loc_historic'].imshow(img)
    axd3['dif_loc_historic'].set_title('(c): Resampling from\n red cheese location', pad=8, fontsize=24)

    # part d
    # different location in the bottom right

    bottom_right = (0, size + padding)

    patches = resample_activations(seed, cheese_channels, cheese_loc=bottom_right)
    padding = maze.get_padding(maze.get_inner_grid_from_seed(seed))
    #viz.plot_pixel_dot(axd3['dif_loc_bottom_right'], 12, 0, hidden_padding=padding)

    venv = create_venv(1,seed,1)
    obs = t.tensor(venv.reset(), dtype=t.float32)

    with hook.use_patches(patches):
        hook.network(obs)
        patched_vfield = viz.vector_field(venv, hook.network)
    img = viz.plot_vf_mpp(patched_vfield, ax=axd3['dif_loc_bottom_right'], save_img=False)
    axd3['dif_loc_bottom_right'].imshow(img)
    axd3['dif_loc_bottom_right'].set_title('(d): Resampling from\n red cheese location', pad=8, fontsize=24)


    # add title
    #plt.suptitle(f'Seed {seed}')
    plt.show()
    #plt.savefig(f'playground/paper_graphics/visualizations/{seed}_fig_3.pdf', bbox_inches="tight", format='pdf')
    #plt.savefig(f'playground/paper_graphics/visualizations/fig_3_by_seed/{seed}_fig_3.png', bbox_inches="tight", format='png')
    #plt.close()
    # except:
    #     #continue #in the for loop
    #     print(f"excepted at seed {seed}")
    #     pass


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

    fig4a, ax4a = plt.subplots()
    fig4b, ax4b = plt.subplots()
    fig4c, ax4c = plt.subplots()
    fig4d, ax4d = plt.subplots()


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

    fig4.show()
    img = viz.plot_vf_mpp(patched_vfield, ax=ax4a, save_img=False)
    viz.plot_pixel_dot(ax4a, u, v, hidden_padding=padding)
    ax4a.imshow(img)
    ax4a.set_xticks([])
    ax4a.set_yticks([])
    fig4a.show()

    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer,channel=55, value=5, coord=success_b_pos)
    with hook.use_patches(patches):
        patched_vfield = viz.vector_field(venv, hook.network)
    img = viz.plot_vf_mpp(patched_vfield, ax=axd4['success_b'], save_img=False)

    u, v = linear_mapping(success_b_pos)
    u, v = inner_grid_coords_from_full_grid_coords((u,v))
    padding = maze.get_padding(maze.get_inner_grid_from_seed(seed))
    viz.plot_pixel_dot(axd4['success_b'], u, v, hidden_padding=padding)
    axd4['success_b'].imshow(img)

    img = viz.plot_vf_mpp(patched_vfield, ax=ax4b, save_img=False)
    viz.plot_pixel_dot(ax4b, u, v, hidden_padding=padding)
    ax4b.imshow(img)
    ax4b.set_xticks([])
    ax4b.set_yticks([])

    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer,channel=55, value=5, coord=success_c_pos)
    with hook.use_patches(patches):
        patched_vfield = viz.vector_field(venv, hook.network)
    img = viz.plot_vf_mpp(patched_vfield, ax=axd4['success_c'], save_img=False)

    u, v = linear_mapping(success_c_pos)
    u, v = inner_grid_coords_from_full_grid_coords((u,v))
    padding = maze.get_padding(maze.get_inner_grid_from_seed(seed))
    viz.plot_pixel_dot(axd4['success_c'], u, v, hidden_padding=padding)
    axd4['success_c'].imshow(img)

    img = viz.plot_vf_mpp(patched_vfield, ax=ax4c, save_img=False)
    viz.plot_pixel_dot(ax4c, u, v, hidden_padding=padding)
    ax4c.imshow(img)
    ax4c.set_xticks([])
    ax4c.set_yticks([])

    patches = patch_utils.get_channel_pixel_patch(layer_name=default_layer,channel=55, value=5, coord=fail)
    with hook.use_patches(patches):
        patched_vfield = viz.vector_field(venv, hook.network)
    img = viz.plot_vf_mpp(patched_vfield, ax=axd4['fail'], save_img=False)

    u, v = linear_mapping(fail)
    u, v = inner_grid_coords_from_full_grid_coords((u,v))
    padding = maze.get_padding(maze.get_inner_grid_from_seed(seed))
    viz.plot_pixel_dot(axd4['fail'], u, v, hidden_padding=padding)
    axd4['fail'].imshow(img)


    img = viz.plot_vf_mpp(patched_vfield, ax=ax4d, save_img=False)
    viz.plot_pixel_dot(ax4d, u, v, hidden_padding=padding)
    ax4d.imshow(img)
    ax4d.set_xticks([])
    ax4d.set_yticks([])

    fig4a.savefig('playground/paper_graphics/visualizations/fig_4a.pdf', bbox_inches="tight", format='pdf')
    fig4b.savefig('playground/paper_graphics/visualizations/fig_4b.pdf', bbox_inches="tight", format='pdf')
    fig4c.savefig('playground/paper_graphics/visualizations/fig_4c.pdf', bbox_inches="tight", format='pdf')
    fig4d.savefig('playground/paper_graphics/visualizations/fig_4d.pdf', bbox_inches="tight", format='pdf')


    #plt.show()
    #plt.savefig('playground/paper_graphics/visualizations/fig_4.pdf', bbox_inches="tight", format='pdf')
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
        figsize=(AX_SIZE * 4, AX_SIZE*1.25),
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

    font_dict = {'fontname': 'Times New Roman', 'fontsize': 24}

    axd5['success_a'].set_title('Maze A', pad=8, **font_dict)#, fontproperties=font_prop)
    axd5['success_b'].set_title('Maze B', pad=8, **font_dict)
    axd5['fail_a'].set_title('Maze C', pad=8, **font_dict)
    axd5['fail_b'].set_title('Maze D', pad=8, **font_dict)

    plt.savefig('playground/paper_graphics/visualizations/fig_5.pdf', bbox_inches="tight", format='pdf')
    #plt.show()

#fig_5()

# ------------------------------------------- fig 5 with subfigures ----------------------------------------------

def fig_5_subfigures():

    fig = plt.figure(figsize=(AX_SIZE * 4, AX_SIZE * 1.15), layout='constrained')
    subfigs = fig.subfigures(1, 2, wspace=0.05)

    historic_a = 2
    historic_b = 543
    different_a = 128
    different_b = 167

    font_dict = {'fontname': 'Times New Roman', 'fontsize': 24}

    subfigs[0].suptitle('Historic Goal Location', **font_dict)
    subfigs[1].suptitle('Different Goal Location', **font_dict)
    # subfigs[2].suptitle('Patched Activations', fontsize=18, fontproperties=font_prop)
    # subfigs[3].suptitle('Patched MPP', fontsize=18, fontproperties=font_prop)

    # Historic A
    ax0 = subfigs[0].subplots(1, 2)
    venv = create_venv(1,historic_a, 1)
    vf = viz.vector_field(venv, policy)

    img = viz.plot_vf_mpp(vf, ax=ax0[0], save_img=False)
    ax0[0].imshow(img)

    # Historic B
    venv = create_venv(1,historic_b,1)
    vf = viz.vector_field(venv, policy)

    img = viz.plot_vf_mpp(vf, ax=ax0[1], save_img=False)
    ax0[1].imshow(img)

    # Different A
    ax1 = subfigs[1].subplots(1, 2)
    venv = create_venv(1,different_a,1)
    vf = viz.vector_field(venv, policy)

    img = viz.plot_vf_mpp(vf, ax=ax1[0], save_img=False)
    ax1[0].imshow(img)

    # Different B
    venv = create_venv(1,different_b,1)
    vf = viz.vector_field(venv, policy)

    img = viz.plot_vf_mpp(vf, ax=ax1[1], save_img=False)
    ax1[1].imshow(img)

    plt.savefig('playground/paper_graphics/visualizations/fig_5_subfigures.pdf', bbox_inches="tight", format='pdf')

#fig_5_subfigures()

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
    axdx1['seed_48_channel_cheese'].set_title("(a): Seed 48, Cheese", pad=8, fontsize=24)#, font="Times New Roman")
    plot_heatmap(48, "cheese", axdx1['seed_48_channel_cheese'])

    axdx1['seed_48_channel_55'].set_title("(b): Seed 48, Channel 55", pad=8, fontsize=24)#, font="Times New Roman")
    plot_heatmap(48, "55", axdx1['seed_48_channel_55'])

    axdx1['seed_48_channel_all'].set_title("(c): Seed 48, Channel All", pad=8, fontsize=24)#, font="Times New Roman")
    plot_heatmap(48, "all", axdx1['seed_48_channel_all'])

    axdx1['seed_48_channel_none'].set_title("(d): Seed 48, Base", pad=8, fontsize=24)#, font="Times New Roman")
    plot_heatmap(48, "normal", axdx1['seed_48_channel_none'])

    plt.show()
    #plt.savefig('playground/paper_graphics/visualizations/fig_x1b.pdf', bbox_inches="tight", format='pdf')


#fig_x1b()

def fig_x1bv2():
    figx1, axdx1 = plt.subplot_mosaic(
        [['seed_0_channel_55', 'seed_0_channel_all', 'seed_48_channel_55', 'seed_48_channel_all']],
        figsize=(AX_SIZE * 4, AX_SIZE*1.5), #increase y to fit titles
        tight_layout=True,
    )

    # loading the data

    #give title to the axes
    axdx1['seed_0_channel_55'].set_title("(a): Seed 0\n Channel 55", pad=8, fontsize=24)#, font="Times New Roman")
    plot_heatmap(0, "55", axdx1['seed_0_channel_55'])

    axdx1['seed_0_channel_all'].set_title("(b): Seed 0\n All Channels", pad=8, fontsize=24)#, font="Times New Roman")
    plot_heatmap(0, "all", axdx1['seed_0_channel_all'])

    axdx1['seed_48_channel_55'].set_title("(c): Seed 48\n Channel 55", pad=8, fontsize=24)#, font="Times New Roman")
    plot_heatmap(48, "55", axdx1['seed_48_channel_55'])

    axdx1['seed_48_channel_all'].set_title("(d): Seed 48\n All Channels", pad=8, fontsize=24)#, font="Times New Roman")
    plot_heatmap(48, "all", axdx1['seed_48_channel_all'])

    #plt.show()
    plt.savefig('playground/paper_graphics/visualizations/fig_x1bv2.pdf', bbox_inches="tight", format='pdf')

#fig_x1bv2()

# -------------------------------------------------- fig 6 ----------------------------------------------------
"""
Alright so instead of the below retargeting analyses I'm going to find better heatmaps. I want heatmaps that show
different things based off of channel size.
Rigth now I'm thinking of doing 2 heatmaps for smaller mazes and 2 more for medium size mazes.
Actually, let me read the Overleaf and see what might work best there.
Alright, have each of the mazes above be by seed too but want to look at the magnitude.

all from channel 55 - scratch that. Alex's data only has magnitude 5.5 for channel 55 only.
Do not want to rerun it if I can help it. Let me do all targets.
Okay actually I do need to rerun it some. The effective channels only have 2.3, all only has 1.0, 55 only has 5.5

Well ... perhaps can do as follows
Then I don't need to amend my heatmap util function either. Fastest route.

First visualize the first 100 seeds.

So heatmap_a is small, effective at 2.3
heatmap_b is small, 55 at 5.5
heatmap_c is large, effective at 2.3
heatmap_d is large, 55 at 5.5
"""
def fig_6():



    fig6, axd6 = plt.subplot_mosaic(
        [['small_low', 'small_high', 'large_low', 'large_high']],
        figsize=(AX_SIZE * 4, AX_SIZE*1.5), #increase y to fit titles
        tight_layout=True,
    )
    #give title to the axes
    axd6['small_low'].set_title("Activation 2.3, Effective Channels", fontsize=14)#, font="Times New Roman")
    plot_heatmap(27, "effective", axd6['small_low'])

    axd6['small_high'].set_title("Activation 5.5, Channel 55", fontsize=14)#, font="Times New Roman")
    plot_heatmap(27, "55", axd6['small_high'])

    axd6['large_low'].set_title("Activation 2.3, Effective Channels", fontsize=14)#, font="Times New Roman")
    plot_heatmap(45, "effective", axd6['large_low'])

    axd6['large_high'].set_title("Activation 5.5, Cheese Channel", fontsize=14)#, font="Times New Roman")
    plot_heatmap(45, "55", axd6['large_high'])

    #plt.show()
    plt.savefig('playground/paper_graphics/visualizations/fig_6.pdf', bbox_inches="tight", format='pdf')

#fig_6()

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

    # # data hardcoded. starting with 3x3 up to 25x25.
    # heatmap_avg_per_size_all = [0.799889956501738, 0.7775709501219863, 0.7797912188517916, 0.7588451193811928, 0.7406733268789705, 0.6959504505843259, 0.6425305479030984, 0.6545479388988915, 0.5824753527436064, 0.5832421962896295, 0.6167109448569562, 0.5299525163283456]
    # heatmap_avg_per_size_effective = [0.7608124070512822, 0.7041304093124999, 0.76901413325, 0.7024797808854166, 0.7087437639591837, 0.7041753128385418, 0.6503412165784831, 0.6596821581796875, 0.6037907865088384, 0.6189248616309523, 0.6371125867599068, 0.5756655757924108]
    # heatmap_avg_per_size_55 = [0.700428774165744, 0.7370594442260178, 0.7010945262579837, 0.6721659938343914, 0.6751070216559201, 0.647880930832588, 0.5939934102651221, 0.6109084206556188, 0.540930443222974, 0.5470496808198526, 0.5846496184297417, 0.5023088997834109]
    # ratio_avg_per_size_effective = [1.8489007321794872, 3.4623849059999996, 1.7517762465, 2.7921403650173606, 1.8093523609387756, 2.8955814034765623, 2.458575891675485, 2.8985679961171877, 2.566675839248737, 2.5696256181845234, 3.013872167534965, 1.8687405686830356]
    #
    # x_values = np.linspace(0, 1, len(heatmap_avg_per_size_all))
    # #x_labels = ['3x3', '5x5', '7x7', '9x9', '11x11', '13x13', '15x15', '17x17', '19x19', '21x21', '23x23', '25x25']
    # x_labels = ['3', '5', '7', '9', '11', '13', '15', '17', '19', '21', '23', '25']
    #
    #
    # #initial box plots
    # fig, ax = plt.subplots()#1, 2, figsize=(AX_SIZE * 2, AX_SIZE), tight_layout=True)
    # #ax.boxplot([heatmap_avg_per_size_all.values(), heatmap_avg_per_size_effective.values(), heatmap_avg_per_size_55.values()], positions = [1, 2, 3], widths = 0.6)
    # ax.plot(x_values, heatmap_avg_per_size_all, marker='o', markersize=6, color="black")
    # ax.plot(x_values, heatmap_avg_per_size_effective, marker='x', markersize=6, color="red")
    # ax.plot(x_values, heatmap_avg_per_size_55, marker='+', markersize=6, color="green")
    #
    # ax.set_xticks(x_values)
    # ax.set_xticklabels(x_labels)
    #
    # plt.savefig('playground/paper_graphics/visualizations/heat_map_avg_bad_idea.png', bbox_inches="tight", format='png')
    #
    # fig, ax = plt.subplots()
    # ax.plot(x_values, ratio_avg_per_size_effective, marker='x', markersize=6, color="red")
    # ax.set_xticks(x_values)
    # ax.set_xticklabels(x_labels)
    # plt.savefig('playground/paper_graphics/visualizations/ratio_avg_also_not_great.png', bbox_inches="tight", format='png')
    #
    #
    # # Alex had an idea to try plotting path distance vs. ratio. Try that here.
    # # ratio only exists for effective, cheese, and normal. Just try effective channels for now to get an idea
    #
    # # 50 steps
    # path_distance_ratios = {i: [] for i in range(100)}
    #
    # for seed in range(100):
    #     data = heatmap_data_by_seed_and_prob_type(seed, "effective")
    #     path_distance = data['d_to_coord']
    #     ratios = data['ratio']
    #
    #     if len(path_distance) != len(ratios):
    #         raise Exception("path distance and ratios are not the same length")
    #     for i in range(len(path_distance)):
    #         # print(path_distance)
    #         # print("values: ", path_distance.values)
    #         # print(path_distance[i][1])
    #         # print(ratios[i][1])
    #         path_distance_ratios[path_distance.values[i]].append(ratios.values[i])
    #
    # #     print()
    # # print()
    #
    # #alright, now average over all the values for each ratio
    # for i in path_distance_ratios.keys():
    #     #if isinstance(path_distance_ratios[i], list):
    #     if len(path_distance_ratios[i]) != 0:
    #         path_distance_ratios[i] = np.mean(path_distance_ratios[i])
    #     else:
    #         path_distance_ratios[i] = 0
    #
    # print()
    #
    #
    #
    # fig, ax = plt.subplots()
    # ax.plot(path_distance_ratios.keys(), path_distance_ratios.values(), marker='x', markersize=6, color="red")
    # #ax.set_xticks(x_values)
    # #ax.set_xticklabels(x_labels)
    # plt.savefig('playground/paper_graphics/visualizations/ratio_v_path_distance.png', bbox_inches="tight", format='png')
    #


    # Redo this plot for probability of successful retargeting vs. distance from top right path
    tr_path_by_probs = {i: [] for i in range(51)}

    for seed in range(100):
        data = heatmap_data_by_seed_and_prob_type(seed, "all")
        tr_path_dist = data['topright_path_divergence']
        probs = data['probability']

        if len(tr_path_dist) != len(probs):
            raise Exception("path distance and ratios are not the same length")
        for i in range(len(tr_path_dist)):
            # print(path_distance)
            # print("values: ", path_distance.values)
            # print(path_distance[i][1])
            # print(ratios[i][1])
            if tr_path_dist.values[i] <= 50:
                tr_path_by_probs[tr_path_dist.values[i]].append(probs.values[i])
            #path_distance_ratios[path_distance.values[i]].append(ratios.values[i])

    #     print()
    # print()

    #alright, now average over all the values for each ratio
    for i in tr_path_by_probs.keys():
        #if isinstance(path_distance_ratios[i], list):
        if len(tr_path_by_probs[i]) != 0:
            tr_path_by_probs[i] = np.mean(tr_path_by_probs[i])
        else:
            tr_path_by_probs[i] = 0

    print()



    fig, ax = plt.subplots()
    ax.plot(tr_path_by_probs.keys(), tr_path_by_probs.values(), marker='x', markersize=6, color="red")
    ax.set_xlabel("Distance from Top Right Path", fontsize=14)
    ax.set_ylabel("Probability of Successful Retargeting", fontsize=14)
    #ax.set_xticks(x_values)
    #ax.set_xticklabels(x_labels)
    plt.savefig('playground/paper_graphics/visualizations/tr_path_dist_v_retargetability.pdf', bbox_inches="tight", format='pdf')
    #plt.show()


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
    #fig, ax = plt.subplots(1, 4, figsize=(15, 5))

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
    # vfields = [
    #     pickle.load(open(f, "rb"))
    #     for f in glob("experiments/statistics/data/vfields/cheese/seed-0*.pkl")
    # ]
    # probs_original_same, probs_patched_same = vfield_stats.get_probs_original_and_patched(
    #     vfields, coeff=-1.0
    # )
    # probs_original_same, probs_patched_same = probs_original_same[:, 0], probs_patched_same[:, 0]

    # plot twist this is supposed to be same cheese location.

    # so my previous attempt to get data was ill conceived. I only have vfield stats for the first 100 seeds.
    # so I should get the cheese loc for each and print them out.

    # alright, out of the first 100 seeds the cheese location that is most prevalent (in outer grid)
    # is (12, 13), with 6 occurences at seeds [1, 11, 18, 29, 30, 38]

    vfields = []
    #for i in [1, 11, 18, 29, 30, 38]:
    for f in glob(f"experiments/statistics/data/vfields/cheese/seed-1_*.pkl"):
        vfields.append(pickle.load(open(f, "rb")))
    for f in glob(f"experiments/statistics/data/vfields/cheese/seed-11_*.pkl"):
        vfields.append(pickle.load(open(f, "rb")))
    for f in glob(f"experiments/statistics/data/vfields/cheese/seed-18_*.pkl"):
        vfields.append(pickle.load(open(f, "rb")))
    for f in glob(f"experiments/statistics/data/vfields/cheese/seed-29_*.pkl"):
        vfields.append(pickle.load(open(f, "rb")))
    for f in glob(f"experiments/statistics/data/vfields/cheese/seed-30_*.pkl"):
        vfields.append(pickle.load(open(f, "rb")))
    for f in glob(f"experiments/statistics/data/vfields/cheese/seed-38_*.pkl"):
        vfields.append(pickle.load(open(f, "rb")))


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
    # for i, (data_var1, data_var2) in enumerate([(probs_original_plus_1, probs_patched_plus_1),
    #                                             (probs_original_minus_1, probs_patched_minus_1),
    #                                             (probs_original_same, probs_patched_same),
    #                                             (probs_small_seed_orig, probs_small_seed_patched)]):
    #
    #     # mean = np.mean(data_var1), np.mean(data_var2)
    #
    #     # Plot the boxes with different colors
    #     bp = ax[i].boxplot([data_var1, data_var2], labels=['Var1', 'Var2'], patch_artist=True)
    #
    #     # Set colors for the boxes
    #     for count, patch in enumerate(bp['boxes']):
    #         #print(patch)
    #         if count == 0:
    #             patch.set_facecolor('lightblue')  # Left part of the box
    #         elif count == 1:
    #             patch.set_facecolor('orange')
    #         patch.set_edgecolor('black')
    #
    #     # TODO - these markers aren't at the actual mean. Try changing the boxplot data struct directly
    #     # Plot markers for mean without lines
    #     # ax[i].plot([1, 2], mean, 'kD', markersize=10, label='Mean', linestyle='None')
    #
    #     # if i == 1:
    #     #     ax[i].set_title(f'Boxplot {i+1}')
    #     # elif i == 2:
    #     #     ax[i].set_title(f'Boxplot {i+1}')
    #     # elif i == 3:
    #     #     ax[i].set_title(f'Boxplot {i+1}')
    #     # else i == 4:
    #     #     ax[i].set_title(f'Boxplot {i+1}')
    #
    # ax[0].set_title("Coeff = 1.0")
    # ax[1].set_title("Coeff = -1.0")
    # ax[2].set_title("Coeff = -1.0, Same Seed")
    # ax[3].set_title("Coeff = -1.0, Small Seeds")
    #
    # # Hide x-tick labels and ticks for all boxplots except the first one
    # for i in range(0, len(ax)):
    #     ax[i].set_xticklabels([])
    #     ax[i].tick_params(axis='x', length=0)

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

    fig, axs = plt.subplots(1, 2, figsize=(15, 5), tight_layout=True)

    boxplot_data_cheese = [probs_original_plus_1, probs_patched_plus_1,
                    probs_patched_minus_1]#,
                    #probs_small_seed_orig, probs_small_seed_patched]
    labels = ['P(Cheese|Decision Square)', 'Coeff = 1.0',
              'Coeff = -1.0']#,
              #'Coeff = -1.0, Small Seed', 'Coeff = -1.0, Small Seeds']
    colors = ['lightblue', 'orange', 'red']#, 'pink', 'green']
    medianprops = dict(linestyle='-', linewidth=2.5, color='black')

    bp1 = axs[0].boxplot(boxplot_data_cheese, medianprops=medianprops, patch_artist=True) #labels=labels)

    for box in bp1:
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)

    orig_patch = mpatches.Patch(color='lightblue', label='Original')
    pos_patch = mpatches.Patch(color='orange', label='Added')
    neg_patch = mpatches.Patch(color='red', label='Subtracted')
    handles = [orig_patch, pos_patch, neg_patch]

    # Display the legend below the subplots
    #fig.legend(handles=[orig_patch, patch_patch], loc='lower center', ncol=2, fontsize=10)
    #fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.3))

    axs[0].set_xticks([])
    #axs[0].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],fontsize=30)
    axs[0].tick_params(labelsize=14)
    axs[0].set_ylabel("P(Cheese | Decision Square)", labelpad=5, fontsize=20)
    axs[0].set_title("Cheese Vector", pad=10, fontsize=30)
    axs[0].legend(handles=handles, loc='upper right', ncol=1, fontsize=18, handlelength=1)
    #axs[0].legend(bp1['boxes'][0], ['Original', 'Added Vector', 'Subtracted Vector'], loc='upper right', ncol=3, fontsize=18)

    # here add stuff from top right vector to place side by side

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

    orig_tr, pos_tr, min_tr = probs_original_plus_1, probs_patched_plus_1, probs_patched_minus_1

    boxplot_data_tr = [probs_original_plus_1, probs_patched_plus_1, probs_patched_minus_1]

    bp2 = axs[1].boxplot(boxplot_data_tr, medianprops=medianprops, patch_artist=True) #labels=labels)

    for box in bp2:
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)

    axs[1].set_ylabel("P(Top Right | Decision Square)", labelpad=5, fontsize=20)
    axs[1].set_xticks([])
    axs[1].tick_params(labelsize=14)
    axs[1].set_title("Top Right Vector", pad=10, fontsize=30)

    #fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.3))
    #fig.legend([bp1['boxes'][0], bp2['boxes'][0]], handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.1),
    #           fancybox=True, shadow=True, ncol=2)
    #fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=18, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0,1.1,1,1]) # left bottom right top

    # Display the plot
    #plt.show()
    #plt.savefig('playground/paper_graphics/visualizations/cheese_vector.pdf', bbox_inches="tight", format='pdf')
    plt.savefig('playground/paper_graphics/visualizations/vector_figure.pdf', bbox_inches="tight", format='pdf')


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

# finding cheese loc with most occurences in the limited vfield data we have

# dict_of_seeds = {str(i): [0, 0] for i in range(100)}
# dict_of_seeds_2 = {}
# for i in range(100):
#     grid = maze.get_full_grid_from_seed(i)
#     cheese_pos = maze.get_cheese_pos(grid)
#     # print(dict_of_seeds[str(i)])
#     # dict_of_seeds[str(i)][0] = cheese_pos
#     # dict_of_seeds[str(i)][1] += 1
#     if cheese_pos not in dict_of_seeds_2.keys():
#         dict_of_seeds_2[cheese_pos] = [1, i]
#     else:
#         dict_of_seeds_2[cheese_pos][0] += 1
#     # print(dict_of_seeds_2)
# max = 0
# cheese_pos = 0
# seeds = []
# for key in dict_of_seeds_2.keys():
#     if dict_of_seeds_2[key][0] > max:
#         max = dict_of_seeds_2[key][0]
#         cheese_pos = key
#
# for i in range(100):
#     grid = maze.get_full_grid_from_seed(i)
#     cheese_pos_test = maze.get_cheese_pos(grid)
#     if cheese_pos_test == cheese_pos:
#         seeds.append(i)
# print(cheese_pos, max, seeds)

# (12, 13) position of cheese in outer grid has 6 occurences.

# --------------------------------------- base heatmap small figure ----------------------------------------------
def base_heatmap_small_fig():
    fig, ax = plt.subplots()

    plot_heatmap(48, "normal", ax)

    #trying png to get latex wrapfigure package to work. - yes. it can't handle PDFs for some reason
    plt.savefig('playground/paper_graphics/visualizations/base_heatmap.png', bbox_inches="tight", format='png')

#base_heatmap_small_fig()

def table_for_appendix_b():

    # this was a dud

    # Data for the table
    data = [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Hide axes
    ax.axis('off')

    colLabels = ['Size of random region', 'Steps between cheese \n and decision square',
                 'Euclidean distance between \n cheese and decision square',
                 ]

    # Create the table
    table = ax.table(cellText=data, loc='center', cellLoc='center', colLabels=colLabels)

    # Format the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)  # Adjust the size if needed

    plt.show()

#table_for_appendix_b()