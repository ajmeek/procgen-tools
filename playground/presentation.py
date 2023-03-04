# %%
%reload_ext autoreload
%autoreload 2

# %%
try:
    import procgen_tools
except ImportError or ModuleNotFoundError:
    get_ipython().run_line_magic(magic_name='pip', line='install -U git+https://github.com/ulissemini/procgen-tools')

from procgen_tools.utils import setup

setup() # create directory structure and download data 

from procgen_tools.imports import *
from procgen_tools import visualization, patch_utils, maze


# %% Setup code for the rest of the notebook
cheese_channels = [77, 113, 44, 88, 55, 42, 7, 8, 82, 99] 
effective_channels = [77, 113, 88, 55, 8, 82, 89]

SAVE_DIR = 'playground/visualizations'
AX_SIZE = 6
gif_dir = f'{SAVE_DIR}/pixel_gifs'

def save_channel_patch_image(seed : int, value : float, row : int, col : int, channels : List[int], default : float = None):
    """ Save an image of the maze with a pixel patch on the given channel at (row, col) in the block2.res1.resadd_out channel. """
    venv = patch_utils.get_cheese_venv_pair(seed=seed)
    padding = maze.get_padding(maze.get_inner_grid_from_seed(seed))

    patch = patch_utils.combined_pixel_patch(layer_name=default_layer, channels=channels, value=value, coord=(row, col), default=default) 
    fig, axs, info = patch_utils.compare_patched_vfields(venv, patch, hook, render_padding=False, ax_size=AX_SIZE, show_original=False) # Don't plot the original vfields
    
    channel_str = ', '.join(str(channel) for channel in channels)
    fig.suptitle(f'Synthetic patch on channel{"s" if len(channels) > 1 else ""} {channel_str}', fontsize=20)
    # Ensure the title is large and close to the figures
    fig.subplots_adjust(top=1)

    # Draw a red pixel at the location of the patch
    visualization.plot_dots(axs, (row, col), color='cyan', hidden_padding = padding)
    save_dir = f'{gif_dir}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fname = f'{save_dir}/seed{seed}_col{col}.png'
    fig.savefig(fname)
    plt.close(fig) # TODO just render image directly, don't save to file
    
    return fname 

# %%
import imageio

def get_z_coords(seed : int, channel_coords : bool = False):
    """ Get a list of (row, col) coordinates which trace the inner grid of the maze, first running along the top edge, then along the top-right to bottom-left diagonal, then to the bottom-right. """ 
    grid = maze.get_inner_grid_from_seed(seed)
    padding = maze.get_padding(grid)
    length = grid.shape[0] # Length of maze

    start = (padding, padding)
    coords = [start]
    coords += list((start[0], start[1]+col) for col in range(1, length))
    coords += list((start[0]+coord, (start[1]+length)-coord) for coord in range(1, length))
    coords += list((start[0]+length-1, start[1]+col) for col in range(1, length))
    
    bound = maze.WORLD_DIM - padding - 1 # The last row and column of the maze, in inner grid coordinates
    assert coords[-1] == (bound, bound), f'Last coordinate {coords[-1]} must be the bottom-right corner of the maze.'

    if channel_coords: # Convert to channel coordinates
        channel_coords_converted = [visualization.get_channel_from_grid_pos(coord) for coord in coords]
        # Remove duplicates
        coords = list(dict.fromkeys(channel_coords_converted))

    return coords

def save_sweep(channels : List[int], seed : int, coords : List[Tuple[int, int]] = None, value : float = 5.6, channels_str : str = None, default : float = None):
    """ Save a sweep of the maze to a GIF, where the pixel patch moves from one coordinate to the next. 
    
    Args:
        channels: The channels to patch.
        seed: The seed of the maze.
        coords: The coordinates to sweep through. If None, the default z-shaped coordinates are used.
        value: The value to patch the channels with at the given coords.
        channels_str: The string to use in the filename. If None, the channels are used.
    """
    if coords is None:
        coords = get_z_coords(seed, channel_coords = True)

    images = []
    grid = maze.get_inner_grid_from_seed(seed)
    for row, col in coords:
        fname = save_channel_patch_image(seed=seed, value=value, row=row, col=col, channels=channels, default=default)
        images.append(imageio.imread(fname))
        # Delete the file
        os.remove(fname)

    channels_str = channels_str or f'c{str(channels)}'
    target = f'{gif_dir}/{channels_str}_seed{seed}_val{str(value).replace(".", "_")}_default{default}.gif'
    # Overwrite if necessary
    if os.path.exists(target):
        os.remove(target)
    imageio.mimsave(target, images, duration=0.5)
    print(f'Saved {target}.')

# %%
for seed in (0, 20, 60):
    for channel in [55, 42]:
        save_sweep(channels=[channel], seed=seed)

# %%
for seed in (0, 20, 60):
    save_sweep(channels=cheese_channels, seed=seed, value=.9, channels_str='cheese_channels')

# %%

# Randomly generate 5 mazes and save a GIF for each
for seed in np.random.randint(0, 100000, 5):
    save_sweep(channels=cheese_channels, seed=seed, value=.9, channels_str='cheese_channels')