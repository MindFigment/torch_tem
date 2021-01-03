import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def plot_rate_maps(cells_firing_rates, symbol_locations, height=5, width=4):
    n_cells = cells_firing_rates.shape[1]
    squared = np.sqrt(n_cells)
    n_rows = int(np.floor(squared))
    n_cols = int(np.ceil(squared))

    _, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols + 1, n_rows), subplot_kw={'xticks': [], 'yticks': []})

    print('Symbol locations: ', ', '.join(['{}'.format(str(sym_loc)) for sym_loc in symbol_locations]))
    
    # Drawing rate maps    
    for ax, i in zip(axes.flat, list(range(n_cells))):
        ax.imshow(cells_firing_rates[:, i].reshape(height, width), interpolation='hanning', cmap='Blues')
        ax.set_title(str(i))

        # Computing symbol and reward boxes
        colors = ['red', 'green', 'orange', 'yellow']
        for j, sym_loc in enumerate(symbol_locations):
            ax.add_patch(Rectangle(((sym_loc % 4) - 0.45, (sym_loc // 4) - 0.45), 1, 1, edgecolor=colors[j], facecolor='blue', fill=False, lw=1))
            ax.add_patch(Rectangle((j - 0.45, 4 - 0.45), 1, 1, edgecolor=colors[j], facecolor='blue', fill=False, lw=1))

    plt.tight_layout()
    plt.show()