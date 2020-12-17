#!/usr/bin/env python3

from torch_tem.world import World
from torch_tem.plot import plot_map, plot_actions, plot_walk
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

env = World('./torch_tem/envs/first-experiment4x4.json')

walks = env.generate_walks(walk_length=10, n_walk=1)
pprint(walks)

ax1 = plot_map(env, values=np.ones(env.n_locations), do_plot_actions=False)
# plt.show(ax)

# ax = plot_actions(env)
# plt.show(ax)

ax = plot_walk(env, walks[0], max_steps=None, n_steps=1, ax=ax1)
plt.show(ax)
