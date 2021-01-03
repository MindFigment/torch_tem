import numpy as np
import torch
from pprint import pprint
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import click
from datetime import datetime
import glob
import copy

import analyse
from my_utils import load_model
import world
import plot
from my_plot import plot_rate_maps


# @click.command()
# @click.option('--load', default=False, type=bool)
# @click.option('--date', default=datetime.today().strftime('%Y-%m-%d'), type=str)
# @click.option('--run', default='0', type=str)
# @click.argument('envs', nargs=-1)
def test_first_experiment():

    date = '2020-12-17'
    run = '0'
    envs = ['first-experiment4x4']

    tem, params, _, _ = load_model(date, run, envs)
    tem.eval()

    # Make list of all the environments that this model was trained on
    envs = ['./envs/' + env + '.json' for env in envs]
    # Set the number of walks to execute in parallel (batch size)
    n_walks = len(envs)
    # Select environments from the environments included in training
    environments = [world.World(graph, randomise_observations=True, shiny=None) for env_i, graph in enumerate(np.random.choice(envs, n_walks))]
    # Determine the length of each walk
    walk_len = np.median([env.n_locations * 5 for env in environments]).astype(int)
    # And generate walks for each environment
    walks = [env.generate_walks(walk_len, 1)[0] for env in environments]

    # print(walks)
    # print(walks[0][0])

    # for i, env in enumerate(environments):
    #     print(f'Env{i + 1} symbol locations: {env.symbol_locations}')

    # Generate model input from specified walk and environment: group steps from all environments together to feed to model in parallel
    model_input = [[[[walks[i][j][k]][0] for i in range(len(walks))] for k in range(3)] for j in range(walk_len)]
    for i_step, step in enumerate(model_input):
        model_input[i_step][1] = torch.stack(step[1], dim=0)

    # print(len(model_input), len(model_input[0]), model_input[0])

    # Run a forward pass through the model using this data, without accumulating gradients
    with torch.no_grad():
        forward = tem(model_input, prev_iter=None)

    # print(forward)

    # # Decide whether to include stay-still actions as valid occasions for inference
    # include_stay_still = False
    # # Choose which environment to plot
    env_to_plot = 0
    # Choose which grid or place cell module to plot
    module_to_plot = 0
    # # And when averaging environments, e.g. for calculating average accuracy, decide which environments to include
    # envs_to_avg = [True]

    # # Compare trained model performance to a node agent and an edge agent
    # correct_model, correct_node, correct_edge = analyse.compare_to_agents(forward, tem, environments, include_stay_still=include_stay_still)
    # print('Correct model: {:.4f} ({}/{})'.format(np.mean(correct_model[0]), np.sum(correct_model[0]), len(correct_model[0])))
    # print('Correct model: {:.4f} ({}/{})'.format(np.mean(correct_node[0]), np.sum(correct_node[0]), len(correct_node[0])))
    # print('Correct model: {:.4f} ({}/{})'.format(np.mean(correct_edge[0]), np.sum(correct_edge[0]), len(correct_edge[0])))

    # # Analyse occurrences of zero-shot inference: predict the right observation arriving from a visited node with a new action
    # zero_shot = analyse.zero_shot(forward, tem, environments, include_stay_still=include_stay_still)
    # print('Zero shot: {:.4f}'.format(np.sum(zero_shot[0]) / len(zero_shot[0])))

    # Generate occupancy maps: how much time TEM spends at every location
    occupation = analyse.location_occupation(forward, tem, environments)
    print('Occupation percent: ' + 
          ', '.join('{:.2f}%'.format(occ) for occ in occupation[env_to_plot] / np.sum(occupation[env_to_plot]) * 100) +
          ' (1% = {:.2f} visits)'.format(np.sum(occupation[env_to_plot])/ 100))

    # Generate rate maps
    g, p = analyse.rate_map(forward, tem, environments)

    print(len(g), len(g[env_to_plot]), g[env_to_plot][module_to_plot].shape)
    print(len(p), len(p[env_to_plot]), p[env_to_plot][module_to_plot].shape)

    #################
    #################
    #################

    plot_rate_maps(g[env_to_plot][module_to_plot], 
                   environments[env_to_plot].symbol_locations,
                   environments[env_to_plot].height + 1,
                   environments[env_to_plot].width)

    #################
    #################
    #################

    # Calculate accuracy leaving from and arriving to each location
    from_acc, to_acc = analyse.location_accuracy(forward, tem, environments)

    print('From acc: ' + ', '.join('{:.2f}%'.format(acc * 100) for acc in from_acc[env_to_plot]))
    print('To acc  : ' + ', '.join('{:.2f}%'.format(acc * 100) for acc in to_acc[env_to_plot]))

    # # Plot results of agent comparison and zero-shot inference analysis
    # filt_size = 41
    # plt.figure()
    # plt.plot(analyse.smooth(np.mean(np.array([env for env_i, env in enumerate(correct_model) if envs_to_avg[env_i]]),0)[1:], filt_size), label='tem')
    # plt.plot(analyse.smooth(np.mean(np.array([env for env_i, env in enumerate(correct_node) if envs_to_avg[env_i]]),0)[1:], filt_size), label='node')
    # plt.plot(analyse.smooth(np.mean(np.array([env for env_i, env in enumerate(correct_edge) if envs_to_avg[env_i]]),0)[1:], filt_size), label='edge')
    # plt.ylim(0, 1.05)
    # plt.legend()
    # plt.title('Zero-shot inference: ' + str(np.mean([np.mean(env) for env_i, env in enumerate(zero_shot) if envs_to_avg[env_i]]) * 100) + '%')
    # plt.show()

    # Plot accuracy separated by location
    plt.figure()
    ax = plt.subplot(1,2,1)
    plot.plot_map(environments[env_to_plot], np.array(to_acc[env_to_plot]), ax)
    ax.set_title('Accuracy to location')
    ax = plt.subplot(1,2,2)
    plot.plot_map(environments[env_to_plot], np.array(from_acc[env_to_plot]), ax)
    ax.set_title('Accuracy from location')

    # # Plot occupation per location, then add walks on top
    # ax = plot.plot_map(environments[env_to_plot], np.array(occupation[env_to_plot])/sum(occupation[env_to_plot])*environments[env_to_plot].n_locations, 
    #                 min_val=0, max_val=2, ax=None, shape='square', radius=1/np.sqrt(environments[env_to_plot].n_locations))
    # ax = plot.plot_walk(environments[env_to_plot], walks[env_to_plot], ax=ax, n_steps=max(1, int(len(walks[env_to_plot])/500)))
    # plt.title('Walk and average occupation')


if __name__ == '__main__':
    test_first_experiment()




