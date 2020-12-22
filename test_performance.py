import numpy as np
import torch
from pprint import pprint
import time

from analyse import performance, location_accuracy, rate_map
from my_utils import load_model
import world


def test_performance():
    tem, params, envs = load_model()

    params['n_rollout'] = 100

    adam = torch.optim.Adam(tem.parameters(), lr = params['lr_max'])

    # Make set of environments: one for each batch, randomly choosing to use shiny objects or not
    environments = [world.World(graph, randomise_observations=True, shiny=None, first_experiment=True) for graph in envs]
    # Initialise whether a state has been visited for each world
    visited = [[False for _ in range(env.n_locations)] for env in environments]
    # And make a single walk for each environment, where walk lengths can be any between the min and max length to de-sychronise world switches
    walks = [env.generate_walks(walk_length=params['n_rollout'] * 25, n_walk=1)[0] for env in environments]
    # Initialise the previous iteration as None: we start from the beginning of the walk, so there is no previous iteration yet
    prev_iter = None

    # print(walks)

    loss_weights = params['loss_weights']

    # Get start time for function timing
    start_time = time.time()

    forwards = []
    while all([len(walk) >= params['n_rollout'] for walk in walks]):

        # Make an empty chunk that will be fed to TEM in this backprop iteration
        chunk = []
        # For each environment: fill chunk by popping the first batch_size steps of the walk
        for env_i, walk in enumerate(walks):             
            # Now pop the first n_rollout steps from this walk and append them to the chunk
            for step in range(params['n_rollout']):
                # For the first environment: simply copy the components (g, x, a) of each step
                if len(chunk) < params['n_rollout']:
                    chunk.append([[comp] for comp in walk.pop(0)])
                # For all next environments: add the components to the existing list of components for each step
                else:
                    for comp_i, comp in enumerate(walk.pop(0)):
                        chunk[step][comp_i].append(comp)
        # Stack all observations (x, component 1) into tensors along the first dimension for batch processing
        for i_step, step in enumerate(chunk):
            chunk[i_step][1] = torch.stack(step[1], dim=0)

        # Forward-pass this walk through the network
        forward = tem(chunk, prev_iter)

        # Accumulate loss from forward pass
        loss = torch.tensor(0.0, device=params['device'])
        # Make vector for plotting losses
        plot_loss = 0
        # Collect all losses 
        for step in forward:            
            # Make list of losses included in this step
            step_loss = []        
            # Only include loss for locations that have been visited before
            for env_i, env_visited in enumerate(visited):
                if env_visited[step.g[env_i]['id']]:
                    step_loss.append(loss_weights * torch.stack([l[env_i] for l in step.L]))
                else:
                    env_visited[step.g[env_i]['id']] = True
            # Stack losses in this step along first dimension, then average across that dimension to get mean loss for this step
            step_loss = torch.tensor(0) if not step_loss else torch.mean(torch.stack(step_loss, dim=0), dim=0)
            # Save all separate components of loss for monitoring
            plot_loss = plot_loss + step_loss.detach().cpu().numpy()
            # And sum all components, then add them to total loss of this step
            loss = loss + torch.sum(step_loss)

        # Reset gradients
        adam.zero_grad()
        # Do backward pass to calculate gradients with respect to total loss of this chunk
        loss.backward(retain_graph=True)
        # Then do optimiser step to update parameters of model
        adam.step()
        # Update the previous iteration for the next chunk with the final step of this chunk, removing all operation history
        prev_iter = [forward[-1].detach()]
        
        # Compute model accuracies
        acc_p, acc_g, acc_gt = np.mean([[np.mean(a) for a in step.correct()] for step in forward], axis=0)
        acc_p, acc_g, acc_gt = [a * 100 for a in (acc_p, acc_g, acc_gt)]        
        # Log progress
        print('Finished backprop iter {:d} in {:.2f} seconds.'.format(1, time.time() - start_time))
        print('Loss: {:.2f}. <p_g> {:.2f} <p_x> {:.2f} <x_gen> {:.2f} <x_g> {:.2f} <x_p> {:.2f} <g> {:.2f} <reg_g> {:.2f} <reg_p> {:.2f}'.format(loss.detach().cpu().numpy(), *plot_loss))
        print('Accuracy: <p> {:.2f}% <g> {:.2f}% <gt> {:.2f}%'.format(acc_p, acc_g, acc_gt))
        print('Parameters: <max_hebb> {:.2f} <eta> {:.2f} <lambda> {:.2f} <p2g_scale_offset> {:.2f}'.format(np.max(np.abs(prev_iter[0].M[0].cpu().numpy())), tem.hyper['eta'], tem.hyper['lambda'], tem.hyper['p2g_scale_offset']))
        print('Weights:' + str([w for w in loss_weights.cpu().numpy()]))
        print()

        # all_correct, all_location_frac, all_action_frac, all_locations_stats = performance(forward, tem, environments)

        # accuracy_from, accuracy_to = location_accuracy(forward, tem, environments)

        # print('All correct')
        # print(all_correct)
        # print('All location fraction')
        # print(all_location_frac)
        # # print('All action fractions')
        # # print(all_action_frac)
        # print('Accuracy from')
        # print(accuracy_from)
        # print('Accuracy to')
        # print(accuracy_to)
        # print('All location stats')
        # print(all_locations_stats)

        forwards += forward

    all_g, all_p = rate_map([step.detach() for step in forwards[5:]], tem, environments)

    # pprint(all_g[0][0][:, 0])

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(nrows=5, ncols=6, figsize=(9, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

    print(all_g[0][0].shape)
    print(all_p[0][0].shape)

    for ax, g in zip(axs.flat, list(range(30))):
        ax.imshow(all_g[0][0][:, g].reshape(4, 5), interpolation='hanning', cmap='hot')
        ax.set_title(str(g))

    plt.tight_layout()
    plt.show()


def main():
    test_performance()



if __name__ == "__main__":
    main()




