import importlib
import torch
import glob
import click
import utils
import os
from datetime import datetime
import numpy as np
from collections import defaultdict
import torch
import copy


def load_model(date, run, i_start, option, envs):
    # Choose which trained model to load
    # i_start = 7999
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set all paths from existing run 
    # model_path = './models'
    # script_path = '.'
    # envs_path = './envs'
    # Set all paths from existing run 
    _, _, model_path, _, script_path, envs_path = utils.set_directories(date, run)

    # path_to_model_weights = glob.glob(f'{model_path}/*.pt')[0]
    # try:
    #     i_start = int(path_to_model_weights.split('_')[-1].split('.')[0])
    # except:
    #     i_start = int(path_to_model_weights.split('_')[-2].split('.')[0])
    
    # Load the model: use import library to import module from specified path
    model_spec = importlib.util.spec_from_file_location('model', script_path + '/model.py')
    model = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model)
    
    # Load the parameters of the model
    params = torch.load(model_path + '/params_' + str(i_start) + '_option' + str(option) + '.pt', map_location=torch.device(device))
    # But certain parameters (like total nr of training iterations) may need to be copied from the current set of parameters
    new_params = {
        'device':  device,
        'train_it': 10000
    }
    # Update those in params
    for key in new_params:
        params[key] = new_params[key]
    
    # Create a new tem model with the loaded parameters
    tem = model.Model(params).to(device)

    # Load the model weights after training
    model_weights = torch.load(model_path + '/tem_' + str(i_start) + '_option' + str(option) + '.pt', map_location=torch.device(device))
    # Set the model weights to the loaded trained model weights
    tem.load_state_dict(model_weights)
    
    # Make list of all the environments that this model was trained on
    # envs = list(glob.iglob(envs_path + '/*'))
    # envs = list(glob.iglob(envs_path + '/first-experiment4x4.json'))
    if envs:
        envs = ['./envs/' + env + '.json' for env in envs]
    else:
        envs = list(glob.iglob(envs_path + '/*'))

    # And increase starting iteration by 1, since the loaded model already carried out the current starting iteration
    i_start = i_start + 1

    return tem, params, envs, i_start


# Measure n-shot inference for this model: see if it can predict an observation following a new action to a know location (zero-shot)
def n_shot(forward, model, environments, include_stay_still=False):
    # Get the number of actions in this model
    n_actions = model.hyper['n_actions'] + model.hyper['has_static_action']
    # Track for all opportunities for n-shot inference if the predictions were correct across environments
    all_correct = []
    # Run through environments and check for n-shot inference in each of them
    for env_i, env in enumerate(environments):
        symbol_locations = env.symbol_locations
        reward_locations = env.reward_locations
        # Keep track for each location whether it has been visited
        location_visited = np.zeros(env.n_locations)
        # And for each action in each location whether it has been taken
        action_taken = np.zeros((env.n_locations, n_actions))
        # Make list that for all opportunities for n-shot inference tracks if the predictions were correct
        correct_nshot_from_symbol = defaultdict(lambda: {
                                                        #  'n-shot': 
                                                        #     {
                                                        #      'up': {10: None, 11: None, 20: None, 21: None, 22: None},
                                                        #      'down': {10: None, 11: None, 20: None, 21: None, 22: None},
                                                        #      'left': {10: None, 11: None, 20: None, 21: None, 22: None},
                                                        #      'right': {10: None, 11: None, 20: None, 21: None, 22: None},
                                                        #      'press button': {10: None, 11: None, 20: None, 21: None, 22: None}
                                                        #     },
                                                        #  'count': 0
                                                        # })
                                                        # {
                                                         'n-shot': 
                                                            {
                                                             'up': {},
                                                             'down': {},
                                                             'left': {},
                                                             'right': {},
                                                             'press button': {}
                                                            },
                                                         'count': 0
                                                        })
        correct_nshot_from_reward = copy.deepcopy(correct_nshot_from_symbol)
        correct_nshot_from_other = copy.deepcopy(correct_nshot_from_symbol)
        # Get the very first iteration
        prev_iter = forward[0]
        prev_pred = False
        # Run through iterations of forward pass to check when an action is taken for the first time
        for step in forward[1:]:
            # Get the previous action and previous location location
            prev_a, prev_g = prev_iter.a[env_i], prev_iter.g[env_i]['id']
            # If the previous action was standing still: only count as valid transition standing still actions are included as zero-shot inference
            if model.hyper['has_static_action'] and prev_a == 0 and not include_stay_still:
                prev_a = None
            # Mark the location of the previous iteration as visited
            location_visited[prev_g] += 1
            # Find whether the prediction was correct
            prediction_result = bool(torch.argmax(step.x_gen[2][env_i]) == torch.argmax(step.x[env_i])) # .numpy()
            # Zero shot inference occurs when the current location was visited, but the previous action wasn't taken before
            n_loc_visited = location_visited[step.g[env_i]['id']]
            if n_loc_visited >= 0 and n_loc_visited <= 3 and prev_a is not None: # and action_taken[prev_g, prev_a] == 0:
                    n_action_taken = action_taken[prev_g, prev_a]
                    key = int(10 * n_loc_visited + n_action_taken)
                    if prev_g in symbol_locations:
                        correct_nshot_from_symbol[prev_g]['n-shot'][env.id2action[prev_a]][key] = (prediction_result, prev_pred)
                        correct_nshot_from_symbol[prev_g]['count'] += 1
                    elif prev_g in reward_locations:
                        correct_nshot_from_reward[prev_g]['n-shot'][env.id2action[prev_a]][key] = (prediction_result, prev_pred)
                        correct_nshot_from_reward[prev_g]['count'] += 1
                    else:
                        correct_nshot_from_other[prev_g]['n-shot'][env.id2action[prev_a]][key] = (prediction_result, prev_pred)
                        correct_nshot_from_other[prev_g]['count'] += 1
            prev_pred = prediction_result
            # Update the previous action as taken
            if prev_a is not None:
                action_taken[prev_g, prev_a] += 1
            # And update the previous iteration to the current iteration
            prev_iter = step
        # Having gone through the full forward pass for one environment, add the zero-shot performance to the list of all 
        all_correct.append([correct_nshot_from_symbol, correct_nshot_from_reward, correct_nshot_from_other])
    # Return lists of success of zero-shot inference for all environments
    return all_correct


def make_lstm_directories():
    '''
    Creates directories for storing data during a model training run
    '''    
    # Get current date for saving folder
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    # Initialise the run and dir_check to create a new run folder within the current date
    run = 0
    dir_check = True
    # Initialise all pahts
    train_path, model_path, save_path, script_path, run_path = None, None, None, None, None
    # Find the current run: the first run that doesn't exist yet
    while dir_check:
        # Construct new paths
        run_path = '../Summaries/lstm/' + date + '/run' + str(run) + '/'
        train_path = run_path + 'train'
        model_path = run_path + 'model'
        save_path = run_path + 'save'
        envs_path = run_path + '/envs'
        run += 1
        # And once a path doesn't exist yet: create new folders
        if not os.path.exists(train_path) and not os.path.exists(model_path) and not os.path.exists(save_path):
            os.makedirs(train_path)
            os.makedirs(model_path)
            os.makedirs(save_path)
            os.makedirs(envs_path)
            dir_check = False
    # Return folders to new path
    return run_path, train_path, model_path, save_path, envs_path