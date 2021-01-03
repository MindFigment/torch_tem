import importlib
import torch
import glob
import click
import utils
import os
from datetime import datetime


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