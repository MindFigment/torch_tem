import importlib
import torch
import glob


def load_model():
    # Choose which trained model to load
    i_start = 7999

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Set all paths from existing run 
    model_path = './models'
    script_path = '.'
    envs_path = './envs'
    
    # Load the model: use import library to import module from specified path
    model_spec = importlib.util.spec_from_file_location('model', script_path + '/model.py')
    model = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model)
    
    # Load the parameters of the model
    params = torch.load(model_path + '/params_' + str(i_start) + '.pt', map_location=torch.device(device))
    # But certain parameters (like total nr of training iterations) may need to be copied from the current set of parameters
    new_params = {
        'device':  device
    }
    # Update those in params
    for key in new_params:
        params[key] = new_params[key]
    
    # Create a new tem model with the loaded parameters
    tem = model.Model(params).to(device)

    # Load the model weights after training
    model_weights = torch.load(model_path + '/tem_' + str(i_start) + '.pt', map_location=torch.device(device))
    # Set the model weights to the loaded trained model weights
    tem.load_state_dict(model_weights)
    
    # Make list of all the environments that this model was trained on
    # envs = list(glob.iglob(envs_path + '/*'))
    envs = list(glob.iglob(envs_path + '/first-experiment4x4.json'))
    
    # And increase starting iteration by 1, since the loaded model already carried out the current starting iteration
    i_start = i_start + 1

    return tem, params, envs