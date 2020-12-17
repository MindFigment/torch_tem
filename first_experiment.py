import numpy as np
from pprint import pprint
import json
from collections import defaultdict
import copy
import os


class FirstExperiment:

    def __init__(self, env_params):
        width = env_params['width']
        height = env_params['height']
        n_observations = env_params['n_observations']
        world_type = env_params['world_type']
        stay_still = env_params['stay_still']
        sym2reward = env_params['sym2reward']

        if world_type == 'first experiment':
            world_dict = self._create_first_experiment(width, height, n_observations, sym2reward, stay_still)
        
            filename = ''.join(['first-experiment', str(height), 'x', str(width), '.json'])
            full_path = os.path.join('envs', filename)
            with open(full_path, 'w') as f:
                json.dump(world_dict, f)


    def _create_first_experiment(self, width, height, n_observations, sym2reward, stay_still):

        # Action names and id
        # STAY_STILL = 'stay_still'
        # STAY_STILL_ID = 0
        # UP = 'up'
        # UP_ID = 1
        # DOWN = 'down'
        # DOWN_ID = 3
        # RIGHT = 'right'
        # RIGHT_ID = 2
        # LEFT = 'left'
        # LEFT_ID = 4

        # Number of board locations is equal to the size of the board (width * height)
        board_locations = int(width * height)
        # Number of all locations is equal to the size of the board + the number of symbols, because
        # after pressing a button on location with a symbol a reward will appear which is regarded as 
        # a seprate state
        n_sym = len(sym2reward.keys())
        all_locations = board_locations + n_sym

        # Count how many symbols and rewards there are
        # it's important because we want to exclude them
        # when sampling random observations
        n_sym_reward = 2 * n_sym

        # Specifying the arrangement of sensory observartions on the board
        observations = np.random.choice(n_observations - n_sym_reward, width * height, replace=False)
        # Put symbols on the board by swaping it with random observations
        sym_locations = np.random.choice(board_locations, n_sym, replace=False)
        np.put(observations, ind=sym_locations, v=list(sym2reward.keys()))
        # Create dict storing to which reward location each symbol will take you after pressing a button
        symloc2rewloc = dict(zip(sym_locations, range(board_locations, board_locations + n_sym)))
        rewloc2symloc = dict(zip(symloc2rewloc.values(), symloc2rewloc.keys()))
        rewloc2reward = dict(zip(symloc2rewloc.values(), sym2reward.values()))

        # Matrix with action transitions
        eye = np.eye(all_locations, dtype=np.int16).tolist()
        # Array specyfing a action transition is not possible from the current location
        zeros = np.zeros(all_locations, dtype=np.int16).tolist()

        # Creating adjacency matrix for the environment
        adj = np.zeros((all_locations, all_locations), dtype=np.int16)
        locations = defaultdict(lambda: {'actions': []})
        action1 = dict()
        action2 = dict()
        action3 = dict()
        # Main loop where data structure holding the structure of the environment is created
        # Loop is only over board locations, reward locations are added at the end, because
        # actions avaiable from them are almost the same as from their associated symbols
        # so we can just copy them, the only difference is that we can't move from reward to symbol location
        for i in range(board_locations):
            
            ##################
            ### stay still ###
            ##################

            # Stay still action is assigned id: 0
            action1['id'] = 0
            action1['name'] = 'stay still'
            
            if stay_still:
                adj[i, i] += 1
                action1['transition'] = eye[i]
            else:
                action1['transition'] = zeros
            
            locations[i]['actions'].append(copy.deepcopy(action1))

            # If we are on the location with a symbol let's update adjacency matrix and add action transition for reward location too
            if i in sym_locations:
                adj[symloc2rewloc[i], symloc2rewloc[i]] += 1
                action1['transition'] = eye[symloc2rewloc[i]]
                locations[symloc2rewloc[i]]['actions'].append(copy.deepcopy(action1))

            #################
            ### down & up ###
            #################

            # Down action is assigned id: 3
            action1['id'] = 3
            action1['name'] = 'down'

            # Up action is assigned id: 1
            action2['id'] = 1
            action2['name'] = 'up'

            # Check if down action is possible from current location
            # If so, update adjecency matrix allowing down transition from current location i
            # to location i + width, and analogously allowing up transition from i + width location
            # to current location i
            if i + width < board_locations:
                adj[i, i + width] = 1
                adj[i + width, i] = 1
                
                action1['transition'] = eye[i + width]
                action2['transition'] = eye[i]

                locations[i]['actions'].append(copy.deepcopy(action1))    
                locations[i + width]['actions'].append(copy.deepcopy(action2))

                # if i < width, then we are on the upper border, so we must add impossibility of moving up
                if i < width:
                    action3['id'] = 1
                    action3['name'] = 'up'
                    action3['transition'] = zeros
                    locations[i]['actions'].append(copy.deepcopy(action3))
            else:
                action1['transition'] = zeros          
                locations[i]['actions'].append(copy.deepcopy(action1)) 


            ####################
            ### left & right ###
            ####################

            # Left action is assigned id: 4
            action1['id'] = 4
            action1['name'] = 'left'

            # Right action is assigned id: 2
            action2['id'] = 2
            action2['name'] = 'right'

            # Analogously as for down & up actions
            # if np.mod(i, width) != 0, current location is not on the left boarder
            if np.mod(i, width) != 0:
                adj[i, i - 1] = 1
                adj[i - 1, i] = 1
            
                action1['transition'] = eye[i - 1]
                action2['transition'] = eye[i]

                locations[i]['actions'].append(copy.deepcopy(action1))
                locations[i - 1]['actions'].append(copy.deepcopy(action2))

                # if np.mod(i + 1, width) == 0, then we are on the right border, so we must add impossibility of moving to the right
                if np.mod(i + 1, width) == 0:
                    action3['id'] = 2
                    action3['name'] = 'right'
                    action3['transition'] = zeros
                    locations[i]['actions'].append(copy.deepcopy(action3))
            # if np.mod(i, width) == 0, current location is on the left boarder 
            # so we omit the impossibility of right action from location i - 1 to location i, because
            # location i - 1 doesn't exist
            else:
                action1['transition'] = zeros
                locations[i]['actions'].append(copy.deepcopy(action1))    
                
            
            ####################
            ### press button ###
            ####################

            # press button action is assigned id: 5
            action1['id'] = 5
            action1['name'] = 'press button'

            # if we are on the location i with symbol, pressing button will take us to reward location sym2rewloc[i]
            if i in sym_locations:
                adj[i, symloc2rewloc[i]] = 1

                action1['transition'] = eye[symloc2rewloc[i]]

                # One can't press button when being on a reward location, because button is already pressed
                action2['id'] = 5
                action2['name'] = 'press button'
                action2['transition'] = zeros
                locations[symloc2rewloc[i]]['actions'].append(copy.deepcopy(action2))
            # if we are not a location with symbol, then pressing button is essentialy the same as 'stay still' action
            else:
                adj[i, i] += 1
                action1['transition'] = eye[i]
            
            locations[i]['actions'].append(copy.deepcopy(action1))    

        for rew_i in range(board_locations, board_locations + n_sym):
            # each reward location can move us to the same places as the symbol it is associated with
            # we don't need to take care of adjacency matrix entry taking us from symbol loction rewloc2symloc[rew_i] to reward location rew_i,
            # because now it's analogous to 'stay still' action, unless stay_still == False
            adj[rew_i, :] = adj[rewloc2symloc[rew_i], :]
            if not stay_still:
                adj[rew_i, rew_i] = 0

            # copy up & down & left & right actions from rewloc2symloc[rew_i] location
            a_tmp = copy.deepcopy([transition_dict for transition_dict in locations[rewloc2symloc[rew_i]]['actions'] if transition_dict['name'] not in ['press button', 'stay still']])
            locations[rew_i]['actions'] += a_tmp

        # Adding probabilities for each transition under random walk policy
        # We also sort actions for each location by id, and turn dictionary of dictionaries into list of dictionaries
        # We also need to normalize action transition probablities so they sum up to 1 for each location
        total_probabilities = np.sum(adj, axis=1, dtype=np.int16).tolist()
        final_locations = []
        for loc_id, loc_dict in locations.items():
            loc_dict['actions'].sort(key=lambda x: x['id'])
            for action in loc_dict['actions']:
                if np.sum(action['transition']) > 0:
                    action['probability'] = np.sum(action['transition']) / total_probabilities[loc_id]
                else:
                    action['probability'] = 0

            location = dict()
            location['id'] = loc_id
            location['actions'] = locations[loc_id]['actions']
            location['observation'] = int(observations[loc_id] if loc_id < board_locations else rewloc2reward[loc_id])
            location['in_locations'] = np.flatnonzero(adj[loc_id, :]).tolist()
            location['in_degree'] = len(np.flatnonzero(adj[loc_id, :]))
            location['out_locations'] = np.flatnonzero(adj[loc_id, :]).tolist()
            location['out_degree'] = len(np.flatnonzero(adj[loc_id, :]))
            final_locations.append(location)

        # Creating dictionary containing whole environment structure
        world_dict = dict()
        world_dict['n_locations'] = all_locations
        world_dict['n_actions'] = 6
        world_dict['n_observations'] = n_observations 
        world_dict['adjacency'] = adj.tolist()
        final_locations.sort(key=lambda x: x['id'])
        world_dict['locations'] = final_locations
        world_dict['n_symbols'] = n_sym
        world_dict['symbol_locations'] = [int(s_l) for s_l in sym_locations]
        world_dict['reward_locations'] = list(rewloc2reward.keys())
        world_dict['width'] = width
        world_dict['height'] = height
        world_dict['sym2reward'] = sym2reward

        # x = 1 / (2 * np.max([width, height]))
        # y = 1 / (2 * np.max([width, height]))
        # step = 1 / np.max([width, height])
        # rew_locations = list(rewloc2reward.keys())
        # for i, location in enumerate(world_dict['locations']):
        #     if location['id'] in rew_locations:
        #         location['x'] = world_dict['locations'][rewloc2symloc[i]]['x']
        #         location['y'] = world_dict['locations'][rewloc2symloc[i]]['y']
        #     else:
        #         location['x'] = x + ((i // width) * step)
        #         location['y'] = y + ((i % width) * step)

        return world_dict 



def main():

    env_params = dict()

    sym2reward = {
        37: 41, # 'banana'
        38: 42, # 'orange'
        39: 43, # 'milk'
        40: 44  # 'coke'
    }
    env_params['sym2reward'] = sym2reward

    env_params['width'] = 4
    env_params['height'] = 4
    env_params['n_observations'] = 45
    env_params['world_type'] = 'first experiment'
    env_params['stay_still'] = False

    world = FirstExperiment(env_params)



if __name__ == '__main__':
    main()