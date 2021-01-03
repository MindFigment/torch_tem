import json
import numpy as np
import random

from my_utils import load_model


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
# PRESS = 'press'
# PRESS_ID = 5


class Environment:
    '''
    '''

    def __init__(self, env='./envs/first-experiment4x4.json', seed=None):
        '''
        Initiates environment
        '''

        file = open(env, 'r') 
        json_text = file.read()
        self.env = json.loads(json_text)
        file.close()

        self.PRESS = 'press'
        self.UP = 'up'
        self.DOWN = 'down'
        self.LEFT = 'left'
        self.RIGHT = 'right'

        self.action2id = {
            self.UP: 1,
            self.DOWN: 3,
            self.LEFT: 4,
            self.RIGHT: 2,
            self.PRESS: 5
        }

        self.actions = [self.UP, self.DOWN, self.LEFT, self.RIGHT, self.PRESS]

        self.width = self.env['width']
        self.height = self.env['height']

        self.grid = np.zeros(self.height * self.width)
        self.sym_loc = self.env['symbol_locations']
        self.rew_loc = self.env['reward_locations']
        self.n_sym = self.env['n_symbols']
        self.n_loc = self.env['n_locations']
        self.n_observations = self.env['n_observations']
        self.n_board = self.n_loc - self.n_sym
        self.symloc2rewloc = dict(zip(self.sym_loc, self.rew_loc))
        self.rewloc2symloc = dict(zip(self.rew_loc, self.sym_loc))

        for location in self.env['locations']:
            if location['id'] not in self.rew_loc:
                self.grid[location['id']] = location['observation']

        self._init_agent_position()


    def step(self, action):
        transition = self.env['locations'][self.curr_loc]['actions'][self.action2id[action]]['transition']
        if np.sum(transition) > 0:
            self.curr_loc = np.nonzero(np.array(transition))[0][0]


    def _init_agent_position(self):
        self.curr_loc = np.random.randint(self.height * self.width)

    
    def get_token(self, i):
        token = None
        if i in self.sym_loc:
            token = int(self.grid[i])
        else:
            # token = 'x'
            token = int(self.grid[i])
        return token


    def display_env(self):
        on_reward_loc = False
        if self.curr_loc >= self.n_board:
            self.grid[self.rewloc2symloc[self.curr_loc]] = self.env['locations'][self.curr_loc]['observation']
            on_reward_loc = True
        for i in range(self.n_board):
            token = self.get_token(i)
            if self.curr_loc == i:
                print('({:2})'.format(token), end='')
            elif on_reward_loc and self.rewloc2symloc[self.curr_loc] == i:
                print('[{:2}]'.format(token), end='')
            else:
                print('|{:2}|'.format(token), end='')
            
            if np.mod(i + 1, self.width) == 0:
                print()
        if on_reward_loc:
            self.grid[self.rewloc2symloc[self.curr_loc]] = self.env['locations'][self.rewloc2symloc[self.curr_loc]]['observation']


def main():
    env = Environment('./envs/first-experiment4x4.json')
    env.display_env()

    # date = '2020-12-17'
    # run = '0'
    # envs = []

    # model, _, _, _ = load_model(date, run, envs)

    while True:
        try:
            command = input('Next action: ')
            if command == '\x1b[A':
                # print('arrow up')
                action = env.UP
            elif command == '\x1b[B':
                # print('arrow down')
                action = env.DOWN
            elif command == '\x1b[D':
                # print('arrow left')
                action = env.LEFT
            elif command == '\x1b[C':
                # print('arrow right')
                action = env.RIGHT
            elif command == 'b':
                # print('Button pressed')
                action = env.PRESS
            elif command == '\x1b' or command == 'e':
                break
            else:
                print('.')
            
            env.step(action)
            env.display_env()
            
        except ValueError:
            print('Ups, something went wrong')
            continue



if __name__ == '__main__':
    main()





