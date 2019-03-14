import numpy as np
from game_of_life import create_random_training_episode, GameOfLife

def get_one_hot_2d(grid, height, width):
    """
    Convert the 2d array of the grid into a 3d array where the extra dimension encodes the one-hot (on/off)
    representation of every grid location.

    :param grid: 2d array representing the grid
    :param height: height of the grid
    :param width: width of the grid
    :return:
    """
    layer_idx = np.arange(grid.shape[1]).reshape(grid.shape[1], 1)
    component_idx = np.tile(np.arange(grid.shape[2]), (grid.shape[1], 1))
    one_hot_state_observation = np.zeros([2, width, height])
    one_hot_state_observation[grid, layer_idx, component_idx] = 1
    return one_hot_state_observation

if __name__ == "__main__":

    max_episodes = 500
    max_steps = 100

    width = 10
    height = 10

    game = GameOfLife(width, height)

    episodes = []

    print('Creating training episodes...')

    for i in range(0, max_episodes):

        # Simulate game of life for 'max_steps'
        training_episode = create_random_training_episode(game, max_steps)

        # convert the observations to one-hot encoding
        one_hot_training_episode = [get_one_hot_2d(step, height, width) for step in training_episode]

        episodes.append(np.stack(one_hot_training_episode))

        if i > 0 and i % 100 == 0:
            print('%d episodes recorded' % i)

    filename = 'training_data.npy'

    np.save(filename, np.stack(episodes))

    print('Grid world data saved to %s' % filename)
