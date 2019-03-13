import numpy as np
import pygame as pg

class GameOfLife:

    def __init__(self, width, height):
        self._width = width
        self._height = height
        self.reset()

    def reset(self):
        self._grid = np.random.randint(2, size=[self._width, self._height])

    def observe(self):
        return np.expand_dims(self._grid, 0)


    def step(self):
        """
        https://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/
        """

        nbrs_count = sum(np.roll(np.roll(self._grid, i, 0), j, 1)

                         for i in (-1, 0, 1) for j in (-1, 0, 1)
                         if (i != 0 or j != 0))

        self._grid = (nbrs_count == 3) | (self._grid & (nbrs_count == 2))

class Render(object):

    def __init__(self, width, height, grid_x, grid_y):

        self._width = width
        self._height = height

        self._grid_x = grid_x
        self._grid_y = grid_y

    def init(self):
        self._window = pg.display.set_mode((self._width, self._height))
        self._window.fill((100, 100, 100))

    def save_image(self, filename):
        pg.image.save(self._window, filename)

    def _get_object_color(self, grid_object):
        if grid_object == 0:
            return (0, 0, 0)
        elif grid_object == 1:
            return (0, 0, 255)
        
class SingleRender(Render):

    def draw_observation(self, observation):

        grid_seperator = -1

        box_width = (self._width - (grid_seperator * (self._grid_x + 1))) / self._grid_x
        box_height = (self._height - (grid_seperator * (self._grid_y + 1))) / self._grid_y

        for r in range(0, self._grid_y):
            for c in range(0, self._grid_x):
                color = self._get_object_color(observation[0][c][r])

                pg.draw.rect(
                    self._window,
                    color,
                    [
                        (box_width + grid_seperator) * r + grid_seperator,
                        (box_height + grid_seperator) * (self._grid_y - c - 1) + grid_seperator,
                        box_width,
                        box_height
                    ]
                )
                
class CompareRender(Render):

    def draw_side_by_side(self, observation_left, observation_right):

        grid_seperator = -1

        box_width = (self._width - (grid_seperator * (self._grid_x + 1))) / self._grid_x
        box_height = (self._height - (grid_seperator * (self._grid_y + 1))) / self._grid_y

        # Draw left
        for r in range(0, self._grid_y):
            for c in range(0, self._grid_x):
                color = self._get_object_color(observation_left[0][c][r])

                pg.draw.rect(
                    self._window,
                    color,
                    [
                        (box_width + grid_seperator) * r + grid_seperator,
                        (box_height + grid_seperator) * (self._grid_y - c - 1) + grid_seperator,
                        box_width,
                        box_height
                    ]
                )

        right_offset = self._width + 10 * grid_seperator

        # Draw right
        for r in range(0, self._grid_y):
            for c in range(0, self._grid_x):
                color = self._get_object_color(observation_right[0][c][r])

                pg.draw.rect(
                    self._window,
                    color,
                    [
                        right_offset + (box_width + grid_seperator) * r + grid_seperator,
                        (box_height + grid_seperator) * (self._grid_y - c - 1) + grid_seperator,
                        box_width,
                        box_height
                    ]
                )

def create_random_training_episode(game, max_steps):

    observations = []

    game.reset()

    for i in range(0, max_steps):
        observation = game.observe()
        observations.append(observation)
        game.step()


    return observations