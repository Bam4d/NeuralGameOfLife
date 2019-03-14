import pygame as pg
import numpy as np
from game_of_life import GameOfLife, CompareRender
from model import GameOfLifeAutoEncoder
from create_training_data import get_one_hot_2d

if __name__ == "__main__":

    width = 10
    height = 10

    game = GameOfLife(width, height)
    model = GameOfLifeAutoEncoder(width, height)

    model.load_trained_model('trained_model.tch')

    render = CompareRender(1024, 1024, width, height)

    render.init(pg)

    clock = pg.time.Clock()

    # initialize random game state for real game and game model
    grid_observation = game.observe()
    grid_predicted_observation = np.expand_dims(get_one_hot_2d(grid_observation, width, height), 0)

    for x in range(0, 20000):

        clock.tick(10)

        render.draw_side_by_side(grid_observation, np.argmax(grid_predicted_observation, axis=1))

        # Step the predicted environment
        grid_predicted_observation = model.predict(grid_predicted_observation)

        game.step()
        grid_observation = game.observe()

        pg.display.flip()

    pg.quit()