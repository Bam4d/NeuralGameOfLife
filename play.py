import pygame as pg
from game_of_life import GameOfLife, SingleRender

if __name__ == "__main__":

    width = 100
    height = 100

    game = GameOfLife(width, height)
    render = SingleRender(1024, 1024, width, height)

    render.init()

    clock = pg.time.Clock()

    for x in range(0, 20000):

        grid_observation = game.observe()

        game.step()

        clock.tick(100)

        render.draw_observation(grid_observation)

        pg.display.flip()

    pg.quit()