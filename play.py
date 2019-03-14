import pygame as pg
from game_of_life import GameOfLife, SingleRender

if __name__ == "__main__":

    width = 10
    height = 10

    game = GameOfLife(width, height)
    render = SingleRender(1024, 1024, width, height)

    render.init(pg)

    clock = pg.time.Clock()

    for x in range(0, 20000):

        grid_observation = game.observe()

        game.step()

        clock.tick(10)

        render.draw_observation(grid_observation)

        pg.display.flip()

    pg.quit()