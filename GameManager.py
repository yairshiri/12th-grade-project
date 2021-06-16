import pygame as pg
import os
import wall
import dataSaver
import pickle
from shapely.geometry import polygon
import functools
import time


def load_content():
    # initiating all of the values that need to be initiated
    instance = dataSaver.DataSaver.get_instance()
    ss = (
        round(instance.draw_scaler[0] * instance.maze_shape[0]),
        round(instance.draw_scaler[1] * instance.maze_shape[1]))
    instance.screen_size = ss
    if not instance.config['paths']['maze name'] is None:
        with open(os.path.join("maze builder", f"{instance.config['paths']['maze name']}.pickle"), 'rb') as f:
            for wall_data in pickle.load(f):
                instance.walls.append(wall.Wall(p=wall_data['p'], width=wall_data['width'], height=wall_data['height']))
    instance.background_img = pg.transform.scale(pg.image.load(os.path.join(r"Resources/Sprites", "background.png")),
                                                 instance.screen_size)
    instance.screen = pg.display.set_mode(instance.screen_size)
    instance.icon = os.path.join(r"Resources\Sprites", "icon.png")
    pg.display.set_icon(pg.image.load(instance.icon))
    instance.wall_img = pg.transform.scale(pg.image.load(os.path.join(r"Resources/Sprites", "wall tile.png")),
                                           instance.draw_scaler)
    instance.set_tree()

    instance.window_limit = polygon.Polygon(
        [(0, 0), (instance.maze_shape[0], 0), instance.maze_shape, (0, instance.maze_shape[1])])
    pg.font.init()
