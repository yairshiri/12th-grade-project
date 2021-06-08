from random import randint

from shapely.geometry import polygon
import dataSaver
import os
import pygame as pg


class Entity:
    def __init__(self, x, y, sprite_name=None):
        # if a sprite name is specified, load and save it
        if sprite_name is not None:
            self.sprite = pg.transform.scale(pg.image.load(os.path.join(r"Resources/Sprites", f"{sprite_name}.png")),
                                             self.instance.draw_scaler)
        # getting an instance of the data saver
        self.instance = dataSaver.DataSaver.get_instance()

        # because the x,y we get are maze-scaled values, we need to scale them to the window
        self.x = x
        self.y = y

        # creating the rect for hit detection
        self.rect = polygon.Polygon([(self.x, self.y), (self.x + 1, self.y),
                                     (self.x + 1, self.y + 1),
                                     (self.x, self.y + 1)])

        # saving the starting position for resetting
        self.startingPos = (self.x, self.y)

    def get_pos(self):
        return self.x, self.y

    def set_pos(self, pos):
        self.x, self.y = pos
        self.update_rect()
        self.update_instance()

    def get_mid(self):
        # getting the middle point of the sprite
        return self.x + 0.5, self.y + 0.5

    def act_on_dir(self, dir):
        # 1 is positive in the direction, -1 is negative. index 0 is x, index 1 is y. we multiply the values because they are maze-scaled (move one maze unit, not one pixel).
        # action = [x * self.instance.draw_scaler[i] for i, x in enumerate(dir)]
        self.x += dir[0]
        self.y += dir[1]
        # updating the rect because we updated the position so the rect also moves
        self.update_rect()
        self.update_instance()
        return dir

    def update_rect(self):
        # just sets the rect to the values of the x,y. like in the init method.
        self.rect = polygon.Polygon([(self.x, self.y), (self.x + 1, self.y),
                                     (self.x + 1, self.y +1),
                                     (self.x, self.y + 1)])

    def check_collision(self):
        # the first check is for the outer wall, the wall that surrounds the entire map.
        if not self.rect.within(self.instance.window_limit):
            return True

        # this check is for all of the other walls (the maze, the normal walls). I found this check to work the best because touches is one point on the outside and no point on the inside
        # and intersects is any points that they have in common so not touches and intersects is one geom having at least one point on the inside of the other geom.
        for x in self.instance.STRtree.query(self.rect):
            if not x.touches(self.rect) and x.intersects(self.rect):
                return True
        return False

    def generate_pos(self):
        # we use -1 so it won't generate positions outside the map
        x = randint(0, self.instance.maze_shape[0]-1)
        y = randint(0, self.instance.maze_shape[1]-1)
        rect = polygon.Polygon([(x, y),
                                (x + 1, y),
                                (x + 1, y + 1),
                                (x, y + 1)])
        geom = polygon.Point((-1, -1))
        for wall in self.instance.walls:
            geom = geom.union(wall.rect)
        if rect.intersects(geom):
            return self.generate_pos()
        return rect, x, y

    def update_instance(self):
        raise NotImplemented

    def reset(self):
        self.set_pos(self.startingPos)
        self._reset()

    def _reset(self):
        raise NotImplemented

    def pos_to_scale(self):
        return self.x*self.instance.draw_scaler[0],self.y*self.instance.draw_scaler[1]
