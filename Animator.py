import pygame as pg


class Animator:
    curr_state = ""
    counter = 0

    def __init__(self):
        self.sprites = {}
        self.iter = None

    def add(self, name, sprites):
        self.sprites[name] = sprites
        return self

    def get_iter(self):
        index = 0
        max_len = len(self.sprites[self.curr_state])
        while True:
            yield self.sprites[self.curr_state][index]
            if self.counter < 10:
                self.counter += 1
            else:
                self.counter = 0
                index += 1
                if index >= max_len:
                    index = 0

    def next(self):
        return next(self.iter)

    def set_state(self, state):
        self.curr_state = state
        self.iter = self.get_iter()
        return self.next()

    def reverse(self, state):
        for i, sprite in enumerate(self.sprites[state]):
            self.sprites[state][i] = pg.transform.flip(self.sprites[state][i], True, False)
