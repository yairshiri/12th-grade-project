import EnemyBrain.sensor
import Entity
from random import randint


class Enemy(Entity.Entity):
    prevDistance = 5000
    bestDistance = 5000

    def __init__(self, pos):
        super().__init__(pos[0], pos[1])
        self.moves = self.instance.config['environment']['max steps per episode']
        # (255,0,0) is red
        self.color = (255,0,0)

    def _reset(self):
        self.set_pos(self.generate_pos()[1:])
        self.bestDistance = self.instance.max_distance
        self.prevDistance = self.instance.max_distance
        self.moves = self.instance.config['environment']['max steps per episode']

    def update_pos(self, pos):
        self.set_pos(pos)

    def update_instance(self):
        self.instance.enemy_pos = self.get_pos()
