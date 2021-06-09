from EnemyBrain.BaseClasses import Entity


class Enemy(Entity):

    def __init__(self, x, y):
        super().__init__(x, y)
        self.moves = self.instance.config['environment']['max steps per episode']
        # (255,0,0) is red
        self.color = (255, 0, 0)
        self.prevDistance = self.instance.max_distance
        self.bestDistance = self.instance.max_distance

    def _reset(self):
        self.set_pos(self.generate_pos())
        self.bestDistance = self.instance.max_distance
        self.prevDistance = self.instance.max_distance
        self.moves = self.instance.config['environment']['max steps per episode']

    def update_instance(self):
        self.instance.enemy_pos = self.get_pos()
