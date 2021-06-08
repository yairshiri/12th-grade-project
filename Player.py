import Entity
import os
from EnemyBrain.sensor import SensorGroup


class Player(Entity.Entity):
    def __init__(self, x, y, sprite_name=None):
        super().__init__(x, y, sprite_name)

        # vars is the helping variable for the user to use
        self.vars = [0, 0]
        path = os.path.join(f"{self.instance.config['paths']['algorithem name']}.py")
        try:
            self.algorithem = "".join(open(path, 'r').readlines())
        except Exception as e:
            raise e
        self.sensors = SensorGroup(p=self.get_mid())

    def update(self):
        exec(self.algorithem)
        self.act_on_dir(self.vars[0])
        if self.check_collision():
            self.act_on_dir((-self.vars[0][0], -self.vars[0][1]))
        self.sensors.update(p=self.get_mid())

    def _reset(self):
        self.sensors.update(p=self.get_mid())

    def update_instance(self):
        self.instance.player_pos = self.get_pos()
