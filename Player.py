from EnemyBrain.BaseClasses import Entity
import os
from utils import alert
import numpy as np


def move(algorithm, p_pos, e_pos, args):
    direction = [0, 0]
    try:
        exec(algorithm)
    except Exception as e:
        alert(str(e))
    if not isinstance(direction, list) or len(direction) != 2 or not isinstance(direction[0], int) or not isinstance(
            direction[1],
            int) or np.max(
        direction) > 1 or np.min(direction) < -1:
        alert(
            "Player algorithm returns bad values!\nShould return a list with 2 length with 0,1 or -1 in both cells.")
    return direction, args


class Player(Entity):
    def __init__(self, x, y, sprite_name=None):
        super().__init__(x, y, sprite_name)

        # args is the helping variable for the user to use
        self.args = []
        # loading the player algorithm:
        path = os.path.join(f"{self.instance.config['paths']['algorithm name']}.py")
        if os.path.exists(path):
            self.algorithm = "".join(open(path, 'r').readlines())
            if "return" in self.algorithm:
                alert(f"Player algorithm cannot have a return statement!")
                quit()
        else:
            alert(f"File {path} not found!")
            quit()

    def update(self):
        direction, self.args = move(self.algorithm, self.get_pos(), self.instance.enemy_pos, self.args)
        self.act_on_dir(direction)

    def _reset(self):
        self.set_pos(self.generate_pos())

    def update_instance(self):
        self.instance.player_pos = self.get_pos()
