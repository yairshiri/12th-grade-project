import Entity
import os


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

    def update(self):
        exec(self.algorithem)
        self.act_on_dir(self.vars[0])

    def _reset(self):
        pass

    def update_instance(self):
        self.instance.player_pos = self.get_pos()
