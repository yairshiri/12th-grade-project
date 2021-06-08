pos = self.instance.enemy_pos
dir = [0, 0]
dx = pos[0] - self.x
dy = pos[1] - self.y
if dx > 0:
    dir[0] = -1
elif dx < 0:
    dir[0] = 1
if dy > 0:
    dir[1] = -1
elif dy < 0:
    dir[1] = 1
self.vars[0] = dir
