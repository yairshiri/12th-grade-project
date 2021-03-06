import Enemy
from EnemyBrain.BaseClasses import Env
import Player
import pygame as pg
import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np


class EnemyEnv(Env):
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    CLOCK = pg.time.Clock()
    POSITIVE_REWARD = 500
    NEGATIVE_REWARD = -1

    def __init__(self):
        super().__init__()
        self.enemy = Enemy.Enemy(0, 0)
        self.player = Player.Player(self.enemy.instance.config['environment']['starting player pos']['x'],
                                    self.enemy.instance.config['environment']['starting player pos']['y'])
        if self.enemy.instance.config['agent']['use sensors'] is True:
            self.enemy.set_sensors()
        self.state = self.set_state()
        self.observation_space = (1,len(self.state))
        # saving the hotkeys text so we won't have to generate it each render call
        font = pg.font.SysFont('Arial', 20)
        self.HOTKEYS_TEXT = font.render("Options: O. Load brain: L. Save Brain: S. See tracked data: G.", True,
                                        self.BLACK)

    def _reset(self):
        self.enemy.reset()
        self.player.reset()
        self.state = self.set_state()
        title = f"Episode {self.latest_data['number of episodes']}."
        # if we have a privious episode to display
        if self.latest_data['number of episodes'] > 1:
            title += f" Episode {self.latest_data['number of episodes'] - 1} was a {'win' if self.metrics['wins'][-1] else 'loss'}."
        title += f" {self.enemy.instance.agent_name} on {self.enemy.instance.map_name}."
        pg.display.set_caption(title)

    def _render(self):
        if self.enemy.instance.config['game']['fps cap'] > 0:
            self.CLOCK.tick(self.enemy.instance.config['game']['fps cap'])
        pg.display.update()
        self.enemy.instance.screen.blit(self.enemy.instance.background_img, (0, 0))
        pg.draw.rect(self.enemy.instance.screen, self.enemy.color,
                     (self.enemy.pos_to_scale(), self.enemy.instance.draw_scaler))
        for wall in self.enemy.instance.walls:
            for loc in wall.locs():
                self.enemy.instance.screen.blit(self.enemy.instance.wall_img, loc)
        pg.draw.rect(self.enemy.instance.screen, self.BLACK,
                     (self.player.pos_to_scale(), self.enemy.instance.draw_scaler))
        self.enemy.instance.screen.blit(self.HOTKEYS_TEXT, (0, 0))

        pg.display.flip()
        pg.event.pump()

    def set_state(self):
        enemy_pos = self.enemy.get_pos()
        player_pos = self.player.get_pos()
        ret = enemy_pos + player_pos
        if self.enemy.sensors is not None:
            sensors = tuple(x[0] for x in self.enemy.sensors.get_info())
            ret += sensors
        return ret

    def act(self, dir_num):
        ret = self.instance.switcher[dir_num]
        return ret

    def check_collision(self):
        return self.enemy.check_collision()

    def _step(self, action):
        self.player.update()
        self.enemy.act_on_dir(self.act(action))
        done = False
        self.enemy.moves -= 1
        info = {'won': False}
        distance = self.touches()
        reward = self.NEGATIVE_REWARD
        # if cought
        if distance == 0:
            reward = self.POSITIVE_REWARD
            done = True
            info['won'] = True

        self.state = self.set_state()
        if self.enemy.moves == 0:
            done = True
        self._render()
        return self.state, reward, done, info

    def touches(self):
        return self.enemy.rect.centroid.distance(self.player.rect.centroid)


class EnemyEnvTwo(EnemyEnv):
    def __init__(self):
        super(EnemyEnvTwo, self).__init__()
        self.observation_space = (1,self.instance.maze_shape[0] * self.instance.maze_shape[1])

    def set_state(self):
        map = np.array(self.instance.taken_pixels)
        enemy_pos = self.enemy.get_pos()
        player_pos = self.player.get_pos()
        map[enemy_pos[0]][enemy_pos[1]] = 2
        map[player_pos[0]][player_pos[1]] = 3
        return map.flatten('C')
