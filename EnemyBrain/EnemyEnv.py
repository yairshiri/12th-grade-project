import Enemy
from EnemyBrain.BaseClasses import Env
import Player
import pygame as pg
import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np
from PIL import Image
from utils import time_me


def state_wrapper(f):
    def wrapper(*args, **kwargs):
        self: EnemyEnv = args[0]
        value = f(self, *args[1:], **kwargs)
        try:
            state = self.state
            state = np.delete(state, -1, axis=0)
            state = np.insert(state, 0, value, axis=0)
        except Exception as e:
            state = np.array([value for i in range(self.STATE_FRAMES)])
        return state

    return wrapper


class EnemyEnv(Env):
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    CLOCK = pg.time.Clock()
    POSITIVE_REWARD = 500
    NEGATIVE_REWARD = -1
    STATE_FRAMES = 5

    def __init__(self):
        super().__init__()
        self.enemy = Enemy.Enemy(0, 0)
        self.player = Player.Player(self.enemy.instance.config['environment']['starting player pos']['x'],
                                    self.enemy.instance.config['environment']['starting player pos']['y'])
        if self.enemy.instance.config['agent']['use sensors'] is True:
            self.enemy.set_sensors()
        if self.instance.config['environment']['use conv']:
            self.set_state = self.set_state_conv
        self.state = self.set_state()
        self.observation_space = self.state.shape
        # saving the hotkeys text so we won't have to generate it each render call
        font = pg.font.SysFont('Arial', 20)
        # creating the default image to be displayed every frame
        self.base_surface = pg.Surface(self.instance.screen_size)
        # adding the background
        self.base_surface.blit(self.instance.background_img, (0, 0))
        # adding the walls
        for wall in self.instance.walls:
            for loc in wall.locs():
                self.base_surface.blit(self.instance.wall_img, loc)
        # adding the hotkey text
        self.enemy.instance.screen.blit(
            font.render("Options: O. Load brain: L. Save Brain: S. See tracked data: G.", True,
                        self.BLACK), (0, 0))

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
        self.CLOCK.tick(self.enemy.instance.config['game']['fps cap'])
        self.metrics['fps'][-1] = self.CLOCK.get_fps()
        pg.display.update()
        self.enemy.instance.screen.blit(self.base_surface, (0, 0))
        pg.draw.rect(self.enemy.instance.screen, self.enemy.color,
                     (self.enemy.pos_to_scale(), self.enemy.instance.draw_scaler))
        pg.draw.rect(self.enemy.instance.screen, self.BLACK,
                     (self.player.pos_to_scale(), self.enemy.instance.draw_scaler))
        pg.display.flip()
        pg.event.pump()

    @state_wrapper
    def set_state(self):
        enemy_pos = self.enemy.get_pos()
        player_pos = self.player.get_pos()
        ret = enemy_pos + player_pos
        if self.enemy.sensors is not None:
            sensors = tuple(x[0] for x in self.enemy.sensors.get_info())
            ret += sensors
        return np.array(ret)

    @state_wrapper
    def set_state_conv(self):
        map = self.instance.taken_pixels
        enemy_pos = self.enemy.get_pos()
        player_pos = self.player.get_pos()
        map[enemy_pos[0]][enemy_pos[1]] = [0, 0, 0]
        map[player_pos[0]][player_pos[1]] = [255, 0, 0]
        return map / 255.0  # scaling the values to be between 0 and 1

    def set_state_conv2(self):
        return np.atleast_3d(pg.surfarray.array3d(self.instance.screen).sum(-1))

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
