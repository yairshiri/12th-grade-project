import time

import numpy as np

from EnemyBrain import Metrics
from gym import Env as GymEnv
import tkinter as tk
from random import randint
from shapely.geometry import polygon
import dataSaver
import os
import pygame as pg
from EnemyBrain.sensor import SensorGroup
from EnemyBrain import Policies


class Entity:
    def __init__(self, x, y, sprite_name=None):
        # if a sprite name is specified, load and save it
        if sprite_name is not None:
            self.sprite = pg.transform.scale(pg.image.load(os.path.join(r"../Resources/Sprites", f"{sprite_name}.png")),
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

        self.sensors = None

    def get_pos(self):
        return self.x, self.y

    def set_sensors(self):
        self.sensors = SensorGroup(p=self.get_mid())

    def set_pos(self, pos):
        self.x, self.y = pos
        self.update_rect()
        self.update_instance()
        if self.sensors is not None:
            self.sensors.update(p=self.get_mid())

    def get_mid(self):
        # getting the middle point of the sprite
        return self.x + 0.5, self.y + 0.5

    def act_on_dir(self, dir):
        # 1 is positive in the direction, -1 is negative. index 0 is x, index 1 is y.
        self.x += dir[0]
        self.update_rect()
        if self.check_collision():
            self.x -= dir[0]
            self.update_rect()
        self.y += dir[1]
        self.update_rect()
        if self.check_collision():
            self.y -= dir[1]
            self.update_rect()
        self.update_instance()
        if self.sensors is not None:
            self.sensors.update(p=self.get_mid())
        return dir

    def update_rect(self):
        # just sets the rect to the values of the x,y. like in the init method.
        self.rect = polygon.Polygon([(self.x, self.y), (self.x + 1, self.y),
                                     (self.x + 1, self.y + 1),
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
        x = randint(0, self.instance.maze_shape[0] - 1)
        y = randint(0, self.instance.maze_shape[1] - 1)
        rect = polygon.Polygon([(x, y),
                                (x + 1, y),
                                (x + 1, y + 1),
                                (x, y + 1)])
        geom = polygon.Point((-1, -1))
        for wall in self.instance.walls:
            geom = geom.union(wall.rect)
        if rect.intersects(geom):
            return self.generate_pos()
        return x, y

    def update_instance(self):
        raise NotImplemented

    def reset(self):
        self.set_pos(self.startingPos)
        self._reset()
        if self.sensors is not None:
            self.sensors.update(p=self.get_mid())

    def _reset(self):
        raise NotImplemented

    def pos_to_scale(self):
        return self.x * self.instance.draw_scaler[0], self.y * self.instance.draw_scaler[1]


class Agent:
    def __init__(self, env):
        self.env = env
        self.hyperparameters = dataSaver.DataSaver.get_instance().agent_hyperparameters
        self.limits = self.hyperparameters['train until']
        self.mode = self.hyperparameters['mode']
        nb_actions = self.hyperparameters['number of actions']
        decay = self.hyperparameters['policy']['epsilon']['decay']
        min_value = self.hyperparameters['policy']['epsilon']['min']
        starting = self.hyperparameters['policy']['epsilon']['starting']
        assert self.hyperparameters['policy']['type'].lower() in ['epsgreedy','bepsgreedy','b']
        if self.hyperparameters['policy']['type'].lower() == 'epsgreedy':
            self.policy = Policies.EpsGreedy(nb_actions,decay,min_value,starting)
        elif self.hyperparameters['policy']['type'].lower() == 'bepsgreedy':
            self.policy = Policies.EpsGreedyBoltzmann(nb_actions,decay,min_value,starting)
        elif self.hyperparameters['policy']['type'].lower() == 'b':
            self.policy = Policies.Boltzmann(nb_actions,decay,min_value,starting)
        # verifying the mode
        if self.mode not in ['training', 'testing']:
            raise Exception(
                f"Mode {self.mode} is an unrecognized mode! The only allowed modes are training and testing")

    def learn(self, *args, **kwargs):
        raise NotImplemented

    def action(self, *args, **kwargs):
        raise NotImplemented

    def load(self):
        name = self._get_save_name("Choose policy file name")
        self._load(name)

    def save(self):
        name = self._get_save_name("Choose policy file name")
        self._save(name)

    def _get_save_name(self, text):
        window = tk.Tk()
        window.title(text)
        window.iconphoto(False, tk.PhotoImage(file=self.env.enemy.instance.icon))
        label = tk.Label(window, text=text)
        label.grid(row=0, column=0)
        var = tk.StringVar()
        entry = tk.Entry(window, textvariable=var)
        entry.grid(row=0, column=1)
        button = tk.Button(window, text="Done", command=window.destroy)
        button.grid(row=1, column=0)
        window.mainloop()
        return var.get()

    def _load(self, f):
        raise NotImplemented

    def _save(self, f):
        raise NotImplemented

    def check_stop(self):
        stop = 0
        # getting the data
        trackers = {'max win rate': self.env.latest_data['win rate'],
                    'max win rate over x': self.env.latest_data['win rate over x'],
                    'max steps': self.env.latest_data['number of steps'],
                    'max episodes': self.env.latest_data['number of episodes']}
        # looping over all of the keys
        for key, value in trackers.items():
            if self.limits[key] > 0:
                if value >= self.limits[key]:
                    # checking win rates only after the 50th episode.
                    if not ('win rate' in key and trackers['max episodes'] < 50):
                        stop += 1
                    # if we've met the required amount of limits
                    if stop >= self.limits['to meet']:
                        return False
        return True


class Env(GymEnv):

    def __init__(self):
        self.instance = dataSaver.DataSaver.get_instance()
        self.metrics = {
            'rewards': Metrics.Metric('latest',
                                      self.instance.config['logging']['metrics']['avg episode reward']),
            'fps': Metrics.Metric('latest',
                                  self.instance.config['logging']['metrics']['fps']),
            'steps': Metrics.Metric('latest',
                                    self.instance.config['logging']['metrics']['number of steps']),
            'wins': Metrics.Metric('count',
                                   self.instance.config['logging']['metrics']['number of wins'], True)}
        self.metrics['rewards'].add_value(0)
        self.metrics['steps'].add_value(0)
        self.metrics['fps'].add_value(0)
        self.state = None
        self.tic = time.perf_counter()
        self.callbacks = [key if val is True else "" for key, val in self.instance.config['logging']['metrics'].items()]
        self.callback_func = self.get_callback_func()
        self.latest_data = self.get_metrics_vals()
        self.win_rates = None
        self.win_rates_over_x = None
        if 'win rate' in self.callbacks:
            self.win_rates = []
        if 'win rate over x' in self.callbacks:
            self.win_rates_over_x = []

    def step(self, action):
        ret = self._step(action)
        self.update_metrics(ret)
        ret[3]['metrics str'] = self.get_metrics_string()
        ret[3]['metrics vals'] = self.get_metrics_vals()
        return ret

    def render(self, mode='human'):
        self._render()

    def update_metrics(self, ret):
        reward, done, info = ret[1:]
        self.metrics['rewards'].vals[-1] += reward
        self.metrics['steps'].vals[-1] += 1
        toc = time.perf_counter()
        dif = toc - self.tic
        self.tic = toc
        self.metrics['fps'].add_value(round(1 / dif, 4))
        if done:
            self.metrics['wins'].add_value(info['won'])
            # self.metrics['win rate'].add_value(
            #     self.metrics['number of wins'].get_value() * 100 / len(self.metrics['number of wins'].vals))
            # wins_over_x = self.metrics['number of wins'].get_over_last(self.metrics['win rate over x'].count_val)
            # self.metrics['win rate over x'].add_value(
            #     wins_over_x.count(self.metrics['number of wins'].count_val) * 100 / len(wins_over_x))
            self.metrics['rewards'].vals[-1] /= self.metrics['steps'].vals[-1]
            self.metrics['rewards'].add_value(0)
            self.metrics['steps'].add_value(0)
            self.metrics['fps'].vals = []
            if self.win_rates is not None:
                self.win_rates.append(self.latest_data['win rate'])
            if self.win_rates_over_x is not None:
                self.win_rates_over_x.append(self.latest_data['win rate over x'])
        self.latest_data = self.get_metrics_vals()

    def get_metrics_string(self):
        ret = ""
        for metric, value in self.latest_data.items():
            if metric in self.callbacks:
                ret = ret.__add__(f"{metric}:{value}\n")
        return ret

    def get_metrics_vals(self):
        ret = {}
        for func in self.callback_func():
            val = func()
            ret[val[1]] = val[0]
        return ret

    def _reset(self):
        raise Exception("Not implemented!")

    def _render(self):
        raise Exception("Not implemented!")

    def _step(self, action):
        raise Exception("Not implemented!")

    def reset(self):
        self._reset()
        return self.state

    def get_callback_func(self):
        funcs = []
        funcs.append(lambda: (self.metrics['rewards'].vals[-1], 'avg episode reward'))
        funcs.append(lambda: (
            np.round(np.sum(self.metrics['fps'].vals[-len(self.metrics['fps'].vals):]) / len(self.metrics['fps'].vals),
                     4),
            'fps'))
        funcs.append(lambda: (len(self.metrics['steps'].vals), 'number of episodes'))
        funcs.append(lambda: (self.metrics['steps'].vals[-1], 'number of steps'))
        funcs.append(lambda: (self.metrics['wins'].vals.count(True), 'number of wins'))
        funcs.append(lambda: (
            round(100 * self.metrics['wins'].vals.count(True) / max(len(self.metrics['wins'].vals), 1), 4), 'win rate'))

        def func():
            amount = min(len(self.metrics['wins'].vals), 100)
            return round(100 * self.metrics['wins'].vals[-amount:].count(True) / max(amount, 1), 4), 'win rate over x'

        funcs.append(func)

        return lambda: [x for x in funcs]


class Memory:
    def __init__(self, min_size, max_size, batch_size):
        self.batch_size = batch_size
        self.max_size = max_size
        self.memory = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.min_size = min_size

    def get_batch(self):
        raise NotImplemented

    def get_ids(self):
        raise NotImplemented

    def add_exp(self, *args, **kwargs):
        raise NotImplemented

    def is_ready(self):
        return len(self.memory['s']) >= self.min_size
