import functools
import time

import dataSaver
from EnemyBrain import Metrics
from gym import Env as GymEnv
import tkinter as tk
import numpy as np

class Agent:
    def __init__(self, env):
        self.env = env
        self.hyperparameters = dataSaver.DataSaver.get_instance().agent_hyperparameters
        self.limits = self.hyperparameters['train until']
        self.mode = self.hyperparameters['mode']
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
        trackers = {'max win rate': self.env.get_metric('win rate'),
                    'max win rate over x': self.env.get_metric('win rate over x'),
                    'max steps': self.env.get_metric('number of steps'),
                    'max episodes': self.env.get_metric('number of episodes')}
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
            'avg reward': Metrics.Metric('latest',
                                         self.instance.config['logging']['metrics']['avg episode reward']),
            'fps': Metrics.Metric('latest',
                                  self.instance.config['logging']['metrics']['fps']),
            'number of steps': Metrics.Metric('latest',
                                              self.instance.config['logging']['metrics']['number of steps']),
            'number of episodes': Metrics.Metric('latest',
                                                 self.instance.config['logging']['metrics']['number of episodes']),
            'number of wins': Metrics.Metric('count',
                                             self.instance.config['logging']['metrics']['number of wins'], True),
            'win rate': Metrics.Metric('latest', self.instance.config['logging']['metrics']['win rate']),
            'win rate over x': Metrics.Metric('latest',
                                              self.instance.config['logging']['metrics']['win rate over x'],
                                              self.instance.config['logging']['graph interval'])
        }
        # we initialize the steps and episodes to 1 because the first step is episode 1 step 1 not episode 0 step 0.
        self.metrics['number of steps'].vals[0] = 1
        self.metrics['number of episodes'].vals[0] = 1
        self.state = None
        self.tic = time.perf_counter()

    def step(self, action):
        ret = self._step(action)
        self.update_metrics(ret)
        ret[3]['metrics str'] = self.get_metrics_string()
        ret[3]['metrics vals'] = self.get_metrics_vals()
        return ret

    def render(self, mode='human'):
        self._render()

    def get_metric(self, metric):
        return self.metrics[metric].get_value()

    def update_metrics(self, ret):
        reward, done, info = ret[1:]
        self.metrics['avg reward'].vals[-1] += reward
        self.metrics['number of steps'].vals[-1] += 1
        toc = time.perf_counter()
        dif = toc - self.tic
        self.tic = toc
        self.metrics['fps'].add_value(round(1 / dif, 4))
        if done:
            self.metrics['number of episodes'].vals[0] += 1
            self.metrics['number of wins'].add_value(info['won'])
            self.metrics['win rate'].add_value(
                self.metrics['number of wins'].get_value() * 100 / len(self.metrics['number of wins'].vals))
            wins_over_x = self.metrics['number of wins'].get_over_last(self.metrics['win rate over x'].count_val)
            self.metrics['win rate over x'].add_value(
                wins_over_x.count(self.metrics['number of wins'].count_val) * 100 / len(wins_over_x))
            self.metrics['avg reward'].vals[-1] /= self.metrics['number of steps'].get_value()

    def get_metrics_string(self):
        ret = ""
        for metric, value in self.metrics.items():
            if value.active:
                ret = ret.__add__(f"{metric}:{value.get_value()}\n")
        return ret

    def get_metrics_vals(self):
        ret = {}
        for metric, value in self.metrics.items():
            ret[metric] = value.get_value()
        return ret

    def _reset(self):
        raise Exception("Not implemented!")

    def _render(self):
        raise Exception("Not implemented!")

    def _step(self, action):
        raise Exception("Not implemented!")

    def reset(self):
        self._reset()
        self.metrics['avg reward'].add_value(0)
        self.metrics['number of steps'].add_value(0)
        return self.state


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

    def add_exp(self,*args,**kwargs):
        raise NotImplemented

    def is_ready(self):
        return len(self.memory['s']) >= self.min_size