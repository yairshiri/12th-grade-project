import math
import time

import pygame as pg
import os
import wall
import dataSaver
import pickle
import numpy as np
from shapely.geometry import polygon
import tkinter as tk
import threading
import queue
from sklearn.preprocessing import MinMaxScaler

instance = dataSaver.DataSaver.get_instance()


def softmax(values, limit):
    return (values / np.sum(values)) * min(limit, max(np.max(values), 1))


def squash(values, limit):
    return np.array([(x-np.min(values))/(np.max(values)-np.min(values)) for x in values])*limit


def boltzmann(values, tau=1):
    values = squash(np.atleast_2d(values), 10)
    exp = np.exp(values/tau)
    p = np.round(exp/np.sum(exp),10)
    print(exp,"\n",p,"\n\n")
    return p[0]


def get_var(text):
    window = tk.Tk()
    window.title(text)
    window.iconphoto(False, tk.PhotoImage(file=instance.icon))
    label = tk.Label(window, text=text)
    label.grid(row=0, column=0)
    var = tk.StringVar()
    entry = tk.Entry(window, textvariable=var)
    entry.grid(row=0, column=1)
    button = tk.Button(window, text="Done", command=window.destroy)
    button.grid(row=1, column=0)
    window.mainloop()
    return var.get()


def alert(text):
    window = tk.Tk()
    window.title(text)
    window.iconphoto(False, tk.PhotoImage(file=instance.icon))
    label = tk.Label(window, text=text)
    label.grid(row=0, column=0)
    button = tk.Button(window, text="OK", command=window.destroy)
    button.grid(row=1, column=0)
    window.mainloop()


def ask(text, options):
    window = tk.Tk()
    window.title(text)
    window.iconphoto(False, tk.PhotoImage(file=instance.icon))
    label = tk.Label(window, text=text)
    label.grid(row=0, column=0)
    x, y = 0, 0
    ret = []
    frame = tk.Frame(window)

    def ret_func(index):
        ret.append(index)
        ret.append(options[index])
        window.destroy()

    for i, option in enumerate(options):
        button = tk.Button(frame, text=option, name=option.lower(), command=lambda idx=i: ret_func(idx))
        button.grid(row=y, column=x)
        x += 1
    frame.grid(row=1, column=0)
    window.mainloop()
    return ret


def safe_load(f_type):
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = get_var("Choose save")
            if name == "":
                return
            if os.path.exists(f"{name}.{f_type}"):
                func(args[0], name)
            else:
                alert(f"File {name}.{f_type} not found!")

        return wrapper

    return decorator


def progression_bar(func):
    def wrapper():
        que = queue.Queue()
        t = threading.Thread(target=lambda: que.put(func()))
        pg.font.init()
        if pg.display.get_active():
            display = pg.display.get_surface()
        else:
            display = pg.display.set_mode((200, 200))
        i = 0
        t.start()
        x, y = display.get_size()
        # the best formula that worked for me\/
        font_size = int(x * 200 / 768)
        font = pg.font.SysFont('Arial', font_size)
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        while t.is_alive():
            pg.display.update()
            display.fill(WHITE)
            i += 1
            text = font.render(f"Loading{'.' * ((i % 3) + 1)}", True, BLACK)
            display.blit(text, (0, 0))
            pg.display.flip()
            pg.event.pump()
            time.sleep(0.3)
        return que.get()

    return wrapper


def load_content():
    # initiating all of the values that need to be initiated
    ss = (
        round(instance.draw_scaler[0] * instance.maze_shape[0]),
        round(instance.draw_scaler[1] * instance.maze_shape[1]))
    instance.screen_size = ss
    if not instance.config['paths']['maze name'] is None:
        with open(os.path.join("maze builder", f"{instance.config['paths']['maze name']}.pickle"), 'rb') as f:
            for wall_data in pickle.load(f):
                instance.walls.append(wall.Wall(p=wall_data['p'], width=wall_data['width'], height=wall_data['height']))
    instance.background_img = pg.transform.scale(pg.image.load(os.path.join(r"Resources/Sprites", "background.png")),
                                                 instance.screen_size)
    instance.wall_img = pg.transform.scale(pg.image.load(os.path.join(r"Resources/Sprites", "wall tile.png")),
                                           instance.draw_scaler)
    instance.set_tree()

    instance.window_limit = polygon.Polygon(
        [(0, 0), (instance.maze_shape[0], 0), instance.maze_shape, (0, instance.maze_shape[1])])

    instance.screen = pg.display.set_mode(instance.screen_size)
    pg.display.set_icon(pg.image.load(instance.icon))
    pg.font.init()
