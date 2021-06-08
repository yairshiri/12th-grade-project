import tkinter as tk
import yaml
from math import ceil, sqrt
import dataSaver
import os

instance = dataSaver.DataSaver.get_instance()


def is_leaf(val):
    if isinstance(val, dict):
        return False
    return True


def save_options():
    with open(os.path.join(r"Resources", "config.yml"), 'w') as file:
        yaml.dump(instance.config, file)


def check_iterable(val):
    try:
        val = iter(val)
        return True
    except:
        return False


def check_type(val1, val2):
    try:
        val2 = type(val1)(val2)
        return True
    except:
        return False


def change_type(val1, val2):
    if isinstance(val2, str):
        if not check_iterable(val1):
            return type(val1)(val2)
        else:
            if isinstance(val1, list):
                val2 = type(val1)(val2[1:-1].split(','))
                for i, item in enumerate(val2):
                    val2[i] = item.replace(" ", "")
    return type(val1)(val2)


def make_the_same(val1, val2):
    if not check_type(val1, val2):
        return val1
    val2 = change_type(val1, val2)
    if isinstance(val1, list):
        for i, item in enumerate(iter(val1)):
            val1[i] = make_the_same(val1[i], val2[i])
    else:
        val1 = change_type(val1, val2)
    return val1


class Menu:
    class Tree:
        def __init__(self, name, val, prev):
            self.name = name
            self.value = val
            self.prev = prev

        def as_dict(self):
            return {self.name: self.value}

        def get_tree(self, name):
            return Menu.Tree(name, self.value[name], self)

    def __init__(self, dict):
        self.tree = Menu.Tree("Options", dict, None)
        self.window = None
        self.frame = None
        self.shape = None
        self.changes = {}

    def initialize(self):
        try:
            self.window.title(self.tree.name)
        except:
            self.window = tk.Tk()
            self.window.title(self.tree.name)
            self.frame = tk.Frame(master=self.window)
            self.frame.grid(row=0, column=0)
        self.window.iconphoto(False, tk.PhotoImage(file=instance.icon))

        width = ceil(sqrt(len(self.tree.value.values())))
        width = max(width, 2)
        shape = (width, width + 1)
        self.shape = shape
        new_frame = tk.Frame(master=self.window)
        new_frame.grid(row=2, column=0)
        # return button
        button = tk.Button(master=new_frame, text="return", name="return", command=self.go_back)
        button.grid(row=0, column=0, padx=(0, 10))
        # exit button
        button = tk.Button(master=new_frame, text="exit", name="exit", command=self.exit)
        button.grid(row=0, column=1, padx=(10, 10))

    def render(self):
        self.initialize()
        self.get_buttons()
        self.window.mainloop()

    def get_buttons(self):
        buttons = []
        entries = []
        pady = (2, 2)
        padx = (0, 0)
        for key, value in self.tree.value.items():
            if is_leaf(value):
                entries.append(self.get_entry(key, value))
            else:
                buttons.append(tk.Button(master=self.frame, name=key, text=key,
                                         command=lambda key=key: self.set_node(key)))
        x, y = 0, 1
        for entry, label in entries:
            x = 0
            label.grid(row=y, column=x, pady=pady)
            x += 1
            entry.grid(row=y, column=x, pady=pady)
            y += 1
        x = 0
        for button in buttons:
            button.grid(row=y, column=x, pady=pady, padx=padx)
            x += 1
            if x >= self.shape[0]:
                x = 0
                y += 1
        if x != 0:
            x = 0
            y += 1

        # return button

    def set_node(self, name):
        self.reset()
        self.tree = self.tree.get_tree(name)
        self.render()

    def get_entry(self, key, value):
        label = tk.Label(master=self.frame, text=f"Set {key}:")
        var = tk.Variable()
        ret = tk.Entry(master=self.frame, name=key, textvariable=var)
        var.set(str(value))
        self.changes[key] = var
        return ret, label

    def go_back(self):
        if self.tree.prev is not None:
            self.reset()
            self.tree = Menu.Tree(self.tree.prev.name, self.tree.prev.value, self.tree.prev.prev)
            self.render()

    def reset(self):
        for slave in self.frame.grid_slaves():
            slave.destroy()
        self.frame.update()
        for key, var in self.changes.items():
            if var.get() != str(self.tree.value[key]):
                self.tree.value[key] = make_the_same(self.tree.value[key], var.get())
        save_options()
        self.changes = {}

    def exit(self):
        self.reset()
        self.window.destroy()
        self.window = None
        while self.tree.prev is not None:
            self.tree = Menu.Tree(self.tree.prev.name, self.tree.prev.value, self.tree.prev.prev)
