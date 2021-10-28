import os
import shapely.strtree
import yaml


class DataSaver:
    __instance__ = None
    config = yaml.safe_load(open(os.path.join(r"Resources", "config.yml")))
    agent_name = config['agent']['type']
    map_name = config['paths']['maze name']
    STRtree = None
    screen_size = (config['game']['screen size']['width'], config['game']['screen size']['height'])
    maze_shape = (config['game']['maze shape']['width'], config['game']['maze shape']['height'])
    draw_scaler = (round(screen_size[0] / maze_shape[0]), round(screen_size[1] / maze_shape[1]))
    max_distance = (maze_shape[0] ** 2 + maze_shape[1] ** 2) ** 0.5
    player_pos = None
    enemy_pos = None
    wall_img = None
    screen = None
    info = {'Algorithm': config['agent']['type'],
            'Policy': config['agent']['policy']['type'],
            'Memory': config['agent']['replay buffer']['type'],
            'Map': config['paths']['maze name'],
            'Learning rate':config['agent']['hyperparameters']['alpha'],
            'Discount rate':config['agent']['hyperparameters']['gamma'],
            'Using sensors':config['agent']['use sensors']}

    walls = []
    window_limit = None
    icon = os.path.join(r"Resources\Sprites", "icon.png")
    agent_hyperparameters = config['agent']
    background_img = None
    if agent_hyperparameters['number of actions'] == 4:
        switcher = [
            [0, 1],
            [1, 0],
            [-1, 0],
            [0, -1]
        ]
    elif agent_hyperparameters['number of actions'] == 8:
        switcher = [
            [1, 0],
            [0, 1],
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, -1],
            [0, -1],
            [-1, 0]
        ]
    else:
        raise Exception(
            f"bad number of action for the agent! {agent_hyperparameters['number of actions']} is a bad value, the only values allowed are 4 and 8.")

    def __init__(self):
        if DataSaver.__instance__ is None:
            DataSaver.__instance__ = self
        else:
            raise Exception("You cannot create another Singleton class")

    @staticmethod
    def get_instance():
        if not DataSaver.__instance__:
            DataSaver()
        return DataSaver.__instance__

    def set_tree(self):
        instance = DataSaver.get_instance()
        objcts = [x.rect for x in instance.walls]
        instance.STRtree = shapely.strtree.STRtree(objcts)
