import dataSaver

name = dataSaver.DataSaver.get_instance().config['paths']['agent name']
exec(f"from EnemyBrain.Agent import {name} as Agent")
from EnemyBrain.EnemyEnv import EnemyEnv as Env


def get_agent():
    env = Env()
    # initialising the agents
    agent = Agent(env)
    return agent
