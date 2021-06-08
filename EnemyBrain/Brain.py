import dataSaver

name = dataSaver.DataSaver.get_instance().config['paths']['agent name']
exec(f"from EnemyBrain.Agent import {name} as Agent")
name = dataSaver.DataSaver.get_instance().config['paths']['environment name']
exec(f"from EnemyBrain.EnemyEnv import {name} as Env")


def get_agent():
    env = Env()
    # initialising the agents
    agent = Agent(env)
    return agent
