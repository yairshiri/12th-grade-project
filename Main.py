import os
import psutil
import pygame as pg
import dataSaver
from Option_gui import Menu
from utils import progression_bar, alert, load_content,ask


@progression_bar
def get_agent():
    agent_name = dataSaver.DataSaver.get_instance().config['agent']['type']
    env_name = dataSaver.DataSaver.get_instance().config['environment']['type']
    try:

        exec(f"from EnemyBrain.Agent import {agent_name} as Agent", globals())
        exec(f"from EnemyBrain.EnemyEnv import {env_name} as Env", globals())
        ret_env = Env()
        ret_agent = Agent(ret_env)
        return ret_agent
    except Exception as e:
        alert(f"Couldn't import the {agent_name} agent")
        current_system_pid = os.getpid()
        ThisSystem = psutil.Process(current_system_pid)
        ThisSystem.terminate()


instance = dataSaver.DataSaver.get_instance()

load_content()

menu = Menu(instance.config)

agent = get_agent()

switch = True
# loop over episodes
while True:
    # resetting for a new episode
    state = agent.env.reset()
    # checking if the training portion has ended and switching to testing
    if switch and not agent.check_stop():
        option = ask(f"Won!\n{agent.env.get_metrics_string()}\nDo you want to keep training or start testing?",["Training","Testing"])[1]
        if option == "Testing":
            agent.mode = 'testing'
            agent.save()
        switch = False
    # the step loop
    while True:
        # checking if the program needs to exit or open certain guis
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()  # close the window
                quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_o:
                    menu.render()
                elif event.key == pg.K_g:
                    agent.env.draw_additional_info()
                elif event.key == pg.K_s:
                    agent.save()
                elif event.key == pg.K_l:
                    agent.load()

        action = agent.action(state)
        # getting the timestep
        next_state, reward, done, info = agent.env.step(action)
        # learning only if the mode is training
        if agent.mode == 'training':
            # let's teach our agents to do something! hopefully they learn.
            agent.learn(state, action, reward, next_state, done=done)

        # checking if the episode is done
        if done:
            break

        # update state and action
        state = next_state
