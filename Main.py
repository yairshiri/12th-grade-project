import os
import psutil
import pygame as pg
import dataSaver
from Option_gui import Menu
from utils import progression_bar, alert, load_content


@progression_bar
def get_agent():
    from EnemyBrain.EnemyEnv import EnemyEnv
    name = dataSaver.DataSaver.get_instance().config['agent']['type']
    try:
        env = EnemyEnv()
        exec(f"from EnemyBrain.Agent import {name} as Agent", globals())
        agent = Agent(env)
        return agent
    except Exception as e:
        alert(f"Couldn't import the {name} agent")
        current_system_pid = os.getpid()
        ThisSystem = psutil.Process(current_system_pid)
        ThisSystem.terminate()


instance = dataSaver.DataSaver.get_instance()

load_content()

menu = Menu(instance.config)

agent = get_agent()
# loop over episodes
while True:
    # resetting for a new episode
    state = agent.env.reset()
    # checking if the training portion has ended and switching to testing
    if agent.mode == 'training' and not agent.check_stop():
        agent.mode = 'testing'
        alert(f"Won!\n{agent.env.get_metrics_string()}Switching to testing from training and saving!")
        agent.save()
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
