import Enemy
from EnemyBrain.BaseClasses import Env
import Player
import pygame as pg
import matplotlib.pyplot as plt
import tkinter as tk


class EnemyEnv(Env):
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    CLOCK = pg.time.Clock()

    def __init__(self):
        super().__init__()
        self.enemy = Enemy.Enemy((0, 0))
        self.player = Player.Player(self.enemy.instance.config['environment']['starting player pos']['x'],
                                    self.enemy.instance.config['environment']['starting player pos']['y'])
        # state is the pos + target pos
        self.state = self.set_state()
        # saving the hotkeys text so we won't have to generate it each render call
        font = pg.font.SysFont('Arial', 20)
        self.HOTKEYS_TEXT = font.render("Options: O. Load brain: L. Save Brain: S. See tracked data: G.", True,
                                        self.BLACK)

    def _reset(self):
        self.enemy.reset()
        self.player.reset()
        self.state = self.set_state()
        title = f"Episode {self.metrics['number of episodes'].get_latest()}."
        # if we have a privious episode to display
        if self.metrics['number of episodes'].get_latest() > 1:
            title += f" Episode {self.metrics['number of episodes'].get_latest() - 1} was a {'win' if self.metrics['number of wins'].get_latest() else 'lost'}."
        title += f" {self.enemy.instance.agent_name} on {self.enemy.instance.map_name}."
        pg.display.set_caption(title)

    def _render(self):
        if self.enemy.instance.config['game']['fps cap'] > 0:
            self.CLOCK.tick(self.enemy.instance.config['game']['fps cap'])
        pg.display.update()
        self.enemy.instance.screen.blit(self.enemy.instance.background_img, (0, 0))
        pg.draw.rect(self.enemy.instance.screen, self.enemy.color,
                     (self.enemy.pos_to_scale(), self.enemy.instance.draw_scaler))
        for wall in self.enemy.instance.walls:
            for loc in wall.locs():
                self.enemy.instance.screen.blit(self.enemy.instance.wall_img, loc)
        pg.draw.rect(self.enemy.instance.screen, self.BLACK,
                     (self.player.pos_to_scale(), self.enemy.instance.draw_scaler))
        self.enemy.instance.screen.blit(self.HOTKEYS_TEXT, (0, 0))

        pg.display.flip()
        pg.event.pump()

    def draw_additional_info(self):
        # drawing the
        window = tk.Tk()
        window.iconphoto(False, tk.PhotoImage(file=self.enemy.instance.icon))
        episode_num = self.metrics['number of episodes'].get_latest()
        window.title(f"Episode {episode_num}, Step {self.metrics['number of steps'].get_latest()}")

        def graph():
            win_rates = self.metrics['win rate over x'].vals
            plt.plot(range(episode_num - len(win_rates), episode_num),
                     win_rates)
            plt.xlabel('episode number')
            plt.ylabel('win rate %')
            plt.title(
                f"win rates over the last {self.metrics['win rate over x'].count_val} episodes of the last {len(win_rates)} episodes")
            plt.show()
            plt.plot(range(episode_num), self.metrics['win rate'].vals)
            plt.xlabel('episode number')
            plt.ylabel('win rate %')
            plt.title('win rates over the last {} episodes'.format(episode_num))
            plt.show()

        text = self.get_metrics_string()
        label = tk.Label(window, text=text)
        label.pack()
        button = tk.Button(window, text="Graph", command=graph)
        button.pack()
        button = tk.Button(window, text="Done", command=window.destroy)
        button.pack()
        window.mainloop()

    def set_state(self):
        enemy_pos = self.enemy.get_pos()
        player_pos = self.player.get_pos()
        return (
            enemy_pos[0], enemy_pos[1],
            player_pos[0], player_pos[1])

    def act(self, dir_num):
        ret = self.instance.switcher[dir_num]
        return ret

    def check_collision(self):
        return self.enemy.check_collision()

    def _step(self, action):
        self.player.update()
        self.enemy.act_on_dir(self.act(action))
        if self.enemy.check_collision():
            self.enemy.act_on_dir([-x for x in self.act(action)])
        done = False
        self.enemy.moves -= 1
        info = {'won': False}
        distance = self.touches()
        reward = -1
        if distance == 0:
            reward = 500
            done = True
            info['won'] = True

        # state is the pos  + target pos
        self.state = self.set_state()
        if self.enemy.moves == 0:
            done = True
        self._render()
        return self.state, reward, done, info

    def touches(self):
        return self.enemy.rect.centroid.distance(self.player.rect.centroid)
