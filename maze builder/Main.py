from math import ceil
import pickle
import pygame as pg
import yaml
import tkinter as tk

# loading the data from the config file
config = yaml.safe_load(open(r'D:\Users\owner\PycharmProjects\Project_before_api\Resources\config.yml'))
maze_shape = (config['game']['maze shape']['width'],config['game']['maze shape']['height'])
screen_size = (config['game']['screen size']['width'],config['game']['screen size']['height'])
unit_size = (round(screen_size[0] / maze_shape[0]), round(screen_size[1] / maze_shape[1]))
WALL_TILE = pg.transform.scale(pg.image.load(
    open(r'D:\Users\owner\PycharmProjects\Project_before_api\Resources\Sprites\wall tile.png')), unit_size)
screen_size = (round(unit_size[0] * maze_shape[0]), round(unit_size[1] * maze_shape[1]))
# initializing the pygame screen and clock
pg.init()
screen = pg.display.set_mode(screen_size)
clock = pg.time.Clock()


# initializing the walls array and the starting pos and ending pos variables. they are used to save and create walls
walls = []
starting_pos = []
ending_pos = []


# the method we run until the program is closed
def get_walls():
    # running of capped 60 so it will be easier to see
    clock.tick(60)
    # starting new frame
    pg.display.update()
    # drawing the back ground. (255,255,255) is RGB for white.
    screen.fill((255,255,255))

    # drawing the lines
    for x in range(maze_shape[0]):
        pg.draw.line(screen, (0, 0, 0), (x * unit_size[0], 0), (x * unit_size[0], screen_size[1]))

    for y in range(maze_shape[1]):
        pg.draw.line(screen, (0, 0, 0), (0, y * unit_size[1]), (screen_size[0], y * unit_size[1]))

    for event in pg.event.get():
        if event.type == pg.QUIT:
            # when closing the program
            pg.quit()  # close the window
            # create the name choosing GUI
            window = tk.Tk()
            label = tk.Label(text="Maze name chooser")
            name = tk.StringVar()
            entry = tk.Entry(window, textvariable=name)
            quit_button = tk.Button(window, text="Done", command=window.destroy)
            quit_button.pack()
            label.pack()
            entry.pack()
            window.mainloop()
            # save the walls (the maze) to a file with the chosen name
            with open(f"{name.get()}.pickle", 'wb') as f:
                pickle.dump(walls, f)
            quit()
        elif event.type == pg.MOUSEBUTTONDOWN:
            # if the mouse button was clicked, save the position as the starting position
            global starting_pos
            starting_pos = list(pg.mouse.get_pos())
        elif event.type == pg.MOUSEBUTTONUP:
            # if the mouse button was released, save the position as the ending position
            global ending_pos
            ending_pos = list(pg.mouse.get_pos())
            # call the wall creating method because we have all we need to create a wall
            pos_to_wall()
    for wall in walls:
        for loc in locs(wall):
            screen.blit(WALL_TILE, loc)
    # ending the frame
    pg.display.flip()
    pg.event.pump()


def pos_to_wall():
    # fixing the stating and ending pos to be a corner of a wall
    global starting_pos
    starting_pos[0] //= unit_size[0]
    starting_pos[1] //= unit_size[1]
    global ending_pos
    ending_pos[0] //= unit_size[0]
    ending_pos[1] //= unit_size[1]
    # getting the information about the wall.
    width = abs(starting_pos[0] - ending_pos[0]) + 1
    height = abs(starting_pos[1] - ending_pos[1]) + 1
    # pos is the top left corner of the wall so we do this instead of just taking the starting pos
    pos = (min(starting_pos[0], ending_pos[0]), min(starting_pos[1], ending_pos[1]))
    global walls
    # adding the wall to the walls array
    walls.append({'p': pos, 'width': width, 'height': height})


def locs(wall):
    # get the locations of the places where we need to draw a wall tile
    for x in range(wall['width']):
        for y in range(wall['height']):
            yield (wall['p'][0] + x) * unit_size[0], (wall['p'][1] + y) * unit_size[1]

while True:
    get_walls()
