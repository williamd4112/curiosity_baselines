# Copyright 2017 the pycolab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An implementation of the environments from 'World Discovery Models'[https://arxiv.org/pdf/1902.07685.pdf]. 
Learn to explore!

This environment uses a simple scrolling mechanism: cropping! As far as the pycolab engine is concerned, 
the game world doesn't scroll at all: it just renders observations that are the size
of the entire map. Only later do "cropper" objects crop out a part of the
observation to give the impression of a moving world/partial observability.

Keys: up, down, left, right - move. q - quit.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import curses

import sys
import numpy as np
from copy import deepcopy

from pycolab import ascii_art
from pycolab import cropping
from pycolab import human_ui
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites

# pylint: disable=line-too-long
MAZES_ART = [
    # Each maze in MAZES_ART must have exactly one of the object sprites
    # 'a', 'b', 'c', 'd' and 'e', 'f', 'g', 'h'. I guess if you really don't want them in your maze
    # can always put them down in an unreachable part of the map or something.
    #
    # Make sure that the Player will have no way to "escape" the maze.
    #
    # Legend:
    #     '#': impassable walls.            'a': fixed object A.
    #     'P': player starting location.    'b': fixed object B.
    #     ' ': boring old maze floor.       'c': fixed object C.
    #                                       'd': fixed object D.
    #                                       'e': fixed object E.
    #                                       'f': fixed object F.
    #                                       'g': fixed object G.
    #                                       'h': fixed object H.
    #
    # Room layout:
    # 8 1 2
    # 7 0 3
    # 6 5 4

    ['#####################################',
    '#####################################',
    '#####################################',
    '############           ##############',
    '######### ##           ## ###########',
    '#######   ###         ###   #########',
    '#######   ###         ###   #########',
    '#####      ###       ###      #######',
    '#####      ###       ###      #######',
    '###         ###     ###         #####',
    '###         ###     ###         #####',
    '##           ##     ##           ####',
    '#####        ##     ##        #######',
    '#######       ### ###       #########',
    '#  ######     ##   ##     ######  ###',
    '#    ######             ######    ###',
    '#      ######         ######      ###',
    '#          ##         ##          ###',
    '#          #           #          ###',
    '#                P           a    ###',
    '#          #           #          ###',
    '#          ##         ##          ###',
    '#      ######         ######      ###',
    '#    ######             ######    ###',
    '#  ######     ##   ##     ######  ###',
    '#######       ### ###       #########',
    '#####        ##     ##        #######',
    '##           ##     ##           ####',
    '###         ###     ###         #####',
    '###         ###     ###         #####',
    '#####      ###       ###      #######',
    '#####      ###       ###      #######',
    '#######   ###         ###   #########',
    '#######   ###         ###   #########',
    '######### ##           ## ###########',
    '############           ##############',
    '#####################################',]
]

# These colours are only for humans to see in the CursesUi.
COLOUR_FG = {' ': (0, 0, 0),        # Default black background
             '#': (764, 0, 999),    # Walls of the maze
             'P': (0, 999, 999),    # This is you, the player
             'a': (999, 0, 780)}    # Patroller A

COLOUR_BG = {'@': (0, 0, 0)}  # So the coins look like @ and not solid blocks.

ENEMIES = {'a'} # Globally accessible set of sprites

# Empty coordinates corresponding to each numbered room (width 1 passageways not blocked)
ROOMS = {
  0 : [[15, 14], [15, 15], [15, 16], [15, 17], [15, 18], [15, 19], [15, 20], [16, 13], [16, 14], [16, 15], [16, 16], [16, 17], [16, 18], [16, 19], [16, 20], [16, 21], [17, 13], [17, 14], [17, 15], [17, 16], [17, 17], [17, 18], [17, 19], [17, 20], [17, 21], [18, 13], [18, 14], [18, 15], [18, 16], [18, 17], [18, 18], [18, 19], [18, 20], [18, 21], [19, 13], [19, 14], [19, 15], [19, 16], [19, 17], [19, 18], [19, 19], [19, 20], [19, 21], [20, 13], [20, 14], [20, 15], [20, 16], [20, 17], [20, 18], [20, 19], [20, 20], [20, 21], [21, 13], [21, 14], [21, 15], [21, 16], [21, 17], [21, 18], [21, 19], [21, 20], [21, 21], [22, 13], [22, 14], [22, 15], [22, 16], [22, 17], [22, 18], [22, 19], [22, 20], [22, 21], [23, 14], [23, 15], [23, 16], [23, 17], [23, 18], [23, 19], [23, 20]],
  1 : [[4, 15], [4, 16], [4, 17], [4, 18], [4, 19], [5, 15], [5, 16], [5, 17], [5, 18], [5, 19], [6, 15], [6, 16], [6, 17], [6, 18], [6, 19], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [8, 15], [8, 16], [8, 17], [8, 18], [8, 19], [9, 15], [9, 16], [9, 17], [9, 18], [9, 19], [10, 15], [10, 16], [10, 17], [10, 18], [10, 19]],
  2 : [[7, 26], [8, 25], [8, 26], [8, 27], [9, 24], [9, 25], [9, 26], [9, 27], [9, 28], [10, 23], [10, 24], [10, 25], [10, 26], [10, 27], [10, 28], [10, 29], [11, 22], [11, 23], [11, 24], [11, 25], [11, 26], [11, 27], [11, 28], [12, 23], [12, 24], [12, 25], [12, 26], [12, 27], [13, 24], [13, 25], [13, 26], [14, 25]],
  3 : [[17, 26], [17, 27], [17, 28], [17, 29], [17, 30], [17, 31], [17, 32], [18, 26], [18, 27], [18, 28], [18, 29], [18, 30], [18, 31], [18, 32], [19, 26], [19, 27], [19, 28], [19, 29], [19, 30], [19, 31], [19, 32], [20, 26], [20, 27], [20, 28], [20, 29], [20, 30], [20, 31], [20, 32], [21, 26], [21, 27], [21, 28], [21, 29], [21, 30], [21, 31], [21, 32]],
  4 : [[24, 25], [25, 24], [25, 25], [25, 26], [26, 23], [26, 24], [26, 25], [26, 26], [26, 27], [27, 22], [27, 23], [27, 24], [27, 25], [27, 26], [27, 27], [27, 28], [28, 23], [28, 24], [28, 25], [28, 26], [28, 27], [28, 28], [28, 29], [29, 24], [29, 25], [29, 26], [29, 27], [29, 28], [30, 25], [30, 26], [30, 27], [31, 26]],
  5 : [[28, 15], [28, 16], [28, 17], [28, 18], [28, 19], [29, 15], [29, 16], [29, 17], [29, 18], [29, 19], [30, 15], [30, 16], [30, 17], [30, 18], [30, 19], [31, 15], [31, 16], [31, 17], [31, 18], [31, 19], [32, 15], [32, 16], [32, 17], [32, 18], [32, 19], [33, 15], [33, 16], [33, 17], [33, 18], [33, 19], [34, 15], [34, 16], [34, 17], [34, 18], [34, 19]],
  6 : [[24, 9], [25, 8], [25, 9], [25, 10], [26, 7], [26, 8], [26, 9], [26, 10], [26, 11], [27, 6], [27, 7], [27, 8], [27, 9], [27, 10], [27, 11], [27, 12], [28, 5], [28, 6], [28, 7], [28, 8], [28, 9], [28, 10], [28, 11], [29, 6], [29, 7], [29, 8], [29, 9], [29, 10], [30, 7], [30, 8], [30, 9], [31, 8]],
  7 : [[17, 2], [17, 3], [17, 4], [17, 5], [17, 6], [17, 7], [17, 8], [18, 2], [18, 3], [18, 4], [18, 5], [18, 6], [18, 7], [18, 8], [19, 2], [19, 3], [19, 4], [19, 5], [19, 6], [19, 7], [19, 8], [20, 2], [20, 3], [20, 4], [20, 5], [20, 6], [20, 7], [20, 8], [21, 2], [21, 3], [21, 4], [21, 5], [21, 6], [21, 7], [21, 8]],
  8 : [[7, 8], [8, 7], [8, 8], [8, 9], [9, 6], [9, 7], [9, 8], [9, 9], [9, 10], [10, 5], [10, 6], [10, 7], [10, 8], [10, 9], [10, 10], [10, 11], [11, 6], [11, 7], [11, 8], [11, 9], [11, 10], [11, 11], [11, 12], [12, 7], [12, 8], [12, 9], [12, 10], [12, 11], [13, 8], [13, 9], [13, 10], [14, 9]]
}

def make_game(level):
  """Builds and returns a Better Scrolly Maze game for the selected level."""
  maze_ascii = deepcopy(MAZES_ART[level])
  
  # change location of fixed object in it's starting room
  for row in range(3, 35):
    if 'a' in maze_ascii[row]:
      maze_ascii[row] = maze_ascii[row].replace('a', ' ', 1)
      new_coord = random.sample(ROOMS[3], 1)[0]
      maze_ascii[new_coord[0]] = maze_ascii[new_coord[0]][:new_coord[1]] + 'a' + maze_ascii[new_coord[0]][new_coord[1]+1:]
  return ascii_art.ascii_art_to_game(
      maze_ascii, what_lies_beneath=' ',
      sprites={
          'P': PlayerSprite,
          'a': FixedObject},
      update_schedule=['P', 'a'],
      z_order='aP')

def make_croppers(level):
  """Builds and returns `ObservationCropper`s for the selected level.

  We make one cropper for each level: centred on the player. Room
  to add more if needed.

  Args:
    level: level to make `ObservationCropper`s for.

  Returns:
    a list of all the `ObservationCropper`s needed.
  """
  return [
      # The player view.
      cropping.ScrollingCropper(rows=5, cols=5, to_track=['P']),
  ]

class PlayerSprite(prefab_sprites.MazeWalker):
  """A `Sprite` for our player, the maze explorer."""

  def __init__(self, corner, position, character):
    """Constructor: just tells `MazeWalker` we can't walk through walls or objects."""
    super(PlayerSprite, self).__init__(
        corner, position, character, impassable='#a')

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del backdrop, layers  # Unused

    if actions == 0:    # go upward?
      self._north(board, the_plot)
    elif actions == 1:  # go downward?
      self._south(board, the_plot)
    elif actions == 2:  # go leftward?
      self._west(board, the_plot)
    elif actions == 3:  # go rightward?
      self._east(board, the_plot)
    elif actions == 4:  # stay put? (Not strictly necessary.)
      self._stay(board, the_plot)
    if actions == 5:    # just quit?
      the_plot.terminate_episode()

class FixedObject(plab_things.Sprite):
  """Static object. Doesn't move."""

  def __init__(self, corner, position, character):
    super(FixedObject, self).__init__(
        corner, position, character)

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del actions, backdrop  # Unused.

def main(argv=()):
  level = int(argv[1]) if len(argv) > 1 else 0

  # Build the game.
  game = make_game(level)
  # Build the croppers we'll use to scroll around in it, etc.
  croppers = make_croppers(level)

  # Make a CursesUi to play it with.
  ui = human_ui.CursesUi(
      keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                       curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                       -1: 4,
                       'q': 5, 'Q': 5},
      delay=100, colour_fg=COLOUR_FG, colour_bg=COLOUR_BG,
      croppers=croppers)

  # Let the game begin!
  ui.play(game)


if __name__ == '__main__':
  main(sys.argv)
