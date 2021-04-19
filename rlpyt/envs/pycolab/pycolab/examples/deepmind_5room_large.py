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

from pycolab import ascii_art
from pycolab import cropping
from pycolab import human_ui
from pycolab import things as plab_things
from pycolab.prefab_parts import sprites as prefab_sprites

# pylint: disable=line-too-long
MAZES_ART = [
    # Each maze in MAZES_ART must have exactly one of the object sprites
    # 'a', 'b', 'c', 'd' and 'e'. I guess if you really don't want them in your maze
    # can always put them down in an unreachable part of the map or something.
    #
    # Make sure that the Player will have no way to "escape" the maze.
    #
    # Legend:
    #     '#': impassable walls.            'a': fixed object A.
    #     'P': player starting location.    'b': white noise object B.
    #     ' ': boring old maze floor.
    #
    # Room layout:
    #   2
    # 1 0 3
    #   4

    # Maze #0: (paper: 5 rooms environment)
   ['#############################',
    '##                         ##',
    '# #                       # #',
    '#  #                 @   #  #',
    '#   #                   #   #',
    '#    #                 #    #',
    '#     #               #     #',
    '#      #      e      #      #',
    '#       #           #       #',
    '#        #         #        #',
    '#         #### ####         #',
    '#         #### ####         #',
    '#         ##     ##         #',
    '#         ##     ##         #',
    '#             P             #',
    '#         ##     ##         #',
    '#         ##     ##         #',
    '#         #### ####         #',
    '#         #### ####         #',
    '#        #         #        #',
    '#       #           #       #',
    '#      #      b      #      #',
    '#     #               #     #',
    '#    #                 #    #',
    '#   #                   #   #',
    '#  #                     #  #',
    '# #                       # #',
    '##                         ##',
    '#############################']
]

# These colours are only for humans to see in the CursesUi.
COLOUR_FG = {' ': (0, 0, 0),        # Default black background
             '#': (764, 0, 999),    # Walls of the maze
             'P': (0, 999, 999),    # This is you, the player
             'e': (99, 140, 140),   # Patroller A
             'b': (145, 987, 341)}  # Patroller B

COLOUR_BG = {'@': (0, 0, 0)}  # Target spot

ENEMIES = {'e', 'b'} # Globally accessible set of sprites

# Empty coordinates corresponding to each numbered room (width 1 passageways not blocked)
ROOMS = {
  0: [[12, 12], [12, 13], [12, 14], [12, 15], [12, 16], [13, 12], [13, 13], [13, 14], [13, 15], [13, 16], [14, 12], [14, 13], [14, 14], [14, 15], [14, 16], [15, 12], [15, 13], [15, 14], [15, 15], [15, 16], [16, 12], [16, 13], [16, 14], [16, 15], [16, 16]],
  1: [[4, 2], [5, 2], [5, 3], [6, 2], [6, 3], [6, 4], [7, 2], [7, 3], [7, 4], [7, 5], [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [10, 2], [10, 3], [10, 4], [10, 5], [10, 6], [10, 7], [10, 8], [11, 2], [11, 3], [11, 4], [11, 5], [11, 6], [11, 7], [11, 8], [11, 9], [12, 2], [12, 3], [12, 4], [12, 5], [12, 6], [12, 7], [12, 8], [12, 9], [13, 2], [13, 3], [13, 4], [13, 5], [13, 6], [13, 7], [13, 8], [14, 2], [14, 3], [14, 4], [14, 5], [14, 6], [14, 7], [14, 8], [15, 2], [15, 3], [15, 4], [15, 5], [15, 6], [15, 7], [15, 8], [16, 2], [16, 3], [16, 4], [16, 5], [16, 6], [16, 7], [16, 8], [16, 9], [17, 2], [17, 3], [17, 4], [17, 5], [17, 6], [17, 7], [17, 8], [17, 9], [18, 2], [18, 3], [18, 4], [18, 5], [18, 6], [18, 7], [18, 8], [19, 2], [19, 3], [19, 4], [19, 5], [19, 6], [19, 7], [20, 2], [20, 3], [20, 4], [20, 5], [20, 6], [21, 2], [21, 3], [21, 4], [21, 5], [22, 2], [22, 3], [22, 4], [23, 2], [23, 3], [24, 2]],
  2: [[2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [2, 11], [2, 12], [2, 13], [2, 14], [2, 15], [2, 16], [2, 17], [2, 18], [2, 19], [2, 20], [2, 21], [2, 22], [2, 23], [2, 24], [3, 5], [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [3, 11], [3, 12], [3, 13], [3, 14], [3, 15], [3, 16], [3, 17], [3, 18], [3, 19], [3, 20], [3, 21], [3, 22], [3, 23], [4, 6], [4, 7], [4, 8], [4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 16], [4, 17], [4, 18], [4, 19], [4, 20], [4, 21], [4, 22], [5, 7], [5, 8], [5, 9], [5, 10], [5, 11], [5, 12], [5, 13], [5, 14], [5, 15], [5, 16], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 8], [6, 9], [6, 10], [6, 11], [6, 12], [6, 13], [6, 14], [6, 15], [6, 16], [6, 17], [6, 18], [6, 19], [6, 20], [7, 9], [7, 10], [7, 11], [7, 12], [7, 13], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [8, 10], [8, 11], [8, 12], [8, 13], [8, 14], [8, 15], [8, 16], [8, 17], [8, 18], [9, 11], [9, 12], [9, 16], [9, 17]],
  3: [[4, 26], [5, 25], [5, 26], [6, 24], [6, 25], [6, 26], [7, 23], [7, 24], [7, 25], [7, 26], [8, 22], [8, 23], [8, 24], [8, 25], [8, 26], [9, 21], [9, 22], [9, 23], [9, 24], [9, 25], [9, 26], [10, 20], [10, 21], [10, 22], [10, 23], [10, 24], [10, 25], [10, 26], [11, 19], [11, 20], [11, 21], [11, 22], [11, 23], [11, 24], [11, 25], [11, 26], [12, 19], [12, 20], [12, 21], [12, 22], [12, 23], [12, 24], [12, 25], [12, 26], [13, 20], [13, 21], [13, 22], [13, 23], [13, 24], [13, 25], [13, 26], [14, 20], [14, 21], [14, 22], [14, 23], [14, 24], [14, 25], [14, 26], [15, 20], [15, 21], [15, 22], [15, 23], [15, 24], [15, 25], [15, 26], [16, 19], [16, 20], [16, 21], [16, 22], [16, 23], [16, 24], [16, 25], [16, 26], [17, 19], [17, 20], [17, 21], [17, 22], [17, 23], [17, 24], [17, 25], [17, 26], [18, 20], [18, 21], [18, 22], [18, 23], [18, 24], [18, 25], [18, 26], [19, 21], [19, 22], [19, 23], [19, 24], [19, 25], [19, 26], [20, 22], [20, 23], [20, 24], [20, 25], [20, 26], [21, 23], [21, 24], [21, 25], [21, 26], [22, 24], [22, 25], [22, 26], [23, 25], [23, 26], [24, 26]],
  4: [[19, 11], [19, 12], [19, 16], [19, 17], [20, 10], [20, 11], [20, 12], [20, 13], [20, 14], [20, 15], [20, 16], [20, 17], [20, 18], [21, 9], [21, 10], [21, 11], [21, 12], [21, 13], [21, 14], [21, 15], [21, 16], [21, 17], [21, 18], [21, 19], [22, 8], [22, 9], [22, 10], [22, 11], [22, 12], [22, 13], [22, 14], [22, 15], [22, 16], [22, 17], [22, 18], [22, 19], [22, 20], [23, 7], [23, 8], [23, 9], [23, 10], [23, 11], [23, 12], [23, 13], [23, 14], [23, 15], [23, 16], [23, 17], [23, 18], [23, 19], [23, 20], [23, 21], [24, 6], [24, 7], [24, 8], [24, 9], [24, 10], [24, 11], [24, 12], [24, 13], [24, 14], [24, 15], [24, 16], [24, 17], [24, 18], [24, 19], [24, 20], [24, 21], [24, 22], [25, 5], [25, 6], [25, 7], [25, 8], [25, 9], [25, 10], [25, 11], [25, 12], [25, 13], [25, 14], [25, 15], [25, 16], [25, 17], [25, 18], [25, 19], [25, 20], [25, 21], [25, 22], [25, 23], [26, 4], [26, 5], [26, 6], [26, 7], [26, 8], [26, 9], [26, 10], [26, 11], [26, 12], [26, 13], [26, 14], [26, 15], [26, 16], [26, 17], [26, 18], [26, 19], [26, 20], [26, 21], [26, 22], [26, 23], [26, 24]]
}

def make_game(level):
  """Builds and returns a Better Scrolly Maze game for the selected level."""
  maze_ascii = MAZES_ART[level]

  return ascii_art.ascii_art_to_game(
      maze_ascii, what_lies_beneath=' ',
      sprites={
          'P': PlayerSprite,
          'e': MoveableObject,
          'b': WhiteNoiseObject},
      update_schedule=['P', 'e', 'b'],
      z_order='ebP')

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
        corner, position, character, impassable='#')
    self.last_position = None # store last position for moveable object
    self.last_action = None # store last action for moveable object

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del backdrop, layers  # Unused

    self.last_position = self.position
    self.last_action = actions
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

class WhiteNoiseObject(prefab_sprites.MazeWalker):
  """Randomly sample direction from left/right/up/down"""

  def __init__(self, corner, position, character):
    """Constructor: list impassables, initialise direction."""
    super(WhiteNoiseObject, self).__init__(corner, position, character, impassable='#')
    # Initialize empty space in surrounding radius.
    self._empty_coords = ROOMS[4]

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del actions, backdrop  # Unused.
    self._teleport(self._empty_coords[np.random.choice(len(self._empty_coords))])

class MoveableObject(prefab_sprites.MazeWalker):
  """Moveable object. Can be pushed by agent."""

  def __init__(self, corner, position, character):
    super(MoveableObject, self).__init__(corner, position, character, impassable='#b')

  def update(self, actions, board, layers, backdrop, things, the_plot):
    mr, mc = self.position
    pr, pc = things['P'].last_position
    p_action = things['P'].last_action

    # move up
    if (mc == pc) and (mr - pr == -1) and (p_action == 0):
      moved = self._north(board, the_plot)
      if moved is not None:
        things['P']._south(board, the_plot)

    # move down
    elif (mc == pc) and (mr - pr == 1) and (p_action == 1):
      exiting_room = (self.position == (8, 14))
      if exiting_room == True:
        things['P']._north(board, the_plot)
        self._stay(board, the_plot)
      else:
        moved = self._south(board, the_plot)
        if moved is not None: # obstructed
          things['P']._north(board, the_plot)

    # move right
    elif (mc - pc == 1) and (mr == pr) and (p_action == 3):
      exiting_room = (self.position == (9, 13))
      if exiting_room == True:
        things['P']._west(board, the_plot)
        self._stay(board, the_plot)
      else:
        moved = self._east(board, the_plot)
        if moved is not None: # obstructed
          things['P']._west(board, the_plot)

    # move left
    elif (mc - pc == -1) and (mr == pr) and (p_action == 2):
      exiting_room = (self.position == (9, 15))
      if exiting_room == True:
        things['P']._east(board, the_plot)
        self._stay(board, the_plot)
      else:
        moved = self._west(board, the_plot)
        if moved is not None: # obstructed
          things['P']._east(board, the_plot)

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
