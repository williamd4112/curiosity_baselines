from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from gym import spaces

from pycolab.examples import (better_scrolly_maze, 
                              deepmind_maze, 
                              deepmind_5room, 
                              deepmind_5room_whitenoise,
                              deepmind_5room_flipped,
                              deepmind_5room_all,
                              deepmind_5room_long,
                              deepmind_5room_longunpadded,
                              deepmind_5room_longwide,
                              deepmind_5room_noobj,
                              deepmind_5room_oneobj,
                              deepmind_5room_onewhite,
                              deepmind_5room_randomfixed, 
                              deepmind_5room_bouncing,
                              deepmind_5room_brownian,
                              deepmind_5room_moveable,
                              deepmind_5room_moveable_stoch,
                              deepmind_5room_extint,
                              deepmind_5room_moveable_brownian,
                              deepmind_5roomlarge,
                              deepmind_5roomlarge_weather,
                              deepmind_5roomlarge_enemy,
                              deepmind_5roomlarge_whitenoise,
                              deepmind_5roomlargetext,
                              deepmind_5roomlargetext_whitenoise,
                              deepmind_5roomlarge_randomfixed,
                              deepmind_5roomlargetext_randomfixed,
                              deepmind_5roomlarge_moveable,
                              deepmind_5roomlarge_moveable_ext,
                              deepmind_5roomlargetext_moveable,
                              deepmind_5roomlarge_moveable_stoch,
                              deepmind_5roomlarge_moveable_stoch_ext,
                              deepmind_5roomlargetext_moveable_stoch,
                              deepmind_5roomlarge_moveable_brownian,
                              deepmind_5roomlargetext_moveable_brownian,
                              deepmind_5roomlarge_brownian,
                              deepmind_5roomlargetext_brownian,
                              deepmind_5roomlarge_all,
                              deepmind_5roomlargetext_all,
                              deepmind_8room,
                              deepmind_piano_long
                              )
from pycolab import cropping
from . import pycolab_env

#############################################################################################################################
######################################################## FIVE ROOM ##########################################################
#############################################################################################################################
class DeepmindMazeWorld_5room(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 1.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5room, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=223,
            color_palette=1,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5room_whitenoise(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 1.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5room_whitenoise, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=224,
            color_palette=1,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_whitenoise.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5room_flipped(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 1.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5room_flipped, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=223,
            color_palette=1,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_flipped.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5room_all(pycolab_env.PyColabEnv):
    """Map with all four objects (white noise, brownian, Fixed, moveable)
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5room_all, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=221,
            color_palette=1,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_all.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5room_long(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment with a long corridor.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5room_long, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=236,
            color_palette=1,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_long.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5room_longunpadded(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment with a long corridor.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5room_longunpadded, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=236,
            color_palette=1,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_longunpadded.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5room_longwide(pycolab_env.PyColabEnv):
    """5 room maze with a wider long corridor.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5room_longwide, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=208,
            color_palette=1,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_longwide.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5room_noobj(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 1 with no reachable objects.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5room_noobj, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=224,
            color_palette=1,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_noobj.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5room_oneobj(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 1 with just a fixed object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5room_oneobj, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=224,
            color_palette=1,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_oneobj.make_game(self.level)
    
    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5room_onewhite(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 1 with just a white noise object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5room_onewhite, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=224,
            color_palette=1,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_onewhite.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []


class DeepmindMazeWorld_5room_randomfixed(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 2.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5room_randomfixed, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=223,
            color_palette=1,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_randomfixed.make_game(self.level)
    
    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5room_bouncing(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 3.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a', 'b', 'c']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5room_bouncing, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=222,
            color_palette=1,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_bouncing.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5room_brownian(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 4.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5room_brownian, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=223,
            color_palette=1,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_brownian.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_maze(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 5.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_maze, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            act_null_value=4,
            resize_scale=17,
            visitable_states=150.,
            color_palette=0,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_maze.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5room_moveable(pycolab_env.PyColabEnv):
    """A 5 room environment with an affectable object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['e', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5room_moveable, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            act_null_value=4,
            resize_scale=17,
            visitable_states=223,
            color_palette=1,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_moveable.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5room_moveable_brownian(pycolab_env.PyColabEnv):
    """A 5 room environment with an affectable object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['e', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5room_moveable_brownian, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            act_null_value=4,
            resize_scale=17,
            visitable_states=223,
            color_palette=1,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_moveable_brownian.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5room_moveable_stoch(pycolab_env.PyColabEnv):
    """A 5 room environment with an affectable object that has stochastic movement.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['e', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5room_moveable_stoch, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            act_null_value=4,
            resize_scale=17,
            visitable_states=223,
            color_palette=1,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_moveable_stoch.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5room_extint(pycolab_env.PyColabEnv):
    """A 5 room environment with an affectable object that has stochastic movement.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5room_extint, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            act_null_value=4,
            resize_scale=17,
            visitable_states=223,
            color_palette=1,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_extint.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_piano_long(pycolab_env.PyColabEnv):
    """A 5 room environment with an affectable object that has stochastic movement.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_piano_long, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            act_null_value=4,
            resize_scale=17,
            visitable_states=1516,
            color_palette=1,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_piano_long.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

#############################################################################################################################
###################################################### 5 ROOM LARGE #########################################################
#############################################################################################################################
class DeepmindMazeWorld_5roomlarge(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5roomlarge, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=915,
            color_palette=3,
            reward_switch=[],) # 3, 21

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5roomlarge.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5roomlarge_enemy(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5roomlarge_enemy, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=915,
            color_palette=3,
            reward_switch=[],) # 3, 21

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5roomlarge_enemy.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5roomlarge_weather(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=1.0,):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5roomlarge_weather, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=913,
            color_palette=3,
            reward_switch=['a', 'b', 'c', 'd'],) # 3, 21

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5roomlarge_weather.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5roomlargetext(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5roomlargetext, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=891,
            color_palette=3,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5roomlargetext.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5roomlarge_whitenoise(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5roomlarge_whitenoise, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=916,
            color_palette=3,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5roomlarge_whitenoise.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5roomlargetext_whitenoise(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5roomlargetext_whitenoise, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=892,
            color_palette=3,
            reward_switch=[],) # 3, 21

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5roomlargetext_whitenoise.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5roomlarge_randomfixed(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5roomlarge_randomfixed, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=915,
            color_palette=3,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5roomlarge_randomfixed.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5roomlargetext_randomfixed(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5roomlargetext_randomfixed, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=891,
            color_palette=3,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5roomlargetext_randomfixed.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5roomlarge_moveable(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['e', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5roomlarge_moveable, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=915,
            color_palette=3,
            reward_switch=[],) # 3, 21

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5roomlarge_moveable.make_game(self.level)
    
    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5roomlarge_moveable_ext(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['e']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5roomlarge_moveable_ext, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=915,
            color_palette=3,
            reward_switch=[],) # 3, 21

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5roomlarge_moveable_ext.make_game(self.level)
    
    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5roomlargetext_moveable(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['e', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5roomlargetext_moveable, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=891,
            color_palette=3,
            reward_switch=[],) # 3, 21

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5roomlargetext_moveable.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5roomlarge_moveable_stoch(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['e', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5roomlarge_moveable_stoch, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=915,
            color_palette=3,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5roomlarge_moveable_stoch.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5roomlarge_moveable_stoch_ext(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['e']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5roomlarge_moveable_stoch_ext, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=915,
            color_palette=3,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5roomlarge_moveable_stoch_ext.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5roomlargetext_moveable_stoch(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['e', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5roomlargetext_moveable_stoch, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=891,
            color_palette=3,
            reward_switch=[],) # 3, 21

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5roomlargetext_moveable_stoch.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5roomlarge_moveable_brownian(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['e', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5roomlarge_moveable_brownian, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=915,
            color_palette=3,
            reward_switch=[],) # 3, 21

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5roomlarge_moveable_brownian.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5roomlargetext_moveable_brownian(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['e', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5roomlargetext_moveable_brownian, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=891,
            color_palette=3,
            reward_switch=[],) # 3, 21

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5roomlargetext_moveable_brownian.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5roomlarge_brownian(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5roomlarge_brownian, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=915,
            color_palette=3,
            reward_switch=[],) # 3, 21

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5roomlarge_brownian.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5roomlargetext_brownian(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5roomlargetext_brownian, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=891,
            color_palette=3,
            reward_switch=[],) # 3, 21

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5roomlargetext_brownian.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5roomlarge_all(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5roomlarge_all, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=913,
            color_palette=3,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5roomlarge_all.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_5roomlargetext_all(pycolab_env.PyColabEnv):
    """Large version of the 5 room environment
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_5roomlargetext_all, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=889,
            color_palette=3,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5roomlargetext_all.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

#############################################################################################################################
####################################################### EIGHT ROOM ##########################################################
#############################################################################################################################
class DeepmindMazeWorld_8room(pycolab_env.PyColabEnv):
    """An eight room environment with many fixed objects.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_8room, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            act_null_value=4,
            resize_scale=17,
            visitable_states=633,
            color_palette=2,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_8room.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_8room_rgb(pycolab_env.PyColabEnv):
    """An eight room environment with many fixed objects.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_8room_rgb, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            act_null_value=4,
            resize_scale=17,
            visitable_states=633,
            color_palette=3,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_8room.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []

class DeepmindMazeWorld_8room_diff(pycolab_env.PyColabEnv):
    """An eight room environment with many fixed objects.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        self.obs_type = obs_type
        super(DeepmindMazeWorld_8room_diff, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            act_null_value=4,
            resize_scale=17,
            visitable_states=633,
            color_palette=0,
            reward_switch=[],)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_8room.make_game(self.level)

    def make_croppers(self):
        if self.obs_type in {'rgb', 'mask'}:
          return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]
        elif self.obs_type == 'rgb_full':
          return []
