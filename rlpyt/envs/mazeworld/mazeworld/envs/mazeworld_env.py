from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from gym import spaces

from pycolab.examples import (better_scrolly_maze, 
                              deepmind_maze, 
                              deepmind_5room, 
                              deepmind_5room_long,
                              deepmind_5room_noobj,
                              deepmind_5room_oneobj,
                              deepmind_5room_onewhite,
                              deepmind_5room_randomfixed, 
                              deepmind_5room_bouncing,
                              deepmind_5room_brownian,
                              deepmind_8room,
                              deepmind_8room_oneobj_singleroom,
                              deepmind_8room_oneobj_allrooms,
                              deepmind_5room_moveable,
                              deepmind_5room_moveable_stoch,
                              deepmind_5room_extint
                              )
from pycolab import cropping
from . import pycolab_env

class MazeWorld(pycolab_env.PyColabEnv):
    """Custom maze world game.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 extrinsic_reward=0.0):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e', '@']
        self.state_layer_chars = ['P', '#'] + self.objects
        super(MazeWorld, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            act_null_value=4,
            resize_scale=17,
            visitable_states=150)

    def make_game(self):
        self._croppers = self.make_croppers()
        return better_scrolly_maze.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]

class DeepmindMazeWorld_5room(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 1.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 extrinsic_reward=0.0):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        super(DeepmindMazeWorld_5room, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=223,
            extrinsic_reward=0.0,
            extrinsic_reward_obj=None)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]

class DeepmindMazeWorld_5room_long(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment with a long corridor.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 extrinsic_reward=0.0):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        super(DeepmindMazeWorld_5room_long, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=235,
            extrinsic_reward=0.0,
            extrinsic_reward_obj=None)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_long.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]

class DeepmindMazeWorld_5room_noobj(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 1 with no reachable objects.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 extrinsic_reward=0.0):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        super(DeepmindMazeWorld_5room_noobj, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=224,
            extrinsic_reward=0.0,
            extrinsic_reward_obj=None)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_noobj.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]

class DeepmindMazeWorld_5room_oneobj(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 1 with just a fixed object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 extrinsic_reward=0.0):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        super(DeepmindMazeWorld_5room_oneobj, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=224,
            extrinsic_reward=0.0,
            extrinsic_reward_obj=None)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_oneobj.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]

class DeepmindMazeWorld_5room_onewhite(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 1 with just a white noise object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 extrinsic_reward=0.0):
        self.level = level
        self.objects = ['b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        super(DeepmindMazeWorld_5room_onewhite, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=224,
            extrinsic_reward=0.0,
            extrinsic_reward_obj=None)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_onewhite.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]


class DeepmindMazeWorld_5room_randomfixed(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 2.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 extrinsic_reward=0.0):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        super(DeepmindMazeWorld_5room_randomfixed, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=223,
            extrinsic_reward=0.0,
            extrinsic_reward_obj=None)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_randomfixed.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]

class DeepmindMazeWorld_5room_bouncing(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 3.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 extrinsic_reward=0.0):
        self.level = level
        self.objects = ['a', 'b', 'c']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        super(DeepmindMazeWorld_5room_bouncing, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=222,
            extrinsic_reward=0.0,
            extrinsic_reward_obj=None)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_bouncing.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]

class DeepmindMazeWorld_5room_brownian(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 4.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 extrinsic_reward=0.0):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        super(DeepmindMazeWorld_5room_brownian, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            resize_scale=17,
            visitable_states=223,
            extrinsic_reward=0.0,
            extrinsic_reward_obj=None)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_brownian.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]

class DeepmindMazeWorld_maze(pycolab_env.PyColabEnv):
    """Deepmind World Discovery Models experiment 5.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 extrinsic_reward=0.0):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        super(DeepmindMazeWorld_maze, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            act_null_value=4,
            resize_scale=17,
            visitable_states=150.,
            extrinsic_reward=0.0,
            extrinsic_reward_obj=None)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_maze.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]

class DeepmindMazeWorld_8room(pycolab_env.PyColabEnv):
    """An eight room environment with many fixed objects.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 extrinsic_reward=0.0):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        super(DeepmindMazeWorld_8room, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            act_null_value=4,
            resize_scale=17,
            visitable_states=833,
            extrinsic_reward=0.0,
            extrinsic_reward_obj=None)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_8room.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]

class DeepmindMazeWorld_8room_extrinsic(pycolab_env.PyColabEnv):
    """An eight room environment with many fixed objects, and object D gives extrinsic reward.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 extrinsic_reward=0.0):
        self.level = level
        self.objects = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        super(DeepmindMazeWorld_8room_extrinsic, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            act_null_value=4,
            resize_scale=17,
            visitable_states=833,
            extrinsic_reward=extrinsic_reward,
            extrinsic_reward_obj='d')

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_8room.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]

class DeepmindMazeWorld_8room_oneobj_singleroom(pycolab_env.PyColabEnv):
    """An eight room environment with one fixed object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 extrinsic_reward=0.0):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        super(DeepmindMazeWorld_8room_oneobj_singleroom, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            act_null_value=4,
            resize_scale=17,
            visitable_states=833,
            extrinsic_reward=0.0,
            extrinsic_reward_obj=None)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_8room_oneobj_singleroom.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]

class DeepmindMazeWorld_8room_oneobj_allrooms(pycolab_env.PyColabEnv):
    """An eight room environment with one fixed object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 extrinsic_reward=0.0):
        self.level = level
        self.objects = ['a']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        super(DeepmindMazeWorld_8room_oneobj_allrooms, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            act_null_value=4,
            resize_scale=17,
            visitable_states=833,
            extrinsic_reward=0.0,
            extrinsic_reward_obj=None)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_8room_oneobj_allrooms.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]

class DeepmindMazeWorld_5room_moveable(pycolab_env.PyColabEnv):
    """A 5 room environment with an affectable object.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 extrinsic_reward=0.0):
        self.level = level
        self.objects = ['e', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        super(DeepmindMazeWorld_5room_moveable, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            act_null_value=4,
            resize_scale=17,
            visitable_states=223,
            extrinsic_reward=0.0,
            extrinsic_reward_obj=None)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_moveable.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]


class DeepmindMazeWorld_5room_moveable_stoch(pycolab_env.PyColabEnv):
    """A 5 room environment with an affectable object that has stochastic movement.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 extrinsic_reward=0.0):
        self.level = level
        self.objects = ['e', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        super(DeepmindMazeWorld_5room_moveable_stoch, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            act_null_value=4,
            resize_scale=17,
            visitable_states=223,
            extrinsic_reward=0.0,
            extrinsic_reward_obj=None)

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_moveable_stoch.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]

class DeepmindMazeWorld_5room_extint(pycolab_env.PyColabEnv):
    """A 5 room environment with an affectable object that has stochastic movement.
    """

    def __init__(self,
                 level=0,
                 max_iterations=500,
                 obs_type='mask',
                 default_reward=0.,
                 extrinsic_reward=0.0):
        self.level = level
        self.objects = ['a', 'b']
        self.state_layer_chars = ['#'] + self.objects # each char will produce a layer in the disentangled state
        super(DeepmindMazeWorld_5room_extint, self).__init__(
            max_iterations=max_iterations,
            obs_type=obs_type,
            default_reward=default_reward,
            action_space=spaces.Discrete(4 + 1), # left, right, up, down, no action
            act_null_value=4,
            resize_scale=17,
            visitable_states=223,
            extrinsic_reward=extrinsic_reward,
            extrinsic_reward_obj='b')

    def make_game(self):
        self._croppers = self.make_croppers()
        return deepmind_5room_extint.make_game(self.level)

    def make_croppers(self):
        return [cropping.ScrollingCropper(rows=5, cols=5, to_track=['P'], scroll_margins=(None, None), pad_char=' ')]






        
