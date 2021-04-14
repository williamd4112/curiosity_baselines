from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym

from .envs import (MazeWorld, 
                    DeepmindMazeWorld_maze, 
                    DeepmindMazeWorld_5room, 
                    DeepmindMazeWorld_5room_long,
                    DeepmindMazeWorld_5room_longwide,
                    DeepmindMazeWorld_5room_noobj, 
                    DeepmindMazeWorld_5room_oneobj, 
                    DeepmindMazeWorld_5room_onewhite, 
                    DeepmindMazeWorld_5room_randomfixed, 
                    DeepmindMazeWorld_5room_bouncing,
                    DeepmindMazeWorld_5room_brownian,
                    DeepmindMazeWorld_8room,
                    DeepmindMazeWorld_8room_extrinsic,
                    DeepmindMazeWorld_8room_oneobj_allrooms,
                    DeepmindMazeWorld_8room_oneobj_singleroom,
                    DeepmindMazeWorld_5room_moveable,
                    DeepmindMazeWorld_5room_moveable_stoch,
                    DeepmindMazeWorld_5room_extint)

def register(id, entry_point, max_episode_steps, kwargs):
    env_specs = gym.envs.registry.env_specs
    if id in env_specs.keys():
        del env_specs[id]
    gym.register(id=id, 
                 entry_point=entry_point, 
                 max_episode_steps=max_episode_steps, 
                 kwargs=kwargs)

register(
    id='Maze-v0',
    entry_point='mazeworld.envs:MazeWorld',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5Room-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLong-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_long',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLong-v1',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_longwide',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomNoObj-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_noobj',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomOneObj-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_oneobj',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomOneObj-v1',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_onewhite',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomRandomFixed-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_randomfixed',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomBouncing-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_bouncing',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomBrownian-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_brownian',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='DeepmindMaze-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_maze',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind8Room-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_8room',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind8RoomExtrinsic-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_8room_extrinsic',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind8RoomOneObj-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_8room_oneobj_singleroom',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind8RoomOneObj-v1',
    entry_point='mazeworld.envs:DeepmindMazeWorld_8room_oneobj_allrooms',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomMoveable-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_moveable',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomMoveable-v1',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_moveable_stoch',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomExtInt-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_extint',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})


