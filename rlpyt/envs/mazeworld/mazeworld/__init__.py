from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym

def register(id, entry_point, max_episode_steps, kwargs):
    env_specs = gym.envs.registry.env_specs
    if id in env_specs.keys():
        del env_specs[id]
    gym.register(id=id, 
                 entry_point=entry_point, 
                 max_episode_steps=max_episode_steps, 
                 kwargs=kwargs)

register(
    id='Deepmind5Room-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5Room-v1',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_flipped',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomWhitenoise-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_whitenoise',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomAll-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_all',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomMoveable-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_moveable',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomMoveableStoch-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_moveable_stoch',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomMoveableBrownian-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_moveable_brownian',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomExtInt-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_extint',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLarge-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5roomlarge',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLargeEnemy-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5roomlarge_enemy',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLargeWeather-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5roomlarge_weather',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLargeText-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5roomlargetext',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLargeWhitenoise-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5roomlarge_whitenoise',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLargeTextWhitenoise-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5roomlargetext_whitenoise',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLargeRandomFixed-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5roomlarge_randomfixed',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLargeTextRandomFixed-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5roomlargetext_randomfixed',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLargeMoveable-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5roomlarge_moveable',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLargeMoveable-v1',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5roomlarge_moveable_ext',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLargeTextMoveable-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5roomlargetext_moveable',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLargeTextMoveable-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5roomlargetext_moveable',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLargeMoveableStoch-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5roomlarge_moveable_stoch',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLargeMoveableStoch-v1',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5roomlarge_moveable_stoch_ext',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLargeTextMoveableStoch-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5roomlargetext_moveable_stoch',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLargeMoveableBrownian-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5roomlarge_moveable_brownian',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLargeTextMoveableBrownian-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5roomlargetext_moveable_brownian',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLargeBrownian-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5roomlarge_brownian',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLargeTextBrownian-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5roomlargetext_brownian',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLargeAll-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5roomlarge_all_stoch',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLargeAll-v1',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5roomlarge_all',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLargeTextAll-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5roomlargetext_all',
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
    id='Deepmind5RoomLong-v2',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_longunpadded',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind5RoomLong-v3',
    entry_point='mazeworld.envs:DeepmindMazeWorld_5room_longext',
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
    id='Deepmind8RoomWeather-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_8room_weather',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind8Room-v1',
    entry_point='mazeworld.envs:DeepmindMazeWorld_8room_rgb',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='Deepmind8Room-v2',
    entry_point='mazeworld.envs:DeepmindMazeWorld_8room_diff',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})

register(
    id='DeepmindPianoLong-v0',
    entry_point='mazeworld.envs:DeepmindMazeWorld_piano_long',
    max_episode_steps=500,
    kwargs={'level': 0, 'max_iterations': 500, 'obs_type': 'maze'})


