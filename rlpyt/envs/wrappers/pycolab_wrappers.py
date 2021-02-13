import gym
import numpy as np
from PIL import Image

class Grayscale(gym.ObservationWrapper):
    def __init__(self, env, crop=True):
        super(Grayscale, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0., high=255., shape=(env.width, env.height, 1), dtype=np.uint8)

    def observation(self, obs):
        img = obs[:, :, 0] * 0.299 + obs[:, :, 1] * 0.587 + obs[:, :, 2] * 0.114
        return np.expand_dims(img, axis=2) # (w, h, c)