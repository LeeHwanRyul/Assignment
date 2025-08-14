import gymnasium as gym
import numpy as np


class Wrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
