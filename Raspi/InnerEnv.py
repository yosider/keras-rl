import gym
import numpy as np

class InnerEnv(gym.core.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete()
