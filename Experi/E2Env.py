import gym
import gym.spaces

import Robot
import numpy as np

import time

num_action = 8
img_w = img_h = 80
nb_channels = 3

class E2Env(gym.core.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(num_action)
        state_high = np.ones((img_w,img_h,nb_channels))
        state_low = np.zeros((img_w,img_h,nb_channels))
        self.observation_space = gym.spaces.Box(low=state_low, high=state_high)

        self.robot = Robot.Robot(cam_resolution=(img_w,img_h), action_interval=1)


    def _step(self, action):
        """
        1. get data from sensors.
        2. generate reward.
            (main theme of this experi.)
        """
        # (num_action,) ndarray
        self.robot.act(action)
        # (height, width, channels) ndarray
        self.view = self.robot.get_view()

        self.state = np.array(self.view)

        done, reward = self._reward(self.state)

        return self.state, reward, done, {}


    def _reset(self):
        self.view = self.robot.get_view()
        self.state = self.view
        return self.state


    def _reward(self, state):
        view = self.view
        height, width, channel = np.shape(view)
        # if np.mean(view[height//2-10:height//2+10][width//2-10:width//2+10]) > 0.5:
        if view[0][0][0] > 0.2:
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        return done, reward
