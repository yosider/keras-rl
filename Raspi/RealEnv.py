import gym
import sensories
import numpy as np

class RealEnv(gym.core.Env):
    def __init__(self, target, tolerance, timeout):
        self.action_space = gym.spaces.Discrete(10)
        state_high = np.ones(10000)
        state_low = np.zeros(10000)
        self.observation_space = gym.spaces.Box(low=state_low, high=state_high)

        self.target = target
        self.tolerance = tolerance
        self.timeout = timeout

        self.model_v = sensories.build_visual_model()

        self.time = 0

    def _step(self, action):
        """
        1. get data from sensors.
        2. generate reward.
            (main theme of this experi.)
        """
        self.view = self.get_view()
        self.state = np.array([self.view])

        self.error = np.sqrt(np.sum(np.square(self.target - self.state)))
        # fix
        done = self.error < self.tolerance \
                or self.time < timeout

        if self.error < self.tolerance:
            reward = 1.0
        elif self.time < timeout:
            reward = 0.0
        else:
            reward = 0.0
            self.time += 1

        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.view = self.get_view()
        self.state = np.array([self.view])
        return self.state
