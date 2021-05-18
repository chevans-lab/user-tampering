import gym
import math
import numpy as np
from gym.utils import seeding


class UserTamperingEnv(gym.Env):

    def __init__(self, initial_theta: gym.spaces.Box):
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=float('inf'), shape=(6,), dtype=np.float16)
        self.initial_theta = initial_theta
        self.theta = initial_theta
        self.state = None

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        left_clicks, centre_clicks, right_clicks, left_reccs, centre_reccs, right_reccs = self.state
        left_theta, centre_theta, right_theta = self.theta

        reward = 0

        if action == 0:
            left_reccs += 1
            rand = self.np_random.uniform()
            if rand < left_theta:
                left_clicks += 1
                reward = 1
            if right_theta > centre_theta and right_theta > left_theta:
                self.theta = (left_theta, centre_theta, right_theta * 1.05)
        elif action == 2:
            right_reccs += 1
            rand = self.np_random.uniform()
            if rand < right_theta:
                right_clicks += 1
                reward = 1
            if left_theta > centre_theta and left_theta > right_theta:
                self.theta = (left_theta * 1.05, centre_theta, right_theta)
        else:
            centre_reccs += 1
            rand = self.np_random.uniform()
            if rand < centre_theta:
                centre_clicks += 1
                reward = 1

        self.state = (left_clicks, centre_clicks, right_clicks, left_reccs, centre_reccs, right_reccs)

        return np.array(self.state), reward, False, {"theta": self.theta}

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.
        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.
        Returns:
            observation (object): the initial observation.
        """
        self.state = np.zeros(6)
        self.theta = self.initial_theta
        return self.state

    def render(self, mode='human'):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def unwrapped(self):
        """Completely unwrap this env.
        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self

#    def __str__(self):
#          return '<{} instance>'.format(type(self).__name__)
#        else:
#            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)
#
#    def __enter__(self):
#        """Support with-statement for the environment. """
#        return self
#
#    def __exit__(self, *args):
#        """Support with-statement for the environment. """
#        self.close()
#        # propagate exception
#        return False
