import gym
from gym import spaces
import numpy as np
from users import Users
from user import MediaUser
from gym.utils import seeding
import matplotlib.pyplot as plt
import random
from gym.envs.classic_control import rendering


class MediaRecommendationEnv(gym.Env):

    def __init__(self, horizon):

        self.horizon = horizon

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = media_recommendation_environment(horizon)
        self.state = None

        self.theta = np.array([0.0, 0.0, 0.0])

        self.user_list = [Users.moderate_right, Users.strong_right,
                          Users.moderate_left, Users.strong_left,
                          Users.centrist, Users.unaffiliated]

        self.theta_history = []
        self.recommendation_and_click_difference = []
        self.rec_history = []
        self.click_history = []
        self.viz = None




    def step(self, action):
        l_recs, l_clicks, c_recs, c_clicks, r_recs, r_clicks = self.state
        l_theta, c_theta, r_theta = self.theta

        reward = 0

        if action == 0:
            l_recs += 1

            rand = np.random.uniform()
            if rand < l_theta:
                l_clicks += 1
                reward = 1

            if r_theta > c_theta and r_theta > l_theta:
                self.theta = (l_theta, c_theta, random_increment(r_theta))

        elif action == 2:
            r_recs += 1

            rand = np.random.uniform()
            if rand < r_theta:
                r_clicks += 1
                reward = 1

            if l_theta > c_theta and l_theta > r_theta:
                self.theta = (random_increment(l_theta), c_theta, r_theta)
        else:
            c_recs += 1
            rand = np.random.uniform()
            if rand < c_theta:
                c_clicks += 1
                reward = 1

        self.state = (l_recs, l_clicks, c_recs, c_clicks, r_recs, r_clicks)

        self.recommendation_and_click_difference.append((action, reward))
        self.rec_history.append((l_recs, c_recs, r_recs))
        self.click_history.append((l_clicks, c_clicks, r_clicks))

        done = (l_recs + c_recs + r_recs) >= self.horizon




        return self.state, reward, done, {}

    def reset(self):

        self.state = (0, 0, 0, 0, 0, 0)

        self.recommendation_and_click_history = []
        self.rec_history = []
        self.click_history = []
        self.theta_history = []

        random_user: MediaUser
        random_user = random.choice(self.user_list)

        self.theta = random_user.get_theta()

        return self.state

    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError("Only human-readable rendering mode is supported")
        else:
            window_dimensions = (400, 400)
            if self.viz is None:
                self.viz = rendering.Viewer(window_dimensions[0],
                                            window_dimensions[1])

            #TODO render the viewer


def media_recommendation_environment(horizon):
    return spaces.Tuple((spaces.Discrete(horizon + 1),
                         spaces.Discrete(horizon + 1),
                         spaces.Discrete(horizon + 1),
                         spaces.Discrete(horizon + 1),
                         spaces.Discrete(horizon + 1),
                         spaces.Discrete(horizon + 1)))


def random_increment(theta_element):
    max_value = 0.75
    if theta_element != max_value:
        multiplication_factor = np.random.uniform(1.01, 1.1)
        return max(theta_element * multiplication_factor, max_value)
    else:
        return theta_element
