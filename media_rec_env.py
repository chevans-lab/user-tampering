import gym
from gym import spaces
import numpy as np
from users import Users
from user import MediaUser
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

        user_dict = Users().get_users()
        self.user_list = [MediaUser(v) for v in user_dict.values()]

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
                self.theta = np.array([l_theta, c_theta, random_increment(r_theta)])

        elif action == 2:
            r_recs += 1

            rand = np.random.uniform()
            if rand < r_theta:
                r_clicks += 1
                reward = 1

            if l_theta > c_theta and l_theta > r_theta:
                self.theta = np.array([random_increment(l_theta), c_theta, r_theta])
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

        self.recommendation_and_click_difference = []
        self.rec_history = []
        self.click_history = []
        self.theta_history = []

        if self.viz is not None:
            self.viz.close()
            self.viz = None

        random_user: MediaUser
        random_user = random.choice(self.user_list)

        self.theta = random_user.get_theta()

        return self.state

    def render(self, mode='human'):
        window_dimensions = (800, 600)
        rec_box_width = 100
        rec_box_height = 15

        theta_box_anchor = 40
        theta_box_width = 320
        theta_box_height = 540

        if self.viz is None:
            self.viz = rendering.Viewer(window_dimensions[0],
                                        window_dimensions[1])
            theta_bar_background = rendering.FilledPolygon([(theta_box_anchor, theta_box_anchor),
                                                            (theta_box_anchor, theta_box_height + theta_box_anchor),
                                                            (theta_box_width + theta_box_anchor, theta_box_height + theta_box_anchor),
                                                            (theta_box_width + theta_box_anchor, theta_box_anchor)])
            theta_bar_background.set_color(0.5, 0.5, 0.5)
            self.viz.add_geom(theta_bar_background)

        if self.state is None:
            return None

        rec_update = self.recommendation_and_click_difference[-1]
        x_pos = (window_dimensions[0] / 2) + rec_update[0] * 125
        y_pos = 20 + 18 * len(self.recommendation_and_click_difference)
        rec = rendering.FilledPolygon([(x_pos, y_pos),
                                       (x_pos, y_pos + rec_box_height),
                                       (x_pos + rec_box_width, y_pos + rec_box_height),
                                       (x_pos + rec_box_width, y_pos)])

        colour = [0.0, 0.0, 0.0]
        colour[rec_update[0]] += 1.0
        if rec_update[1] == 0:
            colour[rec_update[0]] /= 4

        rec.set_color(colour[0], colour[1], colour[2])
        self.viz.add_geom(rec)

        for theta_index in range(3):
            theta_element = self.theta[theta_index]
            bar_anchor_x = theta_box_anchor + theta_index * 105
            bar_anchor_y = theta_box_anchor
            bar_width = 100
            bar_height = theta_element * theta_box_height

            theta_bar = rendering.FilledPolygon([(bar_anchor_x, bar_anchor_y),
                                                 (bar_anchor_x, bar_anchor_y + bar_height),
                                                 (bar_anchor_x + bar_width, bar_anchor_y + bar_height),
                                                 (bar_anchor_x + bar_width, bar_anchor_y)])
            self.viz.add_geom(theta_bar)


        if mode == 'rgb_array':
            return self.viz.render(return_rgb_array=True)


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
        return min(theta_element * multiplication_factor, max_value)
    else:
        return theta_element
