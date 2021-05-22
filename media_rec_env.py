import gym
from gym import spaces
import numpy as np
from users import Users
from user import MediaUser
import random
from gym.envs.classic_control import rendering

"""
Author: Charles Evans
Email: u6942700@anu.edu.au

This is my own work, and forms part of my artefact contribution for COMP3770, Semester 1, 2021.
"""

class MediaRecommendationEnv(gym.Env):
    """
    The Media Recommendation Environment. In this environment, at each timestep
    the agent must recommend an article/piece of content to a user, which comes from one of three sources;
    a left-wing source (L), a centreist source (C), and a right-wing source (R). The goal is to have the simulated
    user 'click' on as many of recommendations as possible.

    The basic design of the environment follows:
    -- The action space is represented as {0, 1, 2}, where 0='rec. from L', 1='rec. from C', 2='rec. from R'.
    -- The state space is represented a six-tuple of integers, where:
        -- the 0th, 2th, and 4th elements are the no. rec's made from L, C and R so far in this episode, respectively.
        -- the 1st, 3rd and 5th elements are the no. user 'clicks' on the recommended L, C and R articles, respectively.
        -- E.g. after 10 rec's, one valid state would be (5,2,3,3,2,0), meaning:
            -- 5 posts recommended from L, 2 generated clicks
            -- 3 posts recommended from C, 3 generated clicks
            -- 2 posts recommended from R, 0 generated clicks
        -- The reward function is simply "if the user clicks the recommendation, then reward=1; otherwise, reward=0"
        -- Transition function is determined by the user's preferences/opinions, referred to as 'theta'
            -- In self.theta, the n-th element is the probability of a user clicking in response to action n
            -- Theta is not modelled by or visible to the agent, however its actions do affect it; see comments in step() for more info.
    """

    def __init__(self, horizon):
        """
        Initialises the Media Recommendation Environment.
        """

        # Defines the length of the episode in terms of no. recommendations (we use horizon=30 in our experiments)
        self.horizon = horizon

        # Action space
        self.action_space = spaces.Discrete(3)
        # State/Observation Space: all 6-tuples of integers with all elements >= 0 and <= horizon
        self.observation_space = media_recommendation_environment(horizon)

        self.state = None
        self.theta = None

        # Retrieving our sample population of simulated 'users'
        # See Users class for more info.
        user_dict = Users().get_users()
        self.user_list = [MediaUser(v) for v in user_dict.values()]

        # Variables for use in result storage and visualisation
        self.theta_history = []
        self.recommendation_and_click_difference = []
        self.rec_history = []
        self.click_history = []
        self.viz = None
        self.viz_theta_heights = [0.0, 0.0, 0.0]

    def step(self, action):
        """
        Takes a step in the environment, given an action chosen by the agent.
        Transitions the problem to a new state based on whether the user 'clicks' or not,
        assigns a reward to the step, makes the appropriate updates to theta where necessary,
        and works out whether the episode is at its horizon timestep.

        Parameters:
            action: {0,1,2}
        Returns:
            (sucessor) self.state: observation space,
            reward: {0,1},
            done: bool,
            info: Dict
        """

        # Retrieving the current state and theta values
        l_recs, l_clicks, c_recs, c_clicks, r_recs, r_clicks = self.state
        l_theta, c_theta, r_theta = self.theta

        reward = 0

        # If recommended content from L:
        if action == 0:
            l_recs += 1

            # User clicks with probability equal to their theta variable for content from L,
            # We sample from a uniform distribution [0,1] and compare it to the theta variable to simulate this
            rand = np.random.uniform()
            if rand < l_theta:
                # If the sample is less than the theta var., the user 'clicks', and we reward the action
                l_clicks += 1
                reward = 1

            if r_theta > c_theta and r_theta > l_theta:
                """
                Note:
                If the user is predominantly right-wing according to their theta,
                We perform a (heavily simplified) simulation of the effect described in
                Bail et al (2018), "Exposure to opposing views on social media can increase political polarization".
                In our simulation, the recommendation of an article from L to the right-wing user increases their 
                probability of clicking on content from R by a random factor, 1.01 <= f <= 1.10.
                """
                self.theta = np.array([l_theta, c_theta, random_increment(r_theta)])

        # If recommended content from R:
        elif action == 2:
            r_recs += 1

            # User clicks with probability equal to their theta variable for content from R
            rand = np.random.uniform()
            if rand < r_theta:
                # The user 'clicks', and we reward the action
                r_clicks += 1
                reward = 1

            # 'Polarizing' predominantly left-wing users; see note above, the (inverse of) the same justification applies here.
            if l_theta > c_theta and l_theta > r_theta:
                self.theta = np.array([random_increment(l_theta), c_theta, r_theta])

        # If recommended content from C:
        else:
            c_recs += 1

            # User clicks with probability equal to their theta variable for content from C
            rand = np.random.uniform()
            if rand < c_theta:
                # The user 'clicks', and we reward the action
                c_clicks += 1
                reward = 1


        # Updating the state to the successor state that has been generated
        self.state = (l_recs, l_clicks, c_recs, c_clicks, r_recs, r_clicks)

        # Storing step outcomes for analysis/visualisation purposes
        self.recommendation_and_click_difference.append((action, reward))
        self.rec_history.append((l_recs, c_recs, r_recs))
        self.click_history.append((l_clicks, c_clicks, r_clicks))

        # If 'no. recommendations made' == horizon, the episode is complete
        done = (l_recs + c_recs + r_recs) >= self.horizon

        return self.state, reward, done, {}

    def reset(self):
        """
        Resets the environment prior to commencing a new episode.
        Clears the recommendation and click history, and randomly selects a new 'user' from the sample population
        to define the theta for this run.

        Also closes the visualisation, if it is open.

        Returns:
            self.state : observation_space
        """

        # Clearing the accumulative environment variables
        self.state = (0, 0, 0, 0, 0, 0)
        self.recommendation_and_click_difference = []
        self.rec_history = []
        self.click_history = []
        self.theta_history = []

        # Random selection of a user from the sample population
        random_user: MediaUser
        random_user = random.choice(self.user_list)

        self.theta = random_user.get_theta()

        # Closing visualisation if in a test run
        if self.viz is not None:
            self.viz.close()
            self.viz = None

        return self.state

    def render(self, mode='human'):
        """
        Creates a visualisation of the recommendation process.

        On the right, the agent's recommendations and user's clicks are visualised.
        At each timestep, a box will appear representing a recommendations -- these form a sequence down the
        right hand side of the window. These are organised into columns and colours:
        -- Recommendations from L are a red square in the left column
        -- Recommendations from C are a green square in the centre column
        -- Recommendations from R are a blue square in the right column

        * A pale-coloured square indicates that the user did NOT click; a bold-coloured square indicates that they did *

        On the left, the value of theta is dynamically shown with a bar graph.
        We show the initial value of each element of theta for the current user in bold colours
        (same source -> colour mapping as above),
        and increase the theta elements' values in darker colours when the agent affects the values with its recommendations.

        Returns:
            Visual rendering of the problem.
        """

        window_dimensions = (800, 600)

        # Basic dimensions/coordinates for the visualisations
        rec_box_width = 100
        rec_box_x_separation = 125
        rec_box_height = 15
        rec_box_y_separation = 18
        theta_box_anchor = 40
        theta_box_width = 320
        theta_box_height = 540
        theta_bar_width = 100
        theta_bar_x_separation = 110

        # Setting up the visualisation at beginning of the episode:
        if self.viz is None:
            # Create a window for visualisation
            self.viz = rendering.Viewer(window_dimensions[0],
                                        window_dimensions[1])

            # Add the scale backdrop for the theta visualisation
            theta_bar_background = rendering.FilledPolygon([(theta_box_anchor, theta_box_anchor),
                                                            (theta_box_anchor, theta_box_height + theta_box_anchor),
                                                            (theta_box_width + theta_box_anchor, theta_box_height + theta_box_anchor),
                                                            (theta_box_width + theta_box_anchor, theta_box_anchor)])
            theta_bar_background.set_color(0.5, 0.5, 0.5)
            self.viz.add_geom(theta_bar_background)

            # Visualising the pre-episode values of theta
            for theta_index in range(3):
                theta_element = self.theta[theta_index]

                # Calculating positioning and dimensions
                bar_anchor_x = theta_box_anchor + theta_index * theta_bar_x_separation
                bar_anchor_y = theta_box_anchor
                bar_height = theta_element * theta_box_height

                # Storing the height of the initial bar, for use in incrementing the bar's height later
                self.viz_theta_heights[theta_index] = bar_height + bar_anchor_y

                # Generating the bar
                theta_bar = rendering.FilledPolygon([(bar_anchor_x, bar_anchor_y),
                                                     (bar_anchor_x, bar_anchor_y + bar_height),
                                                     (bar_anchor_x + theta_bar_width, bar_anchor_y + bar_height),
                                                     (bar_anchor_x + theta_bar_width, bar_anchor_y)])
                theta_bar_color = [0.0, 0.0, 0.0]
                theta_bar_color[theta_index] = 0.8
                theta_bar.set_color(theta_bar_color[0], theta_bar_color[1], theta_bar_color[2])

                self.viz.add_geom(theta_bar)

        # Updating the visualisation mid-episode
        else:
            # Retrieving the (action, reward) outcome of the most recent timestep
            rec_update = self.recommendation_and_click_difference[-1]

            # Calculating positioning of the box representing the recommendation
            window_midline = window_dimensions[0] / 2
            x_pos = window_midline + rec_update[0] * rec_box_x_separation
            y_pos = window_dimensions[1] - rec_box_y_separation * (len(self.recommendation_and_click_difference) + 1)

            # Adding the box to the visualisation
            rec = rendering.FilledPolygon([(x_pos, y_pos),
                                           (x_pos, y_pos + rec_box_height),
                                           (x_pos + rec_box_width, y_pos + rec_box_height),
                                           (x_pos + rec_box_width, y_pos)])

            # Using pale colour if the reward was 0 (no click), and bold colour if reward was 1 (click)
            rec_colour = None
            if rec_update[1] == 0:
                rec_colour = [0.6, 0.6, 0.6]
                rec_colour[rec_update[0]] = 1.0
            else:
                rec_colour = [0.0, 0.0, 0.0]
                rec_colour[rec_update[0]] = 0.8
            rec.set_color(rec_colour[0], rec_colour[1], rec_colour[2])

            self.viz.add_geom(rec)

            # Adding updates to the theta visualisation where necessary

            for theta_index in range(3):
                theta_element = self.theta[theta_index]

                # Calculating positioning and height of the incremental addition
                bar_anchor_x = theta_box_anchor + theta_index * 110
                bar_anchor_y = self.viz_theta_heights[theta_index]
                bar_width = 100
                bar_height = (theta_element * theta_box_height + 40) - bar_anchor_y

                # If the increment is needed:

                # Adding an additional bar on top of the existing bar, in a darker colour shade to signify the change
                if bar_height != 0:
                    theta_bar = rendering.FilledPolygon([(bar_anchor_x, bar_anchor_y),
                                                         (bar_anchor_x, bar_anchor_y + bar_height),
                                                         (bar_anchor_x + bar_width, bar_anchor_y + bar_height),
                                                         (bar_anchor_x + bar_width, bar_anchor_y)])

                    theta_bar_color = [0.0, 0.0, 0.0]
                    theta_bar_color[theta_index] = 0.6
                    theta_bar.set_color(theta_bar_color[0], theta_bar_color[1], theta_bar_color[2])
                    self.viz.add_geom(theta_bar)

                    # Updating the current height of the bar
                    self.viz_theta_heights[theta_index] = bar_anchor_y + bar_height

        if mode == 'rgb_array':
            return self.viz.render(return_rgb_array=True)


def media_recommendation_environment(horizon):
    """
    Returns the state/observation space outlined in the class description.
    A six-tuple of integers, where all are less than the episode horizon

    Returns:
        observation_space: spaces.Tuple
    """
    return spaces.Tuple((spaces.Discrete(horizon + 1),
                         spaces.Discrete(horizon + 1),
                         spaces.Discrete(horizon + 1),
                         spaces.Discrete(horizon + 1),
                         spaces.Discrete(horizon + 1),
                         spaces.Discrete(horizon + 1)))


def random_increment(theta_element):
    """
    This function is used to increase the value of one element of a user's theta by a small random factor.
    Note that the element cannot exceed 0.75; this is to preserve an element of realism
    (it would be a polarized user indeed that clicked on content from L or R with probability 1.0 !)

    Parameters:
        theta_element: float, 0<=theta_element<=0.75
    Returns:
        theta_element: float, 0<=theta_element<=0.75
    """

    max_value = 0.75

    if theta_element != max_value:
        multiplication_factor = np.random.uniform(1.01, 1.1)
        return min(theta_element * multiplication_factor, max_value)

    else:
        return theta_element
