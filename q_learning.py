import numpy as np
import argparse
from media_rec_env import MediaRecommendationEnv
import pickle
import random
import time

"""
Author: Charles Evans
Email: u6942700@anu.edu.au

This is my own work, and forms part of my artefact contribution for COMP3770, Semester 1, 2021.
"""

class QLearning:
    """
    A Q-learning implementation for learning a recommendation policy in the Media Recommendation Environment.

    Implements the standard Q-learning algorithm. Also includes the functionality of saving and loading Q-tables
    to and from plain files to allow us to save a trained agent.
    """

    def __init__(self, input_file, output_file):
        """
        Initialises the agent. Creates the Q-table, and sets (hyper)parameters.

        Parameters:
            input_file: str
            output_file: str
        """

        # Storing the IO file paths for use in loading/saving
        self.input_file = input_file
        self.output_file = output_file

        # Setting learning hyperparameters
        self.gamma = 0.999
        self.epsilon = 0.9
        self.alpha = 0.45

        # Introducing the Q-table and action list
        self.actions = [0, 1, 2]
        self.Q = {}

    def load_q_table(self):
        """
        Loads the Q-table stored in the file at the given 'output_file' path.
        into this instance's Q-table.
        """
        with open(self.input_file, "rb") as f:
            self.Q = pickle.load(f)

    def save_q_table(self):
        """
        Writes this instance's Q-table to the file at the given 'input_file'.
        """
        with open(self.output_file, "wb") as f:
            pickle.dump(self.Q, f)

    def train(self, episodes, horizon=30):
        """
        Runs the standard Q-learning algorithm to train the agent over a given number of problem episodes.

        Parameters:
            episodes: int
            horizon: int
        """

        # Loading our custom media recommendation environment
        env = MediaRecommendationEnv(horizon)

        for ep in range(episodes):
            # Decaying alpha on a per-episode basis
            # For non-trivial no. episodes, this will make alpha decay to ~0.05 at the end of training
            self.alpha *= (1 - (2 / episodes))

            # Refreshing the environment for a new training episode
            s = env.reset()

            done = False
            while not done:

                # Selecting an action given the current state with an 'epsilon-greedy' policy
                a = None
                if np.random.uniform() > self.epsilon:
                    # random action with prob. (1 - epsilon)
                    a = np.random.randint(0, 3)
                else:
                    # greedy action with prob. epsilon
                    a = self.greedy_action(s)

                # Simulating a transition in the env., given chosen action
                s_prime, r, done, step_info = env.step(a)

                # Finding the greedy-optimal action from the successor state
                a_prime = self.greedy_action(s_prime)

                # If the current or successor state-action pairs have never been visited previously,
                # adding them to the Q-table
                if not (s, a) in self.Q:
                    self.Q[(s, a)] = 0
                if not (s_prime, a_prime) in self.Q:
                    self.Q[(s_prime, a_prime)] = 0

                # Performing Temporal Difference-based Q-value update for (s, a)
                expected_future_reward = 0
                if not done:
                    expected_future_reward = self.gamma * self.Q[(s_prime, a_prime)]
                self.Q[(s, a)] += self.alpha * (r + expected_future_reward - self.Q[(s, a)])

                # Transitioning to the next state
                s = s_prime

            if ep % 10000 == 0:
                print(f"Completed {ep} training episodes")

    def greedy_action(self, state):
        """
        Calculates the greedy action from the current state (action with highest expected reward).

        Parameters:
            state: env.observation_space
        Returns:
            action: {0,1,2}
        """
        max_action = 0
        max_action_value = 0

        for i, a in enumerate(self.actions):
            # Assigning the state-action pair its value from the Q-table if it has been visited, 0 otherwise
            action_value = 0
            if (state, a) in self.Q:
                action_value = self.Q[(state, a)]
            # Updating the optimal action if this action is the best checked so far
            if action_value > max_action_value:
                max_action = a
                max_action_value = action_value
            # If this action has the same expected reward as the max, randomly breaking the tie
            elif action_value == max_action_value:
                max_action, max_action_value = random.choice([(a, action_value), (max_action, max_action_value)])

        return max_action

    def execute_policy(self, trials, horizon=30):
        """
        Gives a visual demonstration of the policy learned by the agent.
        Undertakes the given number of 'trial' episodes, using a greedy policy.
        The results are rendered to a pop-up window and listed in the terminal. See
        MediaRecommendationEnv.render() for more info. regarding the visualisation.

        Parameters:
            trials: int
            horizon: int
        """

        env = MediaRecommendationEnv(horizon)
        readable_actions = ["L", "C", "R"]

        for t in range(trials):
            time.sleep(1.0)

            # Refreshing the environment for a new episode
            s = env.reset()
            accumulated_reward = 0

            print()
            print(f"--- Trial {t} ---")
            print("User has initial theta values:", env.theta)
            print()

            # Generating the pop-up window
            env.render(mode='rgb_array')

            done = False
            while not done:
                time.sleep(0.25)

                # Selecting an action and reporting it in the terminal
                a = self.greedy_action(s)
                print(f"Agent recommended from {readable_actions[a]} to the User")

                # Simulating the chosen action, recording the generated reward
                s, r, done, step_info = env.step(a)
                accumulated_reward += r

                # Updating the visualisation
                env.render(mode='rgb_array')

            print(f"Generated {accumulated_reward} clicks in trial.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input_q_file", metavar="INPUT",
                        help="If given, load a pre-trained Q-table from this file")
    parser.add_argument("-o", "--output", dest="output_q_file", metavar="OUTPUT",
                        help="If given, save the trained Q-table to this file")

    args = parser.parse_args()
    i = args.input_q_file
    o = args.output_q_file
    q_learning = QLearning(i, o)
    if i:
        print(f"Loading pre-trained Q-table from {i} ...")
        q_learning.load_q_table()
        print("Loaded.")
    else:
        print("Commencing training...")
        q_learning.train(episodes=50000000)
        if o:
            print(f"Saving trained Q-table to {o} ...")
            q_learning.save_q_table()
            print("Saved.")

    q_learning.execute_policy(10)

if __name__ == "__main__":
    main()