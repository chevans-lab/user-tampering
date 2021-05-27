import numpy as np
import argparse
from media_rec_env import MediaRecommendationEnv
import pickle
import random
import time
import matplotlib.pyplot as plt

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

    def execute_demonstration(self, trials, horizon=30):
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

    def execute_evaluation(self, runs_per_user, horizon=30):
        """
        Conducts an evaluation of the q-learning policy and plots the results.

        For each user in the population, it runs the policy over the specified number of episodes. It then produces:
        -- A horizontal bar chart showing the relative likelihood of the agent choosing each action at each timestep
        -- A line chart showing the average accumulated reward up to and including each timestep by following the policy
            -- This chart also includes line charts corresponding to two other policies for comparison. We have:
                -- A random policy (chooses one of the actions with equal probability at each timestep)
                -- A 'baseline' bandit policy (random for the first third of the episode, and
                 the action with the highest average reward so far thereafter)

        Parameters:
            runs_per_user: int
            horizon: int
        """

        env = MediaRecommendationEnv(horizon)

        for i in range(len(env.user_list)):
            user = env.user_list[i]
            user_name = env.user_names[i]

            # Instantiating arrays to store data from the evaluation runs
            recs_per_timestep = np.zeros((3, horizon))
            random_cumulative_reward = np.zeros(horizon + 1)
            baseline_cumulative_reward = np.zeros(horizon + 1)
            tampering_cumulative_reward = np.zeros(horizon + 1)

            for run in range(runs_per_user):
                # Evaluating the q-learned policy first.
                # Running `runs_per_user` episodes, following the policy on all of them,
                # and for each time t, recording both the accumulated reward up to t and the action at t

                s = env.reset_with_user_profile(user)

                t = 0
                cumulative_reward = 0

                done = False
                while not done:
                    a = self.greedy_action(s)
                    recs_per_timestep[a, t] += 1

                    s, r, done, step_info = env.step(a)
                    cumulative_reward += r
                    t += 1

                    tampering_cumulative_reward[t] += cumulative_reward

                # Evaluating the baseline policy next.
                s = env.reset_with_user_profile(user)

                t = 0
                cumulative_reward = 0
                done = False
                while not done:
                    # Our baseline policy:
                    if t < (horizon / 3):
                        # Random actions in the first third of the episode to estimate user click probabilities
                        a = random.choice(self.actions)
                    else:
                        # Expected value maximisation thereafter based on mean reward per action so far
                        exp_value_per_action = [0.5, 0.5, 0.5]
                        if s[0] != 0:
                            exp_value_per_action[0] = float(s[1]) / float(s[0])
                        if s[2] != 0:
                            exp_value_per_action[1] = float(s[3]) / float(s[2])
                        if s[4] != 0:
                            exp_value_per_action[2] = float(s[5]) / float(s[4])

                        a = exp_value_per_action.index(max(exp_value_per_action))

                    s, r, done, step_info = env.step(a)
                    cumulative_reward += r
                    t += 1

                    baseline_cumulative_reward[t] += cumulative_reward

                # Finally, evaluating the random policy.
                s = env.reset_with_user_profile(user)

                t = 0
                cumulative_reward = 0
                done = False
                while not done:
                    a = random.choice(self.actions)

                    s, r, done, step_info = env.step(a)
                    cumulative_reward += r
                    t += 1

                    random_cumulative_reward[t] += cumulative_reward


            # LEARNED POLICY PLOT

            fig = plt.figure(figsize=(20, 5))

            bar = fig.add_subplot(121)
            bar.set_xlabel("Probability of chosen action")
            bar.set_ylabel("Timestep")
            bar.set_title(f"Learned Strategies for Recommending to a {user_name} User")

            # Dividing the no. times each action taken at each timestep to get an estimate of the prob. of taking
            # action a at time t under the learned policy
            x_values = np.flip(recs_per_timestep / runs_per_user, axis=1)

            y_values = list(range(horizon))
            y_ticks = list(range(horizon))
            y_ticks.reverse()
            plt.yticks(range(horizon), y_ticks)

            # Stacking horizontal bar charts to show the proportion of each action taken at each timestep
            plt.barh(y_values, x_values[0, :], color='red', label='Left-wing source')
            plt.barh(y_values, x_values[1, :], left=x_values[0, :], color='green', label='Centrist source')
            plt.barh(y_values, x_values[2, :], left=x_values[0, :] + x_values[1, :], color='blue', label='Right-wing source')
            plt.legend()

            # CUMULATIVE REWARD PLOT

            # Dividing the cumulative reward totals by the number of runs to get an estimate of expected
            # cumulative reward up to each timestep for one run
            random_cumulative_reward /= runs_per_user
            baseline_cumulative_reward /= runs_per_user
            tampering_cumulative_reward /= runs_per_user

            line = fig.add_subplot(122)
            line.set_xlabel("Timestep")
            line.set_ylabel("Average accumulated reward")
            line.set_title(f"Comparison of Policies on a {user_name} User By Cumulative Reward up to each Timestep")

            # Plotting the growth in cumulative reward for each of the three evaluated policies
            plt.plot(range(horizon + 1), tampering_cumulative_reward, color='purple', label='Q-learning policy')
            plt.plot(range(horizon + 1), baseline_cumulative_reward, color='gray', label='Baseline policy')
            plt.plot(range(horizon + 1), random_cumulative_reward, color='orange', label='Random policy')
            plt.legend()

            plt.show()


def main():
    """
    Reads in command line arguments for Q-table loading/saving,
    Generates a Q-learning agent, and either loads or learns a policy as necessary.

    Also either gives a visual demonstration of the agent's policy on a random selection of 10 simulated users (default)
    or conducts a more full-scale evaluation of the learned policy and plots the results.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", dest="input_q_file", metavar="INPUT",
                        help="If given, loads a pre-trained Q-table from this file")
    parser.add_argument("-s", "--save_to", dest="output_q_file", metavar="OUTPUT",
                        help="If given, saves the trained Q-table to this file")
    parser.add_argument("-v", "--visualisation_type", dest="visualisation_type", metavar="VIS",
                        help="Determines the kind of visual output. " +
                             "'demo' for an animated demonstration of the policy on 10 random users, " +
                             "'eval' for plots evaluating the strategy and performance")

    args = parser.parse_args()
    input_q_file = args.input_q_file
    output_q_file = args.output_q_file
    vis_type = args.visualisation_type

    # Generating Q-learning agent instance
    q_learning = QLearning(input_q_file, output_q_file)

    # If a file was given for loading, getting the Q-table from there
    if input_q_file:
        print(f"Loading pre-trained Q-table from {input_q_file} ...")
        q_learning.load_q_table()
        print("Loaded.")
    # Else, the agent begins training
    else:
        print("Commencing training...")
        q_learning.train(episodes=50000000)

    # Saving the Q-table to given location, if one was given
    if output_q_file:
        print(f"Saving trained Q-table to {output_q_file} ...")
        q_learning.save_q_table()
        print("Saved.")


    if vis_type is None or vis_type == 'demo':
        # Running demonstration of policy
        q_learning.execute_demonstration(10)
    elif vis_type == 'eval':
        # Running evaluation of policy
        q_learning.execute_evaluation(10000)

if __name__ == "__main__":
    main()
