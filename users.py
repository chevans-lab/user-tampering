import numpy as np

"""
Author: Charles Evans
Email: u6942700@anu.edu.au

This is my own work, and forms part of my artefact contribution for COMP3770, Semester 1, 2021.
"""


class Users:
    """
     A representative collection of simulated media 'users.'

    On both 'wings,' we have:
    -- A moderate user, who is unlikely to engage with content from the opposite wing, but
        similarly likely to engage with centrist content and content from their own wing.
    -- A strong user, who is unlikely to engage with any content not from their own wing.

    We also have:
    -- A centrist user (likely to click on centrist content,
        but not particularly likely to engage otherwise).
    """

    def __init__(self):
        self.users = {}
        # Main User Population for training and evaluation
        self.users["Moderate Right"] = np.array([0.1, 0.25, 0.3])
        self.users["Strong Right"] = np.array([0.1, 0.1, 0.4])
        self.users["Moderate Left"] = np.array([0.3, 0.25, 0.1])
        self.users["Strong Left"] = np.array([0.4, 0.1, 0.1])
        self.users["Centrist"] = np.array([0.2, 0.4, 0.2])

        # BELOW: Additional users which were not used in training our model,
        # but can be uncommented at evaluation stage to verify that the learned policy is robust to preference profiles
        # which were not encountered in training.

        #self.users["Extremely Right"] = np.array([0.05, 0.05, 0.5])
        #self.users["Right anti-centrist "] = np.array([0.2, 0.05, 0.35])
        #self.users["Left anti-centrist"] = np.array([0.35, 0.05, 0.2])
        #self.users["Extremely Left"] = np.array([0.5, 0.05, 0.05])

    def get_users(self):
        return self.users
