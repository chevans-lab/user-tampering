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
        but not particularly likely to engage otherwise.
    """

    def __init__(self):
        self.users = {}
        self.users["moderate_right"] = np.array([0.1, 0.25, 0.3])
        self.users["strong_right"] = np.array([0.1, 0.1, 0.4])
        self.users["moderate_left"] = np.array([0.3, 0.25, 0.1])
        self.users["strong_left"] = np.array([0.4, 0.1, 0.1])
        self.users["centrist"] = np.array([0.2, 0.4, 0.2])

    def get_users(self):
        return self.users