import numpy as np


class Users:
    """
    Represents a representative collection of media 'users.'

    On both 'wings,' we have:
    -- A moderate user, who is unlikely to engage with content from the opposite wing, but
        similarly likely to engage with centrist content and content from their own wing.
    -- A strong user, who is unlikely to engage with any content not from their own wing.

    We also have:
    -- A centrist user (likely to click on centrist content,
        but not particularly likely to engage otherwise.
    -- An unengaged user (equally not particularly likely to click on any content)
    """

    def __init__(self):
        self.users = {}
        self.users["moderate_right"] = np.array([0.1, 0.25, 0.3])
        self.users["strong_right"] = np.array([0.1, 0.1, 0.4])
        self.users["moderate_left"] = np.array([0.3, 0.25, 0.1])
        self.users["strong_left"] = np.array([0.4, 0.1, 0.1])
        self.users["centrist"] = np.array([0.2, 0.4, 0.2])
        self.users["unengaged"] = np.array([0.2, 0.2, 0.2])

    def get_users(self):
        return self.users