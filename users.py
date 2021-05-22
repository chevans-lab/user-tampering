from enum import Enum
from user import MediaUser
import numpy as np


class Users(Enum):

    """
    Represents a representative collection of media 'users.'

    On both 'wings,' we have:
    -- A moderate user, who is unlikely to engage with content from the opposite wing, but
        similarly likely to engage with centrist content and content from their own wing.
    -- A strong user, who is unlikely to engage with any content not from their own wing.

    We also have:
    -- A centrist user (likely to click on centrist content,
        but not particularly likely to engage otherwise.
    -- An unaffiliated user (equally not particularly likely to click on any content)
    """

    moderate_right = MediaUser(np.array([0.1, 0.3, 0.3]))
    strong_right = MediaUser(np.array([0.1, 0.1, 0.4]))
    moderate_left = MediaUser(np.array([0.3, 0.3, 0.1]))
    strong_left = MediaUser(np.array([0.4, 0.1, 0.1]))
    centrist = MediaUser(np.array([0.2, 0.4, 0.2]))
    unaffiliated = MediaUser(np.array([0.2, 0.2, 0.2]))
