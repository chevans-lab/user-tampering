import numpy as np

"""
Author: Charles Evans
Email: u6942700@anu.edu.au

This is my own work, and forms part of my artefact contribution for COMP3770, Semester 1, 2021.
"""


class MediaUser:
    """
    Represents a media 'user' i.e. consumer for the purposes of our simulation.
    The only relevant quality for the sim. is the theta parameter representing the user's preferences/opinions.
    """

    def __init__(self, theta: np.ndarray):
        self.theta = theta

    def get_theta(self):
        return self.theta
