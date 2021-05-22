import numpy as np


class MediaUser:
    def __init__(self, theta: np.ndarray):
        self.theta = theta

    def get_theta(self):
        return self.theta
