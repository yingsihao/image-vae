import numpy as np


class HParams:
    def __init__(self):
        self.max_length = 64
        self.start_token = np.array([0, 0, 1, 0, 0])
        self.stop_token = np.array([0, 0, 0, 0, 1])


hp = HParams()
