import numpy as np
from elements import Juncture


class Traffic:
    def __init__(self, flag="morning"):
        self.T = 10800                       # 3 hour into seconds 3 * 60 * 60
        self.gamma = 0.8                     # discount rate
        self.dt = 10                         # 10 seconds per epoch
        self.EPOCH = self.T / self.dt        # Number of total epochs
        self.car_len = 3
        self.multiplier = 10                 # taxi / total # of car
        self.flag = flag

        # Penalties
        self.alpha = 0.5
        self.beta = 10
        self.history = []                    # to store the event history

    def create_crossroads(self):
        # In this function, we basically create four junctures called ABCD.
        self.juncs = {"A": Juncture(id="A", flag=self.flag),
                      "B": Juncture(id="B", flag=self.flag),
                      "C": Juncture(id="C", flag=self.flag),
                      "D": Juncture(id="D", flag=self.flag)}

    def one_epoch(self):
        # We will first update the
        pass
