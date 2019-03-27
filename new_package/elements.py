import numpy as np
from parameters import speed_matrix, transition_matrix


class Juncture():

    def __init__(self, id="A", flag="morning"):

        # The 'id' is just the A/B/C/D representing different junctures
        self.id = id
        # flag 是记录是早晨还是晚上
        self.flag = flag

        # Action is the current state of this crossroad
        self.action = np.zeros((4, 3))
        # 4 represents respectively:                West North East South
        # 3 represents three directions in order:   left front right
        # We would assume that turning right is by default

        # This is the Z we used before,
        # representing the number of out-going cars to which direction.
        # The in_detail function includes where cars starts,
        # so that we can update the waiting cars
        self.out_car_in_detail = np.zeros((4, 4))  # From north to south, etc.
        self.out_car = np.zeros((4, 1))  # To West North East South

        # This is the number of cars waiting at this juncture
        # We also provide two variations,
        # Cars in detail is computed with probability matrix P
        # the without detail function simply stores the cars on each direction.
        self.cars_in_detail = np.zeros((4, 4))
        self.cars = np.zeros((4, 1))
        # 4 represents direction in order of: West North East South

        self.P = transition_matrix(self.flag, self.id)
        self.V = speed_matrix(self.flag, self.id)

        # TODO The particular Q-value of this Juncture
        self.Q = np.zeros(1)

    def in_car(self, income_car):
        '''
        In this function, we let car into this juncture.
        :param:
        income_car: a 4x1 array, representing how many cars are coming from
                    the from direction
        '''

        self.cars = self.cars + income_car
        self.cars_in_detail = np.multiply(self.cars, self.P)

    def out_car(self, time, delta_t=10):
        '''
        TODO
        In this function we compute how many cars are going out of the system
        :param:
        time    : Time is very important in the sense that
                  we're now using different speeds in different times
        delta_t : Time interval, by default it's 10 seconds
        '''

        self.out_car_in_detail = np.zeros((4, 4))  # From north to south, etc.
        self.out_car = np.zeros((4, 1))  # To West North East South
