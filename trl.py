import numpy as np

class Traffic:
    def __init__(self):
        self.N = 1000                       # We do 1,000 times trial
        self.T = 600                        # 600 minutes
        self.NACTIONS = 81                  # # of our action space
        self.gamma = 0.8                    # discount rate
        self.get_speed_matrix()             # Some data from the distribution
        self.get_transition_matrix()        # Some data from the distribution
        self.A = np.zeros(4, 3)             # action space
        self.Q = np.zeros(self.NACTIONS)    # 81 actions in all
        self.R = np.zeros(self.NACTIONS)    # 81 actions, so 81 rewards
        self.history = []                   # We use this to store the event history

    def get_speed_matrix(self):
        '''
        TODO: In this function, we compute the speed matrix.
        :return: self.V 是一个字典
        '''

        V_1 = np.zeros(4, 4)
        V_2 = np.zeros(4, 4)
        V_3 = np.zeros(4, 4)
        V_4 = np.zeros(4, 4)

        self.V = {1: V_1,
                  2: V_2,
                  3: V_3,
                  4: V_4}

    def get_transition_matrix(self):
        '''
        TODO: In this function, we compute the transition matrix.
        :return: self.P 是一个字典
        '''

        P_1 = np.zeros(4, 4)
        P_2 = np.zeros(4, 4)
        P_3 = np.zeros(4, 4)
        P_4 = np.zeros(4, 4)

        self.P = {1: P_1,
                  2: P_2,
                  3: P_3,
                  4: P_4}

    def initialize(self):
        '''
        In this function we initialize our state space
        '''
        self.X = np.zeors(4, 4)  # incoming cars
        self.Y = np.zeros(4, 4)  # waiting cars
        self.Z = np.zeros(4, 4)  # Outgoing cars

    def update_X(self):
        '''
        TODO in this function, we update_X the incoming cars
        Notice that X comes from outside our data, and also from the previous state of Z.
        :return: we store the incoming value into self.X
        '''
        self.X = self.X

    def update_Y_with_X(self):
        '''
        In this function, we update the waiting cars.
        Notice that we only contribute X into Y, but we do not subtract Z
        '''
        self.Y = self.Y + self.X

    def compute_reward(self):
        '''
        TODO In this function, we compute the reward function, based on the current Y
        :return:
        '''
        self.R = np.zeros(self.NACTIONS)

    def compute_Q(self):
        '''
        TODO In this function we estimate the value of Q matrix, which is a 81 long
        We will use function self.compute_reward() in this function.
        :return:
        '''

        self.Q = np.zeros(81)

    def pick_strategy(self):
        '''
        TODO: This is a very easy function, where we choose the largest number Q value in Q matrix and find the best stragegy
        :return:
        '''

        self.best_strategy = np.zeors(4,3)

    def execute(self):
        '''
        TODO: In this function, we execute the result from the strategy, by changing the value of Z
              Notice that you do not need to mess up with Y.
        :return:
        '''

        self.Z = np.zeros(4, 4)

    def update_Y_with_Z(self):
        '''
        In this function we update Y wrt Z
        :return:
        '''
        self.Y = self.Y - self.Z

    def optimize_Q(self):
        '''
        TODO: We update our estimation of Q, to be discussed
        :return:
        '''

    def store_history(self):
        '''
        In this function, we store all the history wrt this time t
        :return: history
        '''
        his = {"Q": self.Q,  # We store our Q matrix
               "X": self.X,  # We store how many cars enter the system at the begining of time t
               "Y": self.Y,  # We store how many cars remains in the system at the end of time t
               "Z": self.Z,  # We store how many cars pass each crossroad.
               "A": self.best_strategy  # We store our decision
               }
        self.history.append(his)

    def Q_learning(self):
        for n in range(self.N):
            print("This is the", n, "th test.")
            self.initialize()  # Initialize the game
            for t in range(self.T):
                print("This is the", t, "th time")
                self.update_X()  # Update X
                self.update_Y_with_X()  # Update Y
                self.compute_Q()
                self.pick_strategy()
                print("The best strategy is", self.best_strategy)
                self.execute()
                self.update_Y_with_Z()
                self.store_history()
            self.optimize_Q()

if __name__ =="__main__":
    t = Traffic()
    t.Q_learning()