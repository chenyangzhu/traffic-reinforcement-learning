import numpy as np
from strategies import return_strategy_space, strategy2idx

class Traffic:
    def __init__(self):
        self.N = 1000  # We do 1,000 times trial
        self.T = 600  # 600 minutes
        self.NACTIONS = 81  # # of our action space
        self.gamma = 0.8  # discount rate
        self.get_speed_matrix()  # Some data from the distribution
        self.get_transition_matrix()  # Some data from the distribution
        self.A = np.zeros((4, 3))  # action space
        self.Q = np.zeros(self.NACTIONS)  # 81 actions in all
        self.R = np.zeros(self.NACTIONS)  # 81 actions, so 81 rewards
        self.history = []  # We use this to store the event history
        self.dt = 100
        self.car_len = 7
        self.multiplier = 100  # taxi/total#of car
        self.strategy_space = return_strategy_space()

    def get_speed_matrix(self):
        '''
        TODO: In this function, we compute the speed matrix.
        :return: self.V 是一个字典
        '''

        V_1 = np.array(
            [[6.6457, 7.8258, 8.9372, 7.5126], [8.4115, 6.5528, 8.7679, 7.5184], [9.384, 9.0436, 8.2904, 7.9447],
             [7.484, 7.4867, 8.0066, 5.9959]]) * 0.277778  # Change to m/s
        V_2 = np.array(
            [[7.5451, 7.0053, 8.9476, 8.4978], [7.3584, 5.408, 7.3179, 7.4435], [9.4287, 8.6249, 7.9717, 7.564],
             [8.2523, 7.133, 8.2126, 7.5429]]) * 0.277778
        V_3 = np.array(
            [[7.2576, 8.8443, 8.8698, 8.6086], [9.607, 8.3618, 7.6701, 9.6802], [8.954, 8.3575, 7.0293, 7.1893],
             [9.2592, 10.4052, 7.6567, 7.1195]]) * 0.277778
        V_4 = np.array(
            [[6.6174, 7.9762, 6.4442, 7.0779], [7.4318, 4.1769, 7.0859, 8.1752], [6.6421, 7.3496, 5.524, 6.1823],
             [6.6792, 7.8172, 6.2007, 5.6317]]) * 0.277778

        self.V = {1: V_1,
                  2: V_2,
                  3: V_3,
                  4: V_4}

    def get_transition_matrix(self):
        '''
        TODO: In this function, we compute the transition matrix.
        :return: self.P 是一个字典
        '''

        P_1 = np.array(
            [[0.0152, 0.0607, 0.7909, 0.1332], [0.1054, 0.0041, 0.6108, 0.2797], [0.5771, 0.3029, 0.0245, 0.0954],
             [0.1148, 0.2991, 0.574, 0.0121]])
        P_2 = np.array(
            [[0.0173, 0.0218, 0.5971, 0.3638], [0.1263, 0.0053, 0.4579, 0.4105], [0.7537, 0.0939, 0.0154, 0.137],
             [0.6154, 0.1455, 0.2324, 0.0067]])
        P_3 = np.array(
            [[0.0319, 0.242, 0.4447, 0.2814], [0.2959, 0.0062, 0.0456, 0.6523], [0.4808, 0.0369, 0.0084, 0.4739],
             [0.2577, 0.4082, 0.3238, 0.0103]])
        P_4 = np.array(
            [[0.0029, 0.0068, 0.25, 0.7403], [0.1462, 0.0012, 0.1155, 0.7371], [0.3828, 0.2051, 0.0039, 0.4082],
             [0.3675, 0.4522, 0.177, 0.0034]])

        self.P = {1: P_1,
                  2: P_2,
                  3: P_3,
                  4: P_4}

    def initialize(self):
        '''
        In this function we initialize our state space
        '''

        self.incomerate = np.array([[0.009907, 0.015428, 0.012338, 0.011991], [0.008565, 0.002199, 0.009387, 0.009421],
                                    [0.016505, 0.015035, 0.015069, 0.005926], [0.003831, 0.006921, 0.020162, 0.013669]])

        self.X = np.random.poisson(self.incomerate * self.multiplier)  # incoming cars, this time without masks.
        self.Y = self.X  # Just the innitial state.
        self.Z = np.zeros((4, 4))  # Outgoing cars are zero in the first state.

    def update_X(self):
        '''
        In this function, we update_X the incoming cars
        Notice that X comes from outside our data, and also from the previous state of Z.
        Notice that X only represents cars from outside our system, thus should only have 10 entries.

        To be compatible with y, we still write it into 4x4 matrix. but we would only have the following entry

        [[ *, *, *,0],
         [0, *, 0, 0],
         [0, *, *, *],
         [*, 0, *, *]]

        Where * denotes that there should be a value.
        :return: we store the incoming value into self.X
        '''

        self.X = np.random.poisson(self.incomerate * self.multiplier*self.dt) # Generate randomly with income possibility
        mask = np.array([[1,1,1,0],
                         [0,1,0,0],
                         [0,1,1,1],
                         [1,0,1,1]])
        self.X = np.multiply(self.X, mask)


    def update_Y_with_X(self):
        '''
        In this function, we update the waiting cars.
        Notice that we only contribute X into Y, but we do not subtract Z
        '''
        self.Y = self.Y + self.X


    def compute_reward(self, Y):
        '''
        TODO In this function, we compute the reward function, based on the current Y
        for a in range(3):  # 第一个路口的3个相位
             for b in range(3):  # 第二个路口的3个相位
                for c in range(3):  # 三一个路口的3个相位
                    for d in range(3):  # 第四个路口的3个相位
                        self.A[0][a] = 1
                        self.A[1][b] = 1
                        self.A[2][c] = 1
                        self.A[3][d] = 1
                        for i in range(4):  # 对每个crossroad 从左边的entry开始是entry1 1234
                            if self.A[i][0] == 1:
                                # strategy矩阵的第i行第1列==1 表示第i个crossroad采取 第1种相位
                                # outgoing cars= delta t(1s) * Vij / physical length of a car(3m)

                                self.Z[i][1] = self.dt * self.V[i][0][1] / self.car_len  # crossroad i  entry 1 to 2
                                self.Z[i][3] = self.dt * self.V[i][2][3] / self.car_len  # crossroad i  entry 3 to 4
                            if self.A[i][1] == 1:
                                self.Z[i][2] = self.dt * self.V[i][0][2] / self.car_len  # crossroad i  entry 1 to 3
                                self.Z[i][0] = self.dt * self.V[i][2][0] / self.car_len  # crossroad i  entry 3 to 1
                            if self.A[i][2] == 1:
                                self.Z[i][0] = self.dt * self.V[i][1][0] / self.car_len  # crossroad i  entry 2 to 1
                                self.Z[i][2] = self.dt * self.V[i][1][2] / self.car_len  # crossroad i  entry 2 to 3
                                self.Z[i][3] = self.dt * self.V[i][1][3] / self.car_len  # crossroad i  entry 2 to 4
                                self.Z[i][0] = self.dt * self.V[i][3][0] / self.car_len  # crossroad i  entry 4 to 1
                                self.Z[i][1] = self.dt * self.V[i][3][1] / self.car_len  # crossroad i  entry 4 to 2
                                self.Z[i][2] = self.dt * self.V[i][3][2] / self.car_len

                        self.R[] = (1 / np.sum(self.Y - self.Z))
        '''
        return np.sum(np.exp(-Y))

    def compute_Q(self):
        '''
        TODO In this function we estimate the value of Q matrix, which is a 81 long
        We will use function self.compute_reward() in this function.
        :return:
        '''
        X_next = np.random.poisson(self.incomerate * self.multiplier*self.dt) # Randomly initiate some cars in next t
        Y_next = self.Y + X_next
        for i in range(81): # For each strategy
            # We want to compute the reward and store them to Q
            strategy = self.strategy_space[i]
            Z = self.execute(strategy, Y_next)
            Y_after = Y_next - Z
            r = self.compute_reward(Y_after) # Then we compute the reward
            self.Q[i] = r # And we store it into the Q matrix

    def pick_strategy(self):
        '''
        TODO: This is a very easy function, where we choose the largest number Q value in Q matrix and find the best stragegy
        :return:
        '''
        self.best_strategy = self.strategy_space[np.argmax(self.Q)]

        '''
        找到Q里面的最大的值对应的index_max(idm)  （最优策略在第81个strategy中位于第idm个）
        然后把idm这个index转化成strategy矩阵 self.best_strategy
        cross1 = idm // 27  # cross1表示第1个crossroad选择的相位是第cross1个相位（cross1可能为0，1，2）
        cross2 = (idm - cross1 * 27) // 9
        cross3 = (idm - cross1 * 27 - cross2 * 9) // 3
        cross4 = idm - cross1 * 27 - cross2 * 9 - cross3 * 3
        self.best_strategy[0][cross1] = 1
        self.best_strategy[1][cross2] = 1
        self.best_strategy[2][cross3] = 1
        self.best_strategy[3][cross4] = 1
        '''

    def execute(self, strategy, Y):
        '''
        TODO: In this function, we execute the result from the strategy, by changing the value of Z
              Notice that you do not need to mess up with Y.
        :param:
        strategy : a 4x3 strategy matrix
        :return:
        Z : # of outgoing cars
        '''

        self.Z = np.zeros((4,4))

        for i in range(4):  # 对每个crossroad 从左边的entry开始是entry1 1234
            if strategy[i][0] == 1:
                '''
                strategy矩阵的第i行第1列==1 表示第i个crossroad采取 第1种相位  
                outgoing cars= delta t(1s) * Vij / physical length of a car(3m)
                '''
                self.Z[i][0] = int(self.dt * (self.V[i+1][0][1]+self.V[i+1][0][3]) / self.car_len)  # crossroad i  entry 1 to 2
                self.Z[i][1] = int(self.dt * self.V[i+1][1][0] / self.car_len)
                self.Z[i][2] = int(self.dt * (self.V[i+1][2][1]+self.V[i+1][2][3] )/ self.car_len)
                self.Z[i][3] = int(self.dt * self.V[i+1][3][2] / self.car_len)  # crossroad i  entry 3 to 4

            elif strategy[i][1] == 1:
                # 换成 elif 因为只有可能有一种状态发生。 --klaus [0 0 1]]

                self.Z[i][0] = int(self.dt * (self.V[i+1][0][2]+self.V[i+1][0][3]) / self.car_len)  # crossroad i  entry 1 to 3
                self.Z[i][1] = int(self.dt * self.V[i+1][1][0] / self.car_len)
                self.Z[i][2] = int(self.dt * self.V[i+1][2][1] / self.car_len)
                self.Z[i][3] = int(self.dt * self.V[i+1][3][2] / self.car_len)  # crossroad i  entry 3 to 1

            elif strategy[i][2] == 1:
                self.Z[i][0] = int(self.dt * self.V[i+1][0][3] / self.car_len)  # crossroad i  entry 2 to 1
                self.Z[i][1] = int(self.dt * (self.V[i+1][1][2]+self.V[i+1][1][3]) / self.car_len)  # crossroad i  entry 2 to 3
                self.Z[i][2] = int(self.dt * self.V[i+1][2][1] / self.car_len)  # crossroad i  entry 2 to 4
                self.Z[i][3] = int(self.dt * (self.V[i+1][3][0]+self.V[i+1][3][1]+self.V[i+1][3][2]) / self.car_len)  # crossroad i  entry 4 to 1

        # if Z>Y then, set Z to Y
        # Since the outgoing car cannot be greater than the # of waiting cars
        for i in range(4):
            for j in range(4):
                if self.Z[i,j] > Y[i,j]:
                    self.Z[i,j] = Y[i,j]

        return self.Z

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
        self.initialize()  # Initialize the game
        for t in range(self.T):
            print("This is the", t, "th time")
            self.update_X()  # Update X
            self.update_Y_with_X()  # Update Y
            print("X is\n", self.X)
            print("Y is\n", self.Y)
            self.compute_Q()
            self.pick_strategy()
            print("The best strategy is\n", self.best_strategy)
            self.Z = self.execute(self.best_strategy, self.Y)
            self.update_Y_with_Z()
            print("Z is \n", self.Z)
            self.store_history()

if __name__ == "__main__":
    t = Traffic()
    t.Q_learning()


   
