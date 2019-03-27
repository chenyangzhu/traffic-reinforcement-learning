import numpy as np
import pandas as pd
from strategies import return_strategy_space, strategy2idx
from parameters import speed_matrix, transition_matrix
from crossroads_helper import Y2detailY


class Traffic:
    def __init__(self):
        self.T = 600                          # 60 minutes
        self.NACTIONS = 81                   # of our action space
        self.gamma = 0.8                     # discount rate
        self.V = speed_matrix()              # Some data from the distribution
        self.P = transition_matrix()         # Some data from the distribution
        self.A = np.zeros((4, 3))            # action space
        self.Q = np.zeros(self.NACTIONS)     # 81 actions in all
        self.R = np.zeros(self.NACTIONS)     # 81 actions, so 81 rewards
        self.history = []                    # We use this to store the event history
        self.dt = 5                          # seconds
        self.car_len = 3
        self.multiplier = 10  # taxi/total#of car
        self.strategy_space = return_strategy_space()

        # Penalties
        self.alpha = 0.5 # Penalty
        self.beta = 10

    def initialize(self):
        '''
        In this function we initialize our state space

        :return:
            self.Y  ##注意了，y只是每个路口在等待的车的数量，还需要转换为前往了某个方向。
        '''

        self.incomerate = np.array([[0.009907, 0.015428, 0.012338, 0.011991], [0.008565, 0.002199, 0.009387, 0.009421],
                                    [0.016505, 0.015035, 0.015069, 0.005926], [0.003831, 0.006921, 0.020162, 0.013669]])

        self.X = np.random.poisson(self.incomerate * self.multiplier)  # incoming cars, this time without masks.
        self.Y = self.X  # Just the initial state.
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

        self.X = np.random.poisson(
            self.incomerate * self.multiplier * self.dt)  # Generate randomly with income possibility
        mask = np.array([[1, 1, 1, 0],
                         [0, 1, 0, 0],
                         [0, 1, 1, 1],
                         [1, 0, 1, 1]])
        self.X = np.multiply(self.X, mask)

    def update_Y_with_X(self):
        '''
        In this function, we update the waiting cars.
        Notice that we only contribute X into Y, but we do not subtract Z
        '''
        self.Y = self.Y + self.X

    def compute_reward(self, Y, Z_out, Z_in):
        '''
        In this function, we compute the reward function, based on the current Y
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
        Reward1 = np.exp(-np.sum(np.square(Y)))
        Reward2 = np.sum(Z_out)
        # 如果有超过20辆车，加penalty
        Penalty1 = self.alpha * sum(sum(self.Y > 20)) / 16
        Penalty2 = np.sum(Z_in)

        # 如果重复决策 pentalty
        reward = Reward1 + Reward2 - Penalty1 - Penalty2
        return reward

    def compute_Q(self):
        '''
        In this function we estimate the value of Q matrix, which is a 81 long
        We will use function self.compute_reward() in this function.
        :return:
        '''
        X_next = np.random.poisson(self.incomerate * self.multiplier * self.dt)  # Randomly initiate some cars in next t
        Y_next = self.Y + X_next
        for i in range(81):  # For each strategy
            # We want to compute the reward and store them to Q
            strategy = self.strategy_space[i]
            Z_out, Z_in = self.execute(strategy, Y_next)
            Y_after = Y_next - Z_out + Z_in
            r = self.compute_reward(Y_after, Z_out, Z_in)  # Then we compute the reward
            self.Q[i] = r  # And we store it into the Q matrix

    def pick_strategy(self):
        '''
        This is a very easy function, where we choose the largest number Q value in Q matrix and find the best stragegy
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
        In this function, we execute the result from the strategy, by changing the value of Z
        Notice that you do not need to mess up with Y.
        :param:
        strategy : a 4x3 strategy matrix
        :return:

        Z_out : # of outgoing cars 每个路口走了多少辆车
            Z_out = [[ A1, A2, A3, A4],
                 [ B1, B2, B3, B4],
                 ...
                 [ D1, D2, D3, D4]]

        Z_in: 用来表示系统内各流向的车互相流动的过程，用来表示某个路口流进了多少车，
                因此只有以下几个值要取值
            Z_in = [[ 0, 0, A3, 0],
                    [ B1, 0, B3, B4],
                    [ C1, 0, 0, 0],
                    [ 0, D2, 0, 0]]
        '''

        Z_out = np.zeros((4, 4))

        detail_Y = Y2detailY(Y, self.P)  # 必须要用detail Y
        # print("Detail Y:\n",detail_Y)
        for i in range(4):  # 对每个crossroad 从左边的entry开始是entry1 1234
            if strategy[i][0] == 1:

                '''
                strategy矩阵的第i行第1列==1 表示第i个crossroad采取 第1种相位  
                outgoing cars= delta t(1s) * Vij / physical length of a car(3m)
                
                为了清晰，我写一个，剩下的并在一起了 --Klaus
                '''
                # For A1 to A2
                max_number_of_cars_to_go = int(self.dt * (self.V[i + 1][0][1] + self.V[i + 1][0][3]) / self.car_len)
                we_only_have_these_cars_to_go = detail_Y[i, 0, 1] + detail_Y[i, 0, 3]
                Z_out[i][0] = min(max_number_of_cars_to_go, we_only_have_these_cars_to_go)


                Z_out[i][1] = min(int(self.dt * self.V[i + 1][1][0] / self.car_len), detail_Y[i, 1, 0])
                Z_out[i][2] = min(int(self.dt * (self.V[i + 1][2][1] + self.V[i + 1][2][3]) / self.car_len),
                                  detail_Y[i, 2, 3] + detail_Y[i, 2, 1])
                Z_out[i][3] = min(int(self.dt * self.V[i + 1][3][2] / self.car_len), detail_Y[i, 3, 2])

            elif strategy[i][1] == 1:
                # 换成 elif 因为只有可能有一种状态发生。 --klaus

                Z_out[i][0] = min(int(self.dt * (self.V[i + 1][0][2] + self.V[i + 1][0][3]) / self.car_len),
                                  detail_Y[i, 0, 2] + detail_Y[i, 0, 3])
                Z_out[i][1] = min(int(self.dt * self.V[i + 1][1][0] / self.car_len), detail_Y[i, 1, 0])
                Z_out[i][2] = min(int(self.dt * self.V[i + 1][2][1] / self.car_len), detail_Y[i, 2, 1])
                Z_out[i][3] = min(int(self.dt * self.V[i + 1][3][2] / self.car_len), detail_Y[i, 3, 2])

            elif strategy[i][2] == 1:
                Z_out[i][0] = min(int(self.dt * self.V[i + 1][0][3] / self.car_len),
                                  detail_Y[i, 0, 3])  # crossroad i  entry 2 to 1
                Z_out[i][1] = min(int(
                    self.dt * (self.V[i + 1][1][2] + self.V[i + 1][1][3]) / self.car_len), detail_Y[i, 1, 2] + detail_Y[i, 1, 3])  # crossroad i  entry 2 to 3
                Z_out[i][2] = min(int(self.dt * self.V[i + 1][2][1] / self.car_len),
                                  detail_Y[i, 2, 1])  # crossroad i  entry 2 to 4
                Z_out[i][3] = min(int(self.dt * (self.V[i + 1][3][0] + self.V[i + 1][3][1] + self.V[i + 1][3][
                    2]) / self.car_len), detail_Y[i, 3, 0] + detail_Y[i, 3, 1])  # crossroad i  entry 4 to 1

        # if Z>Y then, set Z to Y
        # Since the outgoing car cannot be greater than the # of waiting cars
        for i in range(4):
            for j in range(4):
                if Z_out[i, j] > Y[i, j]:
                    Z_out[i, j] = Y[i, j]

        Z_in = np.zeros((4, 4))
        Z_detail = Y2detailY(Z_out, self.P)
        # A3 的车全部来自于 B 中到路口1的车流
        Z_in[0, 2] = np.sum(Z_detail[1, :, 0])
        # B1 的车全部来自 A3
        Z_in[1, 0] = np.sum(Z_detail[0, :, 2])
        # B3 的车全部来自 C1
        Z_in[1, 2] = np.sum(Z_detail[2, :, 0])
        # B4 的车全部来自 D2
        Z_in[1, 3] = np.sum(Z_detail[3, :, 1])
        # C1 的车全部来自 B3
        Z_in[2, 0] = np.sum(Z_detail[1, :, 2])
        # D2 的车全部来自 B4
        Z_in[3, 1] = np.sum(Z_detail[1, :, 3])
        Z_in = Z_in.astype(np.int)

        return Z_out, Z_in

    def update_Y_with_Z(self):

        '''
        In this function we update Y wrt Z
        :return:
        '''
        self.Y = self.Y - self.Z_out + self.Z_in
        # Force Y to be integer
        self.Y = self.Y.astype(np.int)

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
            #print("X is\n", self.X)
            print("Y is\n", self.Y)
            self.compute_Q()
            self.pick_strategy()
            print("The best strategy is\np", self.best_strategy)
            self.Z_out, self.Z_in = self.execute(self.best_strategy, self.Y)
            self.update_Y_with_Z()
            #print("Z_out is \n", self.Z_out)
            #print("Z_in is \n", self.Z_in)
            self.store_history()


if __name__ == "__main__":
    t = Traffic()
    t.Q_learning()
    df = pd.DataFrame(t.history)
    df.to_excel("output.xlsx")
