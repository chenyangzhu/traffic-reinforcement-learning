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

        for a in range(3): #第一个路口的3个相位
            for b in range(3):#第二个路口的3个相位
                for c in range(3):#三一个路口的3个相位
                    for d in range(3):#第四个路口的3个相位
                        self.A[0][a] = 1
                        self.A[1][b] = 1
                        self.A[2][c] = 1
                        self.A[3][d] = 1
                        for i in range(4):                                              #对每个crossroad 从左边的entry开始是entry1 1234
                            if self.A[i][0]==1:
                            '''      
                strategy矩阵的第i行第1列==1 表示第i个crossroad采取 第1种相位  
                outgoing cars= delta t(1s) * Vij / physical length of a car(3m)
                '''
                                self.Z[i][1]=dt*self.V[i][0][1]/car_len                 #crossroad i  entry 1 to 2
                                self.Z[i][3]=dt*self.V[i][2][3]/car_len                 #crossroad i  entry 3 to 4
                            if self.A[i][1]==1:
                                self.Z[i][2]=dt*self.V[i][0][2]/car_len                 #crossroad i  entry 1 to 3
                                self.Z[i][0]=dt*self.V[i][2][0]/car_len                 #crossroad i  entry 3 to 1
                            if self.A[i][2]==1:
                                self.Z[i][0]=dt*self.V[i][1][0]/car_len                 #crossroad i  entry 2 to 1
                                self.Z[i][2]=dt*self.V[i][1][2]/car_len                 #crossroad i  entry 2 to 3
                                self.Z[i][3]=dt*self.V[i][1][3]/car_len                 #crossroad i  entry 2 to 4
                                self.Z[i][0]=dt*self.V[i][3][0]/car_len                 #crossroad i  entry 4 to 1
                                self.Z[i][1]=dt*self.V[i][3][1]/car_len                 #crossroad i  entry 4 to 2
                                self.Z[i][2]=dt*self.V[i][3][2]/car_len    
    
                        self.R.append(1/np.sum(self.Y-self.Z))
        return self.R
    
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
        self.best_strategy = np.zeros((4,3))
        tmp=self.Q.tolist() 
        idm=tmp.index(max(tmp))         
        '''
        找到Q里面的最大的值对应的index_max(idm)  （最优策略在第81个strategy中位于第idm个）
        然后把idm这个index转化成strategy矩阵 self.best_strategy
        '''
        cross1=idm//27                                  #cross1表示第1个crossroad选择的相位是第cross1个相位（cross1可能为0，1，2）           
        cross2=(idm-cross1*27)//9
        cross3=(idm-cross1*27-cross2*9)//3
        cross4=idm-cross1*27-cross2*9-cross3*3
        self.best_strategy[0][cross1]=1
        self.best_strategy[1][cross2]=1
        self.best_strategy[2][cross3]=1
        self.best_strategy[3][cross4]=1
        return self.best_strategy
    
    def execute(self):
        '''
        TODO: In this function, we execute the result from the strategy, by changing the value of Z
              Notice that you do not need to mess up with Y.
        :return:
        '''
        self.Z = np.zeros((4, 4))
        dt=1
        car_len=3
        for i in range(4):                                              #对每个crossroad 从左边的entry开始是entry1 1234
            if self.best_strategy[i][0]==1:
                '''
                strategy矩阵的第i行第1列==1 表示第i个crossroad采取 第1种相位  
                outgoing cars= delta t(1s) * Vij / physical length of a car(3m)
                '''
                self.Z[i][1]=dt*self.V[i][0][1]/car_len                 #crossroad i  entry 1 to 2
                self.Z[i][3]=dt*self.V[i][2][3]/car_len                 #crossroad i  entry 3 to 4
            if self.best_strategy[i][1]==1:
                self.Z[i][2]=dt*self.V[i][0][2]/car_len                 #crossroad i  entry 1 to 3
                self.Z[i][0]=dt*self.V[i][2][0]/car_len                 #crossroad i  entry 3 to 1
            if self.best_strategy[i][2]==1:
                self.Z[i][0]=dt*self.V[i][1][0]/car_len                 #crossroad i  entry 2 to 1
                self.Z[i][2]=dt*self.V[i][1][2]/car_len                 #crossroad i  entry 2 to 3
                self.Z[i][3]=dt*self.V[i][1][3]/car_len                 #crossroad i  entry 2 to 4
                self.Z[i][0]=dt*self.V[i][3][0]/car_len                 #crossroad i  entry 4 to 1
                self.Z[i][1]=dt*self.V[i][3][1]/car_len                 #crossroad i  entry 4 to 2
                self.Z[i][2]=dt*self.V[i][3][2]/car_len                 #crossroad i  entry 4 to 3
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
