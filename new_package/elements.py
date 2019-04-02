import numpy as np
from parameters import speed_matrix, transition_matrix, poisson_matrix


class Juncture():

    def __init__(self, id="A", dt=10):

        # The 'id' is just the A/B/C/D representing different junctures
        self.id = id
        # 默认 dt 是间隔是10s
        self.dt = dt

        # 默认开始时间为0
        self.time = 0
        self.update_matrix()

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

        # This is the number of cars waiting at this juncture
        # We also provide two variations,
        # Cars in detail is computed with probability matrix P
        # the without detail function simply stores the cars on each direction.
        self.cars = np.zeros((4, 1))
        # 这个变量永远都是实时更新的
        # 4 represents direction in order of: West North East South

        # TODO The particular Q-value of this Juncture
        self.Q = np.zeros(1)

    @property
    def cars_detail(self):
        '''
        把它写成property以后，我们每次只需要更新self.cars 就可以直接划分车道
        进车的时候，永远是四个方向不分车道的；我们需要自动将他们分车道；
        '''
        return np.multiply(self.cars, self.P)

    @property
    def out_car_to(self):
        '''
        而出车的时候，自动就是每个车道各个方向；我们需要自动统计有多少车出去了。
        注意了和incar正好是相反的，这也非常合乎常识和逻辑。
        out_car 是一个4x1的向量，分别表示到四个方向车辆数（西，北，东，南）
        '''
        return np.sum(self.out_car_in_detail, axis=0)

    @property
    def out_car_from(self):
        '''
        出去的车是从哪里出去的
        '''
        return np.sum(self.out_car_in_detail, axis=1)

    def update_matrix(self):
        '''
        更新矩阵时间，注意了，需要先跟新系统时间，我已经写在主函数里了。
        '''
        self.P = transition_matrix(self.id, self.time)
        self.V = speed_matrix(self.id, self.time)
        self.N = poisson_matrix(self.id, self.time)

    def in_car(self, income_car):
        '''
        In this function, we let car into this juncture.
        :param:
        income_car: a 4x1 array, representing how many cars are coming from
                    the from direction
        '''
        self.cars = self.cars + income_car

    def sample_in_car(self, mask):
        '''
        Sample in car 和 in car的区别在于，sample这个只对系统外的车进行sample
        :param:
            mask: 需要人工手动输入，这个其实就是各个交通路口的连接。
                  例如mask是[1,1,0,1]就是只有西、北、南三个路口进行sample
        # TODO 把这个泊松分布写完，利用self.N，注意要乘上mask
        '''
        self.cars += np.random.poisson()

    def out_car(self):
        '''
        TODO 需要根据速度和转移概率计算有多少车出去了各个方向。
        In this function we compute how many cars are going out of the system
        注意了，这里要用到self.time和速度矩阵，转移矩阵不需要，直接使用cars_detail即可
        '''

        self.out_car_in_detail = np.zeros((4, 4))  # From north to south, etc.
        self.cars -= self.out_car_from  # 减掉那些出去了的车


class Road:
    def __init__(self, length):
        '''
        TODO 还没有写，
        Road 的基本结构，就是写一个队列，入队和离队就是缓冲过程。
        Road 是一个单行道！
        '''
        self.length = length
        self.queue = []
        self.time = 0  # 初始化，已经写好了自动会和主函数保持同步

    def queue_in(self, in_car):
        '''
        number是多少辆车在time时间进来了，储存到内存里
        '''
        pass

    def queue_out(self):
        '''
        有多少辆车在t时刻可以出去，
        '''
