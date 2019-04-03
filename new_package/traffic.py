import numpy as np
from elements import Juncture, Road
import model

model_list = {"Qlearning-long":   model.qlearning.long,
              "Qlearning-greedy": model.qlearning.greedy,
              "sarsa-long":       model.sarsa.long,
              "sarsa-greedy":     model.sarsa.greedy}


class Traffic:
    def __init__(self, model_name="Qlearning"):
        '''
        时间：
            我们的计时方法是每10秒作为一个epoch，
            所有的矩阵15min进行一次更新，也就是900s，也就是90个epoch。
            以天为单位，一天24小时*60分钟*60s/10s，一共是8640个epoch。
            所有的时间单位，用epoch来表示。
        model_name:
            我们要使用的model名称，这里 model name 是直接作用在 pick_strategy 当中，
        '''
        self.T = 8640
        self.dt = 10                         # 默认 10 seconds per epoch

        self.model_name = model_name

        self.history = []                    # to store the event history

    def simulate(self):
        '''
        Main function for simulations.

        '''
        # 构建道路
        self.create_junctures()
        self.create_roads()
        self.initialize_model()

        for t in range(self.T):
            print("This is the", t, "th iteration")
            self.update_time(t)  # 跟新各个路口的时间和矩阵

            # 这一步sample进入系统里的车并且更新in_car
            self.sample_outer_car()

            self.best_strategy = self.pick_strategy(self.juncs)
            strategy_id = strategy2idx(self.best_strategy, self.action_space) 
            # 这一步执行这个策略,并返回下一步的状态
            self.execute()
            
            #下一步的策略
            strategy_ = self.pick_strategy(self.juncs)
            strategy_id_ = strategy2idx(strategy_, self.action_space)
            
            #更新q表
            self.model.learn(self.juncs,strategy_id,strategy_id_ )
            
            # 这一步保存历史
            self.store_history()

    def create_junctures(self):
        '''
        In this function, we basically create four junctures called ABCD.
        '''
        self.juncs = {"A": Juncture(id="A", dt=self.dt),
                      "B": Juncture(id="B", dt=self.dt),
                      "C": Juncture(id="C", dt=self.dt),
                      "D": Juncture(id="D", dt=self.dt)}

    def create_roads(self):
         # TODO 这里的距离是我瞎写的，需要重新改正。
         # 为了方便期间，所以路都是单向同行。
        self.roads = {"AB": Road(length=200),
                      "BA": Road(length=200),
                      "BC": Road(length=200),
                      "CB": Road(length=200),
                      "BD": Road(length=200),
                      "DB": Road(length=200)}

    def initialize_model(self):
        # 获得 model 并且利用自带的 __init__ 进行初始化
        self.model = model_list[self.model_name]

    def update_time(self, t):
        '''
        在这个function里，我们update所有路口的自有时间
        顺带update一下路口的矩阵
        '''
        for each_crossroad in self.juncs:
            each_crossroad.time = t
            if t % 90 == 0:
                # 意味着到达15分钟了，需要跟新一次矩阵
                each_crossroad.update_matrix()
        for each_road in self.roads:
            each_road.time = t

    def sample_outer_car(self):
        '''
        在这一个方程，我们通过sampling，得到从系统外进入系统的车辆数。
        注意了只有系统外进入系统的车辆需要进行sample，系统内的车辆来来往往不需要sample
        系统内的车辆来往在 self.execute() 里更新。
        需要sample的路口是：A1 A2 A4 B2 C2 C3 C4 D1 D3 D4
        '''

        for each_crossroad in self.juncs:
            self.outcar()  # 这里不需要传入时间，因为时间在之前已经update过了。

    def pick_strategy(self):
        '''
        这个function里，我们通过__init__里输入的modelname，选择model，来选择最佳的测路

        注意了，因为我们把model写到了最外面，所以必须要单独把状态空间传递到model里去，
        但这个不是难事情，我们有非常完美的self.juncs，所以一键搞定。
        '''

        return self.model.best_strategy(self.juncs)

    def execute(self):
        '''
        这一步是更新个路口的车辆数，分为以下几个步骤：
            1. 我们算出每一个路口，在10s内可以通过多少辆车，得到out_detail，
               out_detail算完，会自动在等待的车辆数cars减去这个出去的车数，
               同时得到out_car_to。
            2. 我们利用得到的out_car_to，计算in_car_detail。并且更新。
        '''

        # Step1
        for each_crossroad in self.juncs:
            each_crossroad.out_car()
        # Step1 已经写完了。

        # Step 2 这里，需要通过别人的outcar，来计算A 的incar，

        # TODO 事实上，当车来到这个路口值钱，首先需要经过一段时间，也就是上一个路口到这一个路口的举例，
        # 幸运的是，我们一共只有3条路连接，也就是说，除了这三条路之外，车都是瞬移的。
        # 那么我们只需要写这三条路的一个缓冲/等待即可。
        # 为了保证每个路口自身的update简单、完好，我们不把缓冲的车流写在任何一个路口里，
        # 相反的，我们把它写到新的一个类，叫做 elements.Road 作为缓冲

        # 以下只是一个草稿
        # A的incar只有可能来自于B1->A3的车，因此
        # 这里把离开的车全部缓存到road里去，
        self.roads["BA"].queue_in(self.juncs['B'].out_car_to[0])
        # 这里把缓存过的车加入到路口去
        self.juncs["A"].in_car([0, 0, self.roads['BA'].queue_out(), 0])

        # B的incar来自A3->B1, C1->B3, D2->B4


        # TODO B 下面这个还不对，还要修改！
        self.juncs["B"].in_car([self.juncs['A'].out_car_to[2],
                                0,
                                self.juncs['C'].out_car_to[0],
                                self.juncs['D'].out_car_to[1]])

        # TODO 为了让你更好地理解这个程序，请自己完成 C 和 D！
