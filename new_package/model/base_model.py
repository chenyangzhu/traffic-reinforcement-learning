import numpy as np
from elements import Juncture


class Base:
    def __init__(self):
        pass

    def best_strategy(self, state):
        '''
        这里的state，就是traffic函数里的self.juncs，我们只需要这个就可以做最优决策了

        由于我们每一个路口的action现在是4x3的matrix，因此best strategy是一个字典
        dict = {"A": np.zeros((4,3)),
                "B": np.zeros((4,3)),
                "C": np.zeros((4,3)),
                "D": np.zeros((4,3))}
        '''
        self.juncs = state.copy()
        # 这里同样用juncs是为了保证和主函数符号一致，
        # 另外我使用的是copy是因为我希望在选择策略的时候，确保不改变路口的任何信息。
        # 这里应该是“只读”的路口信息。

        # Some computations here...
        strategy = {"A": np.zeros((4, 3)),
                    "B": np.zeros((4, 3)),
                    "C": np.zeros((4, 3)),
                    "D": np.zeros((4, 3))}

        return strategy
