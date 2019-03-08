import numpy as np

def return_strategy_space():
    '''
    In this function, we automatically generate the strategy space.
    :return: 81x4x3 strategy space.
    '''

    possible_solutions = [np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]
    strategies = []
    for comb1 in possible_solutions:
        for comb2 in possible_solutions:
            for comb3 in possible_solutions:
                for comb4 in possible_solutions:
                    strategies.append(np.concatenate([comb1,comb2,comb3,comb4],axis = 0).reshape((4,3)))
    return np.array(strategies)

def strategy2idx(the_strategy,strategy_space):
    '''
    In this function, we input a strategy and output it's index.
    This index is consistent to Action matrix.
    :return:
    '''

    for i in range(len(strategy_space)):
        if np.array_equal(the_strategy,strategy_space[i]):
            return i
