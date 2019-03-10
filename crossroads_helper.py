import numpy as np
import parameters


def Y2detailY(Y, P):
    '''
    :param Y: a 4x4 matrix for the number of cars waiting at the crossroads
    [[ A1, A2, A3, A4],
     [ B1. B2. B3. B4],
     [ C1, C2, C3, C4],
     [ D1, D2, D3, D4]]

    :param P: a 4x4x4 matrix indicating the probability of the flows of cars,
    for a single crossroads:
    P[A] = [[ A1 to A1, A1 to A2, A1 to A3, A1 to A4],
              A2 to A1. A2 to A2, A2 to A3, A2 to A4],
              ...
              A4 to A1, A4 to A2, A4 to A3, A4 to A4]]

    :return: a very detailed matix indicating how many cars are adding,
        detail_Y = 4x(4x4)
        for each crossroads,
        detail_Y[A] = P[A] element-wise-mult Y[A]'
    '''

    detail_Y = []

    for i, each_cross_roads in enumerate(Y):  # Take each row of Y
        detail_matrix = np.multiply(P[i+1],np.transpose(Y[i]))
        detail_Y.append(detail_matrix.astype(np.int))
    return np.array(detail_Y)