import numpy as np


class Games:

    BASIC_RPS = np.array([[0, -1 , 1], [1, 0, -1], [-1, 1, 0]])

    PRISONERS_DILEMMA = np.array([[3, 0], [5, 1]])

    AUGMENTED_RPS = np.array([[0,   -1,   1,       0.2],
                                [1,    0,   -1,       0.2],
                                [-1,   1,   0,        0.2],
                                [0.1, 0.1, 0.1, 0]])

    HAWK_DOVE = np.array([[0, 3], [1, 2]])


