import numpy as np
import random
# matrix is 2D array
# w is 3-D numpy array
# this function returns True or False which tells if the algorithm converged or not


def convergence(matrix, w):
    if len(matrix) == 0:
        raise Exception(
            'The matrix into convergence(...) has length equals to 0.')

    for row in matrix:
        label = row[2]  # 0 or 1
        inputs = row[:2]

        dotProduct = np.dot(w, np.insert(inputs, 1, 1))
        if label == 1 and dotProduct < 0:
            return False

        if label == 0 and dotProduct >= 0:
            return False

    return True


# This function takes a 2-d matrix data and returns the weight vector.
def learn(matrix):
    iterationLimit = 20
    w = np.array([1, 1, 1])

    while iterationLimit > 0 and not convergence(matrix, w):
        row = matrix[random.randrange(0, len(matrix))]

        if len(row) != 3:
            raise Exception('A row in the matrix is invalid.')

        point = row[:2]
        label = row[2]

        dotProduct = np.dot(w, np.insert(point, 1, 1))

        # update the weights
        if label == 1 and dotProduct < 0:
            w = np.add(w, np.insert(point, 1, 1))

        if label == 0 and dotProduct >= 0:
            w = np.subtract(w, np.insert(point, 1, 1))

        iterationLimit -= 1

    if iterationLimit == 0:
        raise Exception('Possibly, not learned the data.')

    return w
