import numpy as np

# matrix is 2D array
# w is 3-D numpy array


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
