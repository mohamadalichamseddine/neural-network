import numpy as np

def sigmoid(z):
    # z = w . a + b
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def vectorized_result(j):
    # Convert a digit from 0 to 9 into a one-hot column vector
    result = np.zeros((10, 1))
    result[j] = 1.0
    return result