import random

import numpy as np

class Network:
    def __init__(self, layerSizes: list[int]):
        self.numLayers = len(layerSizes)
        self.layerSizes = layerSizes        # Number of neurons in the respective layers

        # Create one bias vector for each non-input layer
        self.biases = []
        for layerSize in layerSizes[1:]:
            self.biases.append(np.random.randn(layerSize, 1))

        self.weights = []
        for prev_layer_size, next_layer_size in zip(layerSizes[:-1], layerSizes[1:]):
            weightMatrix = np.random.randn(next_layer_size, prev_layer_size)
            self.weights.append(weightMatrix)

    def sigmoid(z: int):
        # z = w . a + b
        return 1.0 / (1.0 + np.exp(-z))
    
    def feedforward(self, a: np.ndarray) -> np.ndarray :
        # Return the output of the network, given input a.
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, learningRate, test_data = None):
        n: int = len(training_data)

        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches_data = [
                training_data[k : k + mini_batch_size] for k in range(0, n, mini_batch_size)
                ]
            
            for mini_batch_data in mini_batches_data:
                self.update_mini_batch(mini_batch_data, learningRate)

            if test_data:
                numCorrect = self.evaluate(test_data)
                print(f"Epoch {epoch} complete. {numCorrect} / {len(test_data)}")
            else:
                print(f"Epoch {epoch} complete")

        def update_mini_batch(self, mini_batch_data, learningRate):
            return