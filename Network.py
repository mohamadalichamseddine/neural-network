import random

import numpy as np

class Network:
    def __init__(self, layerSizes: list[int]):
        self.numLayers = len(layerSizes)
        self.layerSizes = layerSizes        # Number of neurons in each layer

        # Create one bias vector for each non-input layer
        self.biases = []
        for layerSize in layerSizes[1:]:
            self.biases.append(np.random.randn(layerSize, 1))

        self.weights = []
        for prev_layer_size, next_layer_size in zip(layerSizes[:-1], layerSizes[1:]):
            # Each row corresponds to one neuron in the next layer
            # Each column corresponds to one neuron in the previous layer
            weightMatrix = np.random.randn(next_layer_size, prev_layer_size)
            self.weights.append(weightMatrix)
    
    def feedforward(self, a: np.ndarray) -> np.ndarray :
        # Return the output of the network, given a is the input
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, learningRate, test_data = None):
        n: int = len(training_data)

        for epoch in range(epochs):
            # Spliting training data to different batches: 
            random.shuffle(training_data)
            mini_batches_data = [
                training_data[k : k + mini_batch_size] for k in range(0, n, mini_batch_size)
                ]
            
            # Apply a single step of gradient descent for each mini-batch:
            for mini_batch_data in mini_batches_data:
                # update_mini_batch updates the network weights and biases according to a single 
                # iteration of gradient descent, using just the training data in mini_batch_data
                self.update_mini_batch(mini_batch_data, learningRate)

            if test_data:
                numCorrect = self.evaluate(test_data)
                print(f"Epoch {epoch} complete. Correct %: {numCorrect} / {len(test_data)}")
            else:
                print(f"Epoch {epoch} complete")

    def update_mini_batch(self, mini_batch_data, learningRate):
        """ Averages gradients and updates parameters.
            Update the network's weights and biases using backpropagation to a single mini batch.
            mini_batch_data is a list of tuples (x, y)
        """
        # The purpose is to accumulate the bias and weights gradient from every training example in mini_batch_data:
        del_b = [np.zeros(b.shape) for b in self.biases]    # one zero-filled array for every bias vector
        del_w = [np.zeros(w.shape) for w in self.weights]   # one zero-filled matrix for every weight matrix.

        # Process each training example:
        for x, y in mini_batch_data:
            # Compute gradients using backpropagation:
            batch_del_b, batch_del_w = self.backpropagation(x, y)
            # Accumulate the bias gradients:
            # For each layer, adds the current example’s bias gradient to the accumulated bias gradient
            del_b = [b + bb for b, bb in zip(del_b, batch_del_b)]
            del_w = [w + bw for w, bw in zip(del_w, batch_del_w)]

        # Updating the weights:
        # At this point, del_w contains the sum of all weight gradients from the mini-batch
        # Apply the gradient descent equation:
        # w are the old weights of a layer. nw is the sum of the gradients over all examples  
        self.weights = [w - (learningRate / len(mini_batch_data)) * nw 
                        for w, nw in zip(self.weights, del_w)]
        
        # Updating the biases:
        self.biases = [b - (learningRate / len(mini_batch_data)) * nb
                        for b, nb in zip(self.biases, del_b)]
    
    def backpropagation(self, x, y):
        # Compute the gradients of the cost function
        """ Return a tuple (del_b, del_w) representing the gradient for the
            cost function C_x.
            del_b and del_w are lists of numpy array objects (one for each layer).
        """
        del_b = [np.zeros(b.shape) for b in self.biases]
        del_w = [np.zeros(w.shape) for w in self.weights]

        #  Feedforward
        activation = x
        activationsByLayer = [x]
        weightedInputsByLayer = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            weightedInputsByLayer.append(z)
            activation = sigmoid(z)
            activationsByLayer.append(activation)

        # Backward pass:
        outputLayerActivations = activationsByLayer[-1]
        outputLayerWeightedInputs = weightedInputsByLayer[-1]

        # δ^L = ∂C/∂a^L ⊙ σ'(z^L)
        delta = self.cost_derivative(outputLayerActivations, y) * sigmoid_prime(outputLayerWeightedInputs)

        # ∂C/∂b^L = δ^L
        del_b[-1] = delta

        # ∂C/∂W^L = δ^L (a^(L-1))^T
        del_w[-1] = np.dot(delta, activationsByLayer[-2].transpose())

        # Move backward through the hidden layers
        for layer in range(2, self.numLayers):
            currentLayerWeightedInput = weightedInputsByLayer[-layer]
            sp = sigmoid_prime(currentLayerWeightedInput)

            # δ^l = (W^(l+1))^T δ^(l+1) ⊙ σ'(z^l)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp

            # ∂C/∂b^l = δ^l
            del_b[-layer] = delta

            # ∂C/∂W^l = δ^l (a^(l-1))^T
            del_w[-layer] = np.dot(delta, activationsByLayer[-layer - 1].transpose())
        
        return (del_b, del_w)
    
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives del(C_x)/del(a) for the output activations"""
        return (output_activations - y) 

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

def sigmoid(z):
    # z = w . a + b
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
