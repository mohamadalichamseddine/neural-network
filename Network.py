import random
import numpy as np

from utils import sigmoid, sigmoid_prime, vectorized_result
import cost

class Network:
    def __init__(self, layerSizes: list[int], costFunction=cost.CrossEntropyCost):
        self.numLayers = len(layerSizes)
        self.layerSizes = layerSizes
        self.costFunction = costFunction
        self._initialize_weights()
    
    def _initialize_weights(self):
        self.biases = []
        for layerSize in self.layerSizes[1:]:
            self.biases.append(np.random.randn(layerSize, 1))

        self.weights = []
        for prev_layer_size, next_layer_size in zip(self.layerSizes[:-1], self.layerSizes[1:]):
            weight_matrix = np.random.randn(next_layer_size, prev_layer_size) / np.sqrt(prev_layer_size)
            self.weights.append(weight_matrix)

    def feedforward(self, a: np.ndarray) -> np.ndarray :
        # Computes the network prediction 
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, learningRate, regularization_param=0.0,
            evaluation_data = None,
            monitor_training_cost = False,
            monitor_training_accuracy = False,
            monitor_evaluation_cost = False,
            monitor_evaluation_accuracy = False):
        
        n = len(training_data)

        training_cost = [] if monitor_training_cost else None
        training_accuracy = [] if monitor_training_accuracy else None
        evaluation_cost = [] if monitor_evaluation_cost else None
        evaluation_accuracy = [] if monitor_evaluation_accuracy else None

        for epoch in range(epochs):
            # Spliting training data to batches: 
            random.shuffle(training_data)
            mini_batches_data = [
                training_data[k : k + mini_batch_size] for k in range(0, n, mini_batch_size)
                ]
            
            for mini_batch_data in mini_batches_data:
                self.SGD_mini_batch(mini_batch_data, n, learningRate, regularization_param)

            print(f"Epoch {epoch} training complete")
            self._monitor(training_cost, training_accuracy, evaluation_cost, evaluation_accuracy, 
                    training_data, evaluation_data, regularization_param, epoch)
            
        return (training_cost, training_accuracy, evaluation_cost, evaluation_accuracy)


    def SGD_mini_batch(self, mini_batch_data, n, learningRate, lmbda):
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

            # For each param, adds the current example’s cost / param gradient to the accumulated cost / param gradient
            del_b = [b + bb for b, bb in zip(del_b, batch_del_b)]
            del_w = [w + bw for w, bw in zip(del_w, batch_del_w)]

        # Apply gradient descent param update:
        m = len(mini_batch_data)
        self.weights = [(1 - learningRate * (lmbda / n)) * w - (learningRate / m) * nw
                        for w, nw in zip(self.weights, del_w)]
        
        self.biases = [b - (learningRate / m) * nb
                       for b, nb in zip(self.biases, del_b)]
        
    def backpropagation(self, x, y):
        # Compute the gradients of the cost function
        """ Return a tuple (del_b, del_w) representing the gradient for the
            cost function C_x.
            del_b and del_w are lists of numpy array objects (one for each layer).
        """
        del_b = [np.zeros(b.shape) for b in self.biases]
        del_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward
        activation = x
        activationsByLayer = [x]
        weightedInputsByLayer = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            weightedInputsByLayer.append(z)
            activation = sigmoid(z)
            activationsByLayer.append(activation)

        # Backward pass
        ## Output layer
        outputLayerActivations = activationsByLayer[-1]
        outputLayerWeightedInputs = weightedInputsByLayer[-1]
        ### Output layer error vector δ^L
        delta = self.costFunction.delta(outputLayerWeightedInputs, outputLayerActivations, y)
        ### Bias gradient ∂C/∂b^L = δ^L
        del_b[-1] = delta
        ### Weight gradient ∂C/∂W^L = δ^L * (a^(L-1))^T
        del_w[-1] = np.dot(delta, activationsByLayer[-2].transpose())

        ## Hidden layers
        for layer in range(2, self.numLayers):
            z = weightedInputsByLayer[-layer]
            sp = sigmoid_prime(z)

            # δ^l = (W^(l+1))^T δ^(l+1) ⊙ σ'(z^l)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp

            # ∂C/∂b^l = δ^l
            del_b[-layer] = delta

            # ∂C/∂W^l = δ^l (a^(l-1))^T
            del_w[-layer] = np.dot(delta, activationsByLayer[-layer - 1].transpose())
        
        return (del_b, del_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


    def _monitor(self, training_cost, training_accuracy, evaluation_cost, evaluation_accuracy, 
                    training_data, evaluation_data, lmbda, epoch):
        if training_cost is not None:
            cost = self._compute_total_cost(training_data, lmbda, False)
            training_cost.append(cost)
            print(f"Epoch {epoch} - Cost on training data: {cost}")

        if training_accuracy is not None:
            accuracy = self._calculate_accuracy(training_data, False)
            training_accuracy.append(accuracy)
            print(f"Epoch {epoch} - Accuracy on training data: {accuracy}")

        if evaluation_cost is not None:
            cost = self._compute_total_cost(evaluation_data, lmbda, True)
            evaluation_cost.append(cost)
            print(f"Epoch {epoch} - Cost on evaluation data: {cost}")

        if evaluation_accuracy is not None:
            accuracy = self._calculate_accuracy(evaluation_data, True)
            evaluation_accuracy.append(accuracy)
            print(f"Epoch {epoch} - Accuracy on evaluation data: {accuracy}")

    
    def _calculate_total_cost(self, data, lmbda, vectorize_results: bool):
        cost = 0.0

        for x, y in data:
            prediction = self.feedforward(x)
            if vectorize_results:
                y = vectorized_result(y)
            cost += self.costFunction.fn(prediction, y) / len(data)
                
        # L2 regularization:
        cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w)**2 for w in self.weights)
        
        return cost

    def _calculate_accuracy(self, data, vectorize_results=False):
        # TO-DO
        return 0
