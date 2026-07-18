import json
import numpy as np

from Network import Network

def save_network(net, filename):
    with open(filename, "w") as f:
        json.dump({
            "layerSizes": net.layerSizes,
            "weights": [w.tolist() for w in net.weights],
            "biases": [b.tolist() for b in net.biases],
        }, f)

def load_network(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    net = Network(data["layerSizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net
