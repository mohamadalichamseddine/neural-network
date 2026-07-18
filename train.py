from Network import Network
import mnist_loader
from persistence import save_network

if __name__ == "__main__":
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    neural_net = Network([784, 30, 10])
    neural_net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
    save_network(neural_net, "models/model.json")