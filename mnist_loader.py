import pickle
import gzip

import numpy as np

from utils import vectorized_result 

def load_data():
    # returned tuple
    # │
    # ├── training_data tuple
    # │   ├── images ndarray
    # │   └── labels ndarray
    # │
    # ├── validation_data tuple
    # │   ├── images ndarray
    # │   └── labels ndarray
    # │
    # └── test_data tuple
    #     ├── images ndarray
    #     └── labels ndarray

    with gzip.open("data/mnist.pkl.gz", "rb") as pickle_file:
        loaded_object = pickle.load(pickle_file, encoding="latin1")
        training_data, validation_data, test_data = loaded_object

    return (training_data, validation_data, test_data)

def load_data_wrapper():
    # returned tuple
    # │
    # ├── training_data
    # │   └── 50,000 tuples: (image column vector, one-hot label)
    # │
    # ├── validation_data
    # │   └── 10,000 tuples: (image column vector, integer label)
    # │
    # └── test_data
    #     └── 10,000 tuples: (image column vector, integer label)

    tr_d, va_d, te_d = load_data()

    training_inputs = []
    for image in tr_d[0]:
        reshaped_image = np.reshape(image, (784, 1))
        training_inputs.append(reshaped_image)

    training_results = []
    for label in tr_d[1]:
        one_hot_label = vectorized_result(label)
        training_results.append(one_hot_label)

    training_data = list(zip(training_inputs, training_results))

    validation_inputs = []
    for image in va_d[0]:
        reshaped_image = np.reshape(image, (784, 1))
        validation_inputs.append(reshaped_image)

    validation_data = list(zip(validation_inputs, va_d[1]))

    test_inputs = []
    for image in te_d[0]:
        reshaped_image = np.reshape(image, (784, 1))
        test_inputs.append(reshaped_image)

    test_data = list(zip(test_inputs, te_d[1]))

    return training_data, validation_data, test_data
