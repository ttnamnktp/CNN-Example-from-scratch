# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from dense import Dense
from convolutional import Convolutional
from activation import Tanh, Sigmoid
from reshape import Reshape
from loss import binary_cross_entrophy_prime, binary_cross_entrophy


def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    # two_index = np.where(y == 2)[0][:limit]
    # three_index = np.where(y == 3)[0][:limit]

    # all_indices = np.hstack((zero_index, one_index, two_index, three_index))
    all_indices = np.hstack((zero_index, one_index))

    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    # y = y.reshape(len(y), 4, 1)

    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# neural network
network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid()
]

epochs = 20
learning_rate = 0.1

# train 
for e in range(epochs):
    error = 0 
    correct_pred = 0
    for x, y in zip(x_train, y_train):
        # forward
        output = x
        for i, layer in enumerate(network):
            # print(output.shape)
            output = layer.forward(output)
            
        # error
        error += binary_cross_entrophy(y, output)          
        
        # accuracy
        correct_pred += 1 if np.argmax(output) == np.argmax(y) else 0
        
        #backward
        grad = binary_cross_entrophy_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
            
    error /= len(x_train)
    accuracy = correct_pred/len(y_train)
    print(f"{e+1}/{epochs}, error={error}, accuracy={accuracy*100:.5f}%")

# Quá trình này train qua một tập dữ liệu lớn    
#test
correct_pred = 0
for x, y in zip(x_test, y_test):
    output=x
    for layer in network:
        output = layer.forward(output)
    if np.argmax(output) == np.argmax(y):
        correct_pred += 1
print(f"\nAccuracy on the test data: {correct_pred/len(y_test)*100:.5f}%")
            
