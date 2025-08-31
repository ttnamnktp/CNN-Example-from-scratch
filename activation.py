from layer import Layer
import numpy as np

class Activation(Layer):
    def __init__(self, activation, activation_prime): # activation, derivative
        self.activation = activation
        self.activation_prime = activation_prime
        
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_gradients, learning_rate):
        return np.multiply(output_gradients, self.activation_prime(self.input))
    
class Tanh(Activation):
    def __init__(self):
        tanh = lambda x : np.tanh(x) # định nghĩa hàm tanh
        tanh_prime = lambda x : 1 - np.tanh(x) ** 2
        super.__init__(tanh, tanh_prime)
        
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1/(1 + np.exp(-x))
        
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1-s)
        
        super().__init__(sigmoid, sigmoid_prime)