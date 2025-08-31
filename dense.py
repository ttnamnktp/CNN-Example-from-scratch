from layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradients, learning_rate):
        weight_gradients = np.dot(output_gradients, self.input.T)
        self.weights -= learning_rate * weight_gradients
        self.bias -= learning_rate * output_gradients
        return np.dot(self.weights.T, output_gradients)
        
    
        