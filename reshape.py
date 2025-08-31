import numpy as np
from layer import Layer

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        
    def forward(self, input):
        return np.reshape(input, self.output_shape)
    
    def backward(self, output_gradients, learning_rate):
        return np.reshape(output_gradients, self.input_shape)