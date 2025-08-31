import numpy as np
from layer import Layer
from scipy import signal

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth): # 3 chiều của khối input, chiều kernel, số channel đầu ra
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape =  (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)
        
    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i,j], "valid")
        return self.output
    
    def backward(self, output_gradients, learning_rate):
        kernel_gradients = np.zeros(self.kernels_shape)
        input_gradients = np.zeros(self.input_shape)
        
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernel_gradients[i,j] = signal.correlate2d(self.input[j], output_gradients[i],"valid")
                input_gradients[j] += signal.convolve2d(output_gradients[i], self.kernels[i, j], "full")                
        
        # update        
        self.kernels -= learning_rate * kernel_gradients
        self.biases -= learning_rate * output_gradients
        return input_gradients
        

