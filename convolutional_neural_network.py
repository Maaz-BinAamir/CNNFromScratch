import numpy as np
from scipy.signal import correlate
from scipy.signal import convolve2d
from keras.datasets import mnist
from keras.utils import to_categorical

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        
    def forwardprop(self, input):
        pass
    
    def backwardprop(self, output_gradient, learning_rate):
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)
        
    def forwardprop(self, input):
        self.input = input
        return np.matmul(self.weights, self.input) + self.biases
    
    def backwardprop(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime
        
    def forwardprop(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backwardprop(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))
    
class ConvolutioalLayer(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernals_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernals = np.random.randn(*self.kernals_shape)
        self.biases = np.random.rand(*self.output_shape)
        
    def forwardprop(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += correlate(self.input[j], self.kernals[i, j], mode="valid")
        
        return self.output
    
    def backwardprop(self, output_gradient, learning_rate):
        kernals_gradient = np.zeros(self.kernals_shape)
        input_gradient = np.zeros(self.input_shape)
        
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernals_gradient[i, j] = correlate(self.input[j], output_gradient[i], mode = "valid")
                input_gradient[j] += convolve2d(output_gradient[i], self.kernals[i, j], mode = "full")
                
        self.kernals -= learning_rate * kernals_gradient
        self.biases -= learning_rate * output_gradient
        
        return input_gradient
        
class ReshapeLayer(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        
    def forwardprop(self, input):
        return np.reshape(input, self.output_shape)
    
    def backwardprop(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)
    
def binary_crossentropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_crossentropy_prime(y_true, y_pred):
    # we don't use np.mean here because we need to return an array backwardprop not a scalar
    return ((1 - y_true) / (1 - y_pred) -  (y_true / y_pred)) / np.size(y_true)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)
        
        super().__init__(sigmoid, sigmoid_prime)
        
def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    
    x, y = x[all_indices], y[all_indices]
    
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32")/255
    
    y = to_categorical(y)
    y = y.reshape(len(y), 2, 1)

    return x, y

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 500)
x_test, y_test = preprocess_data(x_test, y_test, 500)
    
network = [
    ConvolutioalLayer((1, 28, 28), 3, 5),
    Sigmoid(),
    ReshapeLayer((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid()
]

epochs = 20
learning_rate = 0.001

for e in range(epochs):
    error = 0
    for x, y in zip(x_train, y_train):
        #forward prop
        output = x
        for layers in network:
            output = layers.forwardprop(output)
        
        #error
        error += binary_crossentropy(y, output)
            
        #backward prop
        grad = binary_crossentropy_prime(y, output)
        for layers in reversed(network):
            grad = layers.backwardprop(grad, learning_rate)
            
    error /= len(x_train)
    print(f"{e + 1}/{epochs}, error = {error}")
    
correct_predictions = 0

#test
for x, y in zip(x_test, y_test):
    output = x
    for layers in network:
        output = layers.forwardprop(output)
    
    if np.argmax(output) == np.argmax(y):
        correct_predictions += 1
    
    # print(f"pred: {np.argmax(output)}, true = {np.argmax(y)}")
    
# current accuracy: 98.6%
print(f"accuracy: {correct_predictions * 100 / len(y_test)}%")