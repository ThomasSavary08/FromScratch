# Libraries
import numpy as np
from abc import ABC, abstractmethod

# Abstract class for activations function
class activation(ABC):

    # Instanciate an activation function
    def __init__(self):
        super().__init__()

    # Compute activation(x)
    @abstractmethod
    def f(self, x):
        '''
        Compute activation(x)
            Parameters:
                x (numpy.ndarray): input of the activation function.
            Returns:
                res (numpy.ndarray): output of the activation function.
        '''
        pass

    # Compute activation'(x)
    @abstractmethod
    def f_prime(self, x):
        '''
        Compute activation'(x)
            Parameters:
                x (numpy.ndarray): input of the derivate of the activation function.
            Returns:
                res (numpy.ndarray): output of the derivate of the activation function.
        '''
        pass

# Hyperbolic tangent
class tanh(activation):

    def f(self, x):
        return np.tanh(x)
    
    def f_prime(self, x):
        return np.ones_like(x) - np.power(np.tanh(x), 2)

# Sigmoïd
class sigmoid(activation):

    def f(self, x):
        return np.power(np.ones_like(x) + np.exp(-x), -1)

    def f_prime(self, x):
        return np.multiply(self.f(x), np.ones_like(x) - self.f(x))

# Identity
class id(activation):

    def f(self, x):
        return x

    def f_prime(self, x):
        return np.ones_like(x)