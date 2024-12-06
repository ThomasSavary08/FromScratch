# Libraries
import numpy as np
from abc import ABC, abstractmethod

# Abstract class for loss functions
class loss(ABC):

    # Instanciate a loss function
    def __init__(self):
        super().__init__()

    # Compute Loss(x,y)
    @abstractmethod
    def Loss(self, x, y):
        '''
        Compute the mean of the loss on the batch.
            Parameters:
                x (numpy.ndarray): output of the network (predictions).
                y (numpy.ndarray): true labels.
            Returns:
                res (numpy.float64): average of the loss function on the batch.
        '''
        pass

    # Compute Jac(L)(x,y)
    @abstractmethod
    def Jac(self, x, y):
        '''
        Compute the jacobian of the loss function for each sample.
            Parameters:
                x (numpy.ndarray): output of the network (predictions).
                y (numpy.ndarray): true labels.
            Returns:
                res (numpy.ndarray): Jacobians for each sample of the batch.
        '''
        pass

# Binary Cross-Entropy
class BCE(loss):

    def Loss(self, x, y):
        res = np.zeros(x.shape[1])
        for i in range(res.shape[0]):
            res[i] = -y[i]*np.log(x[0,i]) - (1. - y[i])*np.log(1. - x[0,i])
        return np.mean(res)
    
    def Jac(self, x, y):
        res = np.zeros((x.shape[1], 1))
        for i in range(res.shape[0]):
            res[i,0] = (x[0,i] - y[i]) / (x[0,i] * (1. - x[0,i]))
        return res

# Cross-Entropy
class neg_logsoftmax(loss):

    def Loss(self, x, y):
        exp = np.exp(x)
        partition = np.sum(exp, axis = 0)
        res = exp[y, np.arange(y.shape[0])]
        res = np.multiply(res, np.power(partition, -1))
        res = -np.log(res)
        return np.mean(res)
    
    def Jac(self, x, y):
        exp = np.exp(np.transpose(x))
        partition = np.repeat(np.expand_dims(np.sum(exp, axis = 1), axis = 1), x.shape[0], axis = 1)
        res = np.multiply(exp, np.power(partition, -1))
        ones = np.zeros((x.shape[1], x.shape[0]))
        ones[np.arange(x.shape[1]), y] = 1.
        res = res - ones
        return res

# MSE
class MSE(loss):

    def Loss(self, x, y):
        res = np.power(x.squeeze(0) - y, 2)
        return np.mean(res)
    
    def Jac(self, x, y):
        res = 2 * (x.squeeze(0) - y)
        return np.reshape(res, (x.shape[1], 1))
