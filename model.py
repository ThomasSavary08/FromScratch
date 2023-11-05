# Libraries
import numpy as np
from tqdm import tqdm
from loss import BCE, neg_logsoftmax, MSE
from activations import activation, tanh, sigmoid, id

# Layer of the network
class Layer:

    # Instanciate a layer
    def __init__(self, input_dim : int, output_dim : int, act : activation):
        '''
        Instanciate a layer.
            Parameters:
                input_dim (int): dimension of the inputs.
                output_dim (int): dimension of the outputs.
                act (activation): activation function of the layer.
        '''
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.W = np.random.normal(loc = 0., scale = 1./(np.sqrt(input_dim)), size = (output_dim, input_dim))
        self.b = np.zeros(shape = (output_dim, 1))
        self.Z = None
        self.A = None
        self.grad_W = None
        self.grad_b = None
    
    # Forward a batch through the layer
    def forward(self, X):
        '''
        Forward a batch of samples through the layer.
            Parameters:
                X (numpy.ndarray): input of the layer of dimension (p,n).
        '''
        self.Z = (self.W @ X) + np.repeat(self.b, X.shape[1], axis = -1)
        self.A = self.act.f(self.Z)
    
    # Clean gradients of a layer
    def zero_grad(self):
        '''
        Clean the gradients of a layer.
        '''
        self.grad_W = None
        self.grad_b = None
    
    # Update parameters with gradients
    def update_params(self, alpha):
        '''
        Update parameters with classical gradient descent (without momentum or anything else).
            Parameters:
                alpha (float): learning rate of the gradient descent.
        '''
        assert (self.grad_W is not None) and (self.grad_b is not None)
        self.W = self.W - alpha * self.grad_W
        self.b = self.b - alpha * self.grad_b

# Neural network
class NeuralNetwork:

    # Instanciate a neural network
    def __init__(self, input_dim : int, hidden_dims : list, output_dim : int, task : str):
        '''
        Instanciate a neural network.
            Parameters:
                input_dim (int): dimension of the inputs.
                hidden_dims (list): list containing dimensions of hidden layers.
                output_dim (int): dimension of the outputs.
                task (str): string describing the task (should be "classification" or "regression").
        '''
        super().__init__()

        # Dimensions, task, last activation function and loss function
        self.input_dim = input_dim
        self.task = task
        if task == "classification":
            self.output_dim = output_dim
            if output_dim == 1:
                self.last_act = sigmoid()
                self.loss_ = BCE()
            else:
                self.last_act = id()
                self.loss_ = neg_logsoftmax()
        elif task == "regression":
            self.output_dim = 1
            self.last_act = id()
            self.loss_ = MSE()
        else:
            raise Exception("Task should be 'regression' or 'classification'.")
        
        # List of layers
        if len(hidden_dims) == 0:
            self.layers = [Layer(self.input_dim, self.output_dim, self.last_act)]
        elif len(hidden_dims) == 1:
            tanh_ = tanh()
            int_dim = hidden_dims[0]
            self.layers = [Layer(self.input_dim, int_dim, tanh_), Layer(int_dim, self.output_dim, self.last_act)]
        else:
            tanh_ = tanh()
            self.layers = [Layer(self.input_dim, hidden_dims[0], tanh_)]
            for i in range(1, len(hidden_dims)):
                self.layers.append(Layer(hidden_dims[i -1], hidden_dims[i], tanh_))
            self.layers.append(Layer(hidden_dims[-1], self.output_dim, self.last_act))
        
    # Forward a batch through the network
    def forward(self, X : np.ndarray):
        '''
        Forward a batch of samples through the network.
            Parameters:
                X (numpy.ndarray): input of the network of dimension (p,n), typically a batch of samples.
        '''
        self.layers[0].forward(X)
        for i in range(1, len(self.layers)):
            self.layers[i].forward(self.layers[i-1].A)
        return self.layers[-1].A
    
    # Clean gradients of the network
    def zero_grad(self):
        '''
        Clean the gradients of rhe network.
        '''
        for layer in self.layers:
            layer.zero_grad()
    
    # Compute the gradient for each parameter of the network (W and b)
    def backward(self, X : np.ndarray, y : np.ndarray):
        '''
        Compute the gradients of each parameter of the network.
            Parameters:
                X (numpy.ndarray): input of the network of dimension (p,n), typically a batch of samples.
                y (numpy.ndarray): true labels or values of dimension (n,).
        '''
        D, J = self.layers[-1].act.f_prime(self.layers[-1].Z), self.loss_.Jac(self.layers[-1].A, y)
        Delta = np.multiply(D, np.transpose(J))
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i].grad_W = (1./X.shape[1]) * (Delta @ np.transpose(self.layers[i-1].A))
            self.layers[i].grad_b = np.expand_dims(np.mean(Delta, axis = 1), axis = 1)
            Delta = np.transpose(self.layers[i].W) @ Delta
            Delta = np.multiply(self.layers[i-1].act.f_prime(self.layers[i-1].Z), Delta)
        self.layers[0].grad_W = (1./X.shape[1]) * (Delta @ np.transpose(X))
        self.layers[0].grad_b = np.expand_dims(np.mean(Delta, axis = 1), axis = 1)
    
    # Update parameters of the network
    def update_params(self, alpha : float):
        '''
        Update parameters of the network with classical gradient descent (without momentum or anything else).
            Parameters:
                alpha (float): learning rate of the gradient descent.
        '''
        for layer in self.layers:
            layer.update_params(alpha)
    
    # Train the network
    def train(self, nb_epochs : int, batch_size : int, alpha : float, X_train : np.ndarray, y_train : np.ndarray, X_val : np.ndarray = None, y_val : np.ndarray = None):
        '''
        Train the network for a given task and data distribution.
            Parameters:
                nb_epochs (int): number of epochs to do.
                batch_size (int): dimension of the batch.
                alpha (float): learning rate for the gradient descent.
                X_train (numpy.ndarray): training dataset of dimension (n_train, input_dimension).
                y_train (numpy.ndarray): training labels of dimension (n_train,).
                X_val (numpy.ndarray): validation dataset of dimension (n_val, input_dimension).
                y_val (numpy.ndarray): validation labels of dimension (n_val,).
        '''
        # List for training and valisation losses
        list_train = []
        if ((X_val is not None) and (y_val is not None)):
            list_val = []

        # Loop on the number of epochs
        for epoch in range(1, nb_epochs + 1):

            # Monitore the training loss
            running_loss = 0.
            n_loss = 0

            # Shuffle the data
            assert X_train.shape[0] == y_train.shape[0]
            index = np.random.permutation(len(X_train))
            X_epoch, y_epoch = X_train[index], y_train[index]

            # Create lists of mini-batches
            list_batch_X, list_batch_y = [], []
            for i in range(0, X_train.shape[0], batch_size):
                list_batch_X.append(X_epoch[i : i+batch_size, :])
                list_batch_y.append(y_epoch[i : i+batch_size])
            assert len(list_batch_X) == len(list_batch_y)

            # Create a tqdm object to monitore the training
            progress_bar = tqdm(enumerate(list_batch_X), total = len(list_batch_X), desc = f'Epoch {epoch}')

            # Loop on lists of mini-batches
            for batch_ind, X in progress_bar:
                X, y = np.transpose(X), list_batch_y[batch_ind]

                # Clean the gradients
                self.zero_grad()

                # Forward the input and compute the loss
                output = self.forward(X)
                batch_loss = self.loss_.Loss(output, y)
                running_loss += batch_loss * X.shape[1]
                n_loss += X.shape[1]

                # Compute the gradients
                self.backward(X, y)

                # Update parameters
                self.update_params(alpha)

                # Update the description of the progress bar with batch_loss
                progress_bar.set_postfix({'Training loss': running_loss / n_loss})
            
            # Update training list
            list_train.append(running_loss / n_loss)
        
            # Monitore the loss on the validation set
            if ((X_val is not None) and (y_val is not None)):
                output = self.forward(np.transpose(X_val))
                val_loss = self.loss_.Loss(output, y_val)
                list_val.append(val_loss)
        
        # Return list of loss(es)
        if ((X_val is not None) and (y_val is not None)):
            return list_train, list_val
        else:
            return list_train
    
    # Make prediction with the network
    def predict(self, X: np.ndarray):
        '''
        Make prediction with the network.
            Parameters:
                X (np.ndarray): input of the network of dimension (n,p).
            Returns:
                res (np.ndarray): predictions of dimension (n,).
        '''
        output = self.forward(np.transpose(X))
        if (self.task == "regression"):
            return output.squeeze(0)
        else:
            if (self.output_dim > 1):
                return np.argmax(output, axis = 0)
            else:
                return (output.squeeze(0) > 0.5).astype(int)