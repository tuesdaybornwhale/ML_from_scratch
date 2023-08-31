"""
Date: 30-08-2023
Author: Ashe Vazquez
Desc: Network class for image_classifier.py
"""

import random
import numpy as np
#sizes specifies the sizes of each layer. len(sizes) is the amount of layers including the input and output layer.

class Network():
    """Network class represents a network with the specified size. self.biases is a matrix where self.biases[i][j] is a float representing
    the bias ith neuron of the jth layer. self.weights is a matrix such that self.weights[i][j][k] contains the weights between
    the ith neuron in the jth layer and the kth neuron in the next layer."""
    def __init__(self, sizes):
        self.num_layers = len(sizes)

        # this generates a list of lists of length y, where y is the specified size of the layer of the network we are iterating through.
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]

        # the following code generates a list of length num_layers which in itself contains a matrix representing the weights between the 
        # neurons in the ith and the jth network. self.weights[i][j][k] contains the connections between the kth neuron in the ith layer and
        # the jth neuron in the (i+1)th layer.
        self.weights = [np.random.randn(y,x) for y, x in zip(sizes[1:], sizes[:-1])]

    def feedforward(self, a):
        """This method feeds an input x, which is a vector of length sizes[0], through the entire network and outputs the
        activations in the final network."""
        for i in range(self.num_layers):
            # vectorized function returns a vector of activations for the next layer
            a = sigmoid(np.dot(a, self.weights[i]) + self.biases[i]) 
        return a


    def SGD_cycle(self, training_data, batch_size, desired_epochs, learning_rate):
        """Stochastic gradient descent: the network updates itself from batches of training examples for the given amount of cycles (epochs)
        """
        finished_epochs = 0
        while finished_epochs < desired_epochs:
            # we pick a random sample of size batch_size from the training data and update the network from that batch.
            random.shuffle(training_data) 
            # 
            self.update_from_batch(training_data[:batch_size], learning_rate)

            # next, we remove training data we've already used
            training_data = training_data[batch_size:]
            finished_epochs+=1
        

    def update_from_batch(self, batch, learning_rate):
        """Updates network using the input batch,b where atch is a list of tuples (x, y) where x are the training inputs and y is
        the correct output."""
        n = len(batch)
        network_outputs = [self.feedforward(x[0]) for x in batch]
        correct_outputs = [y[1] for y in batch]
        # MSE cost function over multiple training examples. Possibly unnecessary. I don't think we need the cost so much as its derivative
        cost = np.sum([(np.linalg.norm(x-y)**2) for x, y in zip(network_outputs, correct_outputs)])/(2*n) 
        # backprop alg fetches us the partial derivatives with respect to the weights and biases in the network.
        nabla_b, nabla_w = self.backprop(network_outputs, correct_outputs) 

        # updating the networks weights and biases based on the partial derivatives (gradient descent). Not sure if this code actually adds
        # the partial derivatives to the weights and biases the way we want them to - still needs to be tested.S
        self.weights -= nabla_w * learning_rate/n
        self.biases -= nabla_b * learning_rate/n

    def cost_derivative(self, output_activations, correct_activations):
        """returns a vector which stores the derivatives of the cost function with respect to a change in the outputs. the derivative of the
        MSE function with respect to any given input is just output-correct_output by the chain rule."""
        # in theory we would also have to divide by n, the amount of training inputs we started with. In practice, however, we already
        # accounted for that by dividing our nabla_w and nabla_b by n in the update_from_batch mathod, so we won't have to worry about it 
        # again!
        return (output_activations-correct_activations)

    def backprop(self, x, y):
        """calculating the gradient of our cost function using the backpropagation algorithm. x is a """
        nabla_b = np.zeros(np.shape(self.weights))
        nabla_w = np.zeros(np.shape(self.biases))
        activation = x # first layer of 'activations' is the input x.
        activations = [x] # list of activations we'll keep adding to. activations[l] returns a list of the activations in the lth layer.
        z_vectors = [] # list of z vectors which are our inputs to our activations
        # filling matrix which stores activations. The activations are important because we need them to calculate the errors in a given layer.
        for i in range(self.num_layers): 
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            z_vectors.append(z)
            activations.append(sigmoid(z))
        # feed backwards, starting with the last layer which gets treated differently to the other ones
        final_layer = activations[-1]
        final_z = z_vectors[-1]
        error_vector = self.cost_derivative(final_layer, y) * sigmoid_prime(final_z)
        nabla_b[-1] = error_vector
        nabla_w[-1] = np.dot(error_vector, final_layer.transpose)
        # now we use our recursive formula from chapter 2 to find the errors and partial derivatives throughout all the network.
        i = 2
        while i < self.num_layers:
            zs_prev_layer = z_vectors[-i]
            error_vector = np.matmul(self.weights.transpose(), error_vector) * sigmoid_prime(zs_prev_layer)
            nabla_b[-i] = error_vector
            nabla_w[-i] = np.dot(activations[-i-1], error_vector)
            i +=1
        return nabla_b, nabla_w


def sigmoid(z):
    """sigmoid is our activation function. R->R"""
    return 1/(1+np.exp(-z))


def sigmoid_prime(z):
    """derivative of sigmoid"""
    f = sigmoid(z)
    return f*(1-f)