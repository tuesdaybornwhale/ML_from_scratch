"""
Date: 30-08-2023
Author: Ashe Vazquez
Desc: Network class for image_classifier.py
"""

#sizes specifies the sizes of each layer. len(sizes) is the amount of layers including the input and output layer.

Class Network():
    """Network class represents a network with the specified size. self.biases is a matrix where self.biases[i][j] is a float representing
    the bias ith neuron of the jth layer. self.weights is a matrix such that self.weights[i][j][k] contains the weights between
    the ith neuron in the jth layer and the kth neuron in the next layer."""
    def __init__(self, sizes):
        self.num_layers = len(sizes)

        # this generates a list of lists of length y, where y is the specified size of the layer of the network we are iterating through.
        self.biases = [np.random.randn(y,1) for y in sizes]

        # the following code generates a list of length num_layers which in itself contains a matrix representing the weights between the 
        # neurons in the ith and the jth network. self.weights[i][j][k] conatins the connections between the kth neuron in the ith layer and
        # the jth neuron in the (i+1)th layer.
        self.weights = [np.random.randn(y,x) for y, x in zip(sizes[1:], sizes[:-1])]

    def feedforward(x, y):
        """This method feeds a training example x, which is a vector of length sizes[0], through the entire network and outputs the
        activations in the final network."""
        