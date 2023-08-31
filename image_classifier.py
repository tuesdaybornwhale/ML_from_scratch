"""
Date: 30-08-2023
Author: Ashe Vazquez
Desc: Image classifier for handwritten digits.
"""

import random
import numpy as np
from network import *


def main():
    """main executable code."""
    # the size of the first layer (input layer) and last (output) layer are determined by the type of data we're receiving (784 greyscale 
    # pixels) and the type of data we want to output (we can only predict among ten digits). Our only degrees of freedom are the amount of
    # hidden layers and the size of the layer(s) 
    sizes = [784, 30, 10]
    net = Network(sizes)

