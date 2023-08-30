"""
Date: 30-08-2023
Author: Ashe Vazquez
Desc: Image classifier for handwritten digits
"""

import random
import numpy as np

def sigmoid(z):
    """sigmoid is our activation function. R->R"""
    return 1/(1+np.exp(-z))

def sigmoid_prime():
    """derivative of sigmoid"""
    