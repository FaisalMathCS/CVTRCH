# example of how I would do it

import numpy as np 
from copy import deepcopy


'''
ACTIVATION FUNCTIONS BEGIN
'''
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def ReLu(x): 
    return np.where(x > 0, x, 0) 

def tanh(x):
    return (np.exp(x) - np.exp(-x) / (np.exp(x) + np.exp(-x))) 

def linear(x): 
    return x


# Softmax is used on the last layer if the target is categorical with multple of classes
def softmax(x): 
    return np.exp(x)/np.sum(np.exp(x)) 

'''
ACTIVATION FUNCTIONS ENDS
'''

'''
LOSS FUNCTION BEGINS
'''

# Cont 
def mse(y, p):
    return np.mean(np.square(y - p)) 

def mae(y, p):
    return np.mean(np.abs(y - p)) 

#Discrete

# USED for binary classes, minimize this function when trying and the problem has two classes
def binary_cross_entropy(y, p):
    return - np.mean((y*np.log(p)+(1-y) * np.log(1-p)))
# multi class loss function
def categorical_cross_entropy(y, p):
    return - np.mean(np.log(p[np.arange(len(y)),y]))

'''
LOSS FUNCTION ENDS
'''



'''
CORE NN functionality BEGINS
'''
'''
This function would take the input and perform all the calucations to get the expected output
and returns the loss (difference between actual and predicted) 
each of which or np arrays, and later they shall be tensors from torch. 
@para
1. inputs: the input vector
2. weights: the weights of the NN, including the bias
3. true output

@returns 
the computed loss
'''
x = np.array() # get the size and the shape later 
y = np.array() # same here 

weights = np.array() # randomly init these 

# following the first chapter this is not generic. 
def feed_forward(inputs, weights, output): 
    pre_hidden = np.dot(inputs, weights[0]) + weights[1] # y = w * x + b 
    activated_pre_hidden = sigmoid(pre_hidden)
    prediction = np.dot(activated_pre_hidden, weights[2]) + weights[3] 
    return mse(prediction, output)


def update_weights(input, weights, output, lr): 
    original_weights = deepcopy(weights)
    temp_weights = deepcopy(weights) 
    updated_weights = deepcopy(weights) 

    original_loss = feed_forward(input, weights, output) 
    for i, layer in enumerate(original_weights): 
        for index, weight in np.ndenumerate(layer): 
            temp_weights[i][index] += 0.0001
            _loss_plus = feed_forward(input, output, temp_weights)
    pass

'''
CORE NN functionality ENDS  
'''