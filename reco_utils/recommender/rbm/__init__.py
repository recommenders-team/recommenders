#Default values for the RBM class
import os

#number of hidden units
HIDDEN = 10

#keep probability for deopout regularization
KEEP_PROB = 0.7

#Initial value of the momentum optimizer
#MOMENTUM = 0.5

#standard deviation for initializing the weights from a normal distribution with zero mean
STDV =  0.1
#Learning rate
ALPHA = 0.004
#number of minibatches
MINIBATCH = 100
#number of epochs for the optimization
EPOCHS = 20

#Set default directory for saving files
DEFAULTPATH = 'saver'

def _user_item_return_type():
    return float


def _predict_column_type():
    return float

def _predict_column_type():
    return float
