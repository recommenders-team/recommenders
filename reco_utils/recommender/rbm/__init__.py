#Default values for the RBM class
import os

#number of hidden units
HIDDEN = 10

#keep probability for deopout regularization
KEEP_PROB = 1

#Initial value of the momentum optimizer
MOMENTUM = 0.9

#standard deviation for initializing the weights from a normal distribution with zero mean
STDV =  0.01
#Learning rate
ALPHA = 0.004
#number of minibatches
MINIBATCH = 100
#number of epochs for the optimization
EPOCHS = 20

#Set default directory 
dir = os.getcwd() #obtain the absolute path
DEFAULTPATH = dir + '/reco_utils/recommender/rbm/'
