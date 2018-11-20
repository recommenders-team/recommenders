'''
Data processing utilities v.0.1 (2018)
==========================================================
Author: Mirco Milletari <mirco.milletari@microsoft.com>

Collection of utitilies for data processing of recommender systems

'''

import numpy as np
import random
import math

from scipy.sparse import coo_matrix
random.seed = 1

#---------------------------------------------------------------------------------------------------------------------

# 1) Generate the ranking matrix (uses scipy's sparse matrix method)

def gen_ranking_matrix(rating):

    '''
    Generate the user/item rating matrix using scipy's sparse matrix method coo_matrix; for referemce see

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html

    This method is 2 order of magnitudes faster than the one implemented with a for loop. The input format is

    coo_matrix((data, (rows, columns)), shape=(rows, columns))

    Arguments:
    rating -- A dataframe containing at least UserID, ItemID, Ratings

    Returns:
    RM -- Rating matrix of dimensions (Nusers, Nitems) in numpy format. Unrated movies are assigned a value of 0.

    '''

    rating = rating.sort_values(by=['userID'])

    #find max user and item index
    Nusers = rating['userID'].max()
    Nitems = rating['MovieId'].max()

    #extract informations from the dataframe as arrays. Note we substract 1 to itm_id and usr_id
    #in order to map to a matrix format

    r_ = rating.Rating.values #ratings
    itm_id =(rating.MovieId-1).values #itm_id serving as columns
    usr_id =(rating.userID-1).values  #usr_id serving as rows

    #check that all 3 vectors have the same dimensions
    assert((usr_id.shape[0]== r_.shape[0]) & (itm_id.shape[0] == r_.shape[0]))

    #generate a sparse matrix representation using scipy's coo_matrix and convert to array format
    RM = coo_matrix((r_, (usr_id, itm_id)), shape= (Nusers, Nitems)).toarray()

    return RM

#2) Generate the ranking Matrix (direct method)

def gen_ranking_matrix_v0(rating):

    '''
    Deprecated. This method is 2 order of magnitude slower than the scipy implementation

    Generate the user/item rating matrix using a direct method.

    Arguments:
    rating -- A dataframe containing at least UserID, MovieID, Ratings

    Returns:
    RM -- Rating matrix of dimensions (Nusers, Nmovies) in numpy format. Unrated movies are assigned a value of 0 and therefore do not
    contribute to the energy.

    Info: it prints the number of movies that have not been rated by anyones.

    '''
    #Find the total number of movies and users in the database
    Nmovies = rating['ItemID'].max()
    Nusers  = rating['UserID'].max()

    #assert(Nusers == len(rating['UserID'].unique()) )

    #if Nmovies != len(rating['MovieID'].unique()):

    #    unrated = Nmovies - len(rating['MovieID'].unique() )

    #    print('There are {0} unrated movies in the dataset'.format(unrated))

    Rating_matrix = []

    for user in range(1,Nusers+1):

        #create a temporary array of 0s to store the ratings. The lenght of the array corresponds to the max number of movies
        movie_id = (np.zeros(Nmovies))

        #select a user
        user_movie = rating[rating['UserID'] == user].sort_values(by= ['ItemID'])

        #populate the movie/rating vector accordigly
        movie_id[user_movie['ItemID']-1 ] = user_movie['Rating']

        Rating_matrix.append(movie_id)

    #convert to a numpy matrix
    RM = np.reshape(np.asarray(Rating_matrix), [Nusers, Nmovies])

    del Nmovies, Nusers, movie_id, Rating_matrix

    return RM


#2-star ratings (Binary)

def binary_rating(A, threshold):

    '''
    Make binary ratings

    This function transforms an original multiple rating, e.g. 1-5 into a binary one, meaning dislike/like. Most schemes
    of this kind use a 0/1 schemes, but here we reserve 0 for the unrated items. All values below the threshold are set
    to 1 and stands for a dislike, while all values greater than the threshold are set to 2 (like)

    Arguments:
    A-- rating matrix
    threshold-- value separating dislike from like

    Returns
    X -- binary version of the Matrix

    '''

    X = A.copy()
    th = threshold

    X[ (np.where((X>0)&(X<=th) )) ] =1
    X[np.where(X>th)]=2

    return X


#scale ratings

def rescale(A, old, new):
    '''
    Rescale data to a new range

    Note: using old=5 and new=2 one obtains the same result as the binary_rating function with threshold =2
    '''

    X=np.ceil((A/old)*new)

    return X


# 2) Train test/split. Note that this is done differently than the usual train/test split.

def train_test_split(X, test_size):

    np.random.seed(1)

    Nusers = X.shape[0]

    #Test set array
    X_tr  = X.copy()
    X_tst = X.copy()

    #find the number of rated movies per user
    rated = np.sum(X_tr !=0, axis=1)

    #for each user, cut down a test_size% for the test set
    tst = (rated*test_size)//100

    for u in range(Nusers):
        #For each user obtain the index of rated movies
        idx_tst = []
        idx = np.asarray(np.where(np.logical_not(X_tr[u,0:] == 0) )).flatten().tolist()

        #extract a random subset of size n from the set of rated movies without repetition
        for i in range(tst[u]):
            sub_el = random.choice(idx)
            idx.remove(sub_el)
            idx_tst.append(sub_el)

        X_tr[u, idx_tst] = 0  #change the selected rated movies to unrated in the train set
        X_tst[u, idx] = 0  #set the movies that appear already in the train set as 0

        assert(np.sum(X_tr[u,:] != 0) + np.sum(X_tst[u,:] !=0) == rated[u])

    del idx, sub_el, idx_tst

    return X_tr , X_tst


def flat_binary(x):

    m = x.shape[0]

    #Flatten the images into a one dimensional array
    X_flat = x.reshape(x.shape[0], -1)

    for u in range(m):
        ex_idx = np.where(X_flat[u] !=0)
        X_flat[u, ex_idx] =1

    return X_flat


#---------------------Random mini batches----------------------------------------------

def random_mini_batches(X, mini_batch_size, seed):
    """
    Creates a list of random minibatches from X

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, ne)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation]

    # Step 2: Partition  Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = mini_batch_X
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m]
        mini_batch = mini_batch_X
        mini_batches.append(mini_batch)

    return mini_batches
