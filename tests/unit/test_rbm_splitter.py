# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import pandas as pd
import numpy as np
import pytest
from sklearn.utils import shuffle

from reco_utils.dataset.rbm_splitters import splitter

from reco_utils.common.constants import(
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)


@pytest.fixture(scope="module")
def test_specs():

    return {
        "number_of_items": 50,
        "number_of_users": 20,
        "seed": 123,
        "ratio": 0.6,
        "tolerance": 0.01,
        "fluctuation": 0.03,}

#generate a syntetic dataset
@pytest.fixture(scope="module")
def python_dataset(test_specs):

    """Get Python labels"""

    def random_date_generator(start_date, range_in_days):
        """Helper function to generate random timestamps.

        Reference: https://stackoverflow.com/questions/41006182/generate-random-dates-within-a
        -range-in-numpy

        """

        days_to_add = np.arange(0, range_in_days)
        random_dates = []

        for i in range(range_in_days):

            random_date = np.datetime64(start_date) + np.random.choice(days_to_add)
            random_dates.append(random_date)

        return random_dates

    #fix the the random seed
    np.random.seed(test_specs["seed"])

    #generates the user/item affinity matrix. Ratings are from 1 to 5, with 0s denoting unrated items
    X = np.random.randint(low= 0, high=6, size= ( test_specs["number_of_users"], test_specs["number_of_items"]) )

    #In the main code, input data are passed as pandas dataframe. Below we generate such df from the above matrix
    userids = []

    for i in range(1, test_specs["number_of_users"]+1):
        userids.extend([i]*test_specs["number_of_items"])


    itemids = [i for i in range(1, test_specs["number_of_items"]+1)]*test_specs["number_of_users"]
    ratings = np.reshape(X, -1)

    #create dataframe
    results = pd.DataFrame.from_dict(
                        {
                            DEFAULT_USER_COL: userids,
                            DEFAULT_ITEM_COL: itemids,
                            DEFAULT_RATING_COL: ratings,
                            DEFAULT_TIMESTAMP_COL: random_date_generator(
                                "2018-01-01", test_specs["number_of_users"]*test_specs["number_of_items"] ),
                         }
                    )

    #here we eliminate the missing ratings to obtain a standard form of the df as that of real data.
    results = results[results.rating !=0]

    #to make the df more realistic, we shuffle the rows of the df
    rating = shuffle(results)

    return rating



def test_random_stratified_splitter(test_specs, python_dataset ):
    '''
    Test the random stratified splitter. No time information is used in this case
    '''

    #generate a syntetic dataset
    df_rating = python_dataset

    #initialize the splitter
    header = {
            "col_user": DEFAULT_USER_COL,
            "col_item": DEFAULT_ITEM_COL,
            "col_rating": DEFAULT_RATING_COL,
        }

    #instantiate the splitter
    split = splitter(DF = df_rating, **header)

    #the splitter returns (in order): train and test user/affinity matrices, train and test datafarmes and user/items to matrix maps
    Xtr, Xtst, train_df, test_df, _ = split.stratified_split(ratio= test_specs['ratio'], seed= test_specs['seed'])

    #Tests
    #check that the generated matrices have the correct dimensions
    assert( (Xtr.shape[0] == df_rating.userID.unique().shape[0]) & (Xtr.shape[1] == df_rating.itemID.unique().shape[0]) )

    assert( (Xtst.shape[0] == df_rating.userID.unique().shape[0]) & (Xtst.shape[1] == df_rating.itemID.unique().shape[0]) )

    #Check the Split ratio
    M = Xtr+Xtst #original matrix

    M_rated = np.sum(M !=0, axis=1) #number of total rated items per user
    Xtr_rated = np.sum(Xtr !=0, axis=1) #number of rated items in the train set
    Xtst_rated = np.sum(Xtst !=0, axis=1) #number of rated items in the test set

    #global split: check that the all dataset is split in the correct ratio
    assert( ( Xtr_rated.sum()/(M_rated.sum() ) <= test_specs["ratio"] + test_specs["tolerance"] )
            & (Xtr_rated.sum()/(M_rated.sum() ) >= test_specs["ratio"] - test_specs["tolerance"] )
            )

    assert( ( Xtst_rated.sum()/(M_rated.sum() ) <= (1-test_specs["ratio"]) + test_specs["tolerance"] )
            & (Xtr_rated.sum()/(M_rated.sum() ) >= (1-test_specs["ratio"]) - test_specs["tolerance"] )
            )

    #This implementation of the stratified splitter performs a random split at the single user level. Here we check
    #that also this more stringent condition is verified. Note that user to user fluctuations in the split ratio
    #are stronger than for the entire dataset due to the random nature of the per user splitting.
    #For this reason we allow a slightly bigger tollerance, as specified in the test_specs()

    assert( ( Xtr_rated/M_rated <= test_specs["ratio"] + test_specs["fluctuation"] ).all()
            & ( Xtr_rated/M_rated >= test_specs["ratio"] - test_specs["fluctuation"] ).all()
            )

    assert( ( Xtst_rated/M_rated <= (1-test_specs["ratio"]) + test_specs["fluctuation"] ).all()
            & ( Xtst_rated/M_rated >= (1-test_specs["ratio"]) - test_specs["fluctuation"] ).all()
            )

    #dataframe tests
    #Test if both train/test datasets contain the same user list.

    users_train = train_df[DEFAULT_USER_COL].unique()
    users_test = test_df[DEFAULT_USER_COL].unique()

    assert set(users_train) == set(users_test)
