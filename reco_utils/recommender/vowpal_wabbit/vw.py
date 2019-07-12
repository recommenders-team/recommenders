# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
This file provides a wrapper to run Vowpal Wabbit from the command line through python.
It is not recommended to use this approach in production, there are python bindings that can be installed from the
repository or pip or the command line can be used. This is merely to demonstrate vw usage in the example notebooks.
"""

import os
from subprocess import run
from tempfile import TemporaryDirectory
import pandas as pd

from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_PREDICTION_COL,
)


class VW:
    """Vowpal Wabbit Class"""

    def __init__(
        self,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_timestamp=DEFAULT_TIMESTAMP_COL,
        col_prediction=DEFAULT_PREDICTION_COL,
        **kwargs,
    ):
        """Initialize model parameters

        Args:
            col_user (str): user column name
            col_item (str): item column name
            col_rating (str): rating column name
            col_timestamp (str): timestamp column name
            col_prediction (str): prediction column name
        """

        # create temporary files
        self.tempdir = TemporaryDirectory()
        self.train_file = os.path.join(self.tempdir.name, "train.dat")
        self.test_file = os.path.join(self.tempdir.name, "test.dat")
        self.model_file = os.path.join(self.tempdir.name, "vw.model")
        self.prediction_file = os.path.join(self.tempdir.name, "prediction.dat")

        # set DataFrame columns
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_timestamp = col_timestamp
        self.col_prediction = col_prediction

        self.logistic = "logistic" in kwargs.values()
        self.train_cmd = self.parse_train_params(params=kwargs)
        self.test_cmd = self.parse_test_params(params=kwargs)

    @staticmethod
    def to_vw_cmd(params):
        """Convert dictionary of parameters to vw command line.

        Args:
            params (dict): key = parameter, value = value (use True if parameter is just a flag)
        
        Returns:
            list[str]: vw command line parameters as list of strings
        """

        cmd = ["vw"]
        for k, v in params.items():
            if v is False:
                # don't add parameters with a value == False
                continue

            # add the correct hyphen to the parameter
            cmd.append(f"-{k}" if len(k) == 1 else f"--{k}")
            if v is not True:
                # don't add an argument for parameters with value == True
                cmd.append("{}".format(v))

        return cmd

    def parse_train_params(self, params):
        """Parse input hyper-parameters to build vw train commands
        
        Args:
            params (dict): key = parameter, value = value (use True if parameter is just a flag)
        
        Returns:
            list[str]: vw command line parameters as list of strings
        """

        # make a copy of the original hyper parameters
        train_params = params.copy()

        # remove options that are handled internally, not supported, or test only parameters
        invalid = [
            "data",
            "final_regressor",
            "invert_hash",
            "readable_model",
            "t",
            "testonly",
            "i",
            "initial_regressor",
            "link",
        ]

        for option in invalid:
            if option in train_params:
                del train_params[option]

        train_params.update(
            {
                "d": self.train_file,
                "f": self.model_file,
                "quiet": params.get("quiet", True),
            }
        )
        return self.to_vw_cmd(params=train_params)

    def parse_test_params(self, params):
        """Parse input hyper-parameters to build vw test commands
        
        Args:
            params (dict): key = parameter, value = value (use True if parameter is just a flag)
        
        Returns:
            list[str]: vw command line parameters as list of strings
        """

        # make a copy of the original hyper parameters
        test_params = params.copy()

        # remove options that are handled internally, ot supported or train only parameters
        invalid = [
            "data",
            "f",
            "final_regressor",
            "initial_regressor",
            "test_only",
            "invert_hash",
            "readable_model",
            "b",
            "bit_precision",
            "holdout_off",
            "c",
            "cache",
            "k",
            "kill_cache",
            "l",
            "learning_rate",
            "l1",
            "l2",
            "initial_t",
            "power_t",
            "decay_learning_rate",
            "q",
            "quadratic",
            "cubic",
            "i",
            "interactions",
            "rank",
            "lrq",
            "lrqdropout",
            "oaa",
        ]
        for option in invalid:
            if option in test_params:
                del test_params[option]

        test_params.update(
            {
                "d": self.test_file,
                "i": self.model_file,
                "quiet": params.get("quiet", True),
                "p": self.prediction_file,
                "t": True,
            }
        )
        return self.to_vw_cmd(params=test_params)

    def to_vw_file(self, df, train=True):
        """Convert Pandas DataFrame to vw input format file
        
        Args:
            df (pd.DataFrame): input DataFrame
            train (bool): flag for train mode (or test mode if False)
        """

        output = self.train_file if train else self.test_file
        with open(output, "w") as f:
            # extract columns and create a new dataframe
            tmp = df[[self.col_rating, self.col_user, self.col_item]].reset_index()

            if train:
                # we need to reset the rating type to an integer to simplify the vw formatting
                tmp[self.col_rating] = tmp[self.col_rating].astype("int64")

                # convert rating to binary value
                if self.logistic:
                    max_value = tmp[self.col_rating].max()
                    tmp[self.col_rating] = tmp[self.col_rating].apply(
                        lambda x: 2 * round(x / max_value) - 1
                    )
            else:
                tmp[self.col_rating] = ""

            # convert each row to VW input format (https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format)
            # [label] [tag]|[user namespace] [user id feature] |[item namespace] [movie id feature]
            # label is the true rating, tag is a unique id for the example just used to link predictions to truth
            # user and item namespaces separate features to support interaction features through command line options
            for _, row in tmp.iterrows():
                f.write(
                    "{rating} {index}|user {userID} |item {itemID}\n".format(
                        rating=row[self.col_rating],
                        index=row["index"],
                        userID=row[self.col_user],
                        itemID=row[self.col_item],
                    )
                )

    def fit(self, df):
        """Train model
        
        Args:
            df (pd.DataFrame): input training data
        """

        # write dataframe to disk in vw format
        self.to_vw_file(df=df)

        # train model
        run(self.train_cmd, check=True)

    def predict(self, df):
        """Predict results
        
        Args:
            df (pd.DataFrame): input test data
        """

        # write dataframe to disk in vw format
        self.to_vw_file(df=df, train=False)

        # generate predictions
        run(self.test_cmd, check=True)

        # read predictions
        return df.join(
            pd.read_csv(
                self.prediction_file,
                delim_whitespace=True,
                names=[self.col_prediction],
                index_col=1,
            )
        )

    def __del__(self):
        self.tempdir.cleanup()
