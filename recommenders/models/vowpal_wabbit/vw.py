# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""
Wrapper around the Vowpal Wabbit python bindings with the fit, predict and recommend_k_items
interface used by the other models in this repository. The model parameters are the vw command
line options, passed as keyword arguments, so `VW(q="ui", b=26)` is the same as `vw -q ui -b 26`.
"""

import os
from tempfile import TemporaryDirectory

import pandas as pd

from recommenders.evaluation.python_evaluation import get_top_k_items
from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_PREDICTION_COL,
)

try:
    import vowpalwabbit
except ImportError:
    vowpalwabbit = None


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
            kwargs: vw command line options, use True for options that are flags
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
        self.train_params = self.parse_train_params(params=kwargs)
        self.test_params = self.parse_test_params(params=kwargs)

        # user item pairs and items seen during training, used by recommend_k_items
        self.seen = None
        self.items = None

    @staticmethod
    def to_vw_params(params):
        """Convert dictionary of parameters to a vw command line string.

        Args:
            params (dict): key = parameter, value = value (use True if parameter is just a flag)

        Returns:
            str: vw command line parameters
        """

        parts = []
        for k, v in params.items():
            if v is False:
                # don't add parameters with a value == False
                continue

            # add the correct hyphen to the parameter
            parts.append(f"-{k}" if len(k) == 1 else f"--{k}")
            if v is not True:
                # don't add an argument for parameters with value == True
                parts.append("{}".format(v))

        return " ".join(parts)

    def parse_train_params(self, params):
        """Parse input hyper-parameters to build vw train parameters

        Args:
            params (dict): key = parameter, value = value (use True if parameter is just a flag)

        Returns:
            str: vw command line parameters
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
        return self.to_vw_params(params=train_params)

    def parse_test_params(self, params):
        """Parse input hyper-parameters to build vw test parameters

        Args:
            params (dict): key = parameter, value = value (use True if parameter is just a flag)

        Returns:
            str: vw command line parameters
        """

        # make a copy of the original hyper parameters
        test_params = params.copy()

        # remove options that are handled internally, not supported or train only parameters
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
            "passes",
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
        return self.to_vw_params(params=test_params)

    def to_vw_file(self, df, train=True):
        """Convert Pandas DataFrame to vw input format file

        Args:
            df (pandas.DataFrame): input DataFrame
            train (bool): flag for train mode (or test mode if False)
        """

        output = self.train_file if train else self.test_file

        if train:
            # we need to reset the rating type to an integer to simplify the vw formatting
            rating = df[self.col_rating].astype("int64")

            # convert rating to binary value
            if self.logistic:
                rating = 2 * (rating / rating.max()).round().astype("int64") - 1

            label = rating.astype(str)
        else:
            label = pd.Series("", index=df.index)

        # convert each row to VW input format (https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format)
        # [label] [tag]|[user namespace] [user id feature] |[item namespace] [movie id feature]
        # label is the true rating, tag is the row index just used to link predictions to truth
        # user and item namespaces separate features to support interaction features through command line options
        lines = (
            label
            + " "
            + pd.Series(df.index.astype(str), index=df.index)
            + "|user "
            + df[self.col_user].astype(str)
            + " |item "
            + df[self.col_item].astype(str)
        )
        with open(output, "w") as f:
            f.write("\n".join(lines))
            f.write("\n")

    @staticmethod
    def run(params):
        """Run vw with the given command line parameters

        Args:
            params (str): vw command line parameters
        """

        if vowpalwabbit is None:
            raise ImportError(
                "vowpalwabbit is required, install it with pip install recommenders[experimental]"
            )

        # creating the workspace runs vw over the data file given with -d,
        # finish() then writes the model and prediction files
        vowpalwabbit.Workspace(params).finish()

    def fit(self, df):
        """Train model

        Args:
            df (pandas.DataFrame): input training data
        """

        # write dataframe to disk in vw format
        self.to_vw_file(df=df)

        # train model
        self.run(self.train_params)

        # keep what was seen during training to build recommendations
        self.seen = df[[self.col_user, self.col_item]].drop_duplicates()
        self.items = self.seen[[self.col_item]].drop_duplicates()

    def predict(self, df):
        """Predict results

        Args:
            df (pandas.DataFrame): input test data

        Returns:
            pandas.DataFrame: input data with the prediction column added
        """

        # write dataframe to disk in vw format
        self.to_vw_file(df=df, train=False)

        # generate predictions
        self.run(self.test_params)

        # read predictions
        return df.join(
            pd.read_csv(
                self.prediction_file,
                sep=r"\s+",
                names=[self.col_prediction],
                index_col=1,
            )
        )

    def recommend_k_items(self, test, top_k=10, remove_seen=False):
        """Recommend top K items for all users which are in the test set

        Every user in the test set is scored against every item seen during training.

        Args:
            test (pandas.DataFrame): users to test
            top_k (int): number of top items to recommend
            remove_seen (bool): flag to remove items seen in training from recommendation

        Returns:
            pandas.DataFrame: top k recommendation items for each user
        """

        users = test[[self.col_user]].drop_duplicates()
        candidates = pd.merge(users, self.items, how="cross")

        if remove_seen:
            candidates = pd.merge(
                candidates,
                self.seen,
                on=[self.col_user, self.col_item],
                how="left",
                indicator=True,
            )
            candidates = candidates[candidates["_merge"] == "left_only"].drop(
                columns="_merge"
            )

        scored = self.predict(candidates.reset_index(drop=True))
        top_items = get_top_k_items(
            scored, col_user=self.col_user, col_rating=self.col_prediction, k=top_k
        )
        return top_items[[self.col_user, self.col_item, self.col_prediction]]

    def __del__(self):
        self.tempdir.cleanup()
