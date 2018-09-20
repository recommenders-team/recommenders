"""
SurpriseDataset
"""
from surprise import Reader, Dataset

from utilities.common.surprise_utils import maybe_download_builtin_dataset
from utilities.datasets.dataset_base import DatasetBase


class SurpriseDataset(DatasetBase):
    """SurpriseDataset Interface"""

    def load_dataset(self, data_location=None, **kwargs):
        """Interface"""
        pass

    def __init__(self, schema='user item rating', rating_scale=(1, 5),
                 skip_lines=1):
        """Initializer of CSV dataset"""
        super().__init__()
        self.skip_lines = skip_lines
        self.rating_scale = rating_scale
        self.schema = schema


class SurpriseBuiltInDataset(SurpriseDataset):
    """Built in datasets in Surprise. Options:
    - Movielens: https://grouplens.org/datasets/movielens/
    - Jester: http://eigentaste.berkeley.edu/dataset/
    Reference: https://github.com/NicolasHug/Surprise/blob/master/surprise/builtin_datasets.py
    """

    def __init__(self, builtin_name, **kwargs):
        """Initializer Movielens dataset
        Args:
            builtin_name (str): Name of the built-in suprise dataset.
        """
        super().__init__(**kwargs)
        available_datasets = ['ml-100k', 'ml-1m', 'jester']
        if builtin_name not in available_datasets:
            raise ValueError(
                "Dataset name not among the available options {}".format(available_datasets))
        self.builtin_name = builtin_name
        self._name = 'surprise_' + builtin_name

    def load_dataset(self, data_location=None, **kwargs):
        """Get a built-in surprise dataset
        Args:
            data_location (str): Location of the data.
        Returns:
            data (surprise.Dataset): Dataset in surprise format.
        """
        # pylint: disable=fixme
        # TODO: Allow for custom data_location
        maybe_download_builtin_dataset(self.builtin_name)
        data = Dataset.load_builtin(self.builtin_name)
        return data


class SurpriseTextDataset(SurpriseDataset):
    """Generic text file dataset
    Creates a dataset in Surprise format using a generic text file.
    """

    # pylint: disable=fixme
    # TODO: Change the arguments to something more generic, i.e,
    # schema=['user', 'item', 'rating'] instead of the current input.
    # pylint: disable=fixme
    # TODO: create common init arguments for all datasets classes

    def __init__(self, schema='user item rating', sep=',', rating_scale=(1, 5),
                 skip_lines=1, **kwargs):
        """Class initializer"""
        super().__init__(**kwargs)
        self._name = 'surprise_textfile'
        self._reader = Reader(sep=sep,
                              line_format=schema,
                              rating_scale=rating_scale,
                              skip_lines=skip_lines)

    def load_dataset(self, data_location=None, **kwargs):
        """Get a built-in surprise dataset
        Args:
            data_location (str): Location of the data.
        Returns:
            data (surprise.Dataset): Dataset in surprise format.
        """
        data = Dataset.load_from_file(data_location, self._reader)
        return data


class SurpriseCSVDataset(SurpriseTextDataset):
    """CSVDataset
    Creates a dataset in Surprise format using a csv file.
    """

    # pylint: disable=fixme
    # TODO: Change the arguments to something more generic, i.e,
    # schema=['user', 'item', 'rating'] instead of the current input.
    # pylint: disable=fixme
    # TODO: create common init arguments for all datasets classes

    def __init__(self, schema='user item rating', rating_scale=(1, 5),
                 skip_lines=1, **kwargs):
        """Initializer of CSV dataset"""
        super().__init__(schema=schema,
                         sep=',',
                         rating_scale=rating_scale,
                         skip_lines=skip_lines,
                         **kwargs)
        self._name = 'surprise_csv'


class SurprisePandasDataset(SurpriseDataset):
    """SurprisePandasDataset
    Creates a dataset in Surprise format using a pandas dataframe.
    """

    # pylint: disable=fixme
    # TODO: Change the arguments to something more generic, i.e,
    # schema=['user', 'item', 'rating'] instead of the current input.
    # pylint: disable=fixme
    # TODO: create common init arguments for all datasets classes

    def __init__(self, schema=None, rating_scale=(1, 5),
                 skip_lines=1):
        """Initializer of Pandas dataset"""
        super().__init__(schema=schema,
                         rating_scale=rating_scale,
                         skip_lines=skip_lines)
        if schema is None:
            schema = ['UserId', 'MovieId', 'Rating']
        self._name = 'surprise_pandas'
        self.rating_scale = rating_scale
        self.schema = schema

    def load_dataset(self, data_location=None, dataframe=None, **kwargs):
        """load Pandas DataFrame
        Args:
            dataframe: (pd.DataFrame): Pandas dataframe
            data_location (str): interface param
        Returns:
            data (surprise.Dataset): Dataset in surprise format.
        """
        reader = Reader(rating_scale=self.rating_scale)
        return Dataset.load_from_df(dataframe[self.schema], reader)

