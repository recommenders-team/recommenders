import warnings
from scipy.io import loadmat
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, isspmatrix_csr
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import normalize
from numba import jit, prange

from IPython import embed
from .GeoIMCutils import length_normalize, reduce_dims


class DataPtr():
    """
    Holds data and its respective indices
    """

    def __init__(self, data, entities):
        assert isspmatrix_csr(data)

        self.data = data
        self.entities = entities
        self.data_indices = None
        self.entity_indices = [None, None]


    def get_data(self):
        if self.data_indices is None:
            return self.data
        return self.data[self.data_indices]


    def get_entity(self, of="row"):
        idx = 0 if of=="row" else 1
        if self.entity_indices[idx] is None:
            return self.entities[idx]
        return self.entities[idx][self.entity_indices[idx]]


class Dataset():
    """
    Base class that holds necessary (minimal) information needed
    """

    def __init__(
            self,
            name,
            features_dim=0,
            normalize=False,
            target_transform=''
    ):
        """Initialize parameters

        Args:
            name (str): Name of the dataset
            features_dim (uint): Dimension of the features. If not 0, PCA is performed
                on the features as the dimensionality reduction technique
            normalize (bool): Normalize the features
            target_transform (str): Transform the target values. Current options are
                'normalize' (Normalize the values), '' (Do nothing), 'binarize' (convert
                the values using a threshold defined per dataset)

        """
        self.name = None
        self.training_data = None
        self.test_data = None
        self.entities = None

        self.features_dim = features_dim
        self.feat_normalize = normalize
        self.target_transform = target_transform


    def load_data(self, path):
        raise NotImplementedError(f"{self.name} should implement it")


    def binarize_rating(self, v):
        return 1 if v >=3 else 0


    def normalize(self):
        if self.feat_normalize:
            for i in range(len(self.entities)):
                if isspmatrix_csr(self.entities[i]):
                    print(f"Normalizing CSR matrix")
                    self.entities[i] = normalize(self.entities[i])
                else:
                    self.entities[i] = length_normalize(self.entities[i])


    def generate_train_test_data(self, data, test_ratio=0.3):
        self.training_data = DataPtr(data, self.entities)
        self.test_data = DataPtr(data, self.entities)

        self.training_data.data_indices, self.test_data.data_indices = train_test_split(
            np.array(range(0, data.shape[0])),
            test_size=test_ratio,
            shuffle=True,
            random_state=0
        )
        self.training_data.entity_indices[0] = self.training_data.data_indices
        self.test_data.entity_indices[0] = self.test_data.data_indices


    def reduce_dims(self):
        if self.features_dim != 0:
            self.entities[0] = reduce_dims(self.entities[0], self.features_dim)
            self.entities[1] = reduce_dims(self.entities[1], self.features_dim)
            print(f"Dimensionality reduced ...")


class ML_100K(Dataset):
    """
    Handles MovieLens-100K
    """

    def __init__(self, **kwargs):
        super().__init__(self.__class__.__name__, **kwargs)


    def df2coo(self, df):
        data = []
        row = list(df['user id']-1)
        col = list(df['item id']-1)
        for idx in range(0, len(df)):
            val = df['rating'].iloc[idx]
            if self.target_transform == 'normalize':
                val = val/7.4162
            elif self.target_transform == 'binarize':
                val = self.binarize_rating(val)
            data += [val]
        # TODO: Get this from `u.info`
        return coo_matrix((data, (row, col)), shape=(943, 1682))


    def _read_from_file(self, path):
        df = pd.read_csv(path, delimiter='\t', names=['user id','item id','rating','timestamp'], encoding="ISO-8859-1")
        df.drop(['timestamp'], axis=1, inplace=True)
        return self.df2coo(df)


    def load_data(self, path, e1_path, e2_path):
        """ Load dataset

        Args:
            path (str): Path to the directory containing ML100K dataset
            e1_path (str): Path to the file containing row (user) features of ML100K dataset
            e2_path (str): Path to the file containing col (movie) features of ML100K dataset
        """
        self.entities = [self._load_features(e1_path, "userFeatures"), self._load_features(e2_path, "itemFeatures")]
        self.normalize()
        self.reduce_dims()
        self.training_data = DataPtr(self._read_from_file(f"{path}/u1.base").tocsr(), self.entities)
        self.test_data = DataPtr(self._read_from_file(f"{path}/u1.test").tocsr(), self.entities)


    def _load_features(self, path, key):
        data = loadmat(path)
        return data[key].toarray()
