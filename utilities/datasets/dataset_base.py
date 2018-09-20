"""
DatasetBase
"""
from abc import ABC, abstractmethod


class DatasetBase(ABC):
    """Dataset Interface"""

    def __init__(self):
        """Interface initializer"""
        self._name = None
        self._reader = None

    @abstractmethod
    def load_dataset(self, data_location=None, **kwargs):
        """Load a dataset to memory in format specified by subclass"""
        pass  # pragma: no cover

    @property
    def name(self):
        """Name of Dataset"""
        return self._name

