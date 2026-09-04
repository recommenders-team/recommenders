# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import abc


class BaseIterator(object):
    """Abstract base iterator class"""

    @abc.abstractmethod
    def parser_one_line(self, line):
        """Abstract method. Parse one string line into feature values.

        Args:
            line (str): A string indicating one instance.
        """
        pass

    @abc.abstractmethod
    def load_data_from_file(self, infile):
        """Abstract method. Read and parse data from a file.

        Args:
            infile (str): Text input file. Each line in this file is an instance.
        """
        pass

    @abc.abstractmethod
    def _convert_data(self, labels, features):
        pass

    @abc.abstractmethod
    def gen_feed_dict(self, data_dict):
        """Abstract method. Construct a dictionary that maps graph elements to values.

        Args:
            data_dict (dict): A dictionary that maps string name to numpy arrays.
        """
        pass
