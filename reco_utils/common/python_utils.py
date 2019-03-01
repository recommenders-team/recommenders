import numpy as np
import os

def exponential_decay(value, max_val, half_life):
    """Compute decay factor for a given value based on an exponential decay
    Values greater than max_val will be set to 1
    Args:
        value (numeric): value to calculate decay factor
        max_val (numeric): value at which decay factor will be 1
        half_life (numeric): value at which decay factor will be 0.5
    Returns:
        float: decay factor
    """

    return np.minimum(1., np.exp(-np.log(2) * (max_val - value) / half_life))


def jaccard(cooccurrence):
    """Helper method to calculate the Jaccard similarity of a matrix of co-occurrences
    Args:
        cooccurrence (np.array): the symmetric matrix of co-occurrences of items
    Returns:
        np.array: The matrix of Jaccard similarities between any two items
    """

    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)

    with np.errstate(invalid='ignore', divide='ignore'):
        result = cooccurrence / (diag_rows + diag_cols - cooccurrence)

    return np.array(result)


def lift(cooccurrence):
    """Helper method to calculate the Lift of a matrix of co-occurrences
    Args:
        cooccurrence (np.array): the symmetric matrix of co-occurrences of items
    Returns:
        np.array: The matrix of Lifts between any two items
    """

    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)

    with np.errstate(invalid='ignore', divide='ignore'):
        result = cooccurrence / (diag_rows * diag_cols)

    return np.array(result)


def _clean_up(filepath):
    """ Remove cached file. Be careful not to erase anything else. """
    try:
        os.remove(filepath)
    except OSError:
        pass