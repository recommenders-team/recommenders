"""
SurpriseEvaluation
"""
import os
import numpy as np
from surprise.builtin_datasets import get_dataset_dir, download_builtin_dataset


def surprise_predictions_to_numpy(predictions):
    """Map prediction object to numpy
    Args:
        predictions (list of surprise.Prediction): A list of predictions
            using surprise format.
    Returns:
        Pair of datasets, truth and predictions
    """
    size = len(predictions)
    y_true = np.zeros(size, dtype=np.float16)
    y_pred = np.zeros(size, dtype=np.float16)
    for i, (_, _, true_r, est, _) in enumerate(predictions):
        y_true[i] = true_r
        y_pred[i] = est
    return y_true, y_pred


def maybe_download_builtin_dataset(data_size):
    """COMENT THIS!!!"""
    dataset_path = os.path.join(get_dataset_dir(), data_size)
    if not os.path.isdir(dataset_path):
        download_builtin_dataset(data_size)  # pragma: no cover
    return dataset_path

