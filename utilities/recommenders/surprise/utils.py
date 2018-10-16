import numpy as np


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