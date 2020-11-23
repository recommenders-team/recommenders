# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import matplotlib.pyplot as plt
from reco_utils.common.plot import line_graph


def test_line_graph():
    """Naive test to run the function without errors"""
    # Multiple graphs
    line_graph(
        values=[[1, 2, 3], [3, 2, 1]],
        labels=["Train", "Valid"],
        x_guides=[0, 1],
        x_name="Epoch",
        y_name="Accuracy",
        legend_loc="best",
    )
    plt.close()

    # Single graph as a subplot
    line_graph(values=[1, 2, 3], labels="Train", subplot=(1, 1, 1))
    plt.close()

    # Single graph with x values
    line_graph(
        values=[(1, 1), (2, 2), (3, 3)],
        labels="Train",
        x_min_max=(0, 5),
        y_min_max=(0, 5),
        plot_size=(5, 5),
    )
    plt.close()
