# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


def pytest_addoption(parser):
    parser.addoption(
        "--token", action="store", default="", help="Access token of the test data"
    )
