# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import urllib.request
import csv
import codecs


def _csv_reader_url(url, delimiter=",", encoding="utf-8"):
    ftpstream = urllib.request.urlopen(url)
    csvfile = csv.reader(codecs.iterdecode(ftpstream, encoding), delimiter=delimiter)
    return csvfile


def load_affinity(file):
    """Loads user affinities from test dataset"""
    reader = _csv_reader_url(file)
    items = next(reader)[1:]
    affinities = np.array(next(reader)[1:])
    return affinities, items


def load_userpred(file, k=10):
    """Loads test predicted items and their SAR scores"""
    reader = _csv_reader_url(file)
    next(reader)
    values = next(reader)
    items = values[1 : (k + 1)]
    scores = np.array([float(x) for x in values[(k + 1) :]])
    return items, scores


def read_matrix(file, row_map=None, col_map=None):
    """read in test matrix and hash it"""
    reader = _csv_reader_url(file)

    # skip the header
    col_ids = next(reader)[1:]
    row_ids = []
    rows = []
    for row in reader:
        rows += [row[1:]]
        row_ids += [row[0]]
    array = np.array(rows)

    # now map the rows and columns to the right values
    if row_map is not None and col_map is not None:
        row_index = [row_map[x] for x in row_ids]
        col_index = [col_map[x] for x in col_ids]
        array = array[row_index, :]
        array = array[:, col_index]
    return array, row_ids, col_ids
