import os
import pytest
import random
from utilities import split_files

def create_temp_file(name, lines):
    with open(name, 'w') as out:
        for i in range(lines):
            out.write('%d\n' % i)

def count_lines(name):
    i = -1
    with open(name) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

@pytest.mark.parametrize("split_count,ratios,expected_line_counts", [
    (2, [0.8, 0.2], [819, 181]),
    (2, [0.8], [819, 181]),
    (3, [0.3,0.3], [307, 307, 386])
])
def test_split_files(tmpdir, split_count, ratios, expected_line_counts):
    os.chdir(tmpdir)

    # make it reproducible
    random.seed(123)

    # create input data
    create_temp_file('input.txt', 1000)

    assert 1000 == sum(expected_line_counts)

    output_files = ['split_%d.txt' % i for i in range(split_count)]

    # execute actual function
    split_files('input.txt', output_files, ratios)

    # compare output
    for fname, expected_cnt in zip(output_files, expected_line_counts):
        actual_cnt = count_lines(fname)

        # print("Counting lines in %s. Line counts %d vs %d" % (fname, actual_cnt, expected_cnt))

        assert actual_cnt == expected_cnt

def test_split_files_max_row(tmpdir):
    create_temp_file('input.txt', 100)

    split_files('input.txt', ['output.txt'], [1], 50)

    assert count_lines('output.txt') == 50