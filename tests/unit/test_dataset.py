import os
import sys
import pytest

# TODO: better solution??
root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
)
sys.path.append(root)
from reco_utils.dataset.url_utils import maybe_download


def test_maybe_download():
    # TODO: change this file to the repo license when it is public
    file_url = "https://raw.githubusercontent.com/Microsoft/vscode/master/LICENSE.txt"
    filepath = "license.txt"
    assert not os.path.exists(filepath)
    filepath = maybe_download(file_url, "license.txt", expected_bytes=1110)
    assert os.path.exists(filepath)
    # TODO: download again and test that the file is already there, grab the log??
    os.remove(filepath)
    with pytest.raises(IOError):
        filepath = maybe_download(file_url, "license.txt", expected_bytes=0)
