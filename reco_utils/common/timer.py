# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from timeit import default_timer
from datetime import timedelta


class Timer(object):
    """Timer class.
    Original code: https://github.com/miguelgfierro/codebase
    
    Examples:
        >>> big_num = 1000
        >>> t = Timer()
        >>> t.start()
        >>> r = 0
        >>> a = [r+i for i in range(big_num)]
        >>> t.stop()
        >>> t.interval < 1
        True
        >>> r = 0
        >>> with Timer() as t:
        ...   a = [r+i for i in range(big_num)]
        >>> t.interval < 1
        True
        >>> "Time elapsed {}".format(t) #doctest: +ELLIPSIS
        'Time elapsed 0:00:...'
    """

    def __init__(self):
        self._timer = default_timer
        self.interval = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __str__(self):
        return str(timedelta(seconds=self.interval))

    def start(self):
        """Start the timer."""
        self.init = self._timer()

    def stop(self):
        """Stop the timer. Calculate the interval in seconds."""
        self.end = self._timer()
        try:
            self.interval = self.end - self.init
        except AttributeError:
            raise ValueError(
                "Timer has not been initialized: use start() or the contextual form with Timer() as t:"
            )
