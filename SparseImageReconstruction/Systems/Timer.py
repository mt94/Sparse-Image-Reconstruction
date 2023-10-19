import time


class Timer(object):
    """
    Utility class from www.huyng.com.
    """

    def __init__(self, verbose=False):
        super(Timer, self).__init__()
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print("elapsed time: {0} ms".format(self.msecs))
