import abc


class AbstractIterationsObserver(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._bRequireFitError = False

    @property
    def RequireFitError(self):
        return self._bRequireFitError

    @abc.abstractmethod
    def CheckTerminateCondition(self, thetaNp1, thetaN, fitErrorN):
        """Check to see if we want to terminate the iterations"""
        raise NotImplementedError("No default abstract method implementation")
