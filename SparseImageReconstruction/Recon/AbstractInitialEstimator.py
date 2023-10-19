import abc
import numpy as np


class AbstractInitialEstimator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(AbstractInitialEstimator, self).__init__()

    @abc.abstractmethod
    def GetInitialEstimate(self, y, H):
        raise NotImplementedError("No default abstract method implemented")


class ZeroInitialEstimator(object):
    def __init__(self):
        super(ZeroInitialEstimator, self).__init__()

    def GetInitialEstimate(self, y, H):
        return np.zeros(y.shape)


class HtyInitialEstimator(object):
    def __init__(self, sparsityLevel=1):
        super(HtyInitialEstimator, self).__init__()
        self._sparsityLevel = sparsityLevel

    """ Assume that H is the psf """

    def GetInitialEstimate(self, y, H):
        if len(H.shape) == 2:
            fnFft = np.fft.fft2
            fnFftInverse = np.fft.ifft2
        else:
            fnFft = np.fft.fftn
            fnFftInverse = np.fft.ifftn
        HFft = fnFft(H)
        yFft = fnFft(y)
        # Calculate H'y
        Hty = fnFftInverse(np.multiply(HFft.conj(), yFft)).real
        # Decide if we want a sparse initial estimator. IF yes, sort the values of Hty and
        # retain the largest _sparsityLevel*100 % of values
        if self._sparsityLevel == 1:
            return Hty
        else:
            raise NotImplementedError("Cannot handle _sparsityLevel < 1")


class InitialEstimatorFactory(object):
    _concreteInitialEstimator = {
        "zero": ZeroInitialEstimator,
        "Hty": HtyInitialEstimator,
    }

    @staticmethod
    def GetInitialEstimator(initialEstimatorDesc):
        if (
            initialEstimatorDesc
            not in InitialEstimatorFactory._concreteInitialEstimator
        ):
            raise NotImplementedError(
                "InitialEstimator " + str(initialEstimatorDesc) + " isn't implemented"
            )
        return InitialEstimatorFactory._concreteInitialEstimator[initialEstimatorDesc]()
