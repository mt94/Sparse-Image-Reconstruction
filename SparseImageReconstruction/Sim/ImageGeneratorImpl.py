import abc
import numpy as np
from ..Sim.AbstractImageGenerator import AbstractImageGenerator

# Input keys
INPUT_KEY_UNIFORMRV_RANGE = "uniformrv_range"
INPUT_KEY_DISCRETE_VALUES = "discrete_values"


class AbstractSparseImageGenerator(AbstractImageGenerator):
    def __init__(self):
        super(AbstractSparseImageGenerator, self).__init__()

    @abc.abstractmethod
    def SampleFromDistribution(self):
        raise NotImplementedError("No default abstract method implementation")

    def Generate(self):
        assert self.imageShape is not None
        img = np.zeros(self.imageShape)
        # Init

        if len(self.imageShape) == 2:
            assert self.borderWidth >= 0
            validInterval = np.array(
                [
                    [self.borderWidth, self.imageShape[0] - self.borderWidth],
                    [self.borderWidth, self.imageShape[1] - self.borderWidth],
                ]
            )
            # SANITY
            if (validInterval[0, 1] < validInterval[0, 0]) or (
                validInterval[1, 1] < validInterval[1, 0]
            ):
                raise ValueError("border width isn't compatible with image shape")
            fnPickLoc = lambda: (
                np.random.randint(validInterval[0, 0], validInterval[0, 1]),
                np.random.randint(validInterval[1, 0], validInterval[1, 1]),
            )
        elif len(self.imageShape) == 3:
            assert len(self.borderWidth) == 3
            # The valid interval definition is the same as in the 2d case for the x and y dimensions.
            # However, for the z dimension, only leave room at the high end of the interval. This
            # peculiarity is adapted to the MRFM psf calculated, where only the tip is taken.
            validInterval = np.array(
                [
                    [self.borderWidth[0], self.imageShape[0] - self.borderWidth[0]],
                    [self.borderWidth[1], self.imageShape[1] - self.borderWidth[1]],
                    [0, self.imageShape[2] - self.borderWidth[2]],
                ]
            )
            # SANITY
            if (
                (validInterval[0, 1] < validInterval[0, 0])
                or (validInterval[1, 1] < validInterval[1, 0])
                or (validInterval[2, 1] < validInterval[2, 0])
            ):
                raise ValueError("border width isn't compatible with image shape")
            fnPickLoc = lambda: (
                np.random.randint(validInterval[0, 0], validInterval[0, 1]),
                np.random.randint(validInterval[1, 0], validInterval[1, 1]),
                np.random.randint(validInterval[2, 0], validInterval[2, 1]),
            )
        else:
            raise NotImplementedError(
                "Cannot handle |imageShape|={0}".format(len(self.imageShape))
            )

        # This is hugely inefficient if the image isn't at all sparse
        numOnes = 0
        while numOnes < self.numNonzero:
            randomLoc = fnPickLoc()
            if img[randomLoc] == 0:
                img[randomLoc] = self.SampleFromDistribution()
                numOnes += 1

        return img


"""
Concrete image generators
"""


class SparseBinaryImageGenerator(AbstractSparseImageGenerator):
    def __init__(self):
        super(SparseBinaryImageGenerator, self).__init__()

    # Abstract method override
    def SampleFromDistribution(self):
        return 1


class SparseDiscreteImageGenerator(AbstractSparseImageGenerator):
    # Generalizes SparseBinaryImageGenerator
    def __init__(self):
        super(SparseDiscreteImageGenerator, self).__init__()
        self._discreteValues = None

    def SetParameters(self, **kwargs):
        assert INPUT_KEY_DISCRETE_VALUES in kwargs
        self._discreteValues = kwargs[INPUT_KEY_DISCRETE_VALUES]
        assert len(self._discreteValues) >= 1
        # Call the base class SetParameters method
        super(SparseDiscreteImageGenerator, self).SetParameters(**kwargs)

    # Abstract method override
    def SampleFromDistribution(self):
        return self._discreteValues[np.random.randint(0, len(self._discreteValues))]


class SparseUniformImageGenerator(AbstractSparseImageGenerator):
    def __init__(self):
        super(SparseUniformImageGenerator, self).__init__()
        self._uniformRvRange = None

    def SetParameters(self, **kwargs):
        assert INPUT_KEY_UNIFORMRV_RANGE in kwargs
        self._uniformRvRange = kwargs[INPUT_KEY_UNIFORMRV_RANGE]
        # Call the base class SetParameters method
        super(SparseUniformImageGenerator, self).SetParameters(**kwargs)

    # Abstract method override
    def SampleFromDistribution(self):
        return self._uniformRvRange[0] + np.random.rand() * (
            self._uniformRvRange[1] - self._uniformRvRange[0]
        )
