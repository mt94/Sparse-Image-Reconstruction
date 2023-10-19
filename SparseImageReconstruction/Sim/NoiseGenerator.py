import abc
import numpy as np
from ..Channel import ChannelBlock as chb


class AbstractNoiseGenerator(chb.AbstractChannelBlock):
    CHANNEL_BLOCK_TYPE = "NoiseGenerator"

    def __init__(self):
        super(AbstractNoiseGenerator, self).__init__(
            AbstractNoiseGenerator.CHANNEL_BLOCK_TYPE
        )

    @abc.abstractmethod
    def Generate(self, y):
        raise NotImplementedError("No default abstract method implementation")


class AbstractAdditiveNoiseGenerator(AbstractNoiseGenerator):
    CHANNEL_BLOCK_TYPE = "AdditiveNoiseGenerator"
    INPUT_KEY_SIGMA = "noise_sigma"
    INPUT_KEY_SNRDB = "snrdb"

    def __init__(self):
        super(AbstractAdditiveNoiseGenerator, self).__init__()
        self.channelBlockType = AbstractAdditiveNoiseGenerator.CHANNEL_BLOCK_TYPE


class GaussianAdditiveNoiseGenerator(AbstractAdditiveNoiseGenerator):
    def __init__(self):
        super(GaussianAdditiveNoiseGenerator, self).__init__()
        self.gaussianNoiseSigma = None
        self.snrDb = None

    def SetParameters(self, **kwargs):
        assert (AbstractAdditiveNoiseGenerator.INPUT_KEY_SIGMA in kwargs) or (
            AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB in kwargs
        )
        if AbstractAdditiveNoiseGenerator.INPUT_KEY_SIGMA in kwargs:
            self.gaussianNoiseSigma = kwargs[
                AbstractAdditiveNoiseGenerator.INPUT_KEY_SIGMA
            ]
        if AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB in kwargs:
            self.snrDb = kwargs[AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB]

    def Generate(self, y):
        assert y is not None
        assert len(y.shape) >= 1

        if self.snrDb is not None:
            yFlat = np.reshape(y, (y.size,))
            yFlatSquaredSum = (yFlat * yFlat).sum()
            if self.gaussianNoiseSigma is None:
                # If the noise sigma isn't defined, use the SNR to figure out what it should be
                snr = np.power(10, float(self.snrDb) / float(10))
                self.gaussianNoiseSigma = np.sqrt(
                    float(yFlatSquaredSum) / float(y.size) / snr
                )
            else:
                # Set the snrDb field. This already may be correctly set, but that doesn't matter
                self.snrDb = 10 * np.log10(
                    float(yFlatSquaredSum)
                    / float(y.size)
                    / np.square(self.gaussianNoiseSigma)
                )

        assert (self.gaussianNoiseSigma is not None) and (self.gaussianNoiseSigma >= 0)

        if self.gaussianNoiseSigma == 0:
            return np.zeros(y.shape)
        else:
            return np.random.standard_normal(y.shape) * self.gaussianNoiseSigma


class NoiseGeneratorFactory(object):
    _concreteNoiseGenerator = {"additive_gaussian": GaussianAdditiveNoiseGenerator}

    @staticmethod
    def GetNoiseGenerator(noiseGeneratorDesc):
        if noiseGeneratorDesc not in NoiseGeneratorFactory._concreteNoiseGenerator:
            raise NotImplementedError(
                "NoiseGenerator " + str(noiseGeneratorDesc) + " isn't implemented"
            )
        return NoiseGeneratorFactory._concreteNoiseGenerator[noiseGeneratorDesc]()
