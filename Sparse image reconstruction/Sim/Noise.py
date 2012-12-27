import abc
import numpy as np

class AbstractNoiseGenerator(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def Generate(self, y):
        raise NotImplementedError('No default abstract method implementation')
    
class GaussianNoiseGenerator(object):
    INPUT_KEY_SIGMA = 'noise_sigma'
    def __init__(self):
        self.gnSigma = None
    def SetParameters(self, **kwargs):
        assert GaussianNoiseGenerator.INPUT_KEY_SIGMA in kwargs
        self.gnSigma = kwargs[GaussianNoiseGenerator.INPUT_KEY_SIGMA]
    def Generate(self, y):
        assert self.gnSigma is not None
        return np.random.randn(y.shape) * self.gnSigma
        
class NoiseGeneratorFactory:
    _concreteNoiseGenerator = {
                               'gaussian': GaussianNoiseGenerator
                               }
    @staticmethod
    def GetNoiseGenerator(noiseGeneratorDesc):
        if noiseGeneratorDesc not in NoiseGeneratorFactory._concreteNoiseGenerator:
            raise NotImplementedError('NoiseGenerator ' + str(noiseGeneratorDesc) + ' isn\'t implemented' )
        return NoiseGeneratorFactory._concreteNoiseGenerator[noiseGeneratorDesc]()        