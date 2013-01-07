import abc
import numpy as np
import Channel.ChannelBlock as chb

class AbstractNoiseGenerator(chb.AbstractChannelBlock):    
    CHANNEL_BLOCK_TYPE = 'NoiseGenerator'     
    
    def __init__(self):        
        super(AbstractNoiseGenerator, self).__init__(AbstractNoiseGenerator.CHANNEL_BLOCK_TYPE)
                    
    @abc.abstractmethod
    def Generate(self, y):
        raise NotImplementedError('No default abstract method implementation')
    
class AbstractAdditiveNoiseGenerator(AbstractNoiseGenerator):
    CHANNEL_BLOCK_TYPE = 'AdditiveNoiseGenerator'
    INPUT_KEY_SIGMA = 'noise_sigma'
    
    def __init__(self):
        super(AbstractAdditiveNoiseGenerator,self).__init__()
        self.channelBlockType = AbstractAdditiveNoiseGenerator.CHANNEL_BLOCK_TYPE
        
class GaussianAdditiveNoiseGenerator(AbstractAdditiveNoiseGenerator):        
    def __init__(self):
        super(GaussianAdditiveNoiseGenerator, self).__init__()
        self.gaussianNoiseSigma = None        
    def SetParameters(self, **kwargs):
        assert AbstractAdditiveNoiseGenerator.INPUT_KEY_SIGMA in kwargs
        self.gaussianNoiseSigma = kwargs[AbstractAdditiveNoiseGenerator.INPUT_KEY_SIGMA]
    def Generate(self, y):
        assert (self.gaussianNoiseSigma is not None) and (self.gaussianNoiseSigma >= 0)
        assert y is not None
        assert len(y.shape) >= 1        
        if self.gaussianNoiseSigma == 0:
            return np.zeros(y.shape)
        else:
            return np.random.standard_normal(y.shape) * self.gaussianNoiseSigma
        
class NoiseGeneratorFactory(object):
    _concreteNoiseGenerator = {
                               'additive_gaussian': GaussianAdditiveNoiseGenerator
                               }
    @staticmethod
    def GetNoiseGenerator(noiseGeneratorDesc):
        if noiseGeneratorDesc not in NoiseGeneratorFactory._concreteNoiseGenerator:
            raise NotImplementedError('NoiseGenerator ' + str(noiseGeneratorDesc) + ' isn\'t implemented' )
        return NoiseGeneratorFactory._concreteNoiseGenerator[noiseGeneratorDesc]()        