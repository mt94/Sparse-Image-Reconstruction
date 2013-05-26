from GaussianBlurWithNoise import GaussianBlurWithNoise
from MrfmBlurWithNoise import MrfmBlurWithNoise

class BlurWithNoiseFactory(object):
    _concreteBlurWithNoise = {
                              'mrfm': MrfmBlurWithNoise,
                              'gaussian': GaussianBlurWithNoise                              
                              }
    @staticmethod
    def GetBlurWithNoise(blurWithNoiseDesc, simParametersDict):
        if blurWithNoiseDesc not in BlurWithNoiseFactory._concreteBlurWithNoise:
            raise NotImplementedError('BlurWithNoise ' + str(blurWithNoiseDesc) + " isn't implemented")
        if (blurWithNoiseDesc == 'mrfm'):
            return MrfmBlurWithNoise(MrfmBlurWithNoise.GetParameterOptimizer(), simParametersDict)
        else:
            return BlurWithNoiseFactory._concreteBlurWithNoise[blurWithNoiseDesc](simParametersDict)
