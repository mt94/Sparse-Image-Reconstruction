from Gaussian2dBlurWithNoise import Gaussian2dBlurWithNoise
from Mrfm2dBlurWithNoise import Mrfm2dBlurWithNoise

class BlurWithNoiseFactory(object):
    _concreteBlurWithNoise = {
                              'mrfm2d': Mrfm2dBlurWithNoise,
                              'gaussian2d': Gaussian2dBlurWithNoise                              
                              }
    @staticmethod
    def GetBlurWithNoise(blurWithNoiseDesc, simParametersDict):
        if blurWithNoiseDesc not in BlurWithNoiseFactory._concreteBlurWithNoise:
            raise NotImplementedError('BlurWithNoise ' + str(blurWithNoiseDesc) + " isn't implemented")
        if (blurWithNoiseDesc == 'mrfm2d'):
            return Mrfm2dBlurWithNoise(Mrfm2dBlurWithNoise.GetBlurParameterOptimizer(), simParametersDict)
        else:
            return BlurWithNoiseFactory._concreteBlurWithNoise[blurWithNoiseDesc](simParametersDict)
