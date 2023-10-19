from .Gaussian2dBlurWithNoise import Gaussian2dBlurWithNoise
from .Mrfm2dBlurWithNoise import Mrfm2dBlurWithNoise
from .Mrfm3dBlurWithNoise import Mrfm3dBlurWithNoise


class BlurWithNoiseFactory(object):
    _concreteBlurWithNoise = {
        "mrfm2d": (Mrfm2dBlurWithNoise, Mrfm2dBlurWithNoise.GetBlurParameterOptimizer),
        "mrfm3d": (Mrfm3dBlurWithNoise, Mrfm3dBlurWithNoise.GetBlurParameterOptimizer),
        "gaussian2d": (Gaussian2dBlurWithNoise, None),
    }

    @staticmethod
    def GetBlurWithNoise(blurWithNoiseDesc, simParametersDict):
        if blurWithNoiseDesc not in BlurWithNoiseFactory._concreteBlurWithNoise:
            raise NotImplementedError(
                "BlurWithNoise " + str(blurWithNoiseDesc) + " isn't implemented"
            )

        clsConstructor, clsInitializer = BlurWithNoiseFactory._concreteBlurWithNoise[
            blurWithNoiseDesc
        ]

        if clsInitializer is not None:
            return clsConstructor(clsInitializer(), simParametersDict)
        else:
            return clsConstructor(simParametersDict)
