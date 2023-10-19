from .MrfmBlur import MrfmBlur
from .SyntheticBlur import SyntheticBlur


class BlurFactory(object):
    _concreteBlur = {"mrfm": MrfmBlur, "synthetic": SyntheticBlur}

    @staticmethod
    def GetBlur(blurDesc, blurType, blurParametersDict=None):
        if blurDesc not in BlurFactory._concreteBlur:
            raise NotImplementedError("Blur " + str(blurDesc) + " isn't implemented")
        return BlurFactory._concreteBlur[blurDesc](blurType, blurParametersDict)
