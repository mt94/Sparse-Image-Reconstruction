import numpy as np
from Recon.Gaussian.AbstractEmgaussReconstructor import AbstractEmgaussReconstructor
from Recon.Gaussian.Thresholding import *

class EmgaussLazeEmpiricalMap1Reconstructor(AbstractEmgaussReconstructor):
    def Mstep(self, x):
        aHat = x.size / np.abs(self._thetaN).sum()
        wHat = (self._thetaN != 0).sum() / x.size
        if (wHat <= 0.5):
            t = ThresholdingHybrid()
            return 
        else:
            