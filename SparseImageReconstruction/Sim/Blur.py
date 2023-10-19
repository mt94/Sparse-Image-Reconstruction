import abc
from ..Channel import ChannelBlock as chb
import math
import numpy as np


class AbstractBlur(chb.AbstractChannelBlock):
    def __init__(self):
        super(AbstractBlur, self).__init__("Blur")
        self._blurPsf = None
        self._blurShift = None
        self._thetaShape = None

    @property
    def BlurShift(self):
        return self._blurShift

    def RemoveShiftFromBlurredImage(self, anImage):
        if self.BlurShift is None:
            raise UnboundLocalError("BlurShift isn't defined")
        else:
            assert len(anImage.shape) == len(self.BlurShift)
            tmp = anImage
            assert np.all(self.BlurShift > 0)
            for axisInd in range(len(self.BlurShift)):
                tmp = np.roll(
                    tmp, int(-math.floor(self.BlurShift[axisInd])), axis=axisInd
                )
            #            return np.matrix(tmp)
            return tmp

    @property
    def ThetaShape(self):
        return self._thetaShape

    @ThetaShape.setter
    def ThetaShape(self, value):
        assert value is not None
        self._thetaShape = value

    @property
    def BlurPsf(self):
        return self._blurPsf

    @property
    def BlurPsfInThetaFrame(self):
        if self._thetaShape is None:
            raise UnboundLocalError("_thetaShape is None")
        psfInThetaFrame = np.zeros(self._thetaShape)
        if self._blurPsf is None:
            raise UnboundLocalError("_blurPsf is None")
        blurPsfShape = self._blurPsf.shape
        # Only support 2-d or 3-d psfs for now
        if len(blurPsfShape) == 2:
            psfInThetaFrame[: blurPsfShape[0], : blurPsfShape[1]] = self._blurPsf
        elif len(blurPsfShape) == 3:
            psfInThetaFrame[
                : blurPsfShape[0], : blurPsfShape[1], : blurPsfShape[2]
            ] = self._blurPsf
        else:
            raise NotImplementedError(
                "Don't know how to embed psf that isn't 2-d or 3-d"
            )
        return psfInThetaFrame

    @abc.abstractmethod
    def BlurImage(self, theta):
        raise NotImplementedError("No default abstract method implemented")
