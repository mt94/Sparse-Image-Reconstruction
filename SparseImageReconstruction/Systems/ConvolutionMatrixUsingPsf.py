import numpy as np
from ..Systems.AbstractConvolutionMatrix import AbstractConvolutionMatrix


class ConvolutionMatrixUsingPsf(AbstractConvolutionMatrix):
    def __init__(self, psfRepConvMatrix):
        assert isinstance(psfRepConvMatrix, np.ndarray)
        fftFunction = ConvolutionMatrixUsingPsf.GetFftFunction(psfRepConvMatrix)
        self._fnFft = fftFunction["fft"]
        self._fnFftInverse = fftFunction["ifft"]
        self._psfFft = self._fnFft(psfRepConvMatrix)

    @staticmethod
    def GetPsfFft(psfRepConvMatrix):
        assert len(psfRepConvMatrix.shape) >= 2
        if len(psfRepConvMatrix.shape) == 2:
            return np.fft.fft2(psfRepConvMatrix)
        else:
            return np.fft.fftn(psfRepConvMatrix)

    @staticmethod
    def GetFftFunction(psfArg):
        if isinstance(psfArg, np.ndarray):
            psfShape = psfArg.shape
        elif isinstance(psfArg, tuple):
            psfShape = psfArg
        else:
            raise TypeError("Cannot get psf shape from psfArg")
        psfShapeLen = len(psfShape)
        assert psfShapeLen >= 2
        if psfShapeLen == 2:
            return {"fft": np.fft.fft2, "ifft": np.fft.ifft2}
        else:
            return {"fft": np.fft.fftn, "ifft": np.fft.ifftn}

    def CalculateMeanAndL2norm(self):
        singlePoint = np.zeros(self._psfFft.shape)
        np.put(singlePoint, 1, 1)
        singlePsf = self.Multiply(singlePoint)
        convMatrixColumnMean = np.mean(np.reshape(singlePsf, (singlePsf.size, 1)))
        columnWithMeanSubtracted = np.array(
            np.reshape(
                singlePsf - convMatrixColumnMean * np.ones(singlePsf.shape),
                (singlePsf.size, 1),
            )
        )
        convMatrixColumnZeroMeanedL2norm = np.sqrt(
            np.sum(columnWithMeanSubtracted * columnWithMeanSubtracted)
        )
        return {
            "mean": convMatrixColumnMean,
            "l2norm": convMatrixColumnZeroMeanedL2norm,
        }

    """ Implement abstract methods """

    @property
    def PsfShape(self):
        return self._psfFft.shape

    """ Convolution with the psf, so calculate y = Hx = h * x """

    def Multiply(self, x):
        assert isinstance(x, np.ndarray)
        assert (len(x.shape) == len(self._psfFft.shape)) and (
            x.shape == self._psfFft.shape
        )
        return self._fnFftInverse(np.multiply(self._psfFft, self._fnFft(x))).real

    """ Multiply by H', so calculate y = H'x """

    def MultiplyPrime(self, x):
        assert isinstance(x, np.ndarray)
        assert (len(x.shape) == len(self._psfFft.shape)) and (
            x.shape == self._psfFft.shape
        )
        return self._fnFftInverse(np.multiply(self._psfFft.conj(), self._fnFft(x))).real
