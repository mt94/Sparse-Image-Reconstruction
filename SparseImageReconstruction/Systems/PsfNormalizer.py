import abc
import numpy as np

from ..Channel import ChannelBlock as chb
from ..Systems.ConvolutionMatrixUsingPsf import ConvolutionMatrixUsingPsf


class AbstractPsfNormalizer(chb.AbstractChannelBlock):
    def __init__(self, channelBlockType):
        super(AbstractPsfNormalizer, self).__init__(channelBlockType)
        self.psfScalar = None

    @abc.abstractmethod
    def NormalizePsf(self, psfRepH):
        pass

    def NormalizeTheta(self, theta):
        if self.psfScalar is None:
            raise UnboundLocalError("Don't know how to normalize theta yet")
        else:
            assert self.psfScalar > 0
            return theta / self.psfScalar


""" Normalize the l_2 matrix norm of the psf convolution matrix """


class PsfMatrixNormNormalizer(AbstractPsfNormalizer):
    CHANNEL_BLOCK_TYPE = "PsfMatrixNormNormalizer"

    def __init__(self, psfMatrixNormL2Desired):
        super(PsfMatrixNormNormalizer, self).__init__(
            PsfMatrixNormNormalizer.CHANNEL_BLOCK_TYPE
        )
        assert psfMatrixNormL2Desired > 0
        self.psfColumnNormL2Desired = psfMatrixNormL2Desired  # Refers to the matrix l_2 norm of the convolution matrix
        self.psfColumnNormL2 = None

    """ Abstract method implementation. Returns a normalized psf. """

    def NormalizePsf(self, psfRepH):
        psfFft = ConvolutionMatrixUsingPsf.GetPsfFft(psfRepH)
        self.psfColumnNormL2 = np.max(np.reshape(np.absolute(psfFft), psfFft.size))
        self.psfScalar = self.psfColumnNormL2Desired / self.psfColumnNormL2
        return psfRepH * self.psfScalar

    """ Compute the spectral radius of G(H) := H*H' """

    def GetSpectralRadiusGramMatrixRowsH(self):
        if self.psfColumnNormL2 is None:
            raise UnboundLocalError("Cannot calculate the spectral radius of H*H'")
        else:
            assert self.psfColumnNormL2 > 0
            return self.psfColumnNormL2 * self.psfColumnNormL2


""" Normalize the l_2 column norms of the psf convolution matrix """


class PsfColumnNormNormalizer(AbstractPsfNormalizer):
    CHANNEL_BLOCK_TYPE = "PsfColumnNormNormalizer"

    def __init__(self, psfColumnNormL2Desired):
        super(PsfColumnNormNormalizer, self).__init__(
            PsfColumnNormNormalizer.CHANNEL_BLOCK_TYPE
        )
        assert psfColumnNormL2Desired > 0
        self.psfColumnNormL2Desired = psfColumnNormL2Desired  # Refers to the l_2 norm of the columns of the convolution matrix
        self.psfColumnNormL2 = None

    """ Abstract method implementation. Returns a normalized psf. """

    def NormalizePsf(self, psfRepH):
        fftFunc = ConvolutionMatrixUsingPsf.GetFftFunction(psfRepH)
        psfFft = fftFunc["fft"](psfRepH)
        x = np.zeros(psfRepH.shape)
        # XXX Unnecessary
        #        self.psfColumnNormL2 = np.zeros((H.size,))
        #        for flatInd in range(H.size):
        #            if (flatInd > 0):
        #                x.flat[flatInd - 1] = 0
        #            x.flat[flatInd] = 1
        #            y = fnFftInverse(HFft*fnFft(x)).real
        #            self.psfColumnNormL2[flatInd] = np.sqrt((y*y).sum())
        x.flat[0] = 1
        y = fftFunc["ifft"](psfFft * fftFunc["fft"](x)).real
        self.psfColumnNormL2 = np.sqrt((y * y).sum())
        self.psfScalar = self.psfColumnNormL2Desired / self.psfColumnNormL2
        return psfRepH * self.psfScalar
