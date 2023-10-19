from ..Channel import ChannelBlock as chb
import numpy as np


class PsfNormalizer(chb.AbstractChannelBlock):
    CHANNEL_BLOCK_TYPE = "PsfNormalizer"

    def __init__(self, psfNormL2Desired):
        super(PsfNormalizer, self).__init__(PsfNormalizer.CHANNEL_BLOCK_TYPE)
        assert psfNormL2Desired > 0
        self.psfNormL2Desired = psfNormL2Desired
        self.psfNormL2 = None
        self.psfScalar = None

    def NormalizePsf(self, H):
        if len(H.shape) == 2:
            HFft = np.fft.fft2(H)
        else:
            HFft = np.fft.fftn(H)
        psfNormL2Current = np.max(np.reshape(np.absolute(HFft), HFft.size))
        self.psfScalar = self.psfNormL2Desired / psfNormL2Current
        self.psfNormL2 = psfNormL2Current
        return H * self.psfScalar

    def NormalizeTheta(self, theta):
        if self.psfScalar is None:
            raise UnboundLocalError("Don't know how to normalize theta yet")
        else:
            assert self.psfScalar > 0
            return theta / self.psfScalar

    def GetSpectralRadiusGramMatrixRowsH(self):
        if self.psfNormL2 is None:
            raise UnboundLocalError("Cannot calculate the spectral radius of H*H^T")
        else:
            assert self.psfNormL2 > 0
            return self.psfNormL2 * self.psfNormL2
