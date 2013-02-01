import numpy as np
from scipy import stats
import Channel.ChannelBlock as chb

class Blur(chb.AbstractChannelBlock):    
    """ Follow JF's code """
    @staticmethod
    def GaussianKernel(fwhm, nkHalf):
        assert fwhm >= 0
        if fwhm == 0:
            assert nkHalf >= 0
            kern = np.zeros(2 * nkHalf + 1)
            kern[nkHalf] = 1
        else:
            assert nkHalf >= fwhm
            sig = fwhm / np.sqrt(np.log(256))
            x = np.arange(-nkHalf, nkHalf + 1)
            kern = stats.norm.cdf(x + 0.5, 0, sig) - stats.norm.cdf(x - 0.5, 0, sig)
        return kern
            
    @staticmethod
    def GaussianBlurSymmetric2d(fwhm, nkHalf):
        psf = np.matrix(Blur.GaussianKernel(fwhm, nkHalf))  # Row vector        
        return np.transpose(psf) * psf
    
    BLUR_GAUSSIAN_SYMMETRIC_2D = 1
    BLUR_GAUSSIAN_SYMMETRIC_2D_FWHM_DEFAULT = 3
    BLUR_GAUSSIAN_SYMMETRIC_2D_NKHALF_DEFAULT = 5
    
    INPUT_KEY_FWHM = 'fwhm'
    INPUT_KEY_NKHALF = 'nkhalf'
    
    CHANNEL_BLOCK_TYPE = 'Blur'
    
    def __init__(self, blurType, blurParametersDict):
        super(Blur, self).__init__(Blur.CHANNEL_BLOCK_TYPE)
        self._blurType = blurType
        self._blurParametersDict = blurParametersDict
        self._blurShift = None
        self._thetaShape = None         
        self._blurPsf = None       
                
    def RemoveShiftFromBlurredImage(self, anImage):
        if self._blurShift is None:
            raise UnboundLocalError('_blurShift isn\'t defined')
        else:
            assert len(anImage.shape) == len(self._blurShift)
            tmp = np.array(anImage)
            blurShift = self._blurShift
            assert np.all(blurShift > 0)            
            for axisInd in range(len(blurShift)):                
                tmp = np.roll(tmp, -blurShift[axisInd], axisInd)
            return np.matrix(tmp)

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
        if (self._blurType == Blur.BLUR_GAUSSIAN_SYMMETRIC_2D):
            if self._thetaShape is None:
                raise NameError('_thetaShape has not yet been set')
            psfInThetaFrame = np.zeros(self._thetaShape)
            blurPsfShape = self._blurPsf.shape
            assert len(blurPsfShape) == 2
            psfInThetaFrame[:blurPsfShape[0],:blurPsfShape[1]] = self._blurPsf
            return psfInThetaFrame
        else:
            raise NotImplementedError('Blur type ' + self._blurType + ' hasn\'t been implemented')
        
    def BlurImage(self, theta):
        # Blur the image using FFTs to do the convolution
        if (self._blurType == Blur.BLUR_GAUSSIAN_SYMMETRIC_2D):
            # Set the value of fwhm, nkHalf
            fwhm = self._blurParametersDict[Blur.INPUT_KEY_FWHM] if Blur.INPUT_KEY_FWHM in self._blurParametersDict \
                else Blur.BLUR_GAUSSIAN_SYMMETRIC_2D_FWHM_DEFAULT
            nkHalf = self._blurParametersDict[Blur.INPUT_KEY_NKHALF] if Blur.INPUT_KEY_NKHALF in self._blurParametersDict \
                else Blur.BLUR_GAUSSIAN_SYMMETRIC_2D_NKHALF_DEFAULT
            
            self._thetaShape = theta.shape
            if len(self._thetaShape) != 2:
                raise ValueError('BLUR_GAUSSIAN_SYMMETRIC_2D requires theta to be 2-d')
            
            self._blurShift = (nkHalf, nkHalf)
            
            psf = Blur.GaussianBlurSymmetric2d(fwhm, nkHalf)
            if (not np.all(self._thetaShape >= psf.shape)):
                raise ValueError('theta\'s shape must be at least as big as the Gaussian blur\'s shape')            
            self._blurPsf = psf            
            
            y = np.fft.ifft2(np.multiply(np.fft.fft2(self.BlurPsfInThetaFrame), np.fft.fft2(theta)))
            return y.real
        else:
            raise NotImplementedError('Blur type ' + self._blurType + ' hasn\'t been implemented')
            
            
            
            
            
            
            
            
        
    
