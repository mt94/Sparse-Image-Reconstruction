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
        self.blurType = blurType
        self.blurParametersDict = blurParametersDict
        self.blurShift = None
        self.thetaShape = None         
        self._blurPsf = None       
                
    def RemoveShiftFromBlurredImage(self, anImage):
        if self.blurShift is None:
            raise UnboundLocalError('blurShift isn\'t defined')
        else:
            assert len(anImage.shape) == len(self.blurShift)
            tmp = np.array(anImage)
            blurShift = self.blurShift
            assert np.all(blurShift > 0)            
            for axisInd in range(len(blurShift)):                
                tmp = np.roll(tmp, -blurShift[axisInd], axisInd)
            return np.matrix(tmp)
                   
    @property
    def BlurPsf(self):
        return self._blurPsf
    
    @property
    def BlurPsfInThetaFrame(self):
        if (self.blurType == Blur.BLUR_GAUSSIAN_SYMMETRIC_2D):
            psfInThetaFrame = np.zeros(self.thetaShape)
            blurPsfShape = self._blurPsf.shape
            assert len(blurPsfShape) == 2
            psfInThetaFrame[:blurPsfShape[0],:blurPsfShape[1]] = self._blurPsf
            return psfInThetaFrame
        else:
            raise NotImplementedError('Blur type ' + self.blurType + ' hasn\'t been implemented')
        
    def BlurImage(self, theta):
        # Blur the image using FFTs to do the convolution
        if (self.blurType == Blur.BLUR_GAUSSIAN_SYMMETRIC_2D):
            # Set the value of fwhm, nkHalf
            fwhm = self.blurParametersDict[Blur.INPUT_KEY_FWHM] if Blur.INPUT_KEY_FWHM in self.blurParametersDict \
                else Blur.BLUR_GAUSSIAN_SYMMETRIC_2D_FWHM_DEFAULT
            nkHalf = self.blurParametersDict[Blur.INPUT_KEY_NKHALF] if Blur.INPUT_KEY_NKHALF in self.blurParametersDict \
                else Blur.BLUR_GAUSSIAN_SYMMETRIC_2D_NKHALF_DEFAULT
            
            self.thetaShape = theta.shape
            self.blurShift = (nkHalf, nkHalf)
            
            psf = Blur.GaussianBlurSymmetric2d(fwhm, nkHalf)
            self._blurPsf = psf
            y = np.fft.ifft2(np.multiply(np.fft.fft2(self.BlurPsfInThetaFrame), np.fft.fft2(theta)))
            return y.real
        else:
            raise NotImplementedError('Blur type ' + self.blurType + ' hasn\'t been implemented')
            
            
            
            
            
            
            
            
        
    
