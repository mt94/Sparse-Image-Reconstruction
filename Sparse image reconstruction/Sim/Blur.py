import numpy as np
from scipy import stats

class Blur:
    
    """ Follow JF's code """
    @staticmethod
    def GaussianKernel(fwhm, nkHalf):
        assert (fwhm >= 0) and (nkHalf >= 0)
        if fwhm == 0:
            kern = np.zeros(2 * nkHalf + 1)
            kern[nkHalf] = 1
        else:
            sig = fwhm / np.sqrt(np.log(256))
            x = range(-nkHalf, nkHalf + 1)
            kern = stats.norm.cdf(x + 0.5, 0, sig) - stats.norm.cdf(x - 0.5, 0, sig)
            
    @staticmethod
    def GaussianBlurSymmetric2d(fwhm, nkHalf):
        psf = Blur.GaussianKernel(fwhm, nkHalf)
        return psf * np.transpose(psf)
    
    BLUR_GAUSSIAN_SYMMETRIC_2D = 1
    BLUR_GAUSSIAN_SYMMETRIC_2D_FWHM_DEFAULT = 3
    BLUR_GAUSSIAN_SYMMETRIC_2D_NKHALF_DEFAULT = 5
    
    INPUT_KEY_FWHM = 'fwhm'
    INPUT_KEY_NKHALF = 'nkhalf'
    
    def __init__(self, blurType, blurParametersDict):
        self.blurType = blurType
        self.blurParametersDict = blurParametersDict
        self.blurShift = None
        self.thetaShape = None        
                
    def RemoveShiftFromBlurredImage(self, anImage):
        if self.blurShift is None:
            raise UnboundLocalError('blurShift isn\'t defined')
        else:
            tmp = anImage
            blurShift = np.floor(self.blurShift)
            assert np.all(blurShift > 0)
            for axisInd in range(len(blurShift)):
                tmp = np.roll(tmp, -blurShift[axisInd], axisInd)
            return tmp
                                  
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
            psfInLargerFrame = np.zeros(self.thetaShape)
            psfSupport = 2*nkHalf + 1
            psfInLargerFrame[:psfSupport,:psfSupport] = psf
            return np.fft.ifft2(np.multiply(np.fft.fft2(psfInLargerFrame), np.fft.fft2(theta)))
        else:
            raise NotImplementedError('Blur type ' + self.blurType + ' hasn\'t been implemented')
            
            
            
            
            
            
            
            
        
    
