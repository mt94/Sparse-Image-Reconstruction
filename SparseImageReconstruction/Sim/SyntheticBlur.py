import numpy as np
from scipy import stats
from .Blur import AbstractBlur

class SyntheticBlur(AbstractBlur):      
    """ 
    STATIC METHODS
    """       
    # Follow JF's code 
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
        psf = np.matrix(SyntheticBlur.GaussianKernel(fwhm, nkHalf))  # Row vector        
        return np.transpose(psf) * psf
    
    """
    CONSTANTS
    """
    INPUT_KEY_FWHM = 'fwhm'    
    INPUT_KEY_NKHALF = 'nkhalf'
        
    # Default values
    FWHM_DEFAULT = 3
    NKHALF_DEFAULT = 5
            
    # 2-d Gaussian symmetric blur
    BLUR_GAUSSIAN_SYMMETRIC_2D = 1
                            
    
    def __init__(self, blurType, blurParametersDict):
        super(SyntheticBlur, self).__init__()
        self._blurType = blurType
        self._blurParametersDict = blurParametersDict        

        if (self._blurType == SyntheticBlur.BLUR_GAUSSIAN_SYMMETRIC_2D):
            fwhm = self._blurParametersDict.get(SyntheticBlur.INPUT_KEY_FWHM, SyntheticBlur.FWHM_DEFAULT)             
            nkHalf = self._blurParametersDict.get(SyntheticBlur.INPUT_KEY_NKHALF, SyntheticBlur.NKHALF_DEFAULT)            
                                    
            # With the 2d Gaussian symmetric psf, we already know the shift that's introduced by blurring
            self._blurShift = (nkHalf, nkHalf)
            
            self._blurPsf = SyntheticBlur.GaussianBlurSymmetric2d(fwhm, nkHalf)
            self._blurPsf = self._blurPsf / np.max(self._blurPsf)
        else:
            raise NotImplementedError("SyntheticBlur type " + self._blurType + " hasn't been implemented")
                                                              
    def BlurImage(self, theta):                        
        if (self._blurType == SyntheticBlur.BLUR_GAUSSIAN_SYMMETRIC_2D):                        
            self._thetaShape = theta.shape
            if len(self._thetaShape) != 2:
                raise ValueError('BLUR_GAUSSIAN_SYMMETRIC_2D requires theta to be 2-d')
            
            if (not np.all(self._thetaShape >= self._blurPsf.shape)):
                raise ValueError("theta's shape must be at least as big as the blur's shape")            
                                    
            y = np.fft.ifft2(np.multiply(np.fft.fft2(self.BlurPsfInThetaFrame), np.fft.fft2(theta)))
            return y.real
        else:
            raise NotImplementedError("SyntheticBlur type " + self._blurType + " hasn't been implemented")
        
