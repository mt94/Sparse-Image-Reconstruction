import numpy as np
from Recon.AbstractReconstructor import AbstractReconstructor
from Systems.ConvolutionMatrixUsingPsf import ConvolutionMatrixUsingPsf

""" Implement the explicit (non-iterative) solution. Requires the psf. """
class L2NormDirectMinimizerReconstructor(AbstractReconstructor):
            
    def __init__(self, constL2PenaltyOnTheta=None):
        super(L2NormDirectMinimizerReconstructor, self).__init__()
        assert constL2PenaltyOnTheta >= 0
        self.constL2PenaltyOnTheta = constL2PenaltyOnTheta
        
    """ Abstract method override """         
    def Estimate(self, y, psfRepH, theta0):
        fftFunction = ConvolutionMatrixUsingPsf.GetFftFunction(psfRepH)                                
        psfFft = fftFunction['fft'](np.array(psfRepH))            
        yFft = fftFunction['fft'](np.array(y))          
    
        S = np.conjugate(psfFft) * psfFft
        if (self.constL2PenaltyOnTheta is not None):
            S = S + self.constL2PenaltyOnTheta * np.identity(S.shape[0])
                        
        #return fnFftInverse(np.linalg.solve(S, HFft.getH()*yFft)).real
        return fftFunction['ifft']((np.conjugate(psfFft)*yFft)/S).real
            
        
            
        
        
            
