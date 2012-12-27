import numpy as np
import Reconstructor

class L2NormMinimizer(Reconstructor.AbstractReconstructor):
            
    def __init__(self, constL2PenaltyOnTheta=None):
        assert constL2PenaltyOnTheta >= 0
        self.constL2PenaltyOnTheta = constL2PenaltyOnTheta
        
    """ Abstract method override. Take H to be the matrix with which theta is
        convolved to get a noiseless version of y. """
    def Estimate(self, y, H, theta0):
        
        if len(H.shape) == 2:            
            fnFft = np.fft.fft2
            fnFftInverse = np.fft.ifft2  
        else:             
            fnFft = np.fft.fftn
            fnFftInverse = np.fft.ifftn
                        
        HFft = fnFft(H)            
        yFft = fnFft(y)          
    
        S = HFft.getH() * HFft
        if (self.constL2PenaltyOnTheta is not None):
            S = S + self.constL2PenaltyOnTheta * np.identity(S.shape[0])
        
        # Assume that S has full rank. If not, then the solve won't work.
        assert np.linalg.matrix_rank(S, 1e-6) == S.shape[0]
                
        return fnFftInverse(np.linalg.solve(S, HFft.getH()*yFft))
            
        
            
        
        
            
