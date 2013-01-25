import numpy as np

from Systems.AbstractConvolutionMatrix import AbstractConvolutionMatrix
from Systems.ConvolutionMatrixUsingPsf import ConvolutionMatrixUsingPsf 
    
class ConvolutionMatrixZeroMeanUnitNormDerivative(AbstractConvolutionMatrix):
    def __init__(self, psfRepH):
        super(ConvolutionMatrixZeroMeanUnitNormDerivative, self).__init__()
        self._convMatrixObj = ConvolutionMatrixUsingPsf(psfRepH)        
        # Calculate the mean of columns of H (they're all the same) as well as the 
        # l_2 norm of the columns after the mean has been subtracted off.
        op = self._convMatrixObj.CalculateMeanAndL2norm()
        self._columnMean = op['mean']
        self._columnZeroMeanedL2norm = op['l2norm']
                                    
    """ Implement abstract methods """
                                        
    def Multiply(self, theta):
        if (theta.shape != self._psfRepH.shape):
            # Assume that theta has the same shape as the psf representing H
            raise ValueError('theta is incompatible with psf')            
        thetaSum = np.sum(theta)
        return (self._convMatrixObj.Multiply(theta)*(1.0/self._columnZeroMeanedL2norm) -  
                self._columnMean/self._columnZeroMeanedL2norm*thetaSum*np.ones(theta.size))
        
    def MultiplyPrime(self, theta):
        if (theta.shape != self._psfRepH.shape):
            # Assume that theta has the same shape as the psf representing H
            raise ValueError('theta is incompatible with psf')            
        thetaSum = np.sum(theta)
        return (self._convMatrixObj.MultiplyPrime(theta)*(1.0/self._columnZeroMeanedL2norm) -  
                self._columnMean/self._columnZeroMeanedL2norm*thetaSum*np.ones(theta.size))
        
        