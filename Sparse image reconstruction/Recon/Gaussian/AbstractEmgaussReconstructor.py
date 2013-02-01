import abc
#import numpy as np

from Recon.AbstractIterationsObserver import AbstractIterationsObserver
from Recon.AbstractReconstructor import AbstractReconstructor
from Systems.AbstractConvolutionMatrix import AbstractConvolutionMatrix
#from Systems.ConvolutionMatrixUsingPsf import ConvolutionMatrixUsingPsf

class AbstractEmgaussReconstructor(AbstractReconstructor):
    
    INPUT_KEY_ALPHA = 'alpha'
    INPUT_KEY_ITERATIONS_OBSERVER = 'iters_observer'
    INPUT_KEY_MAX_ITERATIONS = 'max_iters'
    INPUT_KEY_ESTIMATE_HYPERPARAMETERS_ITERATIONS_INTERVAL = 'est_hyper_iters_interval'
    INPUT_KEY_TAU = 'tau'    
        
    def __init__(self, optimSettingsDict=None):   
        super(AbstractEmgaussReconstructor, self).__init__()     
        assert optimSettingsDict is not None
        self._optimSettingsDict = optimSettingsDict        
        self._terminationReason = None
        self._thetaN = None            
    
    @property
    def TerminationReason(self):
        return self._terminationReason
    
    @abc.abstractmethod
    def SetupBeforeIterations(self):
        pass 
    
    @abc.abstractmethod
    def Mstep(self, x, numIter):
        pass
        
    """ y and theta0 are NumPy matrices of the same shape. """
    def EstimateUsingFft(self, y, convMatrixObj, theta0):
#        assert (y.shape == psfRepH.shape) and (y.shape == theta0.shape)
        assert isinstance(convMatrixObj, AbstractConvolutionMatrix)
        assert (y.shape == convMatrixObj.PsfShape) and (y.shape == theta0.shape)        
                                                    
        # Get the starting point theta0
        self._thetaN = theta0
        
        maxIter = self._optimSettingsDict[AbstractEmgaussReconstructor.INPUT_KEY_MAX_ITERATIONS] \
            if AbstractEmgaussReconstructor.INPUT_KEY_MAX_ITERATIONS in self._optimSettingsDict \
            else 500
        tau = self._optimSettingsDict[AbstractEmgaussReconstructor.INPUT_KEY_TAU] \
            if AbstractEmgaussReconstructor.INPUT_KEY_TAU in self._optimSettingsDict \
            else 1
                
#        fftFunction = ConvolutionMatrixUsingPsf.GetFftFunction(convMatrixObj.PsfShape)                        
#        psfFft = fftFunction['fft'](psfRepH)            
#        yFft = fftFunction['fft'](y)        
        
        # Get the IterationsObserver object
        assert AbstractEmgaussReconstructor.INPUT_KEY_ITERATIONS_OBSERVER in self._optimSettingsDict   
        iterObserver = self._optimSettingsDict[AbstractEmgaussReconstructor.INPUT_KEY_ITERATIONS_OBSERVER]
        assert isinstance(iterObserver, AbstractIterationsObserver)
        
        if iterObserver.RequireFitError == False:
            fnCallIterObserver = lambda tNp1, tN, feN: iterObserver.CheckTerminateCondition(tNp1, tN)
        else:
            #fnCallIterObserver = lambda tNp1, tN, feN: iterObserver.CheckTerminateCondition(tNp1, tN, fftFunction['ifft'](feN).real)
            fnCallIterObserver = lambda tNp1, tN, feN: iterObserver.CheckTerminateCondition(tNp1, tN, feN)

        # Do any initialization
        self.SetupBeforeIterations()
                        
        # Run through the EM iterations
        numIter = 0;
        while numIter < maxIter:
#            fitErrorNFft = yFft - np.multiply(psfFft, fftFunction['fft'](self._thetaN))
#            correction = fftFunction['ifft'](np.multiply(psfFft.conj(), fitErrorNFft)).real
            fitErrorN = y - convMatrixObj.Multiply(self._thetaN)
            correction = convMatrixObj.MultiplyPrime(fitErrorN)
            thetaNp1 = self.Mstep(self._thetaN + tau * correction, numIter)        
            if fnCallIterObserver(thetaNp1, self._thetaN, fitErrorN):                
                self._terminationReason = 'Iterations observer, terminating after ' + str(numIter) + ' iterations'
                break
            numIter += 1
            self._thetaN = thetaNp1
            
        if numIter == maxIter:
            self._terminationReason = 'Max iterations reached'
            
        return self._thetaN
       
    # Abstract method override
    def Estimate(self, y, psfRepH, theta0):
        # Only support the FFT method for now
        return self.EstimateUsingFft(y, psfRepH, theta0)
        
        
        
            
        
    
