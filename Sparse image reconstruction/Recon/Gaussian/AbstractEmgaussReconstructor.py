import abc
import numpy as np
from IterationsObserver import AbstractIterationsObserver
from Recon.Reconstructor import AbstractReconstructor

class AbstractEmgaussReconstructor(AbstractReconstructor):
    
    INPUT_KEY_MAX_ITERATIONS = 'max_iters'
    INPUT_KEY_TAU = 'tau'
    INPUT_KEY_ITERATIONS_OBSERVER = 'iters_observer'
        
    def __init__(self, optimSettingsDict=None):   
        super(AbstractEmgaussReconstructor, self).__init__()     
        assert optimSettingsDict is not None
        self._optimSettingsDict = optimSettingsDict        
        self._terminationReason = None            
    
    @property
    def TerminationReason(self):
        return self._terminationReason
    
    @abc.abstractmethod
    def Mstep(self, x):
        pass
        
    """ y, H, and theta0 are NumPy matrices of the same shape. H is treated as as the 
        matrix with which theta is convolved (for a noiseless version of y). Don't make
        an assumption on the l_2 norm of H. If the caller wants to normalize H s.t. it
        has unit l_2 norm, then it has to be done before calling this method. 
    """
    def EstimateUsingFft(self, y, H, theta0):                                            
        # Get the starting point theta0
        thetaN = theta0
        
        maxIter = self._optimSettingsDict[AbstractEmgaussReconstructor.INPUT_KEY_MAX_ITERATIONS] \
            if AbstractEmgaussReconstructor.INPUT_KEY_MAX_ITERATIONS in self._optimSettingsDict \
            else 500
        tau = self._optimSettingsDict[AbstractEmgaussReconstructor.INPUT_KEY_TAU] \
            if AbstractEmgaussReconstructor.INPUT_KEY_TAU in self._optimSettingsDict \
            else 1
        
        if len(H.shape) == 2:            
            fnFft = np.fft.fft2
            fnFftInverse = np.fft.ifft2  
        else:             
            fnFft = np.fft.fftn
            fnFftInverse = np.fft.ifftn
                        
        HFft = fnFft(H)            
        yFft = fnFft(y)        

        # Get the IterationsObserver object
        assert AbstractEmgaussReconstructor.INPUT_KEY_ITERATIONS_OBSERVER in self._optimSettingsDict   
        iterObserver = self._optimSettingsDict[AbstractEmgaussReconstructor.INPUT_KEY_ITERATIONS_OBSERVER]
        assert isinstance(iterObserver, AbstractIterationsObserver)
        
        if iterObserver.RequireFitError == False:
            fnCallIterObserver = lambda tNp1, tN, feN: iterObserver.CheckTerminateCondition(thetaNp1, thetaN)
        else:
            fnCallIterObserver = lambda tNp1, tN, feN: iterObserver.CheckTerminateCondition(thetaNp1, thetaN, fnFftInverse(feN).real)
                
        # Run through the EM iterations
        numIter = 0;
        while numIter < maxIter:
            fitErrorNFft = yFft - np.multiply(HFft, fnFft(thetaN))
            correction = fnFftInverse(np.multiply(HFft.conj(), fitErrorNFft)).real
            thetaNp1 = self.Mstep(thetaN + tau * correction)        
            if fnCallIterObserver(thetaNp1, thetaN, fitErrorNFft):                
                self._terminationReason = 'Iterations observer, terminating after ' + str(numIter) + ' iterations'
                break
            numIter += 1
            thetaN = thetaNp1
            
        if numIter == maxIter:
            self._terminationReason = 'Max iterations reached'
            
        return thetaN
       
    # Abstract method override
    def Estimate(self, y, H, theta0):
        # Only support the FFT method for now
        return self.EstimateUsingFft(y, H, theta0)
        
        
        
            
        
    
