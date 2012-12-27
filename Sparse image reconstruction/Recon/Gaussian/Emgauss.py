import numpy as np
import types
import IterationsObserver
import Reconstructor


class EmgaussIterationsObserver(IterationsObserver.AbstractIterationsObserver):
    
    TERMINATE_COND_THETA_DELTA = 0
    INPUT_KEY_TERMINATE_COND = 'terminate_cond'
    INPUT_KEY_TERMINATE_TOL = 'terminate_tol'
        
    def __init__(self, inputDict):        
        self.terminateCondition = inputDict[EmgaussIterationsObserver.INPUT_KEY_TERMINATE_COND] if EmgaussIterationsObserver.INPUT_KEY_TERMINATE_COND in inputDict \
        else EmgaussIterationsObserver.TERMINATE_COND_THETA_DELTA
        self.terminateTolerance = inputDict[EmgaussIterationsObserver.INPUT_KEY_TERMINATE_TOL] if EmgaussIterationsObserver.INPUT_KEY_TERMINATE_TOL in inputDict \
        else 1e-3
        
    # Abstract method override
    def CheckTerminateCondition(self, thetaNp1, thetaN, fitErrorN):
        if (self.terminateCondition == EmgaussIterationsObserver.TERMINATE_COND_THETA_DELTA):
            if (np.linalg.norm(thetaNp1-thetaN, 2) <self.terminateTolerance):
                return True
            else:
                return False
        # By default, don't terminate
        return False

class Emgauss(Reconstructor.AbstractReconstructor):
    
    """ Expected keys in optimSettingsDict: max_iter, tau, iter_observer """
    def __init__(self, optimSettingsDict):        
        self.optimSettingsDict = optimSettingsDict
        self._funcMStep = None
             
    @property
    def FuncMStep(self):
        return self._funcMStep
    
    @FuncMStep.setter
    def FuncMStep(self, value):
        self._funcMStep = value
    
    """ y, H, and theta0 are NumPy matrices of the same shape. H is treated as as the 
        matrix with which theta is convolved (for a noiseless version of y). """
    def EstimateUsingFft(self, y, H, theta0):    
        # Type check
        assert self.FuncMStep is not None and isinstance(self.FuncMStep, types.FunctionType)
                                
        # Get the starting point theta0
        thetaN = theta0
        
        maxIter = self.optimSettingsDict['max_iter'] if 'max_iter' in self.optimSettingsDict else 500
        tau = self.optimSettingsDict['tau'] if 'tau' in self.optimSettingsDict else 1
        
        if len(H.shape) == 2:            
            fnFft = np.fft.fft2
            fnFftInverse = np.fft.ifft2  
        else:             
            fnFft = np.fft.fftn
            fnFftInverse = np.fft.ifftn
                        
        HFft = fnFft(H)            
        yFft = fnFft(y)        

        # Get the IterationsObserver object
        assert ('iter_observer' in self.optimSettingsDict)   
        iterObserver = self.optimSettingsDict['iter_observer']
        assert isinstance(iterObserver, IterationsObserver.AbstractIterationsObserver)
        
        # Run through the EM iterations
        numIter = 0;
        while numIter < maxIter:
            fitErrorNFft = yFft - np.multiply(HFft, fnFft(thetaN))
            correction = fnFftInverse(np.multiply(HFft.getH(), fitErrorNFft))
            thetaNp1 = self.FuncMStep(thetaN + tau * correction)
            if iterObserver.CheckTerminateCondition(thetaNp1, thetaN, fnFftInverse(fitErrorNFft)):
                break
            numIter += 1
            thetaN = thetaNp1
            
        return thetaN
       
    # Abstract method override
    def Estimate(self, y, H, theta0):
        # Only support the FFT method for now
        return self.estimateUsingFft(y, H, theta0)
        
        
        
            
        
    
