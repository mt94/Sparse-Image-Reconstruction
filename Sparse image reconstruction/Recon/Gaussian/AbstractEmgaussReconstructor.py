import abc

from ...Recon.AbstractIterationsObserver import AbstractIterationsObserver
from ...Recon.AbstractReconstructor import AbstractReconstructor
from ...Systems.AbstractConvolutionMatrix import AbstractConvolutionMatrix


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
        self._reconArgs = None        
    
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
    def EstimateUsingFft(self, y, convMatrixObj, *args):

        assert ((len(args) == 1) or (len(args) == 2))
        assert isinstance(convMatrixObj, AbstractConvolutionMatrix)
        
        assert (y.shape == convMatrixObj.PsfShape)
        assert (y.shape == args[0].shape)
        
        if (len(args) == 2):        
            assert (y.shape == args[1].shape)   
                                                    
        # Get the starting point theta0
        self._reconArgs = args
        
        maxIter = self._optimSettingsDict.get(AbstractEmgaussReconstructor.INPUT_KEY_MAX_ITERATIONS, 500)
        tau = self._optimSettingsDict.get(AbstractEmgaussReconstructor.INPUT_KEY_TAU, 1)
        
        # Get the IterationsObserver object
        assert AbstractEmgaussReconstructor.INPUT_KEY_ITERATIONS_OBSERVER in self._optimSettingsDict   
        iterObserver = self._optimSettingsDict[AbstractEmgaussReconstructor.INPUT_KEY_ITERATIONS_OBSERVER]
        assert isinstance(iterObserver, AbstractIterationsObserver)

        fnUpdateIterObserver = lambda tNp1, tN, feN: iterObserver.UpdateWithEstimates(tNp1, tN, feN)
        
        # Do any initialization
        self.SetupBeforeIterations()
                               
        numIter = 0
        
        # Run through the EM iterations
        while numIter < maxIter:
            if (len(self._reconArgs) == 2):
                thetaN = self._reconArgs[0] * self._reconArgs[1]
            else:
                thetaN = self._reconArgs[0]
            fitErrorN = y - convMatrixObj.Multiply(thetaN)
            correction = convMatrixObj.MultiplyPrime(fitErrorN)
            reconArgsNext = self.Mstep(thetaN + tau * correction, numIter)
            fnUpdateIterObserver(reconArgsNext, self._reconArgs, fitErrorN)        
            if iterObserver.TerminateIterations:                
                self._terminationReason = 'Iterations observer, terminating after ' + str(numIter) + ' iterations'
                break
            numIter += 1
            self._reconArgs = reconArgsNext
            
        if numIter == maxIter:
            self._terminationReason = 'Max iterations {0} reached'.format(maxIter)
            
        return self._reconArgs
       
    # Abstract method override
    def Estimate(self, y, psfRepH, *args):
        # Only support the FFT method for now
        return self.EstimateUsingFft(y, psfRepH, *args)
        
        
        
            
        
    
