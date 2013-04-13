
from Recon.AbstractIterationsObserver import AbstractIterationsObserver

class McmcIterationEvaluator(AbstractIterationsObserver):
    
    def __init__(self):
        super(McmcIterationEvaluator, self).__init__()
        self._iterationCount = 0
    
    """ Implementation of abstract methods """
    
    @property
    def TerminateIterations(self):
        return False # Never terminate
    
    def UpdateEstimates(self, thetaNp1, thetaN, fitErrorN):
        raise NotImplementedError('Method unimplemented')
    
    def UpdateState(self, stateDict): 
        self._iterationCount += 1
        print("Iteration count {0}".format(self._iterationCount))
        
