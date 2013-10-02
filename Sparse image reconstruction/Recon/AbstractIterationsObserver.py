import abc

class AbstractIterationsObserver(object):
    __metaclass__ = abc.ABCMeta;
    
    def __init__(self):
        self._bRequireFitError = False
            
    @property
    def RequireFitError(self):
        """ Return true if fitErrorN is required in UpdateWithEstimates """
        return self._bRequireFitError
        
    @abc.abstractproperty
    def TerminateIterations(self):
        """ Check to see if we want to terminate the iterations"""
        raise NotImplementedError('No default abstract property implementation')
            
    @abc.abstractproperty
    def HistoryEstimate(self):
        """ If implemented, returns the history of estimates """
        pass
    
    @abc.abstractproperty
    def HistoryState(self):
        """ A more general version of HistoryEstimate. Any per-iteration state can be calculated and saved. """
        pass
                
    @abc.abstractmethod
    def UpdateWithEstimates(self, reconArgsNp1, reconArgsN, fitErrorN):  
        """ Called each iteration to update observations """      
        raise NotImplementedError('No default abstract method implementation')
    
    @abc.abstractmethod
    def UpdateState(self, stateDict):
        """ A more general update method than UpdateWithEstimates """
        raise NotImplementedError('No default abstract method implementation')
