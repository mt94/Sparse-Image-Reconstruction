import abc

class AbstractIterationsObserver(object):
    __metaclass__ = abc.ABCMeta;
    
    def __init__(self):
        self._bRequireFitError = False
        
    """ Return true if fitErrorN is required in UpdateEstimates """
    @property
    def RequireFitError(self):
        return self._bRequireFitError
    
    """Check to see if we want to terminate the iterations"""
    @abc.abstractproperty
    def TerminateIterations(self):
        raise NotImplementedError('No default abstract property implementation')
        
    """ Called each iteration to update observations """
    @abc.abstractmethod
    def UpdateEstimates(self, thetaNp1, thetaN, fitErrorN):        
        raise NotImplementedError('No default abstract method implementation')
    
    @abc.abstractmethod
    def UpdateState(self, stateDict):
        raise NotImplementedError('No default abstract method implementation')
