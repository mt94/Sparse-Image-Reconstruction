import abc

class AbstractIterationsObserver(object):
    __metaclass__ = abc.ABCMeta;
    
    def __init__(self):
        self._bRequireFitError = False
        
    """ Return true if fitErrorN is required in UpdateObservations """
    @property
    def RequireFitError(self):
        return self._bRequireFitError
    
    """Check to see if we want to terminate the iterations"""
    @abc.abstractproperty
    def TerminateIterations(self):
        raise NotImplementedError('No default abstract property implementation')
        
    """ Called each iteration to update observations """
    @abc.abstractmethod
    def UpdateObservations(self, thetaNp1, thetaN, fitErrorN):        
        raise NotImplementedError('No default abstract method implementation')
    
    
