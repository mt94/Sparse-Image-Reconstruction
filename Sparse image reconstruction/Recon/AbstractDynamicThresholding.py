import abc

class AbstractDynamicThresholding(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        super(AbstractDynamicThresholding, self).__init__()
        
    @abc.abstractmethod
    def GetDynamicThreshold(self, hyperparameter, **kwargs):
        raise NotImplementedError('No default abstract method implemented')        