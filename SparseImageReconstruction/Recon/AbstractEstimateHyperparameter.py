import abc

class AbstractEstimateHyperparameter(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        super(AbstractEstimateHyperparameter, self).__init__()
        
    @abc.abstractmethod
    def EstimateHyperparameter(self, args):
        raise NotImplementedError('No default abstract method implemented')    