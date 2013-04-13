import abc

class AbstractMcmcSampler(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        super(object, self).__init__()
        
    @abc.abstractmethod
    def SamplerSetup(self):
        pass
    
    @abc.abstractmethod
    def SamplerRun(self):
        pass
    
    @abc.abstractmethod
    def SamplerGet(self, elementDesc):
        pass
    
    