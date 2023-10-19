import abc
from ..Channel import ChannelBlock as chb

class AbstractReconstructor(chb.AbstractChannelBlock):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        super(AbstractReconstructor, self).__init__('Reconstructor')
        
    @abc.abstractmethod
    def Estimate(self, y, H, theta0):
        raise NotImplementedError('No default abstract method implemented')
    