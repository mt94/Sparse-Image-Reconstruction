import abc

class AbstractReconstructor(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def Estimate(self, y, H, theta0):
        raise NotImplementedError('No default abstract method implemented')
    