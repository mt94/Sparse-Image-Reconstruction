import abc
import numpy as np

# ABC for Thresholding class
class AbstractThresholding(object):
    __metaclass__ = abc.ABCMeta;
    
    @abc.abstractmethod
    def Apply(self, xIn):
        return xIn
    
class ThresholdingIdentity(AbstractThresholding):
    def Apply(self,xIn):
        return super(ThresholdingIdentity, self).Apply(xIn)
    
# Soft thresholding    
class ThresholdingSoft(AbstractThresholding):
    def __init__(self, t):
        assert t >= 0
        self.t = t
    def Apply(self, xIn):
        return np.piecewise(xIn, 
                            [xIn < -self.t, abs(xIn) <= self.t, xIn > self.t], 
                            [lambda x: x + self.t, 0, lambda x: x - self.t])
        
# Hard thresholding
class ThresholdingHard(AbstractThresholding):
    def __init__(self, t):
        assert t >= 0
        self.t = t
    def Apply(self, xIn):
        return np.piecewise(xIn, [xIn < self.t, abs(xIn) <= self.t, xIn > self.t], [lambda x:x, 0, lambda x:x])        
        
    
