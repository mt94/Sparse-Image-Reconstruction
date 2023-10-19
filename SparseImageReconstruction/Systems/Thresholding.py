import abc
import math
import numpy as np

# ABC for Thresholding class
class AbstractThresholding(object):
    __metaclass__ = abc.ABCMeta;
    
    @abc.abstractmethod
    def Apply(self, xIn):
        return xIn
    
class ThresholdingIdentity(AbstractThresholding):
    def Apply(self, xIn):
        return super(ThresholdingIdentity, self).Apply(xIn)
    
class ThresholdingIdentityNonnegative(AbstractThresholding):
    def Apply(self, xIn):
        return np.piecewise(xIn, [xIn < 0, xIn >= 0], [0, lambda x:x])
        
# Soft thresholding    
class ThresholdingSoft(AbstractThresholding):
    def __init__(self, t, bReturnTuple=False):
        assert t >= 0
        self.t = t
        self.bReturnTuple = bReturnTuple
    def Apply(self, xIn):
        xOut = np.piecewise(xIn, 
                            [xIn < -self.t, abs(xIn) <= self.t, xIn > self.t], 
                            [lambda x: x + self.t, 0, lambda x: x - self.t])
        if (not self.bReturnTuple):
            return xOut
        else:
            return (xOut,)
        
# Hard thresholding
class ThresholdingHard(AbstractThresholding):
    def __init__(self, t, bReturnTuple=False):
        assert t >= 0
        self.t = t
        self.bReturnTuple = bReturnTuple
    def Apply(self, xIn):
        xOut = np.piecewise(xIn, [xIn < -self.t, abs(xIn) <= self.t, xIn > self.t], [lambda x:x, 0, lambda x:x])        
        if (not self.bReturnTuple):
            return xOut
        else:
            return (xOut,)
                
# Hybrid thresholding
class ThresholdingHybrid(AbstractThresholding):
    def __init__(self, t1, t2, bReturnTuple=False):
        assert (t1 >= 0) and (t2 >= 0)
        self.t = (t1, t2)
        self.bReturnTuple = bReturnTuple
    def Apply(self, xIn):
        xOut = np.piecewise(xIn, 
                            [xIn < -self.t[0], abs(xIn) <= self.t[0], xIn > self.t[0]], 
                            [lambda x: x + self.t[1], 0, lambda x: x - self.t[1]])
        if (not self.bReturnTuple):
            return xOut
        else:
            return (xOut,)
            
# MAP1 composite thresholding for w < 0.5
class ThresholdingMap1CompositeSmallWeight(AbstractThresholding):
    def __init__(self, a, w, alpha):
        self.a = a
        assert (w >= 0) and (w < 0.5)
        self.w = w
        self.alpha = alpha        
    def Apply(self, xIn):
        zeroOdds = (1.0 - self.w) / self.w
        tSoft = self.a * (self.alpha**2)
        nzThreshold = tSoft + math.sqrt(2.0 * (self.alpha**2) * math.log(zeroOdds))
        nonzeroIndicator = (np.abs(xIn) > nzThreshold)*1
        thresholder = ThresholdingSoft(tSoft)
        xTilde = np.piecewise(xIn, 
                              [nonzeroIndicator > 0.5, nonzeroIndicator <= 0.5], 
                              [thresholder.Apply, 0])
        return (xTilde, nonzeroIndicator)        
        
# MAP1 composite thresholding for w > 0.5
class ThresholdingMap1CompositeLargeWeight(AbstractThresholding):       
    def __init__(self, a, w, alpha):
        self.a = a
        assert (w >= 0.5) and (w <= 1)
        self.w = w
        self.alpha = alpha
    def Apply(self, xIn):     
        nonzeroIndicator = np.ones(xIn.shape)
        thresholder = ThresholdingSoft(self.a * (self.alpha**2))
        xTilde = thresholder.Apply(xIn)
        return (xTilde, nonzeroIndicator)        