import numpy as np

from Recon.AbstractEstimateHyperparameter import AbstractEstimateHyperparameter
from Recon.AbstractDynamicThresholding import AbstractDynamicThresholding
from Recon.Gaussian.Thresholding import ThresholdingHybrid, ThresholdingSoft
from Recon.Gaussian.EmgaussEmpiricalMapReconstructor import EmgaussEmpiricalMapReconstructor

class LazeMap1EstimateHyperparameter(AbstractEstimateHyperparameter):
    def __init__(self):
        super(LazeMap1EstimateHyperparameter, self).__init__()
    def EstimateHyperparameter(self, thetaN):        
        l0Norm = (thetaN != 0).sum()
        l1Norm = np.abs(thetaN).sum()        
        return (float(thetaN.size)/float(l1Norm), float(l0Norm)/float(thetaN.size))

class LazeMap1DynamicThresholding(AbstractDynamicThresholding):    
    def __init__(self):
        super(LazeMap1DynamicThresholding, self).__init__()
    def GetDynamicThreshold(self, hyperparameter, **kwargs):
        assert "alphaVal" in kwargs
        alpha = kwargs['alphaVal']
        (a, w) = hyperparameter # hyperparameter = (a,w)        
        cTmp = a*alpha*alpha                
        if (w <= 0.5):     
            assert w > 0       
            return ThresholdingHybrid(cTmp + alpha*np.sqrt(2*np.log((1-w)/w)), cTmp)            
        else:
            return ThresholdingSoft(cTmp)                

class EmgaussEmpiricalMapLaze1Reconstructor(EmgaussEmpiricalMapReconstructor):
    def __init__(self, optimSettingsDict):
        super(EmgaussEmpiricalMapLaze1Reconstructor, self).__init__(optimSettingsDict,
                                                                    LazeMap1EstimateHyperparameter(),
                                                                    LazeMap1DynamicThresholding()
                                                                    )  
              
class LazeMap2EstimateHyperparameter(AbstractEstimateHyperparameter):
    def __init__(self, r, gSup=None):
        super(LazeMap2EstimateHyperparameter, self).__init__()
        self._r = r
        self._gSup = gSup        
    def EstimateHyperparameter(self, thetaN):
        l0Norm = (thetaN != 0).sum()
        l1Norm = np.abs(thetaN).sum()
        aHat = float(l0Norm)/float(l1Norm)
        wHat = float(l0Norm)/float(thetaN.size)        
        if (self._gSup is None):
            # If gSup isn't specified, return _r
            assert self._r is not None
            return (aHat, wHat, self._r)
        else:
            # If gSup is specified, use that to calculate r
            return (aHat, wHat, float(self._gSup)*2/float(aHat)*(1/float(wHat)-1))

class LazeMap2DynamicThresholding(AbstractDynamicThresholding):
    def __init__(self):
        super(LazeMap2DynamicThresholding, self).__init__()    
    def GetDynamicThreshold(self, hyperparameter, **kwargs):
        assert "alphaVal" in kwargs
        alpha = kwargs['alphaVal']    
        (a, w, r) = hyperparameter # hyperparameter = (a,w,r)        
        cTmp = a*alpha*alpha                
        assert r >= 0        
        if (r >= 1):            
            return ThresholdingHybrid(cTmp + alpha*np.sqrt(2*np.log(r)), cTmp)            
        else:
            return ThresholdingSoft(cTmp)                
        
class EmgaussEmpiricalMapLaze2Reconstructor(EmgaussEmpiricalMapReconstructor):
    def __init__(self, optimSettingsDict, r, gSup):   
        super(EmgaussEmpiricalMapLaze2Reconstructor, self).__init__(optimSettingsDict,
                                                                    LazeMap2EstimateHyperparameter(r, gSup),
                                                                    LazeMap2DynamicThresholding()
                                                                    ) 
