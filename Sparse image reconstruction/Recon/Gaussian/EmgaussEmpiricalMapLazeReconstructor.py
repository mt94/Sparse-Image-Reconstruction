import numpy as np

from ...Recon.AbstractEstimateHyperparameter import AbstractEstimateHyperparameter
from ...Recon.AbstractDynamicThresholding import AbstractDynamicThresholding
from ...Recon.Gaussian.EmgaussEmpiricalMapReconstructor import EmgaussEmpiricalMapReconstructor
from ...Systems.Thresholding import ThresholdingHybrid, ThresholdingSoft, ThresholdingMap1CompositeSmallWeight, ThresholdingMap1CompositeLargeWeight

class LazeMap1EstimateHyperparameter(AbstractEstimateHyperparameter):
    def __init__(self):
        super(LazeMap1EstimateHyperparameter, self).__init__()
    def EstimateHyperparameter(self, args):       
        # For the MAP1 algorithm, both thetaTildeN and nonzeroIndicatorN are used
        if len(args) == 2: 
            thetaTildeN = args[0]
            nonzeroIndicatorN = args[1]
            nonzeroIndicatorL0Norm = (nonzeroIndicatorN != 0).sum()
            thetaTildeL1Norm = np.abs(thetaTildeN).sum()        
            return (float(thetaTildeN.size)/float(thetaTildeL1Norm), float(nonzeroIndicatorL0Norm)/float(thetaTildeN.size))
        else:
            raise TypeError('Unexpected number of args to EstimateHyperparameter: ' + str(len(args)))

class LazeMap1DynamicThresholding(AbstractDynamicThresholding):    
    def __init__(self):
        super(LazeMap1DynamicThresholding, self).__init__()
    def GetDynamicThreshold(self, hyperparameter, **kwargs):
        assert "alphaVal" in kwargs
        alpha = kwargs['alphaVal']
        (a, w) = hyperparameter # hyperparameter = (a,w)        
        #cTmp = a*alpha*alpha                
        if (w <= 0.5):     
            #assert w > 0       
            #return ThresholdingHybrid(cTmp + alpha*np.sqrt(2*np.log((1-w)/w)), cTmp)      
            return ThresholdingMap1CompositeSmallWeight(a, w, alpha)     
        else:
            #return ThresholdingSoft(cTmp)            
            return ThresholdingMap1CompositeLargeWeight(a, w, alpha)         

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
    def EstimateHyperparameter(self, args):
        if len(args) == 1:
            reconArgsN = args[0]
            l0Norm = (reconArgsN != 0).sum()
            l1Norm = np.abs(reconArgsN).sum()
            aHat = float(l0Norm)/float(l1Norm)
            wHat = float(l0Norm)/float(reconArgsN.size)        
            if (self._gSup is None):
                # If gSup isn't specified, return _r
                assert self._r is not None
                return (aHat, wHat, self._r)
            else:
                # If gSup is specified, use that to calculate r
                return (aHat, wHat, float(self._gSup)*2/float(aHat)*(1/float(wHat)-1))
        else:
            raise TypeError('Unexpected number of args to EstimateHyperparameter: ' + str(len(args)))

class LazeMap2DynamicThresholding(AbstractDynamicThresholding):
    def __init__(self):
        super(LazeMap2DynamicThresholding, self).__init__()    
    def GetDynamicThreshold(self, hyperparameter, **kwargs):
        assert "alphaVal" in kwargs
        alpha = kwargs['alphaVal']    
        (a, _, r) = hyperparameter # hyperparameter = (a,w,r)        
        cTmp = a*alpha*alpha                
        assert r >= 0        
        if (r >= 1):            
            return ThresholdingHybrid(cTmp + alpha*np.sqrt(2*np.log(r)), cTmp, bReturnTuple=True)            
        else:
            return ThresholdingSoft(cTmp, bReturnTuple=True)                
        
class EmgaussEmpiricalMapLaze2Reconstructor(EmgaussEmpiricalMapReconstructor):
    def __init__(self, optimSettingsDict, r, gSup):   
        super(EmgaussEmpiricalMapLaze2Reconstructor, self).__init__(optimSettingsDict,
                                                                    LazeMap2EstimateHyperparameter(r, gSup),
                                                                    LazeMap2DynamicThresholding()
                                                                    ) 
