import numpy as np
from Recon.AbstractEstimateHyperparameter import AbstractEstimateHyperparameter
from Recon.AbstractDynamicThresholding import AbstractDynamicThresholding
from Recon.Gaussian.AbstractEmgaussReconstructor import AbstractEmgaussReconstructor

class EmgaussEmpiricalMapReconstructor(AbstractEmgaussReconstructor):        
    def __init__(self, optimSettingsDict, estHyperparameterObj, dynamicThresholdingObj):
        super(EmgaussEmpiricalMapReconstructor, self).__init__(optimSettingsDict)
        
        assert AbstractEmgaussReconstructor.INPUT_KEY_ALPHA in optimSettingsDict
        assert AbstractEmgaussReconstructor.INPUT_KEY_ESTIMATE_HYPERPARAMETERS_ITERATIONS_INTERVAL in optimSettingsDict
        assert optimSettingsDict[AbstractEmgaussReconstructor.INPUT_KEY_ESTIMATE_HYPERPARAMETERS_ITERATIONS_INTERVAL] >= 1
        
        self._hyperparameter = None
        
        assert isinstance(estHyperparameterObj, AbstractEstimateHyperparameter)
        self._estHyperparameterObj = estHyperparameterObj
        assert isinstance(dynamicThresholdingObj, AbstractDynamicThresholding)
        self._dynamicThresholdingObj = dynamicThresholdingObj
        
        self._thresholder = None

    @property
    def Hyperparameter(self):
        return self._hyperparameter
    
    def EstimateHyperparameter(self, reconArgsN):
        assert reconArgsN is not None
        self._hyperparameter = self._estHyperparameterObj.EstimateHyperparameter(reconArgsN)

    def UpdateThresholder(self):
        self.EstimateHyperparameter(self._reconArgs)
        self._thresholder = self._dynamicThresholdingObj.GetDynamicThreshold(
                                                                             self._hyperparameter, 
                                                                             alphaVal=self._optimSettingsDict[AbstractEmgaussReconstructor.INPUT_KEY_ALPHA]
                                                                             )
    """ Abstract methods implementation """
    
    def SetupBeforeIterations(self):
        self.UpdateThresholder()
                                    
    def Mstep(self, x, numIter):
        # Can return a tuple
        if (np.mod(numIter + 1, 
                   self._optimSettingsDict[AbstractEmgaussReconstructor.INPUT_KEY_ESTIMATE_HYPERPARAMETERS_ITERATIONS_INTERVAL]) == 0):
            # Re-estimate the hyperparameters and update the thresholder
            self.UpdateThresholder()
        return self._thresholder.Apply(x)
    

        

            

            