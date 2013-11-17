import numpy as np

from PlazeGibbsSamplerReconstructor import PlazeGibbsSamplerReconstructor

class MapPlazeGibbsSamplerReconstructor(PlazeGibbsSamplerReconstructor):
    """ Returns the MAP estimator """
    
    def __init__(self, Eps, optimSettingsDict):
        super(MapPlazeGibbsSamplerReconstructor, self).__init__(optimSettingsDict)
        self.Eps = Eps
        self.selectionResult = None
        
    """ Implementation of abstract method from AbstractReconstructor """
    
    def SelectOptimum(self, y, convMatrixObj, maxNumSamples):
        """
        Before running this, assume that SamplerRun has been called.
        """
        if not self.bSamplerRun:
            return None
        
        thetaSeq = self.SamplerGet('theta', maxNumSamples)
        
        posteriorProbSeq = np.zeros((len(thetaSeq),))
        
        for ind in range(len(thetaSeq)):
            posteriorProbSeq[ind] = self.ComputePosteriorProb(y, 
                                                              convMatrixObj, 
                                                              thetaSeq[ind], 
                                                              { 'alpha0': self.hyperparameterPriorDict['alpha0'],
                                                                'alpha1': self.hyperparameterPriorDict['alpha1']
                                                               },
                                                              self.Eps
                                                              )
            
        # Find the theta with the largest posterior and return it
        posteriorProbMaxInd = np.argmax(posteriorProbSeq)
        self.selectionResult = (thetaSeq[posteriorProbMaxInd], posteriorProbSeq, posteriorProbMaxInd)
        return thetaSeq[posteriorProbMaxInd]
        
    def Estimate(self, y, convMatrixObj, initializationDict, maxNumSamples = float('inf')):
        """ 
        Notice that instead of theta0, there's initializationDict. That's because MCMC
        based methods might need more than just theta0.
        """
        self.SamplerSetup(convMatrixObj, initializationDict)
        self.SamplerRun(y)        
        self.SelectOptimum(y, convMatrixObj, maxNumSamples)
        return self.selectionResult[0]

        
            
            
