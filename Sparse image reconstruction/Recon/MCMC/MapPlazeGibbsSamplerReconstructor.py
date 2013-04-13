import numpy as np

from PlazeGibbsSamplerReconstructor import PlazeGibbsSamplerReconstructor

class MapPlazeGibbsSamplerReconstructor(PlazeGibbsSamplerReconstructor):
    """ Returns the MAP estimator """
    
    def __init__(self, optimSettingsDict):
        super(MapPlazeGibbsSamplerReconstructor, self).__init__(optimSettingsDict)
        
    """ Implementation of abstract method from AbstractReconstructor """
    
    def Estimate(self, y, convMatrixObj, initializationDict):
        """ 
        Notice that instead of theta0, there's initializationDict. That's because MCMC
        based methods might need more than just theta0.
        """
        self.SamplerSetup(convMatrixObj, initializationDict)
        self.SamplerRun(y)        
        thetaSeq = self.SamplerGet('theta')
        
        posteriorProbSeq = np.zeros((len(thetaSeq),))
        
        for ind in range(len(thetaSeq)):
            posteriorProbSeq[ind] = self.ComputePosteriorProb(y, 
                                                              convMatrixObj, 
                                                              thetaSeq[ind], 
                                                              { 'alpha0': self.hyperparameterPriorDict['alpha0'],
                                                                'alpha1': self.hyperparameterPriorDict['alpha1']
                                                               }
                                                              )
            
        # Find the theta with the largest posterior and return it
        posteriorProbMaxInd = np.argmax(posteriorProbSeq)
        return thetaSeq[posteriorProbMaxInd]
        
            
            
