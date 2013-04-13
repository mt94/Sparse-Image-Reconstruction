import numpy as np
import pylab as plt

from AbstractExample import AbstractExample
from GaussianBlurWithNoise import GaussianBlurWithNoise
from Recon.MCMC.McmcConstants import McmcConstants
from Recon.MCMC.McmcIterationEvaluator import McmcIterationEvaluator
from Recon.MCMC.MapPlazeGibbsSamplerReconstructor import MapPlazeGibbsSamplerReconstructor
from Systems.ComputeEnvironment import ComputeEnvironment
#from Systems.PsfLinearDerivative import ConvolutionMatrixZeroMeanUnitNormDerivative
from Systems.ConvolutionMatrixUsingPsf import ConvolutionMatrixUsingPsf

class MapPlazeGibbsSampleReconstructorOnExample(AbstractExample):

    """ Constants """
    wInit = 0.2
    aInit = 1
    varInitMin = 1e-8
    varInitMax = 10
        
    def __init__(self, iterObserver, snrDb=None):        
        super(MapPlazeGibbsSampleReconstructorOnExample, self).__init__('MAP P-LAZE Gibbs Sampler')
        self.iterObserver = iterObserver
        self.snrDb = snrDb
        
    def RunExample(self):
        
        if (self.snrDb is not None):
            self.gbwn = GaussianBlurWithNoise({GaussianBlurWithNoise.INPUT_KEY_SNR_DB: self.snrDb})
        else:
            self.gbwn = GaussianBlurWithNoise({GaussianBlurWithNoise.INPUT_KEY_NOISE_SIGMA: 0})            
        self.gbwn.RunExample()      
                
        # Store the true value in the iterations observer
        self.iterObserver.xTrue = np.array(self.gbwn.channelChain.intermediateOutput[0].flat)
        
        print("SIM: SNR is {0} dB => noise var is {1}".format(self.snrDb, (self.gbwn.NoiseSigma) ** 2))          
        y = self.gbwn.blurredImageWithNoise
        psfRepH = self.gbwn.channelChain.channelBlocks[1].BlurPsfInThetaFrame
#        convMatrixObj = ConvolutionMatrixZeroMeanUnitNormDerivative(psfRepH)
        convMatrixObj = ConvolutionMatrixUsingPsf(psfRepH)
        
        optimSettingsDict = { McmcConstants.INPUT_KEY_EPS: ComputeEnvironment.EPS,
                              McmcConstants.INPUT_KEY_HYPERPARAMETER_PRIOR_DICT: { 'alpha0': 2e-2, 
                                                                                   'alpha1': 2e-2 },
                              McmcConstants.INPUT_KEY_ITERATIONS_OBSERVER: self.iterObserver,
                              McmcConstants.INPUT_KEY_NVERBOSE: 2                          
                             }
        reconstructor = MapPlazeGibbsSamplerReconstructor(optimSettingsDict)
        
        # NOTE: assume theta.shape = y.shape, so M = P
        M = y.size
        
        initTheta = np.reshape(reconstructor.DoSamplingXPrior(self.wInit, self.aInit, M), y.shape)
        initVar = reconstructor.DoSamplingPriorVariancePrior(self.varInitMin, self.varInitMax)
        print("INITIAL CONDS.: hyper: w={0}, a={1}; var: {2}".format(self.wInit, self.aInit, initVar))
        
        initializationDict = { 'init_theta': initTheta, 'init_var': initVar }
        # Plot the initial theta to be used by the Gibbs' Sampler
        plt.figure(2)
        plt.imshow(initTheta, interpolation='none')
        plt.colorbar();
        plt.title("Initial theta for Gibbs' Sampler");
        plt.show(block=False)
                                      
        self.reconResult = reconstructor.Estimate(y, convMatrixObj, initializationDict)
        
if __name__ == "__main__":    
    iterEvaluator = McmcIterationEvaluator(ComputeEnvironment.EPS, 
                                           (32, 32), 
                                           None,
                                           3,
                                           np.array((1, 1000, 2000))
                                           )
    ex = MapPlazeGibbsSampleReconstructorOnExample(iterEvaluator, 20)
    ex.RunExample()      
    plt.show()  
    
