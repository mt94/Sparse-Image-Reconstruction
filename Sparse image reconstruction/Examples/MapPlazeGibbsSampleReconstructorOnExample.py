import numpy as np
import pylab as plt

from AbstractReconstructorExample import AbstractReconstructorExample
from BlurWithNoiseFactory import BlurWithNoiseFactory
from Recon.MCMC.McmcConstants import McmcConstants
from Recon.MCMC.McmcIterationEvaluator import McmcIterationEvaluator
from Recon.MCMC.MapPlazeGibbsSamplerReconstructor import MapPlazeGibbsSamplerReconstructor
from Sim.NoiseGenerator import AbstractAdditiveNoiseGenerator
from Systems.ComputeEnvironment import ComputeEnvironment
#from Systems.PsfLinearDerivative import ConvolutionMatrixZeroMeanUnitNormDerivative
from Systems.ConvolutionMatrixUsingPsf import ConvolutionMatrixUsingPsf

class MapPlazeGibbsSampleReconstructorOnExample(AbstractReconstructorExample):

    """ Constants """
    wInit = 0.1
    aInit = 0.1
    varInitMin = 1e-8
    varInitMax = 10
        
    def __init__(self, iterObserver, snrDb=None):        
        super(MapPlazeGibbsSampleReconstructorOnExample, self).__init__('MAP P-LAZE Gibbs Sampler')
        self.iterObserver = iterObserver
        self.snrDb = snrDb
        
    def RunExample(self):        
        if (self.experimentObj is None):
            raise NameError('experimentObj is undefined')
        
        self.experimentObj.RunExample() 
                        
        xTrue = self.experimentObj.channelChain.intermediateOutput[0]        
        y = self.experimentObj.blurredImageWithNoise
        psfRepH = self.experimentObj.channelChain.channelBlocks[1].BlurPsfInThetaFrame
#        convMatrixObj = ConvolutionMatrixZeroMeanUnitNormDerivative(psfRepH)
        convMatrixObj = ConvolutionMatrixUsingPsf(psfRepH)
                
        self.iterObserver.xTrue = np.array(xTrue.flat)
        self.iterObserver.y = y
        
        print("SIM: SNR is {0} dB => noise var is {1}".format(self.snrDb, (self.experimentObj.NoiseSigma) ** 2))          
                
        optimSettingsDict = { McmcConstants.INPUT_KEY_EPS: ComputeEnvironment.EPS,
                              McmcConstants.INPUT_KEY_HYPERPARAMETER_PRIOR_DICT: { 'alpha0': 1e-2, 
                                                                                   'alpha1': 1e-2 },
                              McmcConstants.INPUT_KEY_ITERATIONS_OBSERVER: self.iterObserver,                              
                              McmcConstants.INPUT_KEY_NUM_ITERATIONS: 3000,
                              McmcConstants.INPUT_KEY_NUM_THINNING_PERIOD: 5,
                              McmcConstants.INPUT_KEY_NVERBOSE: 0                 
                             }
        reconstructor = MapPlazeGibbsSamplerReconstructor(optimSettingsDict)
        
        # NOTE: assume theta.shape = y.shape, so M = P
        M = y.size
        
        initTheta = np.reshape(reconstructor.DoSamplingXPrior(self.wInit, self.aInit, M), y.shape)
        initVar = reconstructor.DoSamplingPriorVariancePrior(self.varInitMin, self.varInitMax)
#         initVar = (self.experimentObj.NoiseSigma) ** 2
#         initVar = 1e-2
        
        print("INITIAL CONDS.: hyper: w={0}, a={1}; var: {2}".format(self.wInit, self.aInit, initVar))
        
        initializationDict = { 'init_theta': initTheta, 'init_var': initVar }                                                              
        self.reconResult = reconstructor.Estimate(y, convMatrixObj, initializationDict)
          
        # Plot xTrue
        plt.figure(1); plt.imshow(xTrue, interpolation='none'); plt.colorbar()    
        plt.title('xTrue')
        # Plot the reconstructed result
        plt.figure(); plt.imshow(np.reshape(self.reconResult, xTrue.shape), interpolation='none'); plt.colorbar()
        plt.title('Reconstructed x')
        # Plot yErr and its histogram
        yErr = y - np.reshape(reconstructor.hx, y.shape)
        plt.figure(); plt.imshow(yErr, interpolation='none'); plt.colorbar()
        plt.title('yErr')
        plt.figure(); plt.hist(yErr.flat, 20); plt.title('Histogram of yErr')
       
        
if __name__ == "__main__":    
    iterEvaluator = McmcIterationEvaluator(ComputeEnvironment.EPS, 
                                           (42, 42), # Must be the same as the image size in ex.experimentObj
                                           None,
                                           10,
                                           None,
                                           np.array((300, 1000))
                                           #np.arange(100, 1000, 100)
                                           #np.arange(1000)
                                           )
    ex = MapPlazeGibbsSampleReconstructorOnExample(iterEvaluator, 20)
    ex.experimentObj = BlurWithNoiseFactory.GetBlurWithNoise('mrfm2d', 
                                                             {AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB: ex.snrDb}
                                                             )
    
    ex.RunExample()
          
    plt.show()  
    
