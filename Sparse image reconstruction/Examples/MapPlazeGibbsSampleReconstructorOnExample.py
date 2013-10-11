import numpy as np
import pylab as plt

from AbstractReconstructorExample import AbstractReconstructorExample
from BlurWithNoiseFactory import BlurWithNoiseFactory
from Recon.MCMC.McmcConstants import McmcConstants
from Recon.MCMC.McmcIterationEvaluator import McmcIterationEvaluator
from Recon.MCMC.MapPlazeGibbsSamplerReconstructor import MapPlazeGibbsSamplerReconstructor
from Sim.AbstractImageGenerator import AbstractImageGenerator
from Sim.NoiseGenerator import AbstractAdditiveNoiseGenerator
from Systems.ComputeEnvironment import ComputeEnvironment
from Systems.ConvolutionMatrixUsingPsf import ConvolutionMatrixUsingPsf
from Systems.ReconstructorPerformanceCriteria import ReconstructorPerformanceCriteria
from Systems.Timer import Timer

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
                              McmcConstants.INPUT_KEY_NUM_ITERATIONS: 2000,
                              McmcConstants.INPUT_KEY_NUM_BURNIN_SAMPLES: 300,
                              #McmcConstants.INPUT_KEY_NUM_THINNING_PERIOD: 1,
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
                                                                   
        with Timer() as t:                                                               
            self._thetaEstimated = reconstructor.Estimate(y, convMatrixObj, initializationDict)

        # Save run variables            
        self._timingMs = t.msecs
        self._channelChain = self.experimentObj.channelChain        
        self._theta = self._channelChain.intermediateOutput[0]
        self._y = y                            
        self._reconstructor = reconstructor
        
    def Plot2d(self, fignumStart):        
        # Plot xTrue
        plt.figure(fignumStart)
        plt.imshow(self.Theta, interpolation='none')
        plt.colorbar()    
        plt.title('xTrue')
        
        # Plot the reconstructed result
        plt.figure()
        plt.imshow(np.reshape(self.reconResult, self.Theta.shape), interpolation='none')
        plt.colorbar()
        plt.title('Reconstructed x')
        
        # Plot yErr and its histogram
        yErr = self.NoisyObs - np.reshape(self._reconstructor.hx, self.NoisyObs.shape)
        plt.figure()
        plt.imshow(yErr, interpolation='none')
        plt.colorbar()
        plt.title('yErr')
                
        plt.figure()
        plt.hist(yErr.flat, 20)
        plt.title('Histogram of yErr')       
    
        plt.show()
        
if __name__ == "__main__": 
    EXPERIMENT_DESC = 'mrfm2d'   
    SNRDB  = 20
    IMAGESHAPE = (32, 32); #(32, 32, 14) 
    
    iterEvaluator = McmcIterationEvaluator(ComputeEnvironment.EPS, 
                                           IMAGESHAPE, # Must be the same as the image size in exReconstructor.experimentObj
                                           xTrue = None,
                                           xFigureNum = -1,
                                           y = None,
                                           countIterationDisplaySet = np.array((300, 1000)),
                                           bVerbose = False
                                           )
    
    exReconstructor = MapPlazeGibbsSampleReconstructorOnExample(iterEvaluator, SNRDB)
    
    exReconstructor.experimentObj = BlurWithNoiseFactory.GetBlurWithNoise(
                                                             EXPERIMENT_DESC, 
                                                             {
                                                              AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB: exReconstructor.snrDb,
                                                              AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE: IMAGESHAPE
                                                              }
                                                             )
    
    exReconstructor.RunExample()
    
    perfCriteria = ReconstructorPerformanceCriteria(
                                                    exReconstructor.Theta, 
                                                    np.reshape(exReconstructor.ThetaEstimated, exReconstructor.Theta.shape)
                                                    )
    
    fmtString = "Reconstruction performance criteria: {0}/{1}/{2}, timing={3:g}s."
    
    print(fmtString.format(
                           perfCriteria.NormalizedL2ErrorNorm(),
                           perfCriteria.NormalizedDetectionError(),
                           perfCriteria.NormalizedL0Norm(),
                           exReconstructor.TimingMs / 1.0e3
                           ))
          
      
    
