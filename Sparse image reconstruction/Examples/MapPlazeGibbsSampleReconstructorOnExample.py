import numpy as np
import pylab as plt
from multiprocessing import Pool

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
        
    def __init__(self, iterObserver, optimSettingsDict, snrDb=None):        
        super(MapPlazeGibbsSampleReconstructorOnExample, self).__init__('MAP P-LAZE Gibbs Sampler')
        self.iterObserver = iterObserver
        self.snrDb = snrDb
        self.optimSettingsDict = optimSettingsDict
    
    def RunExample(self):        
        if (self.experimentObj is None):
            raise NameError('experimentObj is undefined')
        
        if not self.experimentObj.RunAlready:
            self.experimentObj.RunExample() 
                        
        xTrue = self.experimentObj.channelChain.intermediateOutput[0]        
        y = self.experimentObj.blurredImageWithNoise
        psfRepH = self.experimentObj.channelChain.channelBlocks[1].BlurPsfInThetaFrame
#        convMatrixObj = ConvolutionMatrixZeroMeanUnitNormDerivative(psfRepH)
        convMatrixObj = ConvolutionMatrixUsingPsf(psfRepH)
                
        self.iterObserver.xTrue = np.array(xTrue.flat)
        self.iterObserver.y = y
        
        print("SIM: SNR is {0} dB => noise var is {1}".format(self.snrDb, (self.experimentObj.NoiseSigma) ** 2))          

        hyperparamPriorAlpha = self.optimSettingsDict.get('prior_alpha', 1e-5)
        
        optimSettingsDict = { McmcConstants.INPUT_KEY_EPS: ComputeEnvironment.EPS,
                              McmcConstants.INPUT_KEY_HYPERPARAMETER_PRIOR_DICT: { 'alpha0': hyperparamPriorAlpha, 
                                                                                   'alpha1': hyperparamPriorAlpha },
                              McmcConstants.INPUT_KEY_ITERATIONS_OBSERVER: self.iterObserver,                              
                              McmcConstants.INPUT_KEY_NUM_ITERATIONS: self.optimSettingsDict.get(McmcConstants.INPUT_KEY_NUM_ITERATIONS, 1000),
                              McmcConstants.INPUT_KEY_NUM_BURNIN_SAMPLES: self.optimSettingsDict.get(McmcConstants.INPUT_KEY_NUM_BURNIN_SAMPLES, 300),
                              McmcConstants.INPUT_KEY_NUM_THINNING_PERIOD: self.optimSettingsDict.get(McmcConstants.INPUT_KEY_NUM_THINNING_PERIOD, 1),
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
        
def RunReconstructor(param, bPlot=False):
    [iterationParams, experimentDesc, imageShape, snrDb, numNonzero] = param
          
                                        
    iterEvaluator = McmcIterationEvaluator(ComputeEnvironment.EPS, 
                                           imageShape, # Must be the same as the image size in exReconstructor.experimentObj
                                           xTrue = None,
                                           xFigureNum = -1,
                                           y = None,
                                           countIterationDisplaySet = np.array((300, 1000)),
                                           bVerbose = False
                                           )
    
    exReconstructor = MapPlazeGibbsSampleReconstructorOnExample(
                                                                iterEvaluator,
                                                                { 
                                                                 McmcConstants.INPUT_KEY_NUM_ITERATIONS: iterationParams[0],
                                                                 McmcConstants.INPUT_KEY_NUM_BURNIN_SAMPLES: iterationParams[1]
                                                                 },
                                                                snrDb
                                                                )
    
    exReconstructor.experimentObj = BlurWithNoiseFactory.GetBlurWithNoise(
                                                             experimentDesc, 
                                                             {
                                                              AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB: exReconstructor.snrDb,
                                                              AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE: imageShape,
                                                              AbstractImageGenerator.INPUT_KEY_NUM_NONZERO: numNonzero
                                                              }
                                                             )
    
    exReconstructor.RunExample()
    
    perfCriteria = ReconstructorPerformanceCriteria(
                                                    exReconstructor.Theta, 
                                                    np.reshape(exReconstructor.ThetaEstimated, exReconstructor.Theta.shape)
                                                    )
    return {
            'timing_ms': exReconstructor.TimingMs,            
            # Reconstruction performance criteria
            'normalized_l2_error_norm': perfCriteria.NormalizedL2ErrorNorm(),
            'normalized_detection_error': perfCriteria.NormalizedDetectionError(),
            'normalized_l0_norm': perfCriteria.NormalizedL0Norm()
            }
    
            
if __name__ == "__main__": 
    EXPERIMENT_DESC = 'mrfm2d'   
    SNRDB  = 2
    IMAGESHAPE = (32, 32); #(32, 32, 14) 
    NUM_NONZERO = 16
    
    runArgs = [(1000, 300), EXPERIMENT_DESC, IMAGESHAPE, SNRDB, NUM_NONZERO]
    
    NUMPROC = 3
    NUMTASKS = 30
    
    fmtString = "Iter. param: ({0},{1}), perf. criteria: {2}/{3}/{4}, timing={5:g}s."

    pool = Pool(processes=NUMPROC)
    resultPool = pool.map(RunReconstructor, [runArgs] * NUMTASKS)
    
    for aResult in resultPool:
        print(fmtString.format(
                               runArgs[0][0], runArgs[0][1],
                               aResult['normalized_l2_error_norm'], aResult['normalized_detection_error'], aResult['normalized_l0_norm'],
                               aResult['timing_ms'] / 1.0e3                               
                               )) 
            
          
      
    
