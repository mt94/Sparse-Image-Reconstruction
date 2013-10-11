#import numpy as np
from multiprocessing import Pool

from AbstractReconstructorExample import AbstractReconstructorExample
from BlurWithNoiseFactory import BlurWithNoiseFactory
from Recon.AbstractInitialEstimator import InitialEstimatorFactory
from Recon.Gaussian.AbstractEmgaussReconstructor import AbstractEmgaussReconstructor
from Recon.Gaussian.EmgaussFixedMstepReconstructor import EmgaussFixedMstepReconstructor
from Recon.Gaussian.EmgaussIterationsObserver import EmgaussIterationsObserver
from Sim.AbstractImageGenerator import AbstractImageGenerator
from Sim.NoiseGenerator import AbstractAdditiveNoiseGenerator
from Systems.ConvolutionMatrixUsingPsf import ConvolutionMatrixUsingPsf
from Systems.PsfNormalizer import PsfMatrixNormNormalizer
from Systems.ReconstructorPerformanceCriteria import ReconstructorPerformanceCriteria
from Systems.Timer import Timer

import Systems.Thresholding as Thresholding

class SimpleThresholdingReconstructorExample(AbstractReconstructorExample):
    """
    Demonstrates an iterative threshold reconstructor where only a simple threshold is used 
    """
    
    _concreteReconstructor = {
                              'landweber': (EmgaussFixedMstepReconstructor, Thresholding.ThresholdingIdentity),
                              'landweber_nonneg': (EmgaussFixedMstepReconstructor, Thresholding.ThresholdingIdentityNonnegative)
                              }
        
    def __init__(self, estimatorDesc, noiseSigma=None, snrDb=None):
        super(SimpleThresholdingReconstructorExample, self).__init__('Simple thresholding reconstructor example')
        if estimatorDesc not in SimpleThresholdingReconstructorExample._concreteReconstructor:
            raise NotImplementedError(estimatorDesc + ' is an unrecognized reconstructor')
        else:
            self.estimatorDesc = estimatorDesc        
        self.noiseSigma = noiseSigma
        self.snrDb = snrDb                 
        
    def RunExample(self):
        if (self.experimentObj is None):
            raise NameError('experimentObj is undefined')
        
        # Run the experiment, which is derived from AbstractExample
        self.experimentObj.RunExample()
        
        # Get the inputs needed for the reconstructor                
        y = self.experimentObj.blurredImageWithNoise
        psfRepH = self.experimentObj.channelChain.channelBlocks[1].BlurPsfInThetaFrame # Careful not to use H, which is the convolution matrix
        if (self.noiseSigma is None):
            self.noiseSigma = self.experimentObj.NoiseSigma        

        emgIterationsObserver = EmgaussIterationsObserver({
                                                           EmgaussIterationsObserver.INPUT_KEY_TERMINATE_COND: EmgaussIterationsObserver.TERMINATE_COND_THETA_DELTA_L2,
                                                           EmgaussIterationsObserver.INPUT_KEY_TERMINATE_TOL: 1e-7                                                
                                                           })
        
        # Create an object that will compute the spectral radius
        gbNormalizer = PsfMatrixNormNormalizer(1)
        gbNormalizer.NormalizePsf(psfRepH)   
        psfSpectralRadius = gbNormalizer.GetSpectralRadiusGramMatrixRowsH()       
                            
        optimSettingsDict = \
        {
            AbstractEmgaussReconstructor.INPUT_KEY_MAX_ITERATIONS: 2e5,
            AbstractEmgaussReconstructor.INPUT_KEY_ITERATIONS_OBSERVER: emgIterationsObserver,
            AbstractEmgaussReconstructor.INPUT_KEY_TAU: 1 / psfSpectralRadius,
        }
        
        # Create the reconstructor object
        params = SimpleThresholdingReconstructorExample._concreteReconstructor[self.estimatorDesc]
        clsReconstructor = params[0]
        clsThresholding = params[1]
        reconstructor = clsReconstructor(optimSettingsDict, clsThresholding().Apply)
        
        # Index the first element, since the Estimate method returns a tuple
        with Timer() as t:
            self._thetaEstimated = reconstructor.Estimate(y,
                                                          ConvolutionMatrixUsingPsf(psfRepH),
                                                          InitialEstimatorFactory.GetInitialEstimator('Hty')
                                                                                 .GetInitialEstimate(y, psfRepH) 
                                                          )[0]
                                            
        # Save run variables
        self._timingMs = t.msecs
        self._channelChain = self.experimentObj.channelChain
        self._y = y     
        self._theta = self._channelChain.intermediateOutput[0]   
        self._reconstructor = reconstructor    
        
def RunAlgo(param):
    [snrDb, experimentDesc, imageShape, estimatorDesc] = param
    exReconstructor = SimpleThresholdingReconstructorExample(estimatorDesc, snrDb=snrDb)
    exReconstructor.experimentObj = BlurWithNoiseFactory.GetBlurWithNoise(
                                                                          experimentDesc, 
                                                                          {
                                                                           AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB: exReconstructor.snrDb,
                                                                           AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE: imageShape
                                                                           }
                                                                          )  
    exReconstructor.RunExample()
        
    perfCriteria = ReconstructorPerformanceCriteria(exReconstructor.Theta, exReconstructor.ThetaEstimated)
    
    return {
            'timing_ms': exReconstructor.TimingMs,
            'termination_reason': exReconstructor.TerminationReason,
            'estimator': estimatorDesc,                                     
            # Reconstruction performance criteria
            'normalized_l2_error_norm': perfCriteria.NormalizedL2ErrorNorm(),
            'normalized_detection_error': perfCriteria.NormalizedDetectionError(),
            'normalized_l0_norm': perfCriteria.NormalizedL0Norm()                   
            }      
            
if __name__ == '__main__':            
    """ 
    Run a comparison between standard Landweber iterations and iterations
    with a non-negative thresholding operation.
    """        
    SNRDB = 20;    
    EXPERIMENT_DESC = 'mrfm3d'    
    IMAGESHAPE = (32, 32, 14); #(32, 32)    

#    RunAlgo([SNRDB, EXPERIMENT_DESC, IMAGESHAPE, 'landweber_nonneg'])
    
    pool = Pool(processes=2)
    
    resultPool = pool.map(
                          RunAlgo, 
                          [[SNRDB, EXPERIMENT_DESC, IMAGESHAPE, 'landweber'], 
                           [SNRDB, EXPERIMENT_DESC, IMAGESHAPE, 'landweber_nonneg']]
                          )
    
    fmtString = "{0}: perf. criteria={1}/{2}/{3}, timing={4:g}s. {5}"
    
    for aResult in resultPool:
        print(fmtString.format(
                               aResult['estimator'],       
                               aResult['normalized_l2_error_norm'], aResult['normalized_detection_error'], aResult['normalized_l0_norm'],                                                                                                                                            
                               aResult['timing_ms'] / 1.0e3,
                               aResult['termination_reason']
                               ))                        