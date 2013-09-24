import numpy as np
from multiprocessing import Pool

from AbstractReconstructorExample import AbstractReconstructorExample
from BlurWithNoiseFactory import BlurWithNoiseFactory
from Recon.AbstractInitialEstimator import InitialEstimatorFactory
from Recon.Gaussian.AbstractEmgaussReconstructor import AbstractEmgaussReconstructor
from Recon.Gaussian.EmgaussFixedMstepReconstructor import EmgaussFixedMstepReconstructor
from Recon.Gaussian.EmgaussIterationsObserver import EmgaussIterationsObserver
from Sim.NoiseGenerator import AbstractAdditiveNoiseGenerator
from Systems.ConvolutionMatrixUsingPsf import ConvolutionMatrixUsingPsf
from Systems.PsfNormalizer import PsfMatrixNormNormalizer
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
#            AbstractEmgaussReconstructor.INPUT_KEY_ALPHA: self.noiseSigma / np.sqrt(psfSpectralRadius),
#            AbstractEmgaussReconstructor.INPUT_KEY_ESTIMATE_HYPERPARAMETERS_ITERATIONS_INTERVAL: 500
        }
        
        # Create the reconstructor object
        params = SimpleThresholdingReconstructorExample._concreteReconstructor[self.estimatorDesc]
        clsReconstructor = params[0]
        clsThresholding = params[1]
        reconstructor = clsReconstructor(optimSettingsDict, clsThresholding().Apply)
        
        # Index the first element, since the Estimate method returns a tuple
        self._thetaEstimated = reconstructor.Estimate(y,
                                                      ConvolutionMatrixUsingPsf(psfRepH),
                                                      InitialEstimatorFactory.GetInitialEstimator('Hty')
                                                                             .GetInitialEstimate(y, psfRepH) 
                                                      )[0]
                                            
        # Save results
        self._channelChain = self.experimentObj.channelChain
        
        self._y = y     
        self._theta = self._channelChain.intermediateOutput[0]   
        self._reconstructor = reconstructor    
        
def RunAlgo(param):
    [snrDb, experimentDesc, estimatorDesc] = param
    exReconstructor = SimpleThresholdingReconstructorExample(estimatorDesc, snrDb=snrDb)
    exReconstructor.experimentObj = BlurWithNoiseFactory.GetBlurWithNoise(experimentDesc, 
                                                                          {AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB: exReconstructor.snrDb}
                                                                          )  
    exReconstructor.RunExample()
    return {
            'estimator': estimatorDesc,
            'error_l2_norm': np.linalg.norm(exReconstructor.Theta - exReconstructor.ThetaEstimated, 2),            
            'termination_reason': exReconstructor.TerminationReason            
            }      
            
if __name__ == '__main__':            
    """ 
    Run a comparison between standard Landweber iterations and iterations
    with a non-negative thresholding operation.
    """        
    SNRDB = 20;        

    pool = Pool(processes=2)
    resultPool = pool.map(RunAlgo, [[SNRDB, 'gaussian2d', 'landweber'], [SNRDB, 'gaussian2d', 'landweber_nonneg']])
    for aResult in resultPool:
        print("{0}: l2 norm recon err={1}. {2}".format(
                                                       aResult['estimator'],                                                       
                                                       aResult['error_l2_norm'],
                                                       aResult['termination_reason']
                                                       ))            
        
    
        
