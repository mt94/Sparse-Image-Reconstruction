import numpy as np
from multiprocessing import Pool

from AbstractReconstructorExample import AbstractReconstructorExample
from BlurWithNoiseFactory import BlurWithNoiseFactory
from Recon.Gaussian.AbstractEmgaussReconstructor import AbstractEmgaussReconstructor
from Recon.Gaussian.EmgaussIterationsObserver import EmgaussIterationsObserver
from Recon.AbstractInitialEstimator import InitialEstimatorFactory
from Recon.Gaussian.EmgaussEmpiricalMapLazeReconstructor import EmgaussEmpiricalMapLaze1Reconstructor, EmgaussEmpiricalMapLaze2Reconstructor
from Sim.AbstractImageGenerator import AbstractImageGenerator
from Sim.NoiseGenerator import AbstractAdditiveNoiseGenerator 
from Systems.ConvolutionMatrixUsingPsf import ConvolutionMatrixUsingPsf
from Systems.PsfNormalizer import PsfMatrixNormNormalizer
from Systems.ReconstructorPerformanceCriteria import ReconstructorPerformanceCriteria
from Systems.Timer import Timer

class EmgaussEmpiricalMapLazeReconstructorOnExample(AbstractReconstructorExample):
    """
    Demonstrates the iterative thresholding implementation of the MAP reconstructor 
    that uses the LAZE prior. 
    """
    
    _concreteMapReconstructor = {
                                 'map1': EmgaussEmpiricalMapLaze1Reconstructor,
                                 'map2': EmgaussEmpiricalMapLaze2Reconstructor
                                 }
    
    def __init__(self, estimatorDesc, noiseSigma=None, snrDb=None, r=None, gSup=None):
        super(EmgaussEmpiricalMapLazeReconstructorOnExample, self).__init__('Empirical MAP LAZE Reconstructor example')
        
        if estimatorDesc not in EmgaussEmpiricalMapLazeReconstructorOnExample._concreteMapReconstructor:
            raise NotImplementedError(estimatorDesc + ' is an unrecognized MAP reconstructor')
        else:
            self.estimatorDesc = estimatorDesc        
        self.noiseSigma = noiseSigma
        self.snrDb = snrDb        
        self.r = r
        self.gSup = gSup
        
#        self.theta = None
#        self.hyperparameter = None
        self._y = None
        self._thetaEstimated = None
        self._channelChain = None
        self._reconstructor = None
            
    def RunExample(self): 
        if (self.experimentObj is None):
            raise NameError('experimentObj is undefined')
        
        # Run the experiment 
        self.experimentObj.RunExample()

        # Get the inputs needed for the reconstructor                
        y = self.experimentObj.blurredImageWithNoise
        psfRepH = self.experimentObj.channelChain.channelBlocks[1].BlurPsfInThetaFrame # Careful not to use H, which is the convolution matrix
        if (self.noiseSigma is None):
            self.noiseSigma = self.experimentObj.NoiseSigma
                             
        # DEBUG
#        plt.figure(1); plt.imshow(psfRepH); plt.colorbar()
                
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
            AbstractEmgaussReconstructor.INPUT_KEY_ALPHA: self.noiseSigma / np.sqrt(psfSpectralRadius),
            AbstractEmgaussReconstructor.INPUT_KEY_ESTIMATE_HYPERPARAMETERS_ITERATIONS_INTERVAL: 500
        }        
        
        # Get the class constructor that we'd like to call
        clsReconstructor = EmgaussEmpiricalMapLazeReconstructorOnExample._concreteMapReconstructor[self.estimatorDesc]
        
        initialEstimate = InitialEstimatorFactory.GetInitialEstimator('Hty').GetInitialEstimate(y, psfRepH)      
        
        if self.estimatorDesc == 'map2':
            # The MAP2 LAZE reconstructor ctor accepts  accepts optimSettingsDict, r, and gSup
            assert self.r is not None
            assert self.gSup is not None
            reconstructor = clsReconstructor(optimSettingsDict, self.r, self.gSup)
            estimateArg = (initialEstimate,)
        else:
            # By default, assume the ctor only accepts optimSettingsDict
            reconstructor = clsReconstructor(optimSettingsDict)
            estimateArg = (initialEstimate, (initialEstimate != 0)*1)
        
        with Timer() as t:
            self._thetaEstimated = reconstructor.Estimate(y,
                                                          ConvolutionMatrixUsingPsf(psfRepH),
                                                          *estimateArg
                                                          )
            
        # Save run variables            
        self._timingMs = t.msecs
        self._channelChain = self.experimentObj.channelChain        
        self._theta = self._channelChain.intermediateOutput[0]
        self._y = y                            
        self._reconstructor = reconstructor
                
    @property 
    def Hyperparameter(self):
        if (self._reconstructor is None):
            raise NameError('Trying to access uninitialized field')            
        return self._reconstructor.Hyperparameter
        
def RunReconstructor(param):
    """ Runs the MAP1 or MAP2 algorithm depending on the length of param """
    if len(param) == 3:
        [experimentDesc, imageShape, snrDb] = param
        exReconstructor = EmgaussEmpiricalMapLazeReconstructorOnExample('map1', snrDb=snrDb)        
    elif len(param) == 4:
        [experimentDesc, imageShape, gSup, snrDb] = param
        exReconstructor = EmgaussEmpiricalMapLazeReconstructorOnExample('map2', snrDb=snrDb, r=0, gSup=gSup)        
    else:
        raise NotImplementedError()
    
    if (exReconstructor.noiseSigma is not None) and (exReconstructor.noiseSigma >= 0):       
        exReconstructor.experimentObj = BlurWithNoiseFactory.GetBlurWithNoise(experimentDesc, 
                                                                              {
                                                                               AbstractAdditiveNoiseGenerator.INPUT_KEY_SIGMA: exReconstructor.noiseSigma,
                                                                               AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE: imageShape
                                                                               }
                                                                              )
    elif (exReconstructor.snrDb is not None):
        exReconstructor.experimentObj = BlurWithNoiseFactory.GetBlurWithNoise(experimentDesc, 
                                                                              {
                                                                               AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB: exReconstructor.snrDb,
                                                                               AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE: imageShape
                                                                               }
                                                                              )
    else:
        raise NameError('noiseSigma or snrDb must be set') 
        
    exReconstructor.RunExample()
        
    perfCriteria = ReconstructorPerformanceCriteria(exReconstructor.Theta, exReconstructor.ThetaEstimated)            
        
    return {
            'timing_ms': exReconstructor.TimingMs,
            'termination_reason': exReconstructor.TerminationReason,
            'hyperparameter': exReconstructor.Hyperparameter,                  
            # Reconstruction performance criteria
            'normalized_l2_error_norm': perfCriteria.NormalizedL2ErrorNorm(),
            'normalized_detection_error': perfCriteria.NormalizedDetectionError(),
            'normalized_l0_norm': perfCriteria.NormalizedL0Norm()
            }
                
if __name__ == "__main__":
    EXPERIMENT_DESC = 'mrfm3d'
    IMAGESHAPE = (32, 32, 14);  # (32, 32)
    GSUP = 1/np.sqrt(2)
    SNRDB = 20;
    
    # For MAP1
    runArgs = [EXPERIMENT_DESC, IMAGESHAPE, SNRDB]        
    # For MAP2
#    runArgs = [EXPERIMENT_DESC, IMAGESHAPE, GSUP, SNRDB]
    mapDesc = {3: 'MAP1', 4: 'MAP2'}[len(runArgs)]  
    
    bRunPool = False
    NUMPROC = 4
    NUMTASKS = 10
        
    fmtString = "{0}: est. hyper.={1}, perf. criteria={2}/{3}/{4}, timing={5:g}s. {6}"
    
    if not bRunPool:
        mapResult = RunReconstructor(runArgs)
        print(fmtString.format(
                               mapDesc,
                               mapResult['hyperparameter'],
                               mapResult['normalized_l2_error_norm'], mapResult['normalized_detection_error'], mapResult['normalized_l0_norm'],
                               mapResult['timing_ms'] / 1.0e3,
                               mapResult['termination_reason']
                               ))        
    else:
        pool = Pool(processes=NUMPROC)
        resultPool = pool.map(RunReconstructor, [runArgs] * NUMTASKS)
        for aResult in resultPool:
            print(fmtString.format(
                                   mapDesc,
                                   aResult['hyperparameter'],
                                   mapResult['normalized_l2_error_norm'], mapResult['normalized_detection_error'], mapResult['normalized_l0_norm'],
                                   aResult['timing_ms'] / 1.0e3,
                                   aResult['termination_reason']
                                   ))        
