import numpy as np
from multiprocessing import Pool

from Examples.AbstractExample import AbstractExample
from Examples.GaussianBlurWithNoise import GaussianBlurWithNoise
from Recon.Gaussian.AbstractEmgaussReconstructor import AbstractEmgaussReconstructor
from Recon.Gaussian.EmgaussIterationsObserver import EmgaussIterationsObserver
from Recon.AbstractInitialEstimator import InitialEstimatorFactory
from Recon.Gaussian.EmgaussEmpiricalMapLazeReconstructor import EmgaussEmpiricalMapLaze1Reconstructor, EmgaussEmpiricalMapLaze2Reconstructor
from Sim.NoiseGenerator import AbstractAdditiveNoiseGenerator
from Systems.ConvolutionMatrixUsingPsf import ConvolutionMatrixUsingPsf
from Systems.PsfNormalizer import PsfMatrixNormNormalizer

class EmgaussEmpiricalMapLazeReconstructorOnExample(AbstractExample):
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
        if (self.noiseSigma is not None) and (self.noiseSigma >= 0):       
            gbwn = GaussianBlurWithNoise({AbstractAdditiveNoiseGenerator.INPUT_KEY_SIGMA: self.noiseSigma})
        elif (self.snrDb is not None):
            gbwn = GaussianBlurWithNoise({AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB: self.snrDb})
        else:
            raise NameError('noiseSigma or snrDb must be set')  
        gbwn.RunExample()
        
        # Get variables of interest from the gbwn object
        y = gbwn.blurredImageWithNoise
        psfRepH = gbwn.channelChain.channelBlocks[1].BlurPsfInThetaFrame # Careful not to use H, which is the convolution matrix
        if (self.noiseSigma is None):
            self.noiseSigma = gbwn.NoiseSigma
        
        emgIterationsObserver = EmgaussIterationsObserver({
                                                           EmgaussIterationsObserver.INPUT_KEY_TERMINATE_COND: EmgaussIterationsObserver.TERMINATE_COND_THETA_DELTA_L2,
                                                           EmgaussIterationsObserver.INPUT_KEY_TERMINATE_TOL: 1e-7                                                
                                                           })        
        gbNormalizer = PsfMatrixNormNormalizer(1)
        gbNormalizer.NormalizePsf(psfRepH)        
        optimSettingsDict = \
        {
            AbstractEmgaussReconstructor.INPUT_KEY_MAX_ITERATIONS: 1e5,
            AbstractEmgaussReconstructor.INPUT_KEY_ITERATIONS_OBSERVER: emgIterationsObserver,
            AbstractEmgaussReconstructor.INPUT_KEY_TAU: 1 / gbNormalizer.GetSpectralRadiusGramMatrixRowsH(),
            AbstractEmgaussReconstructor.INPUT_KEY_ALPHA: self.noiseSigma / np.sqrt(gbNormalizer.GetSpectralRadiusGramMatrixRowsH()),
            AbstractEmgaussReconstructor.INPUT_KEY_ESTIMATE_HYPERPARAMETERS_ITERATIONS_INTERVAL: 500
        }
        clsReconstructor = EmgaussEmpiricalMapLazeReconstructorOnExample._concreteMapReconstructor[self.estimatorDesc]
        if self.estimatorDesc == 'map2':
            # The MAP2 LAZE reconstructor ctor accepts  accepts optimSettingsDict and r
            assert self.r is not None
            assert self.gSup is not None
            reconstructor = clsReconstructor(optimSettingsDict, self.r, self.gSup)
        else:
            # By default, assume the ctor only accepts optimSettingsDict
            reconstructor = clsReconstructor(optimSettingsDict)
                    
        self._thetaEstimated = reconstructor.Estimate(y,
                                                      ConvolutionMatrixUsingPsf(psfRepH),
                                                      InitialEstimatorFactory.GetInitialEstimator('Hty')
                                                                             .GetInitialEstimate(y, psfRepH) 
                                                      )
                                            
        # Save results
        self._y = y
#        self.theta = gbwn.channelChain.intermediateOutput[0]                
#        self.hyperparameter = reconstructor.Hyperparameter
        self._channelChain = gbwn.channelChain
        self._reconstructor = reconstructor
        
    @property
    def Theta(self):
        if (self._channelChain is None):
            raise NameError('Trying to access uninitialized field')
        return self._channelChain.intermediateOutput[0]
        
    @property
    def ThetaEstimated(self):
        if (self._thetaEstimated is None):
            raise NameError('Trying to access uninitialized field')
        return self._thetaEstimated
        
    @property 
    def Hyperparameter(self):
        if (self._reconstructor is None):
            raise NameError('Trying to access uninitialized field')            
        return self._reconstructor.Hyperparameter
        
    @property
    def TerminationReason(self):
        if (self._reconstructor is None):
            raise NameError('Trying to access uninitialized field')
        return self._reconstructor.TerminationReason            
        
    @property
    def NoisyObs(self):
        if (self._y is None):
            raise NameError('Trying to access uninitialized field')    
        return self._y

def RunMap1(snrDb):
    exMap1 = EmgaussEmpiricalMapLazeReconstructorOnExample('map1', snrDb=snrDb)
    exMap1.RunExample()
    m1EstimationErrorL2Norm = np.linalg.norm(exMap1.Theta - exMap1.ThetaEstimated, 2)
    return {
            'error_l2_norm': m1EstimationErrorL2Norm,
            'hyperparameter': exMap1.Hyperparameter,
            'termination_reason': exMap1.TerminationReason            
            }

def RunMap2(param):
    [snrDb, gSup] = param
    exMap2 = EmgaussEmpiricalMapLazeReconstructorOnExample('map2', snrDb=snrDb, r=0, gSup=gSup)
    exMap2.RunExample()
    m2EstimationErrorL2Norm = np.linalg.norm(exMap2.Theta - exMap2.ThetaEstimated, 2)
    return {
            'error_l2_norm': m2EstimationErrorL2Norm,
            'hyperparameter': exMap2.Hyperparameter,      
            'termination_reason': exMap2.TerminationReason      
            }
                
if __name__ == "__main__":
    SNRDB = 20;
    GSUP = 1/np.sqrt(2)
    
    pool = Pool(processes=3)
    
#    map1Result = RunMap1(SNRDB)
#    print("MAP1: {0}, est. hyperparameter = {1}".format(map1Result['termination_reason'], map1Result['hyperparameter']))
#    print("MAP1: l2 norm of reconstruction error is {0}".format(map1Result['error_l2_norm']))    
#    result1Pool = pool.map(RunMap1, np.repeat(SNRDB, 10))
#    for r1 in result1Pool:
#        print r1
    
#    map2Result = RunMap2([SNRDB, 1/np.sqrt(2)])
#    print("MAP2: {0}, est. hyperparameter = {1}".format(map2Result['termination_reason'], map2Result['hyperparameter']))
#    print("MAP2(gSup=1/sqrt(2)): l2 norm of reconstruction error is {0}".format(map2Result['error_l2_norm']))
    result2Pool = pool.map(RunMap2, [[SNRDB, GSUP]]*10)
    for r2 in result2Pool:
        print r2
        