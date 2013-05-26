import numpy as np
#import pylab as plt
from multiprocessing import Pool

from AbstractReconstructorExample import AbstractReconstructorExample
from BlurWithNoiseFactory import BlurWithNoiseFactory
from Recon.Gaussian.AbstractEmgaussReconstructor import AbstractEmgaussReconstructor
from Recon.Gaussian.EmgaussIterationsObserver import EmgaussIterationsObserver
from Recon.AbstractInitialEstimator import InitialEstimatorFactory
from Recon.Gaussian.EmgaussEmpiricalMapLazeReconstructor import EmgaussEmpiricalMapLaze1Reconstructor, EmgaussEmpiricalMapLaze2Reconstructor
from Sim.NoiseGenerator import AbstractAdditiveNoiseGenerator
from Systems.ConvolutionMatrixUsingPsf import ConvolutionMatrixUsingPsf
from Systems.PsfNormalizer import PsfMatrixNormNormalizer

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
        # Run the experiment 
        self.experimentObj.RunExample()
                
        y = self.experimentObj.blurredImageWithNoise
        psfRepH = self.experimentObj.channelChain.channelBlocks[1].BlurPsfInThetaFrame # Careful not to use H, which is the convolution matrix
        if (self.noiseSigma is None):
            self.noiseSigma = self.experimentObj.NoiseSigma
        
        emgIterationsObserver = EmgaussIterationsObserver({
                                                           EmgaussIterationsObserver.INPUT_KEY_TERMINATE_COND: EmgaussIterationsObserver.TERMINATE_COND_THETA_DELTA_L2,
                                                           EmgaussIterationsObserver.INPUT_KEY_TERMINATE_TOL: 1e-7                                                
                                                           }) 
        
        # Create an object that will compute the spectral radius
#        plt.figure(1); plt.imshow(psfRepH); plt.colorbar()
        gbNormalizer = PsfMatrixNormNormalizer(1)
        gbNormalizer.NormalizePsf(psfRepH)        
                
        optimSettingsDict = \
        {
            AbstractEmgaussReconstructor.INPUT_KEY_MAX_ITERATIONS: 2e5,
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
        self._channelChain = self.experimentObj.channelChain
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

def RunMap1(param):
    [snrDb, experimentDesc] = param
    exReconstructor = EmgaussEmpiricalMapLazeReconstructorOnExample('map1', snrDb=snrDb)

    if (exReconstructor.noiseSigma is not None) and (exReconstructor.noiseSigma >= 0):       
        exReconstructor.experimentObj = BlurWithNoiseFactory.GetBlurWithNoise(experimentDesc, 
                                                                              {AbstractAdditiveNoiseGenerator.INPUT_KEY_SIGMA: exReconstructor.noiseSigma}
                                                                              )
    elif (exReconstructor.snrDb is not None):
        exReconstructor.experimentObj = BlurWithNoiseFactory.GetBlurWithNoise(experimentDesc, 
                                                                              {AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB: exReconstructor.snrDb}
                                                                              )
    else:
        raise NameError('noiseSigma or snrDb must be set') 
    
    exReconstructor.RunExample()
        
    return {
            'error_l2_norm': np.linalg.norm(exReconstructor.Theta - exReconstructor.ThetaEstimated, 2),
            'hyperparameter': exReconstructor.Hyperparameter,
            'termination_reason': exReconstructor.TerminationReason            
            }

    
def RunMap2(param):
    [snrDb, gSup, experimentDesc] = param
    exReconstructor = EmgaussEmpiricalMapLazeReconstructorOnExample('map2', snrDb=snrDb, r=0, gSup=gSup)
    
    if (exReconstructor.noiseSigma is not None) and (exReconstructor.noiseSigma >= 0):       
        exReconstructor.experimentObj = BlurWithNoiseFactory.GetBlurWithNoise(experimentDesc, 
                                                                              {AbstractAdditiveNoiseGenerator.INPUT_KEY_SIGMA: exReconstructor.noiseSigma}
                                                                              )
    elif (exReconstructor.snrDb is not None):
        exReconstructor.experimentObj = BlurWithNoiseFactory.GetBlurWithNoise(experimentDesc, 
                                                                              {AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB: exReconstructor.snrDb}
                                                                              )
    else:
        raise NameError('noiseSigma or snrDb must be set') 
            
    exReconstructor.RunExample()
        
    return {
            'error_l2_norm': np.linalg.norm(exReconstructor.Theta - exReconstructor.ThetaEstimated, 2),
            'hyperparameter': exReconstructor.Hyperparameter,      
            'termination_reason': exReconstructor.TerminationReason      
            }
                
if __name__ == "__main__":
    SNRDB = 20;
    GSUP = 1/np.sqrt(2)
    BLURDESC = 'gaussian'
    
    mapDesc = 'map1'
    bRunPool = True
        
    if not bRunPool:
        if mapDesc == 'map1':
            mapResult = RunMap1([SNRDB, BLURDESC])
        else:
            mapResult = RunMap2([SNRDB, GSUP, BLURDESC])
        print("{0}: est. hyper.={1}, l2 norm recon err={2}. {3}".format(
                                                                        mapDesc,
                                                                        mapResult['hyperparameter'],
                                                                        mapResult['error_l2_norm'],
                                                                        mapResult['termination_reason']
                                                                        ))        
    else:
        pool = Pool(processes=4)
        if mapDesc == 'map1':
            resultPool = pool.map(RunMap1, [[SNRDB, BLURDESC]]*10)
        else:
            resultPool = pool.map(RunMap2, [[SNRDB, GSUP, BLURDESC]]*10)            
        for aResult in resultPool:
            print("{0}: est. hyper.={1}, l2 norm recon err={2}. {3}".format(
                                                                            mapDesc,
                                                                            aResult['hyperparameter'],
                                                                            aResult['error_l2_norm'],
                                                                            aResult['termination_reason']
                                                                            ))        

#    plt.show()