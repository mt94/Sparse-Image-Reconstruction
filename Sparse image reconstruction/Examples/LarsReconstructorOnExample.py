import cPickle as pickle
import numpy as np
import pylab as plt

from AbstractExample import AbstractExample
from GaussianBlurWithNoise import GaussianBlurWithNoise
from Recon.Stagewise.LarsConstants import LarsConstants
from Recon.Stagewise.LarsIterationEvaluator import LarsIterationEvaluator
from Recon.Stagewise.LarsReconstructorFactory import LarsReconstructorFactory
from Systems.ComputeEnvironment import ComputeEnvironment
from Systems.PsfLinearDerivative import ConvolutionMatrixZeroMeanUnitNormDerivative

class LarsReconstructorOnExample(AbstractExample):
    
    GAUSSIAN_BLUR_WITH_NOISE_DUMP_FILE = 'c:\Users\Mike\LeastAngleRegressionOnExampleGbwn.dump'
    
    def __init__(self, reconstructorDesc, snrDb=None, bRestoreSim=False):
        super(LarsReconstructorOnExample, self).__init__('LARS example')
        self.reconstructorDesc = reconstructorDesc
        self.snrDb = snrDb
        self.bRestoreSim = bRestoreSim
        self.reconResult = None
        self.gbwn = None        
        
    def RunExample(self):
        
        if (not self.bRestoreSim):
            if (self.snrDb is not None):
                self.gbwn = GaussianBlurWithNoise({GaussianBlurWithNoise.INPUT_KEY_SNR_DB: self.snrDb})
            else:
                self.gbwn = GaussianBlurWithNoise({GaussianBlurWithNoise.INPUT_KEY_NOISE_SIGMA: 0})            
            self.gbwn.RunExample()
            gbwn = self.gbwn
            pickle.dump(gbwn, open(LarsReconstructorOnExample.GAUSSIAN_BLUR_WITH_NOISE_DUMP_FILE, 'wb'))
        else:        
            self.gbwn = pickle.load(open(LarsReconstructorOnExample.GAUSSIAN_BLUR_WITH_NOISE_DUMP_FILE, 'rb'))    
                                
        y = self.gbwn.blurredImageWithNoise
        psfRepH = self.gbwn.channelChain.channelBlocks[1].BlurPsfInThetaFrame # Careful not to use H, which is the convolution matrix
        iterObserver = LarsIterationEvaluator(ComputeEnvironment.EPS, self.gbwn.NoiseSigma)
        
        optimSettingsDict = { 
                              LarsConstants.INPUT_KEY_MAX_ITERATIONS: 10,
                              LarsConstants.INPUT_KEY_EPS: ComputeEnvironment.EPS,
                              LarsConstants.INPUT_KEY_NVERBOSE: 1,
                              LarsConstants.INPUT_KEY_ENFORCE_ONEATATIME_JOIN: True,
                              LarsConstants.INPUT_KEY_ITERATIONS_OBSERVER: iterObserver 
                             }
        reconstructor = LarsReconstructorFactory.GetReconstructor(self.reconstructorDesc, optimSettingsDict)
        
#        gbNormalizer = PsfColumnNormNormalizer(1)
#        psfRepHWithUnitColumnNorm = gbNormalizer.NormalizePsf(psfRepH)                
#        self.reconResult = reconstructor.Estimate(y, ConvolutionMatrixUsingPsf(psfRepHWithUnitColumnNorm))

        yZeroMean = y - np.mean(y.flat)*np.ones(y.shape)
        assert np.mean(yZeroMean.flat) < ComputeEnvironment.EPS
        self.reconResult = reconstructor.Estimate(yZeroMean, ConvolutionMatrixZeroMeanUnitNormDerivative(psfRepH))
        
        return iterObserver
        
if __name__ == "__main__":
    
    ex = LarsReconstructorOnExample('lars_lasso', snrDb=20, bRestoreSim=False) # Use bRestoreSim for debugging problem cases
    iterObserver = ex.RunExample()
    
    activeSetDisplay = ["{0}".format(x) for x in ex.reconResult[LarsConstants.OUTPUT_KEY_ACTIVESET]]
    print("Active set: {0}".format(" ".join(activeSetDisplay)))
        
    if LarsConstants.OUTPUT_KEY_SIGN_VIOLATION_NUMITER in ex.reconResult:
        print("Number of Lars-Lasso iteration(s) with a sign violation: {0}".format(ex.reconResult[LarsConstants.OUTPUT_KEY_SIGN_VIOLATION_NUMITER]))
    
    for h in iterObserver.HistoryState:
        print("rss: {0:.4e}, SURE criterion: {1:.5f}, Max corr: {2:.5f}".format(h[LarsIterationEvaluator.STATE_KEY_FIT_RSS], 
                                                                                h[LarsIterationEvaluator.STATE_KEY_CRITERION_SURE], 
                                                                                h[LarsIterationEvaluator.STATE_KEY_CORRHATABS_MAX]))
                
    # Create plots

    # In order to remove the shift, must access the Blur block in the channel chain
    blurredImageWithNoiseForDisplay = ex.gbwn.channelChain \
                                             .channelBlocks[1] \
                                             .RemoveShiftFromBlurredImage(ex.gbwn.blurredImageWithNoise)
                                             
    blurredImageWithNoiseForDisplayZeroMean = blurredImageWithNoiseForDisplay - \
        np.mean(blurredImageWithNoiseForDisplay.flat)*np.ones(blurredImageWithNoiseForDisplay.shape)        
    assert np.mean(blurredImageWithNoiseForDisplayZeroMean) < ComputeEnvironment.EPS

    estimatedMu = np.reshape(ex.reconResult[LarsConstants.OUTPUT_KEY_MUHAT_ACTIVESET], 
                             ex.gbwn.blurredImageWithNoise.shape)
    estimatedMuForDisplay = ex.gbwn.channelChain \
                                   .channelBlocks[1] \
                                   .RemoveShiftFromBlurredImage(estimatedMu)    

    if False:
        plt.ioff()
                                                             
        plt.figure(1)
        plt.imshow(blurredImageWithNoiseForDisplayZeroMean)
        plt.colorbar()
        
        plt.figure(2)
        plt.imshow(estimatedMuForDisplay)
        plt.colorbar()
        
        plt.show()
    
