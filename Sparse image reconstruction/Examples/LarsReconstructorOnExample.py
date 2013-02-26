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
    """
    Demonstrates Lars-based reconstructors
    """
    
    GAUSSIAN_BLUR_WITH_NOISE_DUMP_FILE = 'c:\Users\Mike\LeastAngleRegressionOnExampleGbwn.dump'
    
    def __init__(self, reconstructorDesc, iterObserver, snrDb=None, bRestoreSim=False):
        super(LarsReconstructorOnExample, self).__init__('LARS example')
        self.reconstructorDesc = reconstructorDesc
        self.iterObserver = iterObserver
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
        convMatrixObj = ConvolutionMatrixZeroMeanUnitNormDerivative(psfRepH)
        
        # Need to set variables in the iteration observer
        iterObserver.NoiseSigma = self.gbwn.NoiseSigma
        iterObserver.ThetaTrue = np.array(self.gbwn.channelChain.intermediateOutput[0].flat)
        muTrue = convMatrixObj.Multiply(self.gbwn.channelChain.intermediateOutput[0])
        iterObserver.MuTrue = np.array(muTrue.flat)
        
        optimSettingsDict = { 
                              LarsConstants.INPUT_KEY_MAX_ITERATIONS: 20,
                              LarsConstants.INPUT_KEY_EPS: ComputeEnvironment.EPS,
                              LarsConstants.INPUT_KEY_NVERBOSE: 0,
                              LarsConstants.INPUT_KEY_ENFORCE_ONEATATIME_JOIN: True,
                              LarsConstants.INPUT_KEY_ITERATIONS_OBSERVER: iterObserver 
                             }
        reconstructor = LarsReconstructorFactory.GetReconstructor(self.reconstructorDesc, optimSettingsDict)
        
#        gbNormalizer = PsfColumnNormNormalizer(1)
#        psfRepHWithUnitColumnNorm = gbNormalizer.NormalizePsf(psfRepH)                
#        self.reconResult = reconstructor.Estimate(y, ConvolutionMatrixUsingPsf(psfRepHWithUnitColumnNorm))

        yZeroMean = y - np.mean(y.flat)*np.ones(y.shape)
        assert np.mean(yZeroMean.flat) < ComputeEnvironment.EPS
        self.reconResult = reconstructor.Estimate(yZeroMean, convMatrixObj)
                
if __name__ == "__main__":
    
    MyReconstructorDesc = 'lars_lasso'
    
    iterObserver = LarsIterationEvaluator(ComputeEnvironment.EPS)
    
    if MyReconstructorDesc == 'lars_lasso':
        iterObserver.TrackCriterionL1Sure = True
            
    ex = LarsReconstructorOnExample(MyReconstructorDesc, iterObserver, snrDb=25, bRestoreSim=False) # Use bRestoreSim for debugging problem cases
    ex.RunExample()
    
    activeSetDisplay = ["{0}".format(x) for x in ex.reconResult[LarsConstants.OUTPUT_KEY_ACTIVESET]]
    print("Active set: {0}".format(" ".join(activeSetDisplay)))
        
    if LarsConstants.OUTPUT_KEY_SIGN_VIOLATION_NUMITER in ex.reconResult:
        print("Number of Lars-Lasso iteration(s) with a sign violation: {0}".format(ex.reconResult[LarsConstants.OUTPUT_KEY_SIGN_VIOLATION_NUMITER]))
    
    for h in iterObserver.HistoryState:
        # Output metrics we always expect to have
        msg = "RSS: {0:.4e}, Max corr: {1:.5f}, theta err l_1/l_2 norm = {2:.5f}/{3:.5f}".format(h[LarsIterationEvaluator.OUTPUT_METRIC_FITERR_SS],
                                                                                                 h[LarsIterationEvaluator.OUTPUT_METRIC_CORRHATABS_MAX],
                                                                                                 h[LarsIterationEvaluator.OUTPUT_METRIC_THETAERR_L1],
                                                                                                 h[LarsIterationEvaluator.OUTPUT_METRIC_THETAERR_L2])
        # Output metrics that may not be present
        if (LarsIterationEvaluator.OUTPUT_METRIC_THETA_PROPEXPL in h) and (LarsIterationEvaluator.OUTPUT_METRIC_MU_PROPEXPL in h):
            msg += ", theta/mu prop. expl.: {0:.5f}/{1:.5f}".format(h[LarsIterationEvaluator.OUTPUT_METRIC_THETA_PROPEXPL],
                                                                    h[LarsIterationEvaluator.OUTPUT_METRIC_MU_PROPEXPL])
        if LarsIterationEvaluator.OUTPUT_CRITERION_L1_SURE in h:
            msg += ", SURE criterion: {0:.5f}".format(h[LarsIterationEvaluator.OUTPUT_CRITERION_L1_SURE])        
        print(msg)
                
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

    if True:
        plt.ioff()
                                                             
        plt.figure(1)
        plt.imshow(blurredImageWithNoiseForDisplayZeroMean)
        plt.colorbar()
        
        plt.figure(2)
        plt.imshow(estimatedMuForDisplay)
        plt.colorbar()
        
        plt.show()
    
