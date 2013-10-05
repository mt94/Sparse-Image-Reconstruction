import cPickle as pickle
import numpy as np
import pylab as plt
import warnings

from AbstractReconstructorExample import AbstractReconstructorExample
from BlurWithNoiseFactory import BlurWithNoiseFactory
from Recon.HyperparameterPick import HyperparameterPick
from Recon.Stagewise.LarsConstants import LarsConstants
from Recon.Stagewise.LarsIterationEvaluator import LarsIterationEvaluator
from Recon.Stagewise.LarsReconstructorFactory import LarsReconstructorFactory
from Sim.AbstractImageGenerator import AbstractImageGenerator
from Sim.NoiseGenerator import AbstractAdditiveNoiseGenerator
from Systems.ComputeEnvironment import ComputeEnvironment
from Systems.PsfLinearDerivative import ConvolutionMatrixZeroMeanUnitNormDerivative
from Systems.Timer import Timer

class LarsReconstructorOnExample(AbstractReconstructorExample):
    """
    Demonstrates Lars-based reconstructors
    """
    
    GAUSSIAN_BLUR_WITH_NOISE_DUMP_FILE = ''; #'c:\\tmp\\LeastAngleRegressionOnExampleGbwn.dump'
    
    def __init__(self, reconstructorDesc, iterObserver, bTurnOnWarnings=False, bRestoreSim=False):
        super(LarsReconstructorOnExample, self).__init__('LARS example')        
        self.reconstructorDesc = reconstructorDesc
        self.iterObserver = iterObserver
        self.bTurnOnWarnings = bTurnOnWarnings
        self.bRestoreSim = bRestoreSim 
        
        # Contains a dictionary of the last iteration of LARS. Does not necessarily correspond
        # to the estimated theta.
        self.reconResult = None            
        
    def _RunExperiment(self):
        if (not self.bRestoreSim):
            if self.experimentObj is None:
                raise NameError('experimentObj is undefined')                 
            self.experimentObj.RunExample()      
            if LarsReconstructorOnExample.GAUSSIAN_BLUR_WITH_NOISE_DUMP_FILE:      
                pickle.dump(self.experimentObj, open(LarsReconstructorOnExample.GAUSSIAN_BLUR_WITH_NOISE_DUMP_FILE, 'wb'))
        else:        
            if LarsReconstructorOnExample.GAUSSIAN_BLUR_WITH_NOISE_DUMP_FILE:
                self.experimentObj = pickle.load(open(LarsReconstructorOnExample.GAUSSIAN_BLUR_WITH_NOISE_DUMP_FILE, 'rb')) 
            else:
                # There's a mistake here
                raise NameError("Cannot restore object when dump filename isn't defined")
                    
    def RunExample(self):   
        # First thing to do is to run the experiment
        self._RunExperiment()
         
        y = self.experimentObj.blurredImageWithNoise
        psfRepH = self.experimentObj.channelChain.channelBlocks[1].BlurPsfInThetaFrame # Careful not to use H, which is the convolution matrix
        convMatrixObj = ConvolutionMatrixZeroMeanUnitNormDerivative(psfRepH)
        
        # Need to set variables in the iteration observer
        self.iterObserver.NoiseSigma = self.experimentObj.NoiseSigma
        self.iterObserver.ThetaTrue = np.array(self.experimentObj.channelChain.intermediateOutput[0]) * convMatrixObj.zeroMeanedColumnL2norm
        muTrue = convMatrixObj.Multiply(self.iterObserver.ThetaTrue)
        self.iterObserver.MuTrue = np.array(muTrue)
        
        optimSettingsDict = { 
                              LarsConstants.INPUT_KEY_MAX_ITERATIONS: 30,
                              LarsConstants.INPUT_KEY_EPS: ComputeEnvironment.EPS,
                              LarsConstants.INPUT_KEY_NVERBOSE: 0,
                              LarsConstants.INPUT_KEY_ENFORCE_ONEATATIME_JOIN: True,
                              LarsConstants.INPUT_KEY_ITERATIONS_OBSERVER: self.iterObserver 
                             }
        reconstructor = LarsReconstructorFactory.GetReconstructor(self.reconstructorDesc, optimSettingsDict)
        
#        gbNormalizer = PsfColumnNormNormalizer(1)
#        psfRepHWithUnitColumnNorm = gbNormalizer.NormalizePsf(psfRepH)                
#        self.reconResult = reconstructor.Estimate(y, ConvolutionMatrixUsingPsf(psfRepHWithUnitColumnNorm))

        yZeroMean = y - np.mean(y.flat)*np.ones(y.shape)        
        assert np.mean(yZeroMean.flat) < 1.0e-11
                
        if self.bTurnOnWarnings:
            with Timer() as t:
                self.reconResult = reconstructor.Estimate(yZeroMean, convMatrixObj)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with Timer() as t:
                    self.reconResult = reconstructor.Estimate(yZeroMean, convMatrixObj)
                    
        # Save run variables                    
        self._timingMs = t.msecs
        self._channelChain = self.experimentObj.channelChain        
        self._theta = self._channelChain.intermediateOutput[0]
        self._y = y                            
        self._reconstructor = reconstructor
        
                
    def PrintOutputSummary(self):
        activeSetDisplay = ["{0}".format(x) for x in self.reconResult[LarsConstants.OUTPUT_KEY_ACTIVESET]]
        print("Active set: {0}".format(" ".join(activeSetDisplay)))
            
        if LarsConstants.OUTPUT_KEY_SIGN_VIOLATION_NUMITER in self.reconResult:
            print("Number of Lars-Lasso iteration(s) with a sign violation: {0}".format(
                                                                                        self.reconResult[LarsConstants.OUTPUT_KEY_SIGN_VIOLATION_NUMITER]
                                                                                        )
                  )
            
    def PrintOutputIterations(self):
        lassoSureCriterionPrev = None        
        cntHistory = 1
        
        for h in self.iterObserver.HistoryState:
            # Output metrics we always expect to have
            msg = "[{0}] ".format(cntHistory)
            fmtString = "RSS: {0:.4e}, Max corr: {1:.5f}, |theta err|_1,2 = {2:.5f}/{3:.5f}, |thetaHat|_0 = {4}"
            msg += fmtString.format(h[LarsIterationEvaluator.OUTPUT_METRIC_FITERR_SS],
                                    h[LarsIterationEvaluator.OUTPUT_METRIC_CORRHATABS_MAX],
                                    h[LarsIterationEvaluator.OUTPUT_METRIC_THETAERR_L1],
                                    h[LarsIterationEvaluator.OUTPUT_METRIC_THETAERR_L2],
                                    h[LarsIterationEvaluator.OUTPUT_METRIC_THETAHAT_L0]
                                    )
            # Output metrics that may not be present
            if (LarsIterationEvaluator.OUTPUT_METRIC_THETA_PROPEXPL in h) and (LarsIterationEvaluator.OUTPUT_METRIC_MU_PROPEXPL in h):
                msg += ", theta/mu prop. expl.: {0:.5f}/{1:.5f}".format(h[LarsIterationEvaluator.OUTPUT_METRIC_THETA_PROPEXPL],
                                                                        h[LarsIterationEvaluator.OUTPUT_METRIC_MU_PROPEXPL])
            if LarsIterationEvaluator.OUTPUT_CRITERION_L1_SURE in h:
                lassoSureCriterionCurr = h[LarsIterationEvaluator.OUTPUT_CRITERION_L1_SURE]
                msg += ", SURE criterion: {0:.5f}".format(lassoSureCriterionCurr)   
                if lassoSureCriterionPrev is not None:
                    msg += " ({0:.3f})".format(lassoSureCriterionCurr - lassoSureCriterionPrev)
                lassoSureCriterionPrev = lassoSureCriterionCurr; # Update
                
            print(msg)
            cntHistory += 1
        
        print("Took {0:g}ms".format(self.TimingMs))
            
    def PlotConvolvedImage2d(self, fignumStart):
        experimentObj = self.experimentObj
        
        blurredImageWithNoiseForDisplay = experimentObj.channelChain \
                                                        .channelBlocks[1] \
                                                        .RemoveShiftFromBlurredImage(experimentObj.blurredImageWithNoise)
                                                 
        blurredImageWithNoiseForDisplayZeroMean = blurredImageWithNoiseForDisplay - \
                                                  np.mean(blurredImageWithNoiseForDisplay.flat)*np.ones(blurredImageWithNoiseForDisplay.shape)
                                                          
        assert np.mean(blurredImageWithNoiseForDisplayZeroMean) < 1.0e-11        
    
        estimatedMu = np.reshape(self.reconResult[LarsConstants.OUTPUT_KEY_MUHAT_ACTIVESET], 
                                 experimentObj.blurredImageWithNoise.shape)
        
        estimatedMuForDisplay = experimentObj.channelChain \
                                             .channelBlocks[1] \
                                             .RemoveShiftFromBlurredImage(estimatedMu)    

        plt.ioff()
                                                             
        plt.figure(fignumStart)
        plt.imshow(blurredImageWithNoiseForDisplayZeroMean, interpolation='none')
        plt.colorbar()
        plt.title('Blurred image with noise')
        
        plt.figure(fignumStart + 1)
        plt.imshow(estimatedMuForDisplay, interpolation='none')
        plt.colorbar()
        plt.title('Last estimate of blurred image')
        
def PlotTheta2d(fignumStart, iterObserver, hyperparameterPick):
    plt.ioff()
     
    plt.figure(fignumStart)
    plt.imshow(iterObserver.ThetaTrue, interpolation='none')
    plt.colorbar()
    plt.title('True theta')         
    
    plt.figure(fignumStart + 1)
    
    indBest, thetaBest = hyperparameterPick.GetBestEstimate(LarsIterationEvaluator.OUTPUT_CRITERION_L1_SURE, -0.1)    
    # DEBUG
#     indBest = 0
#     thetaBest = iterObserver.HistoryEstimate[indBest]

    plt.imshow(
               np.reshape(
                          thetaBest,                          
                          iterObserver.ThetaTrue.shape
                          ),
               interpolation='none'
               )
    plt.colorbar()
    plt.title('Estimated theta: iter {0}'.format(1 + indBest))      
        
if __name__ == "__main__":
    EXPERIMENT_DESC = 'mrfm3d'
    IMAGESHAPE = (32, 32, 14); #(32, 32)
    SNRDB = 20
    
    MyReconstructorDesc = 'lars_lasso'
    
    iterObserver = LarsIterationEvaluator(ComputeEnvironment.EPS)
    
    if MyReconstructorDesc == 'lars_lasso':
        iterObserver.TrackCriterionL1Sure = True
     
    # Use bRestoreSim for debugging problem cases        
    ex = LarsReconstructorOnExample(MyReconstructorDesc, iterObserver, bRestoreSim=False)
    # Get the experimental object, which encapsulates the experiment on which to use the LARS reconstructor 
    ex.experimentObj = BlurWithNoiseFactory.GetBlurWithNoise(EXPERIMENT_DESC, 
                                                             {
                                                              AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB: SNRDB,
                                                              AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE: IMAGESHAPE
                                                              }
                                                             )
    ex.RunExample()
    ex.PrintOutputIterations()
    
    hparamPick = HyperparameterPick(iterObserver)
    hparamPick.PlotConvolvedImage2d(LarsIterationEvaluator.OUTPUT_CRITERION_L1_SURE, 1)
    
    if len(iterObserver.ThetaTrue.shape) == 2:
#        ex.PlotConvolvedImage2d(2)        
        PlotTheta2d(2, iterObserver, hparamPick)    
        plt.show()
                

    
