import cPickle as pickle
import numpy as np
import pylab as plt
import warnings
from multiprocessing import Pool

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
from Systems.ReconstructorPerformanceCriteria import ReconstructorPerformanceCriteria
from Systems.Timer import Timer

class LarsReconstructorOnExample(AbstractReconstructorExample):
    """
    Demonstrates Lars-based reconstructors
    """
    
    GAUSSIAN_BLUR_WITH_NOISE_DUMP_FILE = ''; #'c:\\tmp\\LeastAngleRegressionOnExampleGbwn.dump'
    
    def __init__(self, reconstructorDesc, iterObserver, maxIterations=30, bTurnOnWarnings=False, bRestoreSim=False):
        super(LarsReconstructorOnExample, self).__init__('LARS example')        
        self.reconstructorDesc = reconstructorDesc
        self.iterObserver = iterObserver
        self.bTurnOnWarnings = bTurnOnWarnings
        self.bRestoreSim = bRestoreSim 
        self.maxIterations = maxIterations
        
        # Contains a dictionary of the last iteration of LARS. Does not necessarily correspond
        # to the estimated theta.
        self.reconResult = None            
        
    def _RunExperiment(self):
        if (not self.bRestoreSim):
            # Don't restore the experiment object from persistent storage
            if self.experimentObj is None:
                raise NameError('experimentObj is undefined')   
            if not self.experimentObj.RunAlready:              
                self.experimentObj.RunExample()      
            if LarsReconstructorOnExample.GAUSSIAN_BLUR_WITH_NOISE_DUMP_FILE:      
                pickle.dump(self.experimentObj, open(LarsReconstructorOnExample.GAUSSIAN_BLUR_WITH_NOISE_DUMP_FILE, 'wb'))
        else:        
            # Restore the experiment object from a file
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
                              LarsConstants.INPUT_KEY_MAX_ITERATIONS: self.maxIterations,
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
            
    def PlotMetricVsStages(self, fignumStart):
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
        
def PlotTheta2d(fignumStart, iterObserver, indBest, thetaBest):
    plt.ioff()
     
    plt.figure(fignumStart)
    plt.imshow(iterObserver.ThetaTrue, interpolation='none')
    plt.colorbar()
    plt.title('True theta')         
    
    plt.figure(fignumStart + 1)
        
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
        
def RunReconstructor(param, bPlot=False):
    """ Encapsulate the creation and running of the LARS-LASSO reconstructor """
    [reconstructorDesc, maxIterations, experimentDesc, imageShape, snrDb] = param
    
    iterObserver = LarsIterationEvaluator(ComputeEnvironment.EPS)
    
    if reconstructorDesc == 'lars_lasso':
        iterObserver.TrackCriterionL1Sure = True
     
    # Use bRestoreSim for debugging problem cases        
    exReconstructor = LarsReconstructorOnExample(reconstructorDesc, iterObserver, maxIterations, bRestoreSim=False)
    # Get the experimental object, which encapsulates the experiment on which to use the LARS reconstructor 
    exReconstructor.experimentObj = BlurWithNoiseFactory.GetBlurWithNoise(experimentDesc, 
                                                             {
                                                              AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB: snrDb,
                                                              AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE: imageShape
                                                              }
                                                             )
    exReconstructor.RunExample()
    exReconstructor.PrintOutputIterations()
    
    hparamPick = HyperparameterPick(iterObserver)
    hparamPick.PlotMetricVsStages(LarsIterationEvaluator.OUTPUT_CRITERION_L1_SURE, 1)
    
    indBest, thetaBest = hparamPick.GetBestEstimate(LarsIterationEvaluator.OUTPUT_CRITERION_L1_SURE, -0.1)
    
    perfCriteria = ReconstructorPerformanceCriteria(exReconstructor.Theta, np.reshape(thetaBest, exReconstructor.Theta.shape))

    if (bPlot and (len(iterObserver.ThetaTrue.shape) == 2)):    
        PlotTheta2d(2, iterObserver, indBest, thetaBest)    
        plt.show()
        
    return {
            'timing_ms': exReconstructor.TimingMs,            
            'ind_best': indBest,                  
            'x_best': thetaBest,
            # Reconstruction performance criteria
            'normalized_l2_error_norm': perfCriteria.NormalizedL2ErrorNorm(),
            'normalized_detection_error': perfCriteria.NormalizedDetectionError(),
            'normalized_l0_norm': perfCriteria.NormalizedL0Norm()
            }
            
if __name__ == "__main__":
    RECONSTRUCTOR_DESC = 'lars_lasso'
    MAX_LARS_ITERATIONS = 30
    EXPERIMENT_DESC = 'mrfm2d'
    IMAGESHAPE = (32, 32); #(32, 32, 14)
    SNRDB = 20
    
    runArgs = [RECONSTRUCTOR_DESC, MAX_LARS_ITERATIONS, EXPERIMENT_DESC, IMAGESHAPE, SNRDB]
    
    NUMPROC = 3
    NUMTASKS = 30
    
    fmtString = "Best index: {0}/{1}, perf. criteria: {2}/{3}/{4}, timing={5:g}s."
     
    pool = Pool(processes=NUMPROC)
    resultPool = pool.map(RunReconstructor, [runArgs] * NUMTASKS)
    for aResult in resultPool:
        print(fmtString.format(
                               aResult['ind_best'], MAX_LARS_ITERATIONS,                               
                               aResult['normalized_l2_error_norm'], aResult['normalized_detection_error'], aResult['normalized_l0_norm'],
                               aResult['timing_ms'] / 1.0e3                               
                               ))   
                
