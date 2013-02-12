import numpy as np
from Recon.AbstractIterationsObserver import AbstractIterationsObserver

class LarsIterationEvaluator(AbstractIterationsObserver):
    
    # Input metrics
    STATE_KEY_THETA = 'theta'
    STATE_KEY_FIT_ERROR = 'fit_error'    
    STATE_KEY_CORRHATABS_MAX = 'corrhatabs_max'
    
    # Output metrics that the observer will generate
    STATE_KEY_FIT_RSS = 'fit_rss'
    STATE_KEY_FIT_L1 = 'fit_l1'
    STATE_KEY_THETA_L0 = 'theta_l0'
    STATE_KEY_THETA_L1 = 'theta_l1'
    STATE_KEY_CRITERION_SURE = 'criterion_sure'
    
    def __init__(self, EPS, noiseSigma=None):
        super(LarsIterationEvaluator, self).__init__()
        self._historyEstimate = []
        self._historyState = [] 
        self._noiseSigma = noiseSigma
        self._EPS = EPS
        self._bRequireFitError = True
        
    @property
    def HistoryEstimate(self):
        return self._historyEstimate
    
    @property
    def HistoryState(self):
        return self._historyState
            
    """ Implementation of abstract members """
    
    @property
    def TerminateIterations(self):
        return False # Never terminate
        
    def UpdateEstimates(self, thetaNp1, thetaN, fitErrorN):
        raise NotImplementedError('Method unimplemented')
            
    def UpdateState(self, stateDict):
        fitError = stateDict[LarsIterationEvaluator.STATE_KEY_FIT_ERROR]
        thetaHat = stateDict[LarsIterationEvaluator.STATE_KEY_THETA]
                
        rss = np.sum(fitError*fitError)
        thetaHatL0 = np.where(np.abs(thetaHat) > self._EPS)[0].size        
        thetaHatL1 = np.sum(np.abs(thetaHat))        
        
        if self._noiseSigma is None:
            criterionSure = None
        else:
            N = fitError.size
            noiseSigmaSquare = np.square(self._noiseSigma)
            criterionSure = N*noiseSigmaSquare + rss + 2*noiseSigmaSquare*thetaHatL0
        
        self._historyState.append({
                                   LarsIterationEvaluator.STATE_KEY_FIT_RSS: rss,
                                   LarsIterationEvaluator.STATE_KEY_FIT_L1: np.sum(np.abs(fitError)),
                                   LarsIterationEvaluator.STATE_KEY_CORRHATABS_MAX: stateDict[LarsIterationEvaluator.STATE_KEY_CORRHATABS_MAX],
                                   LarsIterationEvaluator.STATE_KEY_THETA_L0: thetaHatL0,
                                   LarsIterationEvaluator.STATE_KEY_THETA_L1: thetaHatL1,
                                   LarsIterationEvaluator.STATE_KEY_CRITERION_SURE: criterionSure                                   
                                   })

        
        