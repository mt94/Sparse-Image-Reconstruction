import numpy as np
from Recon.AbstractIterationsObserver import AbstractIterationsObserver

class EmgaussIterationsObserver(AbstractIterationsObserver):
        
    INPUT_KEY_TERMINATE_COND = 'terminate_cond'
    TERMINATE_COND_THETA_DELTA_L2 = 'tc_theta_delta_l2'
    
    INPUT_KEY_TERMINATE_TOL = 'terminate_tol'
    TERMINATE_TOL_DEFAULT = 1e-6
        
    def __init__(self, inputDict=None):       
        super(EmgaussIterationsObserver, self).__init__() 
        if (inputDict is not None) and (EmgaussIterationsObserver.INPUT_KEY_TERMINATE_COND in inputDict):
            self.terminateCondition = inputDict[EmgaussIterationsObserver.INPUT_KEY_TERMINATE_COND]  
        else: 
            self.terminateCondition = EmgaussIterationsObserver.TERMINATE_COND_THETA_DELTA_L2
            
        if (inputDict is not None) and (EmgaussIterationsObserver.INPUT_KEY_TERMINATE_TOL in inputDict):
            self.terminateTolerance = inputDict[EmgaussIterationsObserver.INPUT_KEY_TERMINATE_TOL]
        else:
            self.terminateTolerance = EmgaussIterationsObserver.TERMINATE_TOL_DEFAULT
            
        self._bTerminate = False

    """ Implementation of abstract members """
    
    @property
    def TerminateIterations(self):
        return self._bTerminate
    
    @property
    def HistoryEstimate(self):
        raise NotImplementedError()
    
    @property
    def HistoryState(self):
        raise NotImplementedError()
                
    def UpdateWithEstimates(self, reconArgsNp1, reconArgsN, fitErrorN=None):
        if (self.terminateCondition == EmgaussIterationsObserver.TERMINATE_COND_THETA_DELTA_L2):                    
            if len(reconArgsN) == 1 and len(reconArgsN) == len(reconArgsNp1):
                # If reconArgsN only has one element, take it to be theta
                thetaDiff = np.reshape(reconArgsNp1[0] - reconArgsN[0], (reconArgsN[0].size,))
            elif len(reconArgsN) == 2 and len(reconArgsN) == len(reconArgsNp1):
                # If reconArgsN has two elements, assume that theta is the element-wise product of the two elementss
                thetaDiff = np.reshape(reconArgsNp1[0]*reconArgsNp1[1] - reconArgsN[0]*reconArgsN[1], (reconArgsN[0].size,))
            else:
                raise TypeError('Method called with inappropriate arguments')
            if np.sqrt((thetaDiff*thetaDiff).sum()) < self.terminateTolerance:                
                self._bTerminate = True
            else:
                self._bTerminate = False
        else:
            raise NotImplementedError('Unrecognized termination condition')
        
    def UpdateState(self, stateDict):
        raise NotImplementedError('Method unimplemented')
        