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

    @property
    def TerminateIterations(self):
        return self._bTerminate
            
    # Abstract method override
    def UpdateObservations(self, thetaNp1, thetaN, fitErrorN=None):
        if (self.terminateCondition == EmgaussIterationsObserver.TERMINATE_COND_THETA_DELTA_L2):
#            if (np.linalg.norm(thetaNp1 - thetaN, 2) < self.terminateTolerance):
            thetaDiff = np.reshape(thetaNp1 - thetaN, (thetaN.size,))            
            if np.sqrt((thetaDiff*thetaDiff).sum()) < self.terminateTolerance:                
                self._bTerminate = True
            else:
                self._bTerminate = False
        else:
            raise NotImplementedError('Unrecognized termination condition')