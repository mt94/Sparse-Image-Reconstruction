import numpy as np

class ReconstructorPerformanceCriteria(object):
    def __init__(self, theta, thetaEstimated):
        super(ReconstructorPerformanceCriteria, self).__init__()
        assert (isinstance(theta, np.ndarray) and isinstance(thetaEstimated, np.ndarray)) \
            or (isinstance(theta, np.matrix) and isinstance(thetaEstimated, np.matrix))                              
        self._theta = theta
        self._thetaEstimated = thetaEstimated
        
    def NormalizedL2ErrorNorm(self):
        thetaDiff = self._theta - self._thetaEstimated
        return np.linalg.norm(thetaDiff.flat, 2) / np.linalg.norm(self._theta.flat, 2)
    
    def NormalizedDetectionError(self):
        M = len(self._theta.flat)
        xorVec = np.logical_xor(self._theta != 0, self._thetaEstimated != 0)
        return float(xorVec.sum()) / float(M)
    
    def NormalizedL0Norm(self):
        return float((self._thetaEstimated != 0).sum()) / float((self._theta != 0).sum()) 
        