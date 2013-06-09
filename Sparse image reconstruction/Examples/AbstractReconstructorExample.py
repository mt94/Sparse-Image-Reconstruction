import AbstractExample

class AbstractReconstructorExample(AbstractExample.AbstractExample):
    def __init__(self, exampleDesc=None):
        super(AbstractReconstructorExample, self).__init__(exampleDesc)
        # Define the attribute that denotes the experiment. Expect the
        # experiment to be derived from AbstractExample. This object
        # contains the experiment whose result will be analyzed by
        # a reconstructor.
        self.experimentObj = None
    
        self._theta = None;             # Image before blur and noise    
        self._y = None;                 # Noisy observation
        
        self._reconstructor = None;     # The reconstructor
        self._thetaEstimated = None;    # Theta as estimated by the reconstructor
    
    @property
    def Theta(self):
        if (self._theta is None):
            raise NameError('Theta is uninitialized')
        return self._theta
        
    @property
    def NoisyObs(self):
        if (self._y is None):
            raise NameError('NoisyObs is uninitialized')    
        return self._y    
    
    @property
    def ThetaEstimated(self):
        if (self._thetaEstimated is None):
            raise NameError('ThetaEstimated is uninitialized')
        return self._thetaEstimated 
    
    @property
    def TerminationReason(self):
        if (self._reconstructor is None):
            return "Termination reason is unknown"
        else:
            return self._reconstructor.TerminationReason        