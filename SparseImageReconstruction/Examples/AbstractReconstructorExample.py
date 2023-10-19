from .AbstractExample import AbstractExample


class AbstractReconstructorExample(AbstractExample):
    def __init__(self, exampleDesc=None):
        super(AbstractReconstructorExample, self).__init__(exampleDesc)
        # Define the attribute that denotes the experiment. Expect the
        # experiment to be derived from AbstractExample. This object
        # contains the experiment whose result will be analyzed by
        # a reconstructor.
        self.experimentObj = None

        self._theta = None
        # Image before blur and noise
        self._y = None
        # Noisy observation

        self._reconstructor = None
        # The reconstructor
        self._thetaEstimated = None
        # Theta as estimated by the reconstructor
        self._blurOperator = None
        # Blur operator
        self._timingMs = None
        # Timing of algorithm

    @property
    def TimingMs(self):
        return self._timingMs

    @property
    def Theta(self):
        if self._theta is None:
            raise NameError("Theta is uninitialized")
        return self._theta

    @property
    def NoisyObs(self):
        if self._y is None:
            raise NameError("NoisyObs is uninitialized")
        return self._y

    @property
    def ThetaEstimated(self):
        if self._thetaEstimated is None:
            raise NameError("ThetaEstimated is uninitialized")
        if (type(self._thetaEstimated) is tuple) or (
            type(self._thetaEstimated) is list
        ):
            if len(self._thetaEstimated) == 1:
                return self._thetaEstimated[0]
            elif len(self._thetaEstimated) == 2:
                return self._thetaEstimated[0] * self._thetaEstimated[1]
            else:
                raise TypeError("Don't know how to calculate ThetaEstimated")
        else:
            # Assume that _thetaEstimated is a numpy array
            return self._thetaEstimated

    @property
    def BlurOperator(self):
        if self._blurOperator is None:
            raise NameError("BlurOperator is uninitialized")
        return self._blurOperator

    @property
    def TerminationReason(self):
        if self._reconstructor is None:
            return "Termination reason is unknown"
        else:
            return self._reconstructor.TerminationReason
