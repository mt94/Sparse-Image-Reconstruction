# import types
from ...Recon.Gaussian.AbstractEmgaussReconstructor import AbstractEmgaussReconstructor


class EmgaussFixedMstepReconstructor(AbstractEmgaussReconstructor):
    def __init__(self, optimSettingsDict, funcMstep):
        super(EmgaussFixedMstepReconstructor, self).__init__(optimSettingsDict)
        #        assert funcMstep is not None and isinstance(funcMstep, types.FunctionType)
        assert funcMstep is not None
        self.funcMstep = funcMstep

    """ Abstract methods implementation """

    def SetupBeforeIterations(self):
        assert len(self._reconArgs) == 1

    def Mstep(self, x, numIter):
        return (self.funcMstep(x),)
