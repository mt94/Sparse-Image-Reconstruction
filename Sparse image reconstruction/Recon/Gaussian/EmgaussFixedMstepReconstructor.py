import types
from Recon.Gaussian.AbstractEmgaussReconstructor import AbstractEmgaussReconstructor

class EmgaussFixedMstepReconstructor(AbstractEmgaussReconstructor):
    def __init__(self, optimSettingsDict, funcMstep):
        super(EmgaussFixedMstepReconstructor, self).__init__(optimSettingsDict)
        assert funcMstep is not None and isinstance(funcMstep, types.FunctionType)
        self.funcMstep = funcMstep
        
    def Mstep(self, x):
        return self.funcMstep(x)