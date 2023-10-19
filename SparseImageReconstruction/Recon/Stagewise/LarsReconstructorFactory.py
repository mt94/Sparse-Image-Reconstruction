from ...Recon.Stagewise.LarsReconstructor import LarsReconstructor
from ...Recon.Stagewise.LarsLassoReconstructor import LarsLassoReconstructor


class LarsReconstructorFactory(object):
    _concreteReconstructorGenerator = {
        "lars": LarsReconstructor,
        "lars_lasso": LarsLassoReconstructor,
    }

    @staticmethod
    def GetReconstructor(reconstructorDesc, optimSettingsDict):
        if (
            reconstructorDesc
            not in LarsReconstructorFactory._concreteReconstructorGenerator
        ):
            raise NotImplementedError(
                "Reconstructor " + str(reconstructorDesc) + " isn't implemented"
            )
        return LarsReconstructorFactory._concreteReconstructorGenerator[
            reconstructorDesc
        ](optimSettingsDict)
