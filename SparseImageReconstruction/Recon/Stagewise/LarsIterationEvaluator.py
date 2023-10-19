import numpy as np
import pylab as plt
from ...Recon.AbstractIterationsObserver import AbstractIterationsObserver


class LarsIterationEvaluator(AbstractIterationsObserver):
    """
    Observer for Lars-based reconstructors. Used to compute the L1-SURE criterion. Can be modified
    to compute other criteria.
    """

    # Input metrics
    STATE_KEY_THETA = "theta"
    STATE_KEY_FIT_ERROR = "fit_error"
    FUNC_KEY_MU_FROM_THETA = "func_mu_from_theta"

    # Output metrics that the observer will generate
    OUTPUT_METRIC_CORRHATABS_MAX = "corrhatabs_max"

    OUTPUT_METRIC_FITERR_SS = "fit_rss"
    OUTPUT_METRIC_FITERR_L1 = "fit_l1"

    OUTPUT_METRIC_THETAHAT_L0 = "theta_hat_l0"
    OUTPUT_METRIC_THETAERR_L1 = "theta_err_l1"
    OUTPUT_METRIC_THETAERR_L2 = "theta_err_l2"

    OUTPUT_METRIC_THETA_PROPEXPL = "theta_prop_expl"
    OUTPUT_METRIC_MU_PROPEXPL = "mu_prop_expl"

    OUTPUT_CRITERION_L1_SURE = "criterion_l1_sure"

    def __init__(self, EPS, noiseSigma=None):
        super(LarsIterationEvaluator, self).__init__()
        self._historyEstimate = []
        self._historyState = []
        self._EPS = EPS
        self._bRequireFitError = True
        # Attributes that the user can set
        self.TrackCriterionL1Sure = False
        self.NoiseSigma = None
        self.ThetaTrue = None
        self.MuTrue = None

    @staticmethod
    def CalculateL0(x, EPS):
        return np.where(np.abs(x) > EPS)[0].size

    """ Implementation of abstract members """

    @property
    def TerminateIterations(self):
        return False  # Never terminate

    @property
    def HistoryEstimate(self):
        return self._historyEstimate

    @property
    def HistoryState(self):
        return self._historyState

    def UpdateWithEstimates(self, thetaNp1, thetaN, fitErrorN):
        raise NotImplementedError("Method unimplemented")

    def UpdateState(self, ipStateDict, bPlotThetaErr=False):
        fitError = ipStateDict[LarsIterationEvaluator.STATE_KEY_FIT_ERROR]
        thetaHat = ipStateDict[LarsIterationEvaluator.STATE_KEY_THETA]
        self._historyEstimate.append(np.array(thetaHat))

        rss = np.sum(fitError * fitError)
        thetaHatL0 = self.CalculateL0(thetaHat, self._EPS)

        if self.ThetaTrue is not None:
            thetaErr = self.ThetaTrue.flat - np.reshape(
                np.array(thetaHat), (1, self.ThetaTrue.size)
            )
            if bPlotThetaErr:  # DEBUG
                plt.figure(1), plt.clf()
                plt.plot(self.ThetaTrue.flat, "k.")
                plt.hold(True)
                plt.plot(thetaHat.flat, "ms")
                plt.show()

        else:
            thetaErr = None

        if (self.NoiseSigma is None) or (self.TrackCriterionL1Sure is False):
            criterionSure = None
        else:
            N = fitError.size
            noiseSigmaSquare = np.square(self.NoiseSigma)
            criterionSure = (
                N * noiseSigmaSquare + rss + 2 * noiseSigmaSquare * thetaHatL0
            )

        opStateDict = {
            LarsIterationEvaluator.OUTPUT_METRIC_CORRHATABS_MAX: ipStateDict[
                LarsIterationEvaluator.OUTPUT_METRIC_CORRHATABS_MAX
            ],
            LarsIterationEvaluator.OUTPUT_METRIC_FITERR_L1: np.sum(np.abs(fitError)),
            LarsIterationEvaluator.OUTPUT_METRIC_FITERR_SS: rss,
            LarsIterationEvaluator.OUTPUT_METRIC_THETAHAT_L0: thetaHatL0,
        }

        # Add additional parameters to the state if they can/are asked to be computed
        if criterionSure is not None:
            opStateDict[LarsIterationEvaluator.OUTPUT_CRITERION_L1_SURE] = criterionSure

        if thetaErr is not None:
            opStateDict[LarsIterationEvaluator.OUTPUT_METRIC_THETAERR_L1] = np.sum(
                np.abs(thetaErr)
            )
            opStateDict[LarsIterationEvaluator.OUTPUT_METRIC_THETAERR_L2] = np.sqrt(
                np.sum(thetaErr * thetaErr)
            )
            # This comes from (3.17) of the LARS paper
            opStateDict[
                LarsIterationEvaluator.OUTPUT_METRIC_THETA_PROPEXPL
            ] = 1 - np.sum(thetaErr * thetaErr) / np.sum(
                self.ThetaTrue * self.ThetaTrue
            )

        if (LarsIterationEvaluator.FUNC_KEY_MU_FROM_THETA in ipStateDict) and (
            self.MuTrue is not None
        ):
            fnMapThetaToMu = ipStateDict[LarsIterationEvaluator.FUNC_KEY_MU_FROM_THETA]
            muErr = self.MuTrue.flat - np.reshape(
                fnMapThetaToMu(thetaHat), (1, self.MuTrue.size)
            )
            opStateDict[LarsIterationEvaluator.OUTPUT_METRIC_MU_PROPEXPL] = 1 - np.sum(
                muErr * muErr
            ) / np.sum(self.MuTrue * self.MuTrue)

        self._historyState.append(opStateDict)
