import numpy as np
import pylab as plt

from ...Recon.AbstractIterationsObserver import AbstractIterationsObserver
from ...Systems.NumericalHelper import NumericalHelper


class McmcIterationEvaluator(AbstractIterationsObserver):
    """
    Observer for MCMC methods
    """

    STATE_KEY_COUNT_ITER = "count_iter"
    STATE_KEY_X_ITER = "x_iter"
    STATE_KEY_HX_ITER = "hx_iter"
    STATE_KEY_W_ITER = "w_iter"
    STATE_KEY_A_ITER = "a_iter"
    STATE_KEY_NOISEVAR_ITER = "noisevar_iter"

    def __init__(
        self,
        Eps,
        xShape,
        xTrue=None,
        xFigureNum=None,
        y=None,
        countIterationDisplaySet=None,
        bVerbose=False,
    ):
        super(McmcIterationEvaluator, self).__init__()

        self.bVerbose = bVerbose
        self.Eps = Eps
        self.xShape = xShape
        self.xTrue = xTrue
        self.y = y

        if (xFigureNum is not None) and (xFigureNum > 0):
            self.xFigureNum = xFigureNum
        else:
            self.xFigureNum = -1

        self.countIteration = 0

        if countIterationDisplaySet is not None:
            self.countIterationDisplaySet = countIterationDisplaySet
        else:
            # Default value
            self.countIterationDisplaySet = np.append(1, np.arange(100, 4000, 100))

        self.wHistory = []
        self.aHistory = []
        self.noiseVarHistory = []
        self.xErrL1NormHistory = []
        self.yErrL2NormHistory = []

    """ Implementation of abstract methods """

    @property
    def TerminateIterations(self):
        return False  # Never terminate

    @property
    def HistoryEstimate(self):
        raise NotImplementedError()

    @property
    def HistoryState(self):
        raise NotImplementedError()

    def UpdateWithEstimates(self, thetaNp1, thetaN, fitErrorN):
        raise NotImplementedError("Method unimplemented")

    def UpdateState(self, stateDict):
        if McmcIterationEvaluator.STATE_KEY_COUNT_ITER in stateDict:
            self.countIteration = stateDict[McmcIterationEvaluator.STATE_KEY_COUNT_ITER]
        else:
            self.countIteration += 1

        keysForMsg = (
            McmcIterationEvaluator.STATE_KEY_W_ITER,
            McmcIterationEvaluator.STATE_KEY_A_ITER,
            McmcIterationEvaluator.STATE_KEY_NOISEVAR_ITER,
        )

        if all(k in stateDict for k in keysForMsg):
            wIter = stateDict[McmcIterationEvaluator.STATE_KEY_W_ITER]
            aIter = stateDict[McmcIterationEvaluator.STATE_KEY_A_ITER]
            noiseVarIter = stateDict[McmcIterationEvaluator.STATE_KEY_NOISEVAR_ITER]
            self.wHistory.append(wIter)
            self.aHistory.append(aIter)
            self.noiseVarHistory.append(noiseVarIter)
            if self.bVerbose:
                print(
                    "=> Iter {0}: hyper samp.: w={1:.5f}, a={2:.5f}; var samp.: {3:.5e}".format(
                        self.countIteration, wIter, aIter, noiseVarIter
                    )
                )
        if (
            self.bVerbose
            and (self.xShape is not None)
            and (len(self.xShape) == 2)
            and (self.xFigureNum > 0)
            and np.in1d(self.countIteration, self.countIterationDisplaySet)[0]
        ):
            bPlot = True
        else:
            bPlot = False

        bPlotToShow = False

        xSampledMsg = "   Sampled x:"
        xInitialLen = len(xSampledMsg)

        if (McmcIterationEvaluator.STATE_KEY_X_ITER in stateDict) and isinstance(
            stateDict[McmcIterationEvaluator.STATE_KEY_X_ITER], np.ndarray
        ):
            xIter = stateDict[McmcIterationEvaluator.STATE_KEY_X_ITER]
            if self.xTrue is None:
                raise ValueError("xTrue variable hasn't been initialized")
            xErr = self.xTrue - xIter[:, 0]
            n0Next, n1Next = NumericalHelper.CalculateNumZerosNonzeros(xIter, self.Eps)
            xErrL1Norm = np.sum(np.abs(xErr))
            self.xErrL1NormHistory.append(xErrL1Norm)
            if self.bVerbose:
                xSampledMsg += (
                    " n0={0} |x|_0={1} |x|_1={2:.5f} |xErr|_1={3:.5f}".format(
                        n0Next, n1Next, np.sum(np.abs(xIter)), xErrL1Norm
                    )
                )
            if bPlot:
                # Plot xTrue
                plt.figure(self.xFigureNum)
                plt.imshow(np.reshape(self.xTrue, self.xShape), interpolation="none")
                plt.title("Actual theta")
                plt.colorbar()
                # Plot xIter
                plt.figure()
                plt.imshow(np.reshape(xIter, self.xShape), interpolation="none")
                plt.title("xIter at iteration {0}".format(self.countIteration))
                if np.max(xIter.flat) > np.min(xIter.flat):
                    plt.colorbar()
                # Plot xErrL1NormHistory
                plt.figure()
                plt.semilogy(self.xErrL1NormHistory, "b-")
                plt.grid()
                plt.title("|xErr|_1 vs. iteration")
                bPlotToShow = True

        if (McmcIterationEvaluator.STATE_KEY_HX_ITER in stateDict) and isinstance(
            stateDict[McmcIterationEvaluator.STATE_KEY_HX_ITER], np.ndarray
        ):
            hxIter = stateDict[McmcIterationEvaluator.STATE_KEY_HX_ITER]
            if self.y is None:
                raise ValueError("y variable hasn't been initialized")
            yErr = self.y - np.reshape(hxIter, self.y.shape)
            yErrFlat = np.array(yErr.flat)
            yErrL2Norm = np.sum(yErrFlat * yErrFlat)
            self.yErrL2NormHistory.append(yErrL2Norm)
            if self.bVerbose:
                xSampledMsg += " |yErr|_2={0:.5f}".format(yErrL2Norm)
            if bPlot:
                # Plot y
                plt.figure(self.xFigureNum + 3)
                plt.imshow(self.y, interpolation="none")
                plt.title("y")
                plt.colorbar()
                # Plot H*xIter
                plt.figure()
                plt.imshow(np.reshape(hxIter, self.xShape), interpolation="none")
                plt.title("H*xIter at iteration {0}".format(self.countIteration))
                if np.max(hxIter.flat) > np.min(hxIter.flat):
                    plt.colorbar()
                # Plot yErr as an image and also its histogram
                plt.figure()
                plt.imshow(yErr, interpolation="none")
                plt.title("(y - H*xIter) at iteration{0}".format(self.countIteration))
                if np.max(yErrFlat) > np.min(yErrFlat):
                    plt.colorbar()
                plt.figure()
                plt.hist(yErrFlat, 20)
                plt.title(
                    "Histogram of (y - H*xIter) values at iteration{0}".format(
                        self.countIteration
                    )
                )
                # Plot yErrL2NormHistory
                plt.figure()
                plt.semilogy(self.yErrL2NormHistory, "r-")
                plt.grid()
                plt.title("|yErr|_2 vs. iteration")
                bPlotToShow = True

        if self.bVerbose and len(xSampledMsg) > xInitialLen:
            print(xSampledMsg)

        if bPlotToShow:
            plt.show()
