import unittest
import numpy as np
import cProfile
import pylab as plt

from ..Examples.Gaussian2dBlurWithNoise import Gaussian2dBlurWithNoise
from ..Recon.Gaussian.AbstractEmgaussReconstructor import AbstractEmgaussReconstructor
from ..Recon.Gaussian.EmgaussFixedMstepReconstructor import (
    EmgaussFixedMstepReconstructor,
)
from ..Recon.Gaussian.EmgaussIterationsObserver import EmgaussIterationsObserver
from ..Recon.Gaussian.Thresholding import ThresholdingIdentity
from ..Recon.NormMinimizer.L2NormMinimizer import L2NormDirectMinimizerReconstructor
from ..Recon.AbstractInitialEstimator import InitialEstimatorFactory
from ..Sim.AbstractImageGenerator import AbstractImageGenerator
from ..Sim.NoiseGenerator import AbstractAdditiveNoiseGenerator
from ..Systems.ConvolutionMatrixUsingPsf import ConvolutionMatrixUsingPsf
from ..Systems.PsfNormalizer import PsfMatrixNormNormalizer


class T_Recon_Noiseless(unittest.TestCase):
    """
    Tests Landweber iterations and the deconvolution reconstructor in the case of a noiseless output.
    Landweber iterations for a GaussianBlurWithNoise output converge slowly.
    """

    testMessages = []

    def setUp(self):
        ex = Gaussian2dBlurWithNoise(
            {
                AbstractAdditiveNoiseGenerator.INPUT_KEY_SIGMA: 0,
                AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE: (32, 32),
            }
        )
        ex.RunExample()
        self.channelChain = ex.channelChain
        self.blurredImage = (
            ex.blurredImageWithNoise
        )  # Since we specified a sigma of 0, the image is noiseless

    def testLandweberIterations(self):
        # Set up Landweber iterations
        gbNormalizer = PsfMatrixNormNormalizer(1)
        gbNormalizer.NormalizePsf(
            self.channelChain.channelBlocks[1].BlurPsfInThetaFrame
        )
        tIdentity = ThresholdingIdentity()
        emgIterationsObserver = EmgaussIterationsObserver(
            {
                EmgaussIterationsObserver.INPUT_KEY_TERMINATE_COND: EmgaussIterationsObserver.TERMINATE_COND_THETA_DELTA_L2,
                EmgaussIterationsObserver.INPUT_KEY_TERMINATE_TOL: 1e-7,
            }
        )
        emg = EmgaussFixedMstepReconstructor(
            {
                AbstractEmgaussReconstructor.INPUT_KEY_MAX_ITERATIONS: 1e5,
                AbstractEmgaussReconstructor.INPUT_KEY_ITERATIONS_OBSERVER: emgIterationsObserver,
                AbstractEmgaussReconstructor.INPUT_KEY_TAU: 1.999
                / gbNormalizer.GetSpectralRadiusGramMatrixRowsH(),  # Don't exceed 2/rho(H'*H)
            },
            lambda x: tIdentity.Apply(x),
        )

        # Run the Landweber iterations and verify that it recovers the original image exactly.
        # This is possible since there's no noise.
        y = self.blurredImage
        psfRepH = self.channelChain.channelBlocks[1].BlurPsfInThetaFrame
        initialEstimator = InitialEstimatorFactory.GetInitialEstimator(
            "Hty"
        ).GetInitialEstimate(y, psfRepH)
        estimated = emg.Estimate(
            y, ConvolutionMatrixUsingPsf(psfRepH), initialEstimator
        )
        thetaEstimated = estimated[0]
        # No need to print this out, we most likely hit the max #iterations
        # T_Recon_Noiseless.testMessages.append("Landweber iterations: termination reason: " + emg.TerminationReason)

        theta = self.channelChain.intermediateOutput[0]
        estimationErrorL2Norm = np.linalg.norm(theta - thetaEstimated, 2)
        # T_Recon_Noiseless.testMessages.append("Landweber iterations: estimation error l_2 norm: " + str(estimationErrorL2Norm))
        # Loose assertion. Landweber iterations converge to the deconvolution solution,
        # but convergence is slow
        self.assertLess(estimationErrorL2Norm, 1.5)

    #        nFigStart = 1
    #        plt.figure(nFigStart)
    #        plt.imshow(theta)
    #        plt.colorbar()
    #        plt.figure(nFigStart + 1)
    #        plt.imshow(self.channelChain.channelBlocks[1].RemoveShiftFromBlurredImage(self.blurredImage))
    #        plt.colorbar()
    #        plt.figure(nFigStart + 2)
    #        plt.imshow(thetaEstimated)
    #        plt.colorbar()

    def testDeconvolution(self):
        deconv = L2NormDirectMinimizerReconstructor(0)
        thetaEstimated = deconv.Estimate(
            self.blurredImage,
            self.channelChain.channelBlocks[1].BlurPsfInThetaFrame,
            None,
        )
        theta = self.channelChain.intermediateOutput[0]
        estimationErrorL2Norm = np.linalg.norm(theta - thetaEstimated, 2)
        # T_Recon_Noiseless.testMessages.append("deconvolution test: estimation error l_2 norm: " + str(estimationErrorL2Norm))
        self.assertLess(estimationErrorL2Norm, 1e-9)

    #        nFigStart = 4
    #        plt.figure(nFigStart)
    #        plt.imshow(theta)
    #        plt.colorbar()
    #        plt.figure(nFigStart + 1)
    #        plt.imshow(self.channelChain.channelBlocks[1].RemoveShiftFromBlurredImage(self.blurredImage))
    #        plt.colorbar()
    #        plt.figure(nFigStart + 2)
    #        plt.imshow(thetaEstimated)
    #        plt.colorbar()

    @classmethod
    def tearDownClass(cls):
        #        plt.show()
        for testMessage in cls.testMessages:
            print(testMessage)


if __name__ == "__main__":
    cProfile.run("unittest.main()")
