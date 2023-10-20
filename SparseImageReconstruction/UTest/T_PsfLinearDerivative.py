import numpy as np
import pylab as plt
import unittest

from ..Sim.SyntheticBlur import SyntheticBlur
from ..Systems.ComputeEnvironment import ComputeEnvironment
from ..Systems.PsfLinearDerivative import ConvolutionMatrixZeroMeanUnitNormDerivative


class T_PsfLinearDerivative(unittest.TestCase):
    """
    Tests ConvolutionMatrixZeroMeanUnitNormDerivative
    """

    TEST_IMAGE_SHAPE = (32, 32)

    def testMultiply(self):
        blurParametersDict = {
            SyntheticBlur.INPUT_KEY_FWHM: 3,
            SyntheticBlur.INPUT_KEY_NKHALF: 5,
        }
        gb = SyntheticBlur(SyntheticBlur.BLUR_GAUSSIAN_SYMMETRIC_2D, blurParametersDict)

        EPS_PRECISION_PLACES = np.int(
            np.floor(-1 * np.log10(ComputeEnvironment.EPS))
        )  # For EPS = 2.22e-16, this evaluates to 15

        # Check op of a random binary image
        imgBinaryRandom = np.random.randint(
            2, size=T_PsfLinearDerivative.TEST_IMAGE_SHAPE
        )
        yBinaryRandomWoCorrection = gb.Blur(imgBinaryRandom)
        self.assertNotAlmostEqual(
            0, np.mean(yBinaryRandomWoCorrection.flat), EPS_PRECISION_PLACES
        )

        gbDerivative = ConvolutionMatrixZeroMeanUnitNormDerivative(
            gb.BlurPsfInThetaFrame
        )
        yBinaryRandom = gbDerivative.Multiply(imgBinaryRandom)
        self.assertAlmostEqual(
            0, np.mean(yBinaryRandom.flat), EPS_PRECISION_PLACES - 1
        )  # Won't pass without -1

        # Check op of an image with a single value set to -1, the other values being set to 0
        imgBinaryRandom1 = np.zeros(T_PsfLinearDerivative.TEST_IMAGE_SHAPE)
        imgBinaryRandom1.flat[2] = -1
        yBinaryRandom1 = gbDerivative.Multiply(imgBinaryRandom1)
        self.assertAlmostEqual(0, np.mean(yBinaryRandom1.flat), EPS_PRECISION_PLACES)
        yBinaryRandom1AsArray = np.array(yBinaryRandom1.flat)
        self.assertAlmostEqual(
            1, np.sum(yBinaryRandom1AsArray * yBinaryRandom1AsArray), 12
        )  # Less accuracy
