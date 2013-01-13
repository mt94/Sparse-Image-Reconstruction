import unittest
import numpy as np
import cProfile
import pylab as plt

#from Channel.ChannelProcessingChain import ChannelProcessingChain
from Examples.GaussianBlurWithNoise import GaussianBlurWithNoise
from Recon.Gaussian.AbstractEmgaussReconstructor import AbstractEmgaussReconstructor
from Recon.Gaussian.EmgaussFixedMstepReconstructor import EmgaussFixedMstepReconstructor
from Recon.Gaussian.EmgaussIterationsObserver import EmgaussIterationsObserver
from Recon.Gaussian.Thresholding import ThresholdingIdentity
from Recon.NormMinimizer.L2NormMinimizer import L2NormMinimizer
from Recon.PsfNormalizer import PsfNormalizer
from Recon.AbstractInitialEstimator import InitialEstimatorFactory
#from Sim.Blur import Blur
#from Sim.ImageGenerator import AbstractImageGenerator, ImageGeneratorFactory

class T_Recon_Noiseless(unittest.TestCase):
    
    testMessages = []
    
    def setUp(self):
        ex = GaussianBlurWithNoise(0)
        ex.RunExample()
        self.channelChain = ex.channelChain        
        self.blurredImage = ex.blurredImageWithNoise # Since we specified a sigma of 0, the image is noiseless
        
    def testLandweberIterations(self):        
        # Set up Landweber iterations
        gbNormalizer = PsfNormalizer(1)
        gbNormalizer.NormalizePsf(self.channelChain.channelBlocks[1].BlurPsfInThetaFrame)
        tIdentity = ThresholdingIdentity()
        emgIterationsObserver = EmgaussIterationsObserver({
                                                           EmgaussIterationsObserver.INPUT_KEY_TERMINATE_COND: EmgaussIterationsObserver.TERMINATE_COND_THETA_DELTA_L2,
                                                           EmgaussIterationsObserver.INPUT_KEY_TERMINATE_TOL: 1e-7                                                
                                                           })
        emg = EmgaussFixedMstepReconstructor(
            {
                AbstractEmgaussReconstructor.INPUT_KEY_MAX_ITERATIONS: 1e6,
                AbstractEmgaussReconstructor.INPUT_KEY_ITERATIONS_OBSERVER: emgIterationsObserver,
                AbstractEmgaussReconstructor.INPUT_KEY_TAU: 1.999 / gbNormalizer.GetSpectralRadiusGramMatrixRowsH()  # Don't exceed 2/rho(H'*H)
            },
            lambda x: tIdentity.Apply(x)
        )
        
        # Run the Landweber iterations and verify that it recovers the original image exactly.
        # This is possible since there's no noise.
        y = self.blurredImage
        H = self.channelChain.channelBlocks[1].BlurPsfInThetaFrame
        thetaEstimated = emg.Estimate(y, 
                                      H,
                                      InitialEstimatorFactory.GetInitialEstimator('Hty')
                                                             .GetInitialEstimate(y, H)                                       
                                      )            
        T_Recon_Noiseless.testMessages.append("Landweber iterations: termination reason: " + emg.TerminationReason)
        
        theta = self.channelChain.intermediateOutput[0]
        estimationErrorL2Norm = np.linalg.norm(theta - thetaEstimated, 2)
        T_Recon_Noiseless.testMessages.append("Landweber iterations: estimation error l_2 norm: " + str(estimationErrorL2Norm))      
        # Loose assertion. Landweber iterations converge to the deconvolution solution,
        # but convergence is slow
        self.assertLess(estimationErrorL2Norm, 1)

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
        deconv = L2NormMinimizer(0)
        thetaEstimated = deconv.Estimate(self.blurredImage,
                                         self.channelChain.channelBlocks[1].BlurPsfInThetaFrame, 
                                         None)
        theta = self.channelChain.intermediateOutput[0]
        estimationErrorL2Norm = np.linalg.norm(theta - thetaEstimated, 2)
        T_Recon_Noiseless.testMessages.append("deconvolution test: estimation error l_2 norm: " + str(estimationErrorL2Norm))             
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
            print testMessage
        
if __name__ == "__main__":
    cProfile.run("unittest.main()")
    
