import unittest
import numpy as np
import cProfile

from Channel.ChannelProcessingChain import ChannelProcessingChain
from Recon.Gaussian.AbstractEmgaussReconstructor import AbstractEmgaussReconstructor
from Recon.Gaussian.EmgaussFixedMstepReconstructor import EmgaussFixedMstepReconstructor
from Recon.Gaussian.EmgaussIterationsObserver import EmgaussIterationsObserver
from Recon.Gaussian.Thresholding import ThresholdingIdentity
from Recon.NormMinimizer.L2NormMinimizer import L2NormMinimizer
from Recon.PsfNormalizer import PsfNormalizer
from Sim.Blur import Blur
from Sim.ImageGenerator import AbstractImageGenerator, ImageGeneratorFactory

import pylab as plt

class T_Recon_Noiseless(unittest.TestCase):
    
    def setUp(self):
        # Construct the processing chain and execute
        channelChain = ChannelProcessingChain(True)
        ig = ImageGeneratorFactory.GetImageGenerator('random_binary_2d')
        ig.SetParameters(**{ 
                            AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE: (32, 32),
                            AbstractImageGenerator.INPUT_KEY_NUM_NONZERO: 8,
                            AbstractImageGenerator.INPUT_KEY_BORDER_WIDTH: 5
                           }
                         )                                                                                                        
        channelChain.channelBlocks.append(ig)
        blurParametersDict = {
                              Blur.INPUT_KEY_FWHM: 3,
                              Blur.INPUT_KEY_NKHALF: 5                              
                              }
        gb = Blur(Blur.BLUR_GAUSSIAN_SYMMETRIC_2D, blurParametersDict)        
        channelChain.channelBlocks.append(gb)
        self.channelChain = channelChain        
        self.blurredImage = channelChain.RunChain() # No noise added        
        
    def testLandweberIterations(self):        
        # Set up Landweber iterations
        gbNormalizer = PsfNormalizer(1)
        gbNormalizer.NormalizePsf(self.channelChain.channelBlocks[1].BlurPsfInThetaFrame)
        tIdentity = ThresholdingIdentity()
        emgIterationsObserver = EmgaussIterationsObserver({
                                                           EmgaussIterationsObserver.INPUT_KEY_TERMINATE_COND: EmgaussIterationsObserver.TERMINATE_COND_THETA_DELTA_L2,
                                                           EmgaussIterationsObserver.INPUT_KEY_TERMINATE_TOL: 1e-6                                                
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
        thetaEstimated = emg.Estimate(self.blurredImage, 
                                      self.channelChain.channelBlocks[1].BlurPsfInThetaFrame, 
                                      np.zeros(self.blurredImage.shape)
                                      )            
        print "Landweber iterations: termination reason: ", emg.TerminationReason
        
        theta = self.channelChain.intermediateOutput[0]
        estimationErrorL2Norm = np.linalg.norm(theta - thetaEstimated, 2)
        print "Landweber iterations: estimation error l_2 norm: ", estimationErrorL2Norm      
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
        print "deconvolution test: estimation error l_2 norm: ", estimationErrorL2Norm             
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
        pass
        
if __name__ == "__main__":
    cProfile.run("unittest.main()")
    
