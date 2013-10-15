import math
import numpy as np
import pylab as plt

from AbstractBlurWithNoise import AbstractBlurWithNoise
from MrfmBlurExample import MrfmBlurExample
from Sim.AbstractImageGenerator import AbstractImageGenerator
from Sim.ImageGeneratorFactory import ImageGeneratorFactory 
from Sim.MrfmBlur import MrfmBlur
from Sim.MrfmBlurParameterOptimizer import MrfmBlurParameterOptimizer
from Sim.NoiseGenerator import AbstractAdditiveNoiseGenerator

class Mrfm2dBlurWithNoise(AbstractBlurWithNoise):
    """
    Simulates 2d MRFM blur and optionally adds AWGN.
    """    
    @staticmethod
    def GetBlurParameterOptimizer():
        # Generate MRFM psf used in 04/f/sp_img_recon.m (less realistic parameters than those used in psf_sim_sing.m)
        opti = MrfmBlurParameterOptimizer(deltaB0=100)
        opti.bUseSmallerR0 = True
        opti.bUseSmallerDeltaB0 = False
        opti.CalcOptimalValues(1e4, 6, R0=4)
        return opti
        
    def __init__(self, optimizer, simParametersDict):
        super(Mrfm2dBlurWithNoise, self).__init__(optimizer, simParametersDict, '2-d Mrfm blur with additive Gaussian noise example')
        self.debugMessages = []  
                               
    """ Abstract method override """
    def GetBlur(self):
        numPointsInMesh = np.min([32, self.ImageShape[0]])
        blurEx = MrfmBlurExample(self._optimizer, (numPointsInMesh,), MrfmBlur.BLUR_2D, '').RunExample()
        self._psfSupport = blurEx.Blur.PsfSupport
        return blurEx.Blur
         
    def GetImageGenerator(self):
        # Get the ImageGenerator after we've constructed the psf. This is b/c we need to figure
        # out the border width so that convolution with the psf doesn't result in spillover.
        ig = ImageGeneratorFactory.GetImageGenerator('random_binary')
        igBorderWidth = int(math.ceil((max(self._psfSupport[0]) - min(self._psfSupport[0]))/2.0)) + 1; # 1 shouldn't be necessary really
        ig.SetParameters(**{ 
                            AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE: self.ImageShape,
                            AbstractImageGenerator.INPUT_KEY_NUM_NONZERO: self.NumNonzero,
                            AbstractImageGenerator.INPUT_KEY_BORDER_WIDTH: igBorderWidth
                           }
                         )
        self.debugMessages.append("Border width in image generator is {0}".format(igBorderWidth))           
        return ig
    
    def RunExample(self):
        super(Mrfm2dBlurWithNoise, self).RunExample()
        self.debugMessages.append("Blur shift is: {0}".format(self.channelChain.channelBlocks[1].BlurShift))
        
    def Plot(self):
        # In order to remove the shift, must access the SyntheticBlur block in the channel chain
        blurredImageWithNoiseForDisplay = self.channelChain \
                                              .channelBlocks[1] \
                                              .RemoveShiftFromBlurredImage(self.blurredImageWithNoise)
                                
        # Random sparse image                                
        plt.figure(1); plt.imshow(self.channelChain.intermediateOutput[0], interpolation='none'); plt.colorbar()
        # Blur
        plt.figure(); plt.imshow(self.blurPsfInThetaFrame, interpolation='none'); plt.colorbar()      
        # After convolution with the blur psf + noise
        plt.figure(); plt.imshow(blurredImageWithNoiseForDisplay, interpolation='none'); plt.colorbar()
        plt.show()     
                    
if __name__ == "__main__":           
    # Construct the example object and invoke its RunExample method
    ex = Mrfm2dBlurWithNoise(Mrfm2dBlurWithNoise.GetBlurParameterOptimizer(),
                             { 
                              AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB: 20,
                              AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE: (42, 42),
                              AbstractImageGenerator.INPUT_KEY_NUM_NONZERO: 16
                              }
                             )
    ex.RunExample()    
    print "\n".join(ex.debugMessages)
    print("Channel block timing [ms]: {0}".format(
                                                  ", ".join(
                                                            [str(tElapsed) for tElapsed in ex.channelChain.channelBlocksTiming]
                                                            )
                                                  )
          )
    ex.Plot()
    
    

        
              
