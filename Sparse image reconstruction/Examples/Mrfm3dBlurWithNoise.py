import math
#import numpy as np
import pylab as plt

from AbstractBlurWithNoise import AbstractBlurWithNoise
from MrfmBlurExample import MrfmBlurExample
from Sim.AbstractImageGenerator import AbstractImageGenerator
from Sim.ImageGeneratorFactory import ImageGeneratorFactory 
from Sim.MrfmBlur import MrfmBlur
from Sim.MrfmBlurParameterOptimizer import MrfmBlurParameterOptimizer
from Sim.NoiseGenerator import AbstractAdditiveNoiseGenerator

class Mrfm3dBlurWithNoise(AbstractBlurWithNoise):
    """
    Simulates 3d MRFM blur and optionally adds AWGN.
    """ 
    @staticmethod
    def GetBlurParameterOptimizer():
        # Generate MRFM psf used in 04/f/sp_img_recon.m (less realistic parameters than those used in psf_sim_sing.m)
        opti = MrfmBlurParameterOptimizer(deltaB0=100)
        opti.bUseSmallerR0 = True
        opti.bUseSmallerDeltaB0 = False
        opti.CalcOptimalValues(1e4, 3)
        return opti   

    @staticmethod  
    def _PlotXySlices(fignumStart, img):
        imgShape = img.shape
        assert len(imgShape) == 3        
        # Plot the 3-d psf by plotting all x-y slices
        for zInd in range(imgShape[2]):
            plt.figure(fignumStart + zInd)
            plt.imshow(img[:, :, zInd], interpolation='none')        
            plt.xlabel('x'), plt.ylabel('y') 
#            plt.colorbar()                
            plt.title("slice " + str(zInd + 1))        
            
    # Constants
    IMG_INITIAL = 0
    IMG_BLUR_PSF = 1
    IMG_BLURRED = 2
    
    def __init__(self, optimizer, simParametersDict):
        super(Mrfm3dBlurWithNoise, self).__init__(optimizer, simParametersDict, '3-d Mrfm blur with additive Gaussian noise example')    
        assert len(self.ImageShape) == 3
        assert self.ImageShape[0] == self.ImageShape[1]; # Constrain the x and y dims to have equal length
        self.debugMessages = []    
            
    """ Abstract method override """
    def GetBlur(self):   
        blurEx = MrfmBlurExample(self._optimizer, (self.ImageShape[0], self.ImageShape[2]), MrfmBlur.BLUR_3D, '').RunExample() 
        self._psfSupport = blurEx.Blur.PsfSupport
        return blurEx.Blur
        
    def GetImageGenerator(self):
        ig = ImageGeneratorFactory.GetImageGenerator('random_binary_3d')
        # 1 shouldn't be necessary really
        igBorderWidth = [(int(math.ceil((max(suppVec) - min(suppVec))/2.0)) + 1) for suppVec in self._psfSupport]                                                  
        ig.SetParameters(**{ 
                            AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE: self.ImageShape,
                            AbstractImageGenerator.INPUT_KEY_NUM_NONZERO: self.NumNonzero,
                            AbstractImageGenerator.INPUT_KEY_BORDER_WIDTH: igBorderWidth
                           }
                         )
        self.debugMessages.append("Border width in image generator is {0}".format(igBorderWidth))           
        return ig
        
    def RunExample(self):
        super(Mrfm3dBlurWithNoise, self).RunExample()
        self.debugMessages.append("Blur shift is: {0}".format(ex.channelChain.channelBlocks[1].BlurShift))
                      
    def Plot(self, plotType, fignumStart=1):
        # Can't create plots of the 3 images as in Mrfm2dBlurWithNoise. The user has to select one
        # image to plot
        imgToPlot = {
                     Mrfm3dBlurWithNoise.IMG_INITIAL: self.channelChain.intermediateOutput[0],
                     Mrfm3dBlurWithNoise.IMG_BLUR_PSF: self.blurPsfInThetaFrame,
                     Mrfm3dBlurWithNoise.IMG_BLURRED: self.channelChain.channelBlocks[1].RemoveShiftFromBlurredImage(self.blurredImageWithNoise)
                     }.get(plotType, None)                 
        if imgToPlot is not None:
            self._PlotXySlices(fignumStart, imgToPlot)
    
if __name__ == "__main__":
    # Construct the example object and invoke its RunExample method
    ex = Mrfm3dBlurWithNoise(Mrfm3dBlurWithNoise.GetBlurParameterOptimizer(),
                             { 
                              AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB: 20,
                              AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE: (32, 32, 12)
                              }
                             )
    ex.RunExample()  
    print "\n".join(ex.debugMessages)
     
#    ex.Plot(Mrfm3dBlurWithNoise.IMG_BLUR_PSF, 1)
    ex.Plot(Mrfm3dBlurWithNoise.IMG_INITIAL, 1)   
    ex.Plot(Mrfm3dBlurWithNoise.IMG_BLURRED, 101)
    plt.show()      
