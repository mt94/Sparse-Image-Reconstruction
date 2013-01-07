import pylab as plt

from Channel.ChannelProcessingChain import ChannelProcessingChain
from Example import AbstractExample
from Sim.Blur import Blur
from Sim.ImageGenerator import AbstractImageGenerator, ImageGeneratorFactory 
from Sim.NoiseGenerator import AbstractAdditiveNoiseGenerator, NoiseGeneratorFactory
from Recon.PsfNormalizer import PsfNormalizer

class GaussianBlurWithNoise(AbstractExample):   
    def __init__(self):
        super(GaussianBlurWithNoise, self).__init__('Gaussian Blur with additive Gaussian noise')
        self.blurredImageWithNoise = None
        self.channelChain = None
         
    def RunExample(self):        
        # Construct the processing chain
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
        ng = NoiseGeneratorFactory.GetNoiseGenerator('additive_gaussian')
        ng.SetParameters(**{
                            AbstractAdditiveNoiseGenerator.INPUT_KEY_SIGMA: 2e-3
                            }
                         )
        channelChain.channelBlocks.append(ng)
        
        # Run
        self.channelChain = channelChain
        self.blurredImageWithNoise = channelChain.RunChain()
                    
        """ Calculate the spectral radius of H*H^T. Must do this after running the chain,
            since gb.blurPsf is only created when the Blur channel block gets called. This
            isn't an issue since PsfNormalizer is intended to be used in reconstruction,
            hence another processing chain.
        """
        gbNormalizer = PsfNormalizer(1)
        gbNormalizer.NormalizePsf(gb.BlurPsfInThetaFrame)
        print 'Spectral radius of H*H^T is:', gbNormalizer.GetSpectralRadiusGramMatrixRowsH()
                
        
    
if __name__ == "__main__":    
    ex = GaussianBlurWithNoise()
    ex.RunExample()
    # In order to remove the shift, must access the Blur block in the channel chain
    blurredImageWithNoiseForDisplay = ex.channelChain \
                                        .channelBlocks[1] \
                                        .RemoveShiftFromBlurredImage(ex.blurredImageWithNoise)
    plt.figure(1)
    plt.imshow(ex.channelChain.intermediateOutput[0])
    plt.figure(2)
    plt.imshow(blurredImageWithNoiseForDisplay)
    # Optionally, plt.colorbar()
    # Run plt.ion() followed by plt.show() in ipython

    
