import pylab as plt

from Channel.ChannelProcessingChain import ChannelProcessingChain
from AbstractExample import AbstractExample
from Sim.Blur import Blur
from Sim.ImageGenerator import AbstractImageGenerator, ImageGeneratorFactory 
from Sim.NoiseGenerator import AbstractAdditiveNoiseGenerator, NoiseGeneratorFactory
from Systems.PsfNormalizer import PsfMatrixNormNormalizer

class GaussianBlurWithNoise(AbstractExample):   
    def __init__(self, noiseSigma=None, snrDb=None):
        super(GaussianBlurWithNoise, self).__init__('Gaussian Blur with additive Gaussian noise example')
        self.blurredImageWithNoise = None
        self.channelChain = None        
        self.noiseSigma = noiseSigma
        self.snrDb = snrDb
         
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
        if (self.noiseSigma is not None) and (self.noiseSigma >= 0):
            ng.SetParameters(**{
                                AbstractAdditiveNoiseGenerator.INPUT_KEY_SIGMA: self.noiseSigma
                                }
                             )
        elif (self.snrDb is not None):
            ng.SetParameters(**{
                                AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB: self.snrDb
                                }
                             )
        else:
            raise NameError('noiseSigma or snrDb must be set')            
        channelChain.channelBlocks.append(ng)
        
        # Run
        self.channelChain = channelChain
        self.blurredImageWithNoise = channelChain.RunChain()
        if (self.noiseSigma is not None) and (self.snrDb is None):
            # Update snrDb
            self.snrDb = ng.snrDb
        elif (self.noiseSigma is None) and (self.snrDb is not None):
            # Update noiseSigma
            self.noiseSigma = ng.gaussianNoiseSigma
                    
        """ Calculate the spectral radius of H*H^T. Must do this after running the chain,
            since gb.blurPsf is only created when the Blur channel block gets called. This
            isn't an issue since PsfMatrixNormNormalizer is intended to be used in reconstruction,
            hence another processing chain.
        """
        gbNormalizer = PsfMatrixNormNormalizer(1)
        gbNormalizer.NormalizePsf(gb.BlurPsfInThetaFrame)
#        print 'Spectral radius of H*H^T is:', gbNormalizer.GetSpectralRadiusGramMatrixRowsH()
                
        
    
if __name__ == "__main__":    
    ex = GaussianBlurWithNoise(snrDb=20)
    ex.RunExample()
    # In order to remove the shift, must access the Blur block in the channel chain
    blurredImageWithNoiseForDisplay = ex.channelChain \
                                        .channelBlocks[1] \
                                        .RemoveShiftFromBlurredImage(ex.blurredImageWithNoise)
    plt.figure(1)
    plt.imshow(ex.channelChain.intermediateOutput[0])
    plt.figure(2)
    plt.imshow(blurredImageWithNoiseForDisplay)
    plt.colorbar()
    plt.show()
    # Optionally, plt.colorbar()
    # Run plt.ion() followed by plt.show() in ipython

    
