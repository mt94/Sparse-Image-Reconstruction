import pylab as plt

from Channel.ChannelProcessingChain import ChannelProcessingChain
from AbstractExample import AbstractExample
from Sim.SyntheticBlur import SyntheticBlur
from Sim.ImageGenerator import AbstractImageGenerator, ImageGeneratorFactory 
from Sim.NoiseGenerator import AbstractAdditiveNoiseGenerator, NoiseGeneratorFactory
from Systems.PsfNormalizer import PsfMatrixNormNormalizer

class GaussianBlurWithNoise(AbstractExample):
    """
    Simulates 2d Gaussian blur and optionally adds AWGN.
    """
    
    INPUT_KEY_NOISE_SIGMA = 'noiseSigma'
    INPUT_KEY_SNR_DB = 'snrDb'
           
    def __init__(self, simParametersDict):
        super(GaussianBlurWithNoise, self).__init__('Gaussian SyntheticBlur with additive Gaussian noise example')
        self._simParametersDict = simParametersDict
        self.blurredImageWithNoise = None
        self.channelChain = None
        self.blurPsfInThetaFrame = None        

    @property
    def NoiseSigma(self):
        if GaussianBlurWithNoise.INPUT_KEY_NOISE_SIGMA in self._simParametersDict:
            return self._simParametersDict[GaussianBlurWithNoise.INPUT_KEY_NOISE_SIGMA]
        else:
            return None

    @property
    def SnrDb(self):
        if GaussianBlurWithNoise.INPUT_KEY_SNR_DB in self._simParametersDict:
            return self._simParametersDict[GaussianBlurWithNoise.INPUT_KEY_SNR_DB]
        else:
            return None

    """ Abstract method override """                
    def RunExample(self):        
        # Get simulation parameters if present; otherwise, assume default values
        try:
            imageShape = self._simParametersDict[AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE]
        except KeyError:
            imageShape = (32, 32)
            
        try:
            blurFwhm = self._simParametersDict[SyntheticBlur.INPUT_KEY_FWHM]
            blurNkhalf = self._simParametersDict[SyntheticBlur.INPUT_KEY_NKHALF]
        except KeyError:
            blurFwhm = 3
            blurNkhalf = 5
            
        try:
            numNonzero = self._simParametersDict[AbstractImageGenerator.INPUT_KEY_NUM_NONZERO]
        except KeyError:
            numNonzero = 8
            
        noiseSigma = self.NoiseSigma            
        snrDb = self.SnrDb
             
        # Construct the processing chain
        channelChain = ChannelProcessingChain(True)
        
        ig = ImageGeneratorFactory.GetImageGenerator('random_binary_2d')
        ig.SetParameters(**{ 
                            AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE: imageShape,
                            AbstractImageGenerator.INPUT_KEY_NUM_NONZERO: numNonzero,
                            AbstractImageGenerator.INPUT_KEY_BORDER_WIDTH: blurNkhalf
                           }
                         )                                                                                                        
        channelChain.channelBlocks.append(ig)
        
        blurParametersDict = {
                              SyntheticBlur.INPUT_KEY_FWHM: blurFwhm,
                              SyntheticBlur.INPUT_KEY_NKHALF: blurNkhalf                              
                              }
        gb = SyntheticBlur(SyntheticBlur.BLUR_GAUSSIAN_SYMMETRIC_2D, blurParametersDict)        
        channelChain.channelBlocks.append(gb)
        
        ng = NoiseGeneratorFactory.GetNoiseGenerator('additive_gaussian')
        if (noiseSigma is not None) and (noiseSigma >= 0):
            ng.SetParameters(**{
                                AbstractAdditiveNoiseGenerator.INPUT_KEY_SIGMA: noiseSigma
                                }
                             )
        elif (snrDb is not None):
            ng.SetParameters(**{
                                AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB: snrDb
                                }
                             )
        else:
            raise NameError('noiseSigma or snrDb must be set')            
        channelChain.channelBlocks.append(ng)
        
        # Run
        self.channelChain = channelChain
        self.blurredImageWithNoise = channelChain.RunChain()
        
        if (noiseSigma is not None) and (snrDb is None):
            # Update snrDb
            self._simParametersDict[GaussianBlurWithNoise.INPUT_KEY_SNR_DB] = ng.snrDb
        elif (noiseSigma is None) and (snrDb is not None):
            # Update noiseSigma
            self._simParametersDict[GaussianBlurWithNoise.INPUT_KEY_NOISE_SIGMA] = ng.gaussianNoiseSigma
                    
        self.blurPsfInThetaFrame = gb.BlurPsfInThetaFrame
                
        
    
if __name__ == "__main__":    
    ex = GaussianBlurWithNoise({'snrDb': 20})
    ex.RunExample()
    
    """ Calculate the spectral radius of H*H^T. Must do this after running the chain,
        since gb.blurPsf is only created when the SyntheticBlur channel block gets called. This
        isn't an issue since PsfMatrixNormNormalizer is intended to be used in reconstruction,
        hence another processing chain.
    """
    gbNormalizer = PsfMatrixNormNormalizer(1)
    gbNormalizer.NormalizePsf(ex.blurPsfInThetaFrame)
    print 'Spectral radius of H*H^T is:', gbNormalizer.GetSpectralRadiusGramMatrixRowsH()

    
    # In order to remove the shift, must access the SyntheticBlur block in the channel chain
    blurredImageWithNoiseForDisplay = ex.channelChain \
                                        .channelBlocks[1] \
                                        .RemoveShiftFromBlurredImage(ex.blurredImageWithNoise)
    plt.figure(1)
    plt.imshow(ex.channelChain.intermediateOutput[0], interpolation='none')
    plt.figure(2)
    plt.imshow(blurredImageWithNoiseForDisplay, interpolation='none')
    plt.colorbar()
    plt.show()
    # Optionally, plt.colorbar()
    # Run plt.ion() followed by plt.show() in ipython

    
