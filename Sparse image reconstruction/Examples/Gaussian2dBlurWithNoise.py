import pylab as plt

from .AbstractBlurWithNoise import AbstractBlurWithNoise
from ..Sim.AbstractImageGenerator import AbstractImageGenerator
from ..Sim.ImageGeneratorFactory import ImageGeneratorFactory
from ..Sim.ImageGeneratorImpl import INPUT_KEY_DISCRETE_VALUES
from ..Sim.NoiseGenerator import AbstractAdditiveNoiseGenerator
from ..Sim.SyntheticBlur import SyntheticBlur
from ..Systems.PsfNormalizer import PsfMatrixNormNormalizer


class Gaussian2dBlurWithNoise(AbstractBlurWithNoise):
    """
    Simulates 2d Gaussian blur and optionally adds AWGN.
    """
    def __init__(self, simParametersDict):
        super(Gaussian2dBlurWithNoise, self).__init__(None, simParametersDict, 'Gaussian SyntheticBlur with additive Gaussian noise example')        
      
    """ Abstract method override """  
    def GetBlur(self):
        # Get parameters for the Gaussian blur            
        try:
            blurFwhm = self._simParametersDict[SyntheticBlur.INPUT_KEY_FWHM]
            blurNkhalf = self._simParametersDict[SyntheticBlur.INPUT_KEY_NKHALF]
        except KeyError:
            blurFwhm = 3
            blurNkhalf = 5

        blurParametersDict = {
                              SyntheticBlur.INPUT_KEY_FWHM: blurFwhm,
                              SyntheticBlur.INPUT_KEY_NKHALF: blurNkhalf                              
                              }
        self._psfSupport = (blurNkhalf,)        
        return SyntheticBlur(SyntheticBlur.BLUR_GAUSSIAN_SYMMETRIC_2D, blurParametersDict) 
      
    def GetImageGenerator(self):
        ig = ImageGeneratorFactory.GetImageGenerator(self.ImageType)
        parameterDict = { 
                         AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE: self.ImageShape,
                         AbstractImageGenerator.INPUT_KEY_NUM_NONZERO: self.NumNonzero,
                         AbstractImageGenerator.INPUT_KEY_BORDER_WIDTH: self._psfSupport[0]
                         }
        if ((self.ImageDiscreteNonzeroValues is not None) and (len(self.ImageDiscreteNonzeroValues) > 0)):
            parameterDict[INPUT_KEY_DISCRETE_VALUES] = self.ImageDiscreteNonzeroValues
        ig.SetParameters(**parameterDict)                         
        return ig
                                                        
if __name__ == "__main__":    
    ex = Gaussian2dBlurWithNoise({
                                  AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB: 2,
                                  AbstractImageGenerator.INPUT_KEY_IMAGE_TYPE: 'random_discrete',
                                  AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE: (32, 32),                                  
                                  AbstractImageGenerator.INPUT_KEY_NUM_NONZERO: 12,
                                  AbstractImageGenerator.INPUT_KEY_IMAGE_DISCRETE_NZVALUES: [1, -1]                          
                                  })    
    ex.RunExample()
    
    """ Calculate the spectral radius of H*H^T. Must do this after running the chain,
        since gb.blurPsf is only created when the SyntheticBlur channel block gets called. This
        isn't an issue since PsfMatrixNormNormalizer is intended to be used in reconstruction,
        hence another processing chain.
    """
    gbNormalizer = PsfMatrixNormNormalizer(1)
    gbNormalizer.NormalizePsf(ex.blurPsfInThetaFrame)
    print('Spectral radius of H*H^T is:', gbNormalizer.GetSpectralRadiusGramMatrixRowsH())
    
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

    
