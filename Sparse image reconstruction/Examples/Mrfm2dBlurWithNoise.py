import math
import numpy as np
import pylab as plt

from Channel.ChannelProcessingChain import ChannelProcessingChain
from AbstractExample import AbstractExample
from Sim.AbstractImageGenerator import AbstractImageGenerator
from Sim.ImageGeneratorFactory import ImageGeneratorFactory 
from Sim.MrfmBlur import MrfmBlur
from Sim.MrfmBlurParameterOptimizer import MrfmBlurParameterOptimizer
from Sim.NoiseGenerator import AbstractAdditiveNoiseGenerator, NoiseGeneratorFactory

class Mrfm2dBlurWithNoise(AbstractExample):
    """
    Simulates 2d MRFM blur and optionally adds AWGN.
    """
    
    @staticmethod
    def GetParameterOptimizer():
        # Generate MRFM psf used in 04/f/sp_img_recon.m (less realistic parameters than those used in psf_sim_sing.m)
        opti = MrfmBlurParameterOptimizer(deltaB0=100)
        opti.bUseSmallerR0 = True
        opti.bUseSmallerDeltaB0 = False
        opti.CalcOptimalValues(1e4, 6, R0=4)
        return opti
        
    def __init__(self, optimizer, simParametersDict):
        super(Mrfm2dBlurWithNoise, self).__init__('MrfmBlur with additive Gaussian noise example')
        self._optimizer = optimizer
        self._simParametersDict = simParametersDict
        self.blurredImageWithNoise = None
        self.channelChain = None
        self.blurPsfInThetaFrame = None      
                
    @property
    def NoiseSigma(self):
        return self._simParametersDict.get(AbstractAdditiveNoiseGenerator.INPUT_KEY_SIGMA)

    @property
    def SnrDb(self):
        return self._simParametersDict.get(AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB)
    
    """ Abstract method override """                
    def RunExample(self): 
        try:
            numNonzero = self._simParametersDict[AbstractImageGenerator.INPUT_KEY_NUM_NONZERO]
        except KeyError:
            numNonzero = 8
                    
        try:
            imageShape = self._simParametersDict[AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE]
        except KeyError:
            imageShape = (42, 42)

        noiseSigma = self.NoiseSigma            
        snrDb = self.SnrDb

        # Use numpy.mgrid to generate 3-d grid mesh          
        xyzMesh = MrfmBlur.GetXyzMeshFor2d(self._optimizer.xSpan, 
                                           self._optimizer.z0, 
                                           np.min([32, imageShape[0]])
                                           )
                    
        # Create the MRFM blur      
        mb = MrfmBlur(MrfmBlur.BLUR_2D, 
                      {
                       MrfmBlur.INPUT_KEY_BEXT: self._optimizer.Bext,
                       MrfmBlur.INPUT_KEY_BRES: self._optimizer.Bres,
                       MrfmBlur.INPUT_KEY_SMALL_M: self._optimizer.m,
                       MrfmBlur.INPUT_KEY_XPK: self._optimizer.xPk,
                       MrfmBlur.INPUT_KEY_XMESH: np.array(xyzMesh[1], dtype=float),                                                 
                       MrfmBlur.INPUT_KEY_YMESH: np.array(xyzMesh[0], dtype=float),
                       MrfmBlur.INPUT_KEY_ZMESH: np.array(xyzMesh[2], dtype=float)
                       }
                      )
        mb._GetBlurPsf()
    
        # Construct the processing chain
        channelChain = ChannelProcessingChain(True)
        
        ig = ImageGeneratorFactory.GetImageGenerator('random_binary_2d')
        igBorderWidth = int(math.ceil((max(mb.PsfSupport[0]) - min(mb.PsfSupport[0]))/2.0)) + 1; # 1 shouldn't be necessary really
        ig.SetParameters(**{ 
                            AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE: imageShape,
                            AbstractImageGenerator.INPUT_KEY_NUM_NONZERO: numNonzero,
                            AbstractImageGenerator.INPUT_KEY_BORDER_WIDTH: igBorderWidth
                           }
                         )
#        print("Border width in image generator is {0}".format(igBorderWidth))
        
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
        
        channelChain.channelBlocks.append(ig); # image generator                        
        channelChain.channelBlocks.append(mb); # MRFM blur
        channelChain.channelBlocks.append(ng); # noise generator
        
        # Run
        self.channelChain = channelChain
        self.blurredImageWithNoise = channelChain.RunChain()

        # Either noiseSigma or SNR dB must be specified. Update the other qty            
        if (noiseSigma is not None) and (snrDb is None):
            # Update snrDb
            self._simParametersDict[AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB] = ng.snrDb
        elif (noiseSigma is None) and (snrDb is not None):
            # Update noiseSigma
            self._simParametersDict[AbstractAdditiveNoiseGenerator.INPUT_KEY_SIGMA] = ng.gaussianNoiseSigma
                    
        self.blurPsfInThetaFrame = mb.BlurPsfInThetaFrame      
                
if __name__ == "__main__":           
    # Construct the example object
    ex = Mrfm2dBlurWithNoise(Mrfm2dBlurWithNoise.GetParameterOptimizer(), 
                             { 
                              AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB: 20 
                              }
                             )
    ex.RunExample()
       
    # In order to remove the shift, must access the SyntheticBlur block in the channel chain
    print("Blur shift is: " + str(ex.channelChain.channelBlocks[1].BlurShift))
    blurredImageWithNoiseForDisplay = ex.channelChain \
                                        .channelBlocks[1] \
                                        .RemoveShiftFromBlurredImage(ex.blurredImageWithNoise)
    plt.figure(1); plt.imshow(ex.channelChain.intermediateOutput[0], interpolation='none'); plt.colorbar();
    plt.figure(); plt.imshow(ex.blurPsfInThetaFrame, interpolation='none'); plt.colorbar()
    plt.figure(); plt.imshow(blurredImageWithNoiseForDisplay, interpolation='none'); plt.colorbar()
    plt.show()               