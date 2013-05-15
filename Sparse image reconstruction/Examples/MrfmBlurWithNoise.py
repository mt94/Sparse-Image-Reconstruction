import numpy as np
import pylab as plt

from Channel.ChannelProcessingChain import ChannelProcessingChain
from AbstractExample import AbstractExample
from Sim.MrfmBlur import MrfmBlur
from Sim.MrfmBlurParameterOptimizer import MrfmBlurParameterOptimizer
from Sim.ImageGenerator import AbstractImageGenerator, ImageGeneratorFactory 
from Sim.NoiseGenerator import AbstractAdditiveNoiseGenerator, NoiseGeneratorFactory

class MrfmBlurWithNoise(AbstractExample):
    """
    Simulates 2d MRFM blur and optionally adds AWGN.
    """
    def __init__(self, simParametersDict):
        super(MrfmBlurWithNoise, self).__init__('MrfmBlur with additive Gaussian noise example')
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
            imageShape = (32, 32)

        noiseSigma = self.NoiseSigma            
        snrDb = self.SnrDb

        # Use numpy.mgrid to generate 3-d grid mesh          
        xyzMesh = MrfmBlur.GetXyzMeshFor2d(self._simParametersDict['xspan'], 
                                           self._simParametersDict['z0'], 
                                           np.min([16, np.floor(imageShape[0]/2)])
                                           )
        self._simParametersDict[MrfmBlur.INPUT_KEY_XMESH] = np.array(xyzMesh[0], dtype=float)
        self._simParametersDict[MrfmBlur.INPUT_KEY_YMESH] = np.array(xyzMesh[1], dtype=float)
        self._simParametersDict[MrfmBlur.INPUT_KEY_ZMESH] = np.array(xyzMesh[2], dtype=float)
                    
        # Create the MRFM blur      
        mb = MrfmBlur(MrfmBlur.BLUR_2D, self._simParametersDict)
    
        # Construct the processing chain
        channelChain = ChannelProcessingChain(True)
        
        ig = ImageGeneratorFactory.GetImageGenerator('random_binary_2d')
        ig.SetParameters(**{ 
                            AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE: imageShape,
                            AbstractImageGenerator.INPUT_KEY_NUM_NONZERO: numNonzero,
                            AbstractImageGenerator.INPUT_KEY_BORDER_WIDTH: max(mb.BlurShift)
                           }
                         )
        
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
        
        channelChain.channelBlocks.append(ig)                               
        channelChain.channelBlocks.append(mb)            
        channelChain.channelBlocks.append(ng)       
        
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
    opti = MrfmBlurParameterOptimizer()
    opti.CalcOptimalValues(3e4, 3)   
    # Construct the example object
    ex = MrfmBlurWithNoise({
                            'snrdb': 20,
                            MrfmBlur.INPUT_KEY_BEXT: opti.Bext,
                            MrfmBlur.INPUT_KEY_BRES: opti.Bres,
                            MrfmBlur.INPUT_KEY_SMALL_M: opti.m,
                            MrfmBlur.INPUT_KEY_XPK: opti.xPk,
                            'xspan': opti.xSpan,
                            'z0': opti.z0
                            })
    ex.RunExample()              
