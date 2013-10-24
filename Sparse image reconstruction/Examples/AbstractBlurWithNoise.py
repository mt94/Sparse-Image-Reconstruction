import abc

from AbstractExample import AbstractExample
from Channel.ChannelProcessingChain import ChannelProcessingChain
from Sim.AbstractImageGenerator import AbstractImageGenerator
from Sim.NoiseGenerator import AbstractAdditiveNoiseGenerator, NoiseGeneratorFactory

class AbstractBlurWithNoise(AbstractExample):
        
    RUN_ONCE_ONLY = 'run_once_only'
    
    def __init__(self, optimizer, simParametersDict, desc):
        super(AbstractBlurWithNoise, self).__init__(desc)
        self._optimizer = optimizer
        self._simParametersDict = simParametersDict
        self._psfSupport = None
        self._bRunAlready = False
        self.blurredImageWithNoise = None
        self.channelChain = None
        self.blurPsfInThetaFrame = None     
        
    @property
    def RunAlready(self):
        return self._bRunAlready
    
    @property
    def NoiseSigma(self):
        return self._simParametersDict.get(AbstractAdditiveNoiseGenerator.INPUT_KEY_SIGMA)

    @property
    def SnrDb(self):
        return self._simParametersDict.get(AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB)
    
    @property
    def NumNonzero(self):
        try:
            numNonzero = self._simParametersDict[AbstractImageGenerator.INPUT_KEY_NUM_NONZERO]
        except KeyError:
            numNonzero = 8
        return numNonzero
  
    @property
    def ImageType(self):
        try:
            imageType = self._simParametersDict[AbstractImageGenerator.INPUT_KEY_IMAGE_TYPE]
        except KeyError:
            imageType = 'random_binary'
        return imageType
    
    @property
    def ImageShape(self):                           
        return self._simParametersDict[AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE]
     
    """ Abstract methods """
    @abc.abstractmethod
    def GetImageGenerator(self):
        raise NotImplementedError('No default abstract method implementation')
    
    @abc.abstractmethod
    def GetBlur(self):
        raise NotImplementedError('No default abstract method implementation')
        
    """ Abstract method override """                
    def RunExample(self):
        
        if self._simParametersDict.get(AbstractBlurWithNoise.RUN_ONCE_ONLY, False) and self._bRunAlready:
            raise RuntimeError('RunExample already called')
                 
        # Construct the blur object first, then the image generator
        blr = self.GetBlur()
        ig = self.GetImageGenerator()
        
        # Next, the noise generator
        ng = NoiseGeneratorFactory.GetNoiseGenerator('additive_gaussian')
        if (self.NoiseSigma is not None) and (self.NoiseSigma >= 0):
            ng.SetParameters(**{
                                AbstractAdditiveNoiseGenerator.INPUT_KEY_SIGMA: self.NoiseSigma
                                }
                             )
        elif (self.SnrDb is not None):
            ng.SetParameters(**{
                                AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB: self.SnrDb
                                }
                             )
        else:
            raise NameError('noiseSigma or snrDb must be set') 
        
        # Construct the processing chain
        channelChain = ChannelProcessingChain(True)          
        channelChain.channelBlocks.append(ig); # image generator                        
        channelChain.channelBlocks.append(blr); # blur 
        channelChain.channelBlocks.append(ng); # noise generator
        
        # Run
        self.channelChain = channelChain
        self.blurredImageWithNoise = channelChain.RunChain()

        # Either noiseSigma or SNR dB must be specified. Update the other qty            
        if (self.NoiseSigma is not None) and (self.SnrDb is None):
            # Update snrDb
            self._simParametersDict[AbstractAdditiveNoiseGenerator.INPUT_KEY_SNRDB] = ng.snrDb
        elif (self.NoiseSigma is None) and (self.SnrDb is not None):
            # Update noiseSigma
            self._simParametersDict[AbstractAdditiveNoiseGenerator.INPUT_KEY_SIGMA] = ng.gaussianNoiseSigma
                    
        self.blurPsfInThetaFrame = blr.BlurPsfInThetaFrame
                
        if not self._bRunAlready:
            self._bRunAlready = True           