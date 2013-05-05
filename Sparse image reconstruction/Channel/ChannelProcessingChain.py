from ChannelBlock import AbstractChannelBlock
from Sim.ImageGenerator import AbstractImageGenerator
from Sim.Blur import AbstractBlur
from Sim.NoiseGenerator import AbstractAdditiveNoiseGenerator

class ChannelProcessingChain(object):
    @staticmethod
    def ProcessImageGenerator(channelBlock):
        if not issubclass(channelBlock.__class__, AbstractImageGenerator):
            raise TypeError('Expect channelBlock to be an AbstractImageGenerator. Instead, it\'s a ' +
                            channelBlock.__class__.__name__);
        else:
            return channelBlock.Generate()
    
    @staticmethod
    def ProcessBlur(channelBlock, theta):
        if not issubclass(channelBlock.__class__, AbstractBlur):
            raise TypeError('Expect channelBlock to be a Blur. Instead, it\'s a ' + 
                            channelBlock.__class__.__name__);
        else:
            return channelBlock.BlurImage(theta)
        
    @staticmethod
    def ProcessAdditiveNoiseGenerator(channelBlock, y):        
        if not issubclass(channelBlock.__class__, AbstractAdditiveNoiseGenerator):
            raise TypeError('Expect channelBlock to be a AbstractAdditiveNoiseGenerator. Instead, it\'s a ' +
                            channelBlock.__class__.__name__)
        else:
            return y + channelBlock.Generate(y)                

    _channelBlockFunctionAcceptsInputDict = {
                                             'ImageGenerator': False,
                                             'Blur': True,
                                             'AdditiveNoiseGenerator': True
                                            }
    _channelBlockFunctionDict = {
                                 'ImageGenerator': ProcessImageGenerator,
                                 'Blur': ProcessBlur,
                                 'AdditiveNoiseGenerator': ProcessAdditiveNoiseGenerator
                                }
    
    def __init__(self, bSaveAllIntermediateOutput=False):
        self.channelBlocks = []
        self.bSaveAllIntermediateOutput = bSaveAllIntermediateOutput
        self.intermediateOutput = []
            
    def RunChain(self, chainInput=None):
        cbInput = chainInput
        cbOutput = None
                
        for channelBlock in self.channelBlocks:            
            
            if not isinstance(channelBlock, AbstractChannelBlock):
                raise TypeError('channelBlocks must consist of objects derived from AbstractChannelBlock')            
            
            if channelBlock.channelBlockType in ChannelProcessingChain._channelBlockFunctionDict:
                assert channelBlock.channelBlockType in ChannelProcessingChain._channelBlockFunctionAcceptsInputDict
                cbFunc = ChannelProcessingChain._channelBlockFunctionDict[channelBlock.channelBlockType]
                # XXX Must call the __get__ method to convert a descriptor to a callable object                
                if not ChannelProcessingChain._channelBlockFunctionAcceptsInputDict[channelBlock.channelBlockType]:
                    cbOutput = cbFunc.__get__(None, ChannelProcessingChain)(channelBlock)
                else:
                    assert cbInput is not None                    
                    cbOutput = cbFunc.__get__(None, ChannelProcessingChain)(channelBlock, cbInput)
            else:
                # Only recognize several types of AbstractChannelBlock
                raise NotImplementedError('Don\'t know how to handle channel block: ' +
                                          channelBlock.channelBlockType) 

            """ Save the block output only if:
                a) it's a source that occurs at the very first block -- or --
                b) we're being asked to save all intermediate outputs
            """
            if ((cbInput is None) and (cbOutput is not None)) or (self.bSaveAllIntermediateOutput is True):
                self.intermediateOutput.append(cbOutput)
                
            cbInput = cbOutput
                        
        # Save the last output
        self.intermediateOutput.append(cbOutput)
        return cbOutput
                    
                
                
                
                
                
                
            
            
            
