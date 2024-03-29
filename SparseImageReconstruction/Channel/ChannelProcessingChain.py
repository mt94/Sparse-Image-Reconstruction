from .ChannelBlock import AbstractChannelBlock
from ..Sim.AbstractImageGenerator import AbstractImageGenerator
from ..Sim.Blur import AbstractBlur
from ..Sim.NoiseGenerator import AbstractAdditiveNoiseGenerator, AbstractNoiseGenerator
from ..Systems.Timer import Timer


class ChannelProcessingChain(object):
    @staticmethod
    def ProcessImageGenerator(channelBlock):
        if not issubclass(channelBlock.__class__, AbstractImageGenerator):
            raise TypeError(
                "Expect channelBlock to be an AbstractImageGenerator. Instead, it's a "
                + channelBlock.__class__.__name__
            )
        else:
            return channelBlock.Generate()

    @staticmethod
    def ProcessBlur(channelBlock, theta):
        if not issubclass(channelBlock.__class__, AbstractBlur):
            raise TypeError(
                "Expect channelBlock to be a Blur. Instead, it's a "
                + channelBlock.__class__.__name__
            )
        else:
            return channelBlock.Blur(theta)

    @staticmethod
    def ProcessNoiseGenerator(channelBlock, y):
        if issubclass(channelBlock.__class__, AbstractAdditiveNoiseGenerator):
            return y + channelBlock.Generate(y)
        elif issubclass(channelBlock.__class__, AbstractNoiseGenerator):
            return channelBlock.Generate(y)
        else:
            raise TypeError(
                "Expect channelBlock to be a AbstractAdditiveNoiseGenerator. Instead, it's a "
                + channelBlock.__class__.__name__
            )

    @staticmethod
    def ProcessDefault(channelBlock, x):
        if callable(getattr(channelBlock, "Process")):
            return channelBlock.Process(x)
        raise TypeError("Expect a Default channelBlock to have a Process method.")

    _channelBlockFunctionAcceptsInputDict = {
        "ImageGenerator": False,
        "Blur": True,
        "AdditiveNoiseGenerator": True,
        "NoiseGenerator": True,
        "Default": True,
    }

    _channelBlockFunctionDict = {
        "ImageGenerator": ProcessImageGenerator,
        "Blur": ProcessBlur,
        "AdditiveNoiseGenerator": ProcessNoiseGenerator,
        "NoiseGenerator": ProcessNoiseGenerator,
        "Default": ProcessDefault,
    }

    def __init__(self, bSaveAllIntermediateOutput=False):
        self.channelBlocks = []
        self.channelBlocksTiming = []
        self.bSaveAllIntermediateOutput = bSaveAllIntermediateOutput
        self.intermediateOutput = []

    def RunChain(self, chainInput=None):
        cbInput = chainInput
        cbOutput = None

        for channelBlock in self.channelBlocks:
            if not isinstance(channelBlock, AbstractChannelBlock):
                raise TypeError(
                    "channelBlocks must consist of objects derived from AbstractChannelBlock"
                )

            if (
                channelBlock.channelBlockType
                in ChannelProcessingChain._channelBlockFunctionDict
            ):
                assert (
                    channelBlock.channelBlockType
                    in ChannelProcessingChain._channelBlockFunctionAcceptsInputDict
                )
                cbFunc = ChannelProcessingChain._channelBlockFunctionDict[
                    channelBlock.channelBlockType
                ]
                with Timer() as t:
                    # XXX Must call the __get__ method to convert a descriptor to a callable object
                    if not ChannelProcessingChain._channelBlockFunctionAcceptsInputDict[
                        channelBlock.channelBlockType
                    ]:
                        cbOutput = cbFunc.__get__(None, ChannelProcessingChain)(
                            channelBlock
                        )
                    else:
                        assert cbInput is not None
                        cbOutput = cbFunc.__get__(None, ChannelProcessingChain)(
                            channelBlock, cbInput
                        )
                self.channelBlocksTiming.append(t.msecs)
            else:
                # Only recognize several types of AbstractChannelBlock
                raise NotImplementedError(
                    "Don't know how to handle channel block: "
                    + channelBlock.channelBlockType
                )

            """ Save the block output only if:
                a) it's a source that occurs at the very first block -- or --
                b) we're being asked to save all intermediate outputs
            """
            if ((cbInput is None) and (cbOutput is not None)) or (
                self.bSaveAllIntermediateOutput is True
            ):
                self.intermediateOutput.append(cbOutput)

            cbInput = cbOutput

        # Save the last output
        self.intermediateOutput.append(cbOutput)
        return cbOutput
