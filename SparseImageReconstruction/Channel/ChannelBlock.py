import abc


class AbstractChannelBlock(object):
    __metaclass__ = abc.ABCMeta

    DEFAULT_CHANNEL_BLOCK_TYPE = "Default"

    def __init__(self, channelBlockType=None):
        super(AbstractChannelBlock, self).__init__()
        self.channelBlockType = channelBlockType
