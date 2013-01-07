import abc

class AbstractChannelBlock(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, channelBlockType=None):
        self.channelBlockType = channelBlockType
        
    
