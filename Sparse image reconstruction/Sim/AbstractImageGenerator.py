import abc
from ..Channel import ChannelBlock as chb

class AbstractImageGenerator(chb.AbstractChannelBlock):        
    # Input keys
    INPUT_KEY_IMAGE_TYPE = 'image_type'
    INPUT_KEY_IMAGE_SHAPE = 'image_shape'
    INPUT_KEY_IMAGE_DISCRETE_NZVALUES = 'image_discrete_nzvalues'
    INPUT_KEY_NUM_NONZERO = 'num_nonzero'
    INPUT_KEY_BORDER_WIDTH = 'border_width'
    
    CHANNEL_BLOCK_TYPE = 'ImageGenerator'

    def __init__(self):
        super(AbstractImageGenerator,self).__init__(AbstractImageGenerator.CHANNEL_BLOCK_TYPE)
        self.imageShape = None
        self.numNonzero = 0
        self.borderWidth = 0        

    def SetParameters(self, **kwargs):
        # Ensure that mandatory keys are present
        assert AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE in kwargs
        assert AbstractImageGenerator.INPUT_KEY_NUM_NONZERO in kwargs
        
        self.imageShape = kwargs[AbstractImageGenerator.INPUT_KEY_IMAGE_SHAPE]
        assert (len(self.imageShape) == 2) or (len(self.imageShape) == 3)        
        
        # Assume that we want less than half the image as ones
        self.numNonzero = kwargs[AbstractImageGenerator.INPUT_KEY_NUM_NONZERO]
        assert (self.numNonzero >= 0) and (self.numNonzero < self.imageShape[0] * self.imageShape[1] * 0.5)
        
        # Look for optional keys
        if AbstractImageGenerator.INPUT_KEY_BORDER_WIDTH in kwargs:
            self.borderWidth = kwargs[AbstractImageGenerator.INPUT_KEY_BORDER_WIDTH]
            # Don't do an assert here -- let border width be a tuple, for ex.
            #assert self.borderWidth >= 0
                
    @abc.abstractmethod
    def Generate(self):
        raise NotImplementedError('No default abstract method implementation')