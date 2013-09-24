import numpy as np
from Sim.AbstractImageGenerator import AbstractImageGenerator

class RandomBinary2dImageGenerator(AbstractImageGenerator):    
    def __init__(self):
        super(RandomBinary2dImageGenerator, self).__init__()                    
    def Generate(self):
        assert self.imageShape is not None
        img = np.zeros(self.imageShape)
        
        rowInterval = (self.borderWidth, self.imageShape[0] - self.borderWidth)
        colInterval = (self.borderWidth, self.imageShape[1] - self.borderWidth)
        if (rowInterval[1] < rowInterval[0]) or (colInterval[1] < colInterval[0]):
            raise ValueError('border width isn\'t compatible with image shape')
            
        # This is hugely inefficient if the image isn't at all sparse
        numOnes = 0
        while numOnes < self.numNonzero:
            rowRand = np.random.randint(rowInterval[0], rowInterval[1])
            colRand = np.random.randint(colInterval[0], colInterval[1])
            if img[rowRand, colRand] == 0:
                img[rowRand, colRand] = 1
                numOnes += 1
        return img
    
class RandomUniform2dImageGenerator(AbstractImageGenerator):
    # Input keys
    INPUT_KEY_UNIFORMRV_RANGE = 'uniformrv_range'
    
    def __init__(self):
        super(RandomUniform2dImageGenerator, self).__init__()
        self.uniformRvRange = None
        
    def SetParameters(self, **kwargs):
        assert RandomUniform2dImageGenerator.INPUT_KEY_UNIFORMRV_RANGE in kwargs
        self.uniformRvRange = kwargs[RandomUniform2dImageGenerator.INPUT_KEY_UNIFORMRV_RANGE]
        super(RandomUniform2dImageGenerator, self).SetParameters(**kwargs)
        
    def Generate(self):
        assert self.imageShape is not None
        img = np.zeros(self.imageShape)
        
        rowInterval = (self.borderWidth, self.imageShape[0] - self.borderWidth)
        colInterval = (self.borderWidth, self.imageShape[1] - self.borderWidth)
        if (rowInterval[1] < rowInterval[0]) or (colInterval[1] < colInterval[0]):
            raise ValueError('border width isn\'t compatible with image shape')
                    
        uniformRvIid = self.uniformRvRange[0] + np.random.rand(1, self.numNonzero) * (self.uniformRvRange[1] - self.uniformRvRange[0])
        
        # This is hugely inefficient if the image isn't at all sparse
        numNonzero = 0
        while numNonzero < self.numNonzero:
            rowRand = np.random.randint(rowInterval[0], rowInterval[1])
            colRand = np.random.randint(colInterval[0], colInterval[1])
            if img[rowRand, colRand] == 0:
                img[rowRand, colRand] = uniformRvIid[0, numNonzero] 
                numNonzero += 1
        return img