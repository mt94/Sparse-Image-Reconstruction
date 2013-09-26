import abc
import numpy as np
from Sim.AbstractImageGenerator import AbstractImageGenerator

"""
2d image generators
"""
class AbstractSparse2dImageGenerator(AbstractImageGenerator):    
    def __init__(self):
        super(AbstractSparse2dImageGenerator, self).__init__()          
    @abc.abstractmethod
    def SampleFromDistribution(self):
        raise NotImplementedError('No default abstract method implementation')
    def Generate(self):
        assert (self.imageShape is not None) and (len(self.imageShape) == 2)
        img = np.zeros(self.imageShape)
        
        assert self.borderWidth >= 0
        validInterval = np.array([[self.borderWidth, self.imageShape[0] - self.borderWidth],
                                  [self.borderWidth, self.imageShape[1] - self.borderWidth]]
                                 )        
        # SANITY
        if (validInterval[0, 1] < validInterval[0, 0]) or (validInterval[1, 1] < validInterval[1, 0]):
            raise ValueError("border width isn't compatible with image shape")
            
        # This is hugely inefficient if the image isn't at all sparse
        numOnes = 0
        while numOnes < self.numNonzero:
            pickRandom = (np.random.randint(validInterval[0, 0], validInterval[0, 1]),
                          np.random.randint(validInterval[1, 0], validInterval[1, 1]))            
            if img[pickRandom] == 0:
                img[pickRandom] = self.SampleFromDistribution()
                numOnes += 1
                
        return img
    
class SparseBinary2dImageGenerator(AbstractSparse2dImageGenerator):    
    def __init__(self):
        super(SparseBinary2dImageGenerator, self).__init__()                    
    # Abstract method override
    def SampleFromDistribution(self):
        return 1
    
class SparseUniform2dImageGenerator(AbstractSparse2dImageGenerator):
    # Input keys
    INPUT_KEY_UNIFORMRV_RANGE = 'uniformrv_range'
    
    def __init__(self):
        super(SparseUniform2dImageGenerator, self).__init__()
        self.uniformRvRange = None
        
    def SetParameters(self, **kwargs):
        assert SparseUniform2dImageGenerator.INPUT_KEY_UNIFORMRV_RANGE in kwargs
        self.uniformRvRange = kwargs[SparseUniform2dImageGenerator.INPUT_KEY_UNIFORMRV_RANGE]
        super(SparseUniform2dImageGenerator, self).SetParameters(**kwargs)
        
    # Abstract method override
    def SampleFromDistribution(self):
        sampleOfUniformRv = self.uniformRvRange[0] + np.random.rand() * (self.uniformRvRange[1] - self.uniformRvRange[0])
        return sampleOfUniformRv

        
"""
3d image generators
"""
        
class AbstractSparse3dImageGenerator(AbstractImageGenerator): 
    def __init__(self):
        super(AbstractSparse3dImageGenerator, self).__init__()
    @abc.abstractmethod
    def SampleFromDistribution(self):
        raise NotImplementedError('No default abstract method implementation')        
    def Generate(self):
        assert (self.imageShape is not None) and (len(self.imageShape) == 3)
        img = np.zeros(self.imageShape)
        
        assert len(self.borderWidth) == 3
        # The valid interval definition is the same as in the 2d case for the x and y dimensions.
        # However, for the z dimension, only leave room at the high end of the interval. This 
        # peculiarity is adapted to the MRFM psf calculated, where only the tip is taken.
        validInterval = np.array([[self.borderWidth[0], self.imageShape[0] - self.borderWidth[0]],
                                  [self.borderWidth[1], self.imageShape[1] - self.borderWidth[1]],
                                  [0, self.imageShape[2] - self.borderWidth[2]]]
                                 )      
        # SANITY
        if ((validInterval[0, 1] < validInterval[0, 0]) or 
            (validInterval[1, 1] < validInterval[1, 0]) or
            (validInterval[2, 1] < validInterval[2, 0])):
            raise ValueError("border width isn't compatible with image shape")        
          
        numOnes = 0
        while numOnes < self.numNonzero:
            pickRandom = (np.random.randint(validInterval[0, 0], validInterval[0, 1]),
                          np.random.randint(validInterval[1, 0], validInterval[1, 1]),
                          np.random.randint(validInterval[2, 0], validInterval[2, 1]))            
            if img[pickRandom] == 0:
                img[pickRandom] = self.SampleFromDistribution()
                numOnes += 1
                          
        return img        
    
class SparseBinary3dImageGenerator(AbstractSparse3dImageGenerator):
    def __init__(self):
        super(SparseBinary3dImageGenerator, self).__init__() 
    # Abstract method override
    def SampleFromDistribution(self):
        return 1            
        

