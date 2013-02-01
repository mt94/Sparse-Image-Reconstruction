import numpy as np
import pylab as plt

from AbstractExample import AbstractExample
from GaussianBlurWithNoise import GaussianBlurWithNoise
from Sim.ImageGenerator import AbstractImageGenerator 
from Systems.PsfLinearDerivative import ConvolutionMatrixZeroMeanUnitNormDerivative

class VisualizeCorrelations(AbstractExample):
    def __init__(self, numNonzero, noiseSigma=0):
        super(VisualizeCorrelations, self).__init__('Visualize correlations with output')
        self._numNonzero = numNonzero        
        self._noiseSigma = noiseSigma
        self.output = None
        self.gbwn = None

    """ Abstract method override """        
    def RunExample(self):
        # Set up the test
        gbwn = GaussianBlurWithNoise({GaussianBlurWithNoise.INPUT_KEY_NOISE_SIGMA: self._noiseSigma,
                                      AbstractImageGenerator.INPUT_KEY_NUM_NONZERO: self._numNonzero
                                      }) 
        gbwn.RunExample()
        self.gbwn = gbwn        
                        
        psfRepH = gbwn.channelChain.channelBlocks[1].BlurPsfInThetaFrame # Careful not to use H, which is the convolution matrix
        convMatrixObj = ConvolutionMatrixZeroMeanUnitNormDerivative(psfRepH)
                 
        fnConvolveWithPsfPrime = lambda x: convMatrixObj.MultiplyPrime(x) # Define convenience function
        y = gbwn.blurredImageWithNoise             
        HPrimey = fnConvolveWithPsfPrime(y)
        corrSorted = np.unique(HPrimey)
                
        print "There are " + str(corrSorted.size) + " unique correlations in a vector of length " + str(HPrimey.size)
        self.output = { 'HPrimey': HPrimey, 'corrSorted': corrSorted }              
        
if __name__ == "__main__":
    myImshow = lambda img: plt.imshow(img, interpolation='none')
    plt.close('all')    
    
    exNnz10 = VisualizeCorrelations(10)
    exNnz10.RunExample()
    
    exNnz1 = VisualizeCorrelations(1)
    exNnz1.RunExample()
    plt.figure()
    blurredImageWithNoiseForDisplay = exNnz1.gbwn.channelChain \
                                                 .channelBlocks[1] \
                                                 .RemoveShiftFromBlurredImage(exNnz1.gbwn.blurredImageWithNoise)    
    myImshow(blurredImageWithNoiseForDisplay)
    plt.colorbar()
    plt.figure()
    myImshow(exNnz1.output['HPrimey'])
    plt.colorbar()
    plt.figure()
    corrSorted = exNnz1.output['corrSorted'].flat
    plt.hist(np.log10(np.abs(corrSorted)), bins=30)
         
    exNnz1Noisy = VisualizeCorrelations(1, 40)
    exNnz1Noisy.RunExample()         

    plt.ioff()
    plt.show()