import numpy as np
import pylab as plt

from Example import AbstractExample
from Examples.GaussianBlurWithNoise import GaussianBlurWithNoise
from Recon.PsfNormalizer import PsfColumnNormNormalizer
from Recon.Stagewise.LeastAngleRegressionReconstructor import LeastAngleRegressionReconstructor

class LeastAngleRegressionOnExample(AbstractExample):
    def __init__(self, snrDb=None):
        super(LeastAngleRegressionOnExample, self).__init__('LARS example')
        self.snrDb = snrDb
        self.reconResult = None
        self.gbwn = None
        
    def RunExample(self):
        if (self.snrDb is not None):
            self.gbwn = GaussianBlurWithNoise(snrDb=self.snrDb)
        else:
            self.gbwn = GaussianBlurWithNoise(noiseSigma=0)
        self.gbwn.RunExample()
                
        y = self.gbwn.blurredImageWithNoise
        H = self.gbwn.channelChain.channelBlocks[1].BlurPsfInThetaFrame
        
        gbNormalizer = PsfColumnNormNormalizer(1)
        HWithUnitColumnNorm = gbNormalizer.NormalizeLinearOperator(H)
        
        optimSettingsDict = { LeastAngleRegressionReconstructor.INPUT_KEY_MAX_ITERATIONS: 7,
                              LeastAngleRegressionReconstructor.INPUT_KEY_EPS: np.spacing(1),
                              LeastAngleRegressionReconstructor.INPUT_KEY_NVERBOSE: 1
                             }
        reconstructor = LeastAngleRegressionReconstructor(optimSettingsDict)
        self.reconResult = reconstructor.Estimate(y, HWithUnitColumnNorm)
#        self.reconResult = reconstructor.Estimate(y, H)
        
if __name__ == "__main__":
    ex = LeastAngleRegressionOnExample()
    ex.RunExample()
    
    activeSetDisplay = ["{0}".format(x) for x in ex.reconResult[LeastAngleRegressionReconstructor.OUTPUT_KEY_ACTIVESET]]
    print("Active set: {0}".format(" ".join(activeSetDisplay)))
    
    corrHatHistoryDisplay = ["{0:.5f}".format(x) for x in ex.reconResult[LeastAngleRegressionReconstructor.OUTPUT_KEY_MAX_CORRHAT_HISTORY]]
    print("Max corr: {0}".format(" ".join(corrHatHistoryDisplay)))

    # In order to remove the shift, must access the Blur block in the channel chain
    blurredImageWithNoiseForDisplay = ex.gbwn.channelChain \
                                             .channelBlocks[1] \
                                             .RemoveShiftFromBlurredImage(ex.gbwn.blurredImageWithNoise)
                                            
    plt.figure(1)
    plt.imshow(blurredImageWithNoiseForDisplay)
    plt.colorbar()
    
    estimatedMu = np.reshape(ex.reconResult[LeastAngleRegressionReconstructor.OUTPUT_KEY_MUHAT_ACTIVESET], 
                             ex.gbwn.blurredImageWithNoise.shape)
    estimatedMuForDisplay = ex.gbwn.channelChain \
                                   .channelBlocks[1] \
                                   .RemoveShiftFromBlurredImage(estimatedMu)    
    plt.figure(2)
    plt.imshow(estimatedMuForDisplay)
    plt.colorbar()
    
    plt.show()
