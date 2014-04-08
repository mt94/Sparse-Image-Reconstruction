import math
import numpy as np
import unittest

from Recon.MCMC.McmcConstants import McmcConstants
from Recon.MCMC.PlazeGibbsSamplerReconstructor import PlazeGibbsSamplerReconstructor
from Sim.SyntheticBlur import SyntheticBlur
from Systems.ComputeEnvironment import ComputeEnvironment
from Systems.ConvolutionMatrixUsingPsf import ConvolutionMatrixUsingPsf

class TestGibbsSamplerReconstructor(PlazeGibbsSamplerReconstructor):
    """
    Test class
    """        
    def __init__(self, optimSettingsDict):
        super(TestGibbsSamplerReconstructor, self).__init__(optimSettingsDict)        
    """ Implementation of abstract method from AbstractReconstructor """    
    def Estimate(self, y, convMatrixObj, initializationDict):
        # Do nothing
        pass
        
class T_PlazeGibbsSamplerReconstructor(unittest.TestCase):
    
    def setUp(self):
        # Construct a SyntheticBlur object        
        blurParametersDict = {
                              SyntheticBlur.INPUT_KEY_FWHM: 3,
                              SyntheticBlur.INPUT_KEY_NKHALF: 5                              
                              }
        gb = SyntheticBlur(SyntheticBlur.BLUR_GAUSSIAN_SYMMETRIC_2D, blurParametersDict)
        gb.BlurImage(np.zeros((32, 32))) # Have to invoke BlurImage for the BlurPsfInThetaFrame property to work
        convMatrixObj = ConvolutionMatrixUsingPsf(gb.BlurPsfInThetaFrame)
        
        # Construct a TestGibbsSamplerReconstructor object       
        optimSettingsDict = { McmcConstants.INPUT_KEY_EPS: ComputeEnvironment.EPS,
                             McmcConstants.INPUT_KEY_HYPERPARAMETER_PRIOR_DICT: { 'alpha0': 1e-2, 'alpha1': 1e-2 },                             
                             McmcConstants.INPUT_KEY_NUM_ITERATIONS: 300,
                             McmcConstants.INPUT_KEY_NVERBOSE: 1                 
                             }       
        self.reconstructor = TestGibbsSamplerReconstructor(optimSettingsDict)
        initializationDict = { 'init_theta': np.zeros((32, 32)), 'init_var': 0 }
        self.reconstructor.SamplerSetup(convMatrixObj, initializationDict)            
                
    def testSamplingSpecificXConditionedAll(self):
        # TODO
        pass
    
    def testSamplingXConditionedAll(self):
        # TODO
        pass    
        
