import unittest
from ..Sim.MrfmBlurParameterOptimizer import MrfmBlurParameterOptimizer

class T_MrfmBlurParameterOptimizer(unittest.TestCase):

    def testDefaultValues(self):
        """
        Check that we get the same values as the Matlab function call:
        >psf_param3(0)
        """
        opti = MrfmBlurParameterOptimizer()
        # Check default values
        self.assertEquals(opti.Bext, 20000)
        self.assertEquals(opti.Bres, 22500)
        self.assertEquals(opti.R0, 60.17)
        self.assertEquals(opti.M, 1700)
        self.assertAlmostEquals(opti.m, 1551234893.31723, 5)
        self.assertAlmostEquals(opti.xPk, 0.2)
        self.assertEquals(opti.xOpt, None)
        self.assertEquals(opti.z0, 83.5)
        self.assertEquals(opti.d, None)
        self.assertAlmostEquals(opti.GMax, None)
        self.assertAlmostEquals(opti.xSpan, 10)
            
    def testCalcOptimalValues(self):
        """
        Check that we get the same values as the Matlab function call: 
        >psf_param3(1,3e4,3,1)
        """
        opti = MrfmBlurParameterOptimizer()        
        opti.CalcOptimalValues(3e4, 3)
        # Check its values
        self.assertAlmostEquals(opti.Bext, 28834.86571346)
        self.assertEquals(opti.Bres, 30000)
        self.assertEquals(opti.R0, 3)
        self.assertEquals(opti.M, 1700)
        self.assertAlmostEquals(opti.m, 192265.47039970)
        self.assertAlmostEquals(opti.xPk, 0.049151815)
        self.assertAlmostEquals(opti.xOpt, -2.335771259)
        self.assertEquals(opti.z0, 6)
        self.assertEquals(opti.d, 3)
        self.assertAlmostEquals(opti.GMax, 406.902571151)
        self.assertAlmostEquals(opti.xSpan, 4.671542519)
        
    def testCalcOptimalValues2(self):
        """
        Check that we get the same values as the Matlab function call:
        >psf_param4('Bres', 1e4, ...
                    'd', 6, ...
                    'nVerboseLevel', 1, ...
                    'bUseSmallerR0', 1, ...
                    'bUseSmallerB0', 0, ...
                    'bComputeOptR0', 0, ...
                    'R0', 4, ...
                    'bComputeOptBext', 1)
        """
        opti = MrfmBlurParameterOptimizer(deltaB0=100)
        opti.bUseSmallerR0 = True
        opti.bUseSmallerDeltaB0 = False
        opti.CalcOptimalValues(1e4, 6, R0=4)
        # Check its values
        self.assertAlmostEquals(opti.Bext, 9401.008494816)
        self.assertEquals(opti.Bres, 10000)
        self.assertEquals(opti.R0, 4)
        self.assertEquals(opti.M, 1700)
        self.assertAlmostEquals(opti.m, 455740.374280759)
        self.assertAlmostEquals(opti.xPk, 0.799996993)
        self.assertAlmostEquals(opti.xOpt, -3.892952099)
        self.assertEquals(opti.z0, 10)
        self.assertEquals(opti.d, 6)
        self.assertAlmostEquals(opti.GMax, 125.000469858)
        self.assertAlmostEquals(opti.xSpan, 7.785904198)
        
         
        

        
    