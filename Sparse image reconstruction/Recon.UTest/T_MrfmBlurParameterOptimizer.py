import unittest
from Sim.MrfmBlur import MrfmBlur
from Sim.MrfmBlurParameterOptimizer import MrfmBlurParameterOptimizer

class T_MrfmBlurParameterOptimizer(unittest.TestCase):

    def testDefaultValues(self):
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
        opti = MrfmBlurParameterOptimizer()
        # Calculate optimal values using the specified args and with default flags
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
        

        
    