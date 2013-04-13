import math
import unittest

from Recon.MCMC.PlazeGibbsSamplerReconstructor import PlazeGibbsSamplerReconstructor
from Systems.NumericalHelper import NumericalHelper

class T_PlazeGibbsSamplerReconstructor(unittest.TestCase):
    
    def testNumericalHelper(self):
        w = 0.1
        a = 1
        muInd = -7.5
        etaIndSquared = 1
        
        y = -muInd / PlazeGibbsSamplerReconstructor.CONST_SQRT_2 / math.sqrt(etaIndSquared)
                
        uIndSimplistic = w / a * PlazeGibbsSamplerReconstructor._C(muInd, etaIndSquared) * math.exp(y ** 2)
        self.assertGreaterEqual(uIndSimplistic, 0)
                    
        uIndApprox = w / a * math.sqrt(etaIndSquared) * PlazeGibbsSamplerReconstructor.CONST_SQRT_HALF_PI * \
            NumericalHelper.CalculateSmallBigExpressionUsingApprox(y)
#            NumericalHelper.CalculateSmallBigExpressionUsingSeries(y, PlazeGibbsSamplerReconstructor.CONST_SERIES_TRUNC_N)
            
        self.assertGreater(uIndApprox, 0)
            
        self.assertAlmostEqual(uIndSimplistic, uIndApprox, 6)
        
    def testNumericalHelper2(self):
        value = NumericalHelper.CalculateSmallBigExpressionUsingApprox(15)
        print("{0}".format(value))
        
        
