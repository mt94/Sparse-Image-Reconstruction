import math
import numpy as np
import operator
import scipy.misc as spmisc

class NumericalHelper(object):
    
    @staticmethod
    def CalculateNumZerosNonzeros(x, eps):
        """ Return (n0, n1), where n0 ~ number of zeros in x, n1 ~ number of non-zeros in x """
        I0 = np.where(np.abs(x) < eps)[0]
        n0 = I0.size
        ##I1 = np.setdiff1d(np.arange(xLast.size), I0)
        ##n1 = I1.size
        n1 = x.size - n0    
        assert (n0 >= 0) and (n0 <= x.size)
        assert (n1 >= 0) and (n1 <= x.size)                
        return (n0, n1)   
    
    _SQRT_PI = math.sqrt(math.pi)
    
    """ Min value of v when calling CalculateSmallBigExpressionUsingSeries below """
    SMALL_BIG_VMIN = 5
        
    @staticmethod
    def CalculateSmallBigExpressionUsingSeries(v, numTerms): 
        """ 
        Calculate erfc(v)*exp(v^2) for a large positive v. Try to absorb the large term in the small term by using                     
        a truncated series expansion or approximation of the small term. If the truncated series expansion is 
        accurate enough, we should be ok.         
        """
        assert v >= NumericalHelper.SMALL_BIG_VMIN
        assert numTerms >= 1
        vv = -1 / 2 / (v ** 2)
        partialSum = 1;
        for n in range(1, numTerms):
            partialSum += ((vv ** n) * spmisc.factorial2(2 * n - 1))
        return partialSum / (v * NumericalHelper._SQRT_PI)
    
    """ Coefficients needed by CalculateSmallBigExpressionUsingApprox """
    _APPROX_COEFF_A = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]
    _APPROX_COEFF_P = 0.3275911
    
    @staticmethod
    def CalculateSmallBigExpressionUsingApprox(v):
        """
        Calculate erfc(v)*exp(v^2) for a large positive v by using an approx. for erfc(v).
        """
        assert v >= NumericalHelper.SMALL_BIG_VMIN
        t = 1/(1 + NumericalHelper._APPROX_COEFF_P*v)
        tPowers = [t, t ** 2, t ** 3, t ** 4, t ** 5]
        return sum(map(operator.mul, NumericalHelper._APPROX_COEFF_A, tPowers))
        
        
