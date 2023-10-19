import math
import numpy as np
import operator
import warnings
import scipy.misc as spmisc
from scipy.stats import norm
from scipy.stats import truncnorm

class NumericalHelper(object):
    """
    Class that contains methods which must be custom implemented due to numerical problems 
    encountered using the standard methods.
    """
    
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
    _SMALL_BIG_VMIN = 5
     
    #   
    # /// DEPRECATED
    #
#     @staticmethod
#     def CalculateSmallBigExpressionUsingSeries(v, numTerms): 
#         """ 
#         Calculate erfc(v)*exp(v^2) for a large positive v. Try to absorb the large term in the small term by using                     
#         a truncated series expansion or approximation of the small term. If the truncated series expansion is 
#         accurate enough, we should be ok.         
#         """
#         warnings.warn("Approximation method isn't reliable: will be removed", DeprecationWarning)
#         assert v >= NumericalHelper._SMALL_BIG_VMIN
#         assert numTerms >= 1
#         vv = -1 / 2 / (v ** 2)
#         partialSum = 1;
#         for n in range(1, numTerms):
#             partialSum += ((vv ** n) * spmisc.factorial2(2 * n - 1))
#         return partialSum / (v * NumericalHelper._SQRT_PI)
    
    """ Coefficients needed by CalculateSmallBigExpressionUsingApprox """
    _APPROX_COEFF_A = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]
    _APPROX_COEFF_P = 0.3275911

    #   
    # /// DEPRECATED
    #    
#     @staticmethod
#     def CalculateSmallBigExpressionUsingApprox(v):
#         """
#         Calculate erfc(v)*exp(v^2) for a large positive v by using an approx. for erfc(v).
#         """
#         warnings.warn("Approximation method isn't reliable: will be removed", DeprecationWarning)
#         assert v >= NumericalHelper._SMALL_BIG_VMIN
#         t = 1/(1 + NumericalHelper._APPROX_COEFF_P*v)
#         tPowers = [t, t ** 2, t ** 3, t ** 4, t ** 5]
#         return sum(map(operator.mul, NumericalHelper._APPROX_COEFF_A, tPowers))
    
    """ If both a, b are s.t. |a|, |b| < 7, then use rtruncnorm from Scipy """
    _NORMAL_STANDARD_LIMIT = 7
    
    """ If there isn't a finite upper or lower limit, take the other finite endpoint and use the following default range """
    _TRUNCATED_DEFAULT_RANGE = 30
    
    @staticmethod
    def RandomTruncatedStandardNormal(a=float('-inf'), b=float('inf'), n=1):
        """
        Use rejection sampling to generate a truncated standard normal rv
        """
        if (a >= b):
            raise ValueError('Must have a < b')
        
        if ((a == float('-inf') and (b== float('inf')))):
            return norm.rvs(size=n)
        
        aAbs = math.fabs(a)
        bAbs = math.fabs(b)
        if ((aAbs <= NumericalHelper._NORMAL_STANDARD_LIMIT) and (bAbs <= NumericalHelper._NORMAL_STANDARD_LIMIT)):
            return truncnorm.rvs(a, b, size=n)
        
        if (b == float('inf')):
            limitLower = a
            limitUpper = a + NumericalHelper._TRUNCATED_DEFAULT_RANGE
        elif (a == float('-inf')):
            limitUpper = b
            limitLower = b - NumericalHelper._TRUNCATED_DEFAULT_RANGE
        else:
            limitLower = a
            limitUpper = b
            
        assert limitLower < limitUpper
        
        """ 
        Use rejection sampling
        """

        bFlipSign = False # Initialize
                        
        if (limitLower*limitUpper >= 0):                                       
            if (limitUpper <= 0):
                tmp = limitLower
                limitLower = -limitUpper
                limitUpper = -tmp
                bFlipSign = True
            qMaxPoint = limitLower               
        else:
            # limitLower and limitUpper straddle the origin
            qMaxPoint = 0
            
        # Q ~ U(limitLower, limitUpper) 
        qLower = limitLower
        qRange = limitUpper - limitLower        

        randomVariates = np.zeros((n,))                    
        
        for ind in range(n):
            while True:
                qVariate = np.random.rand()*qRange + qLower
                uVariate = np.random.rand()
                if (math.log(uVariate) < -0.5 * (qVariate ** 2 - qMaxPoint ** 2)):
                    if not bFlipSign:
                        randomVariates[ind] = qVariate
                    else:
                        randomVariates[ind] = -qVariate
                    break
                
        return randomVariates        
        
    @staticmethod
    def RandomNonnegativeNormal(muValue, varValue):
        """ Generate a sample from the non-negative truncated N(mu,var) """

        assert varValue > 0
        sdValue = math.sqrt(varValue)
        a = -muValue/sdValue
        
        # Using RandomTruncatedStandardNorm
#         rv = NumericalHelper.RandomTruncatedStandardNormal(a)
#         return rv*sdValue + muValue
    
        # Using scipy 
        rv = truncnorm(a, float('inf'), loc=muValue, scale=sdValue)
        return rv.rvs()