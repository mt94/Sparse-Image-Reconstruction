from __future__ import print_function

import math
import numpy as np
import pymc
import scipy.linalg as splinalg
import scipy.misc as spmisc
import scipy.special as spspecial

from AbstractMcmcSampler import AbstractMcmcSampler
from McmcConstants import McmcConstants
from McmcIterationEvaluator import McmcIterationEvaluator
from Recon.AbstractReconstructor import AbstractReconstructor
from Systems.NumericalHelper import NumericalHelper

class PlazeGibbsSamplerReconstructor(AbstractMcmcSampler, AbstractReconstructor):
    """ 
    This class does not define an implementation for the abstract Estimate inherited from AbstractReconstructor.
    It's up to a derived class to do so. What's furnished here is the Gibbs Sampler implementation assuming that
    the image is i.i.d. positive-LAZE. 
    """
     
    def __init__(self, optimSettingsDict):
        super(PlazeGibbsSamplerReconstructor, self).__init__()

        if not(McmcConstants.INPUT_KEY_EPS in optimSettingsDict):
            raise KeyError('Missing keys in optimSettingsDict')        
        self.Eps = optimSettingsDict[McmcConstants.INPUT_KEY_EPS]        

        if not(McmcConstants.INPUT_KEY_HYPERPARAMETER_PRIOR_DICT in optimSettingsDict):
            raise KeyError('Missing keys in optimSettingsDict')        
        self.hyperparameterPriorDict = optimSettingsDict[McmcConstants.INPUT_KEY_HYPERPARAMETER_PRIOR_DICT]
            
        # Use defaults of 2000 iterations and 300 burn-in iterations    
        self.Iterations = optimSettingsDict[McmcConstants.INPUT_KEY_NUM_ITERATIONS] \
            if McmcConstants.INPUT_KEY_NUM_ITERATIONS in optimSettingsDict \
            else 2000
            
        self.BurninSamples = optimSettingsDict[McmcConstants.INPUT_KEY_NUM_BURNIN_SAMPLES] \
            if McmcConstants.INPUT_KEY_NUM_BURNIN_SAMPLES in optimSettingsDict \
            else 300           
             
        # By default, don't do any thinning, although this isn't recommended
        self.ThinningPeriod = optimSettingsDict[McmcConstants.INPUT_KEY_NUM_THINNING_PERIOD] \
            if McmcConstants.INPUT_KEY_NUM_THINNING_PERIOD in optimSettingsDict \
            else 1 
            
        self.iterObserver = optimSettingsDict[McmcConstants.INPUT_KEY_ITERATIONS_OBSERVER] \
            if McmcConstants.INPUT_KEY_ITERATIONS_OBSERVER in optimSettingsDict \
            else None
            
        self.nVerbose = optimSettingsDict[McmcConstants.INPUT_KEY_NVERBOSE] \
            if McmcConstants.INPUT_KEY_NVERBOSE in optimSettingsDict \
            else 0                           
            
    # Constants for the static method _C 
    CONST_SQRT_HALF_PI = math.sqrt(math.pi/2)
    CONST_SQRT_2 = math.sqrt(2)
        
    @staticmethod
    def _C(m, sSquared):
        """ C(m,s^2) as given by (25) """
        s = math.sqrt(sSquared)
        assert s > 0
        return PlazeGibbsSamplerReconstructor.CONST_SQRT_HALF_PI * s * (1 + math.erf(m / PlazeGibbsSamplerReconstructor.CONST_SQRT_2 / s))
                            
    CONST_SERIES_TRUNC_N = 20
    
    # Sample xInd|w, a, sigma^2, y, x\xInd
    def _DoSamplingSpecificXConditionedAll(self, w, a, ind, errInd):
        assert (a > 0) and (w > 0)        
        
        lastVar = self.varianceSeq[-1]
        assert lastVar > 0
        etaIndSquared = lastVar / self._hNormSquared[ind]
                
        dotProduct = np.dot(np.array(errInd.flat), self._h[:, ind])
        muInd = etaIndSquared * (dotProduct / lastVar - 1 / a)
        
        """
        When y is big, the simplistic method doesn't work since we're trying to multiply a
        very small number (which equals 0 due to finite floating point representation) and
        a very large number. The final result is 0.
        """  
        y = -muInd / PlazeGibbsSamplerReconstructor.CONST_SQRT_2 / math.sqrt(etaIndSquared)
        
        if y < NumericalHelper.SMALL_BIG_VMIN:
            """ Use the simplistic method """
            try:
                uInd = w / a * PlazeGibbsSamplerReconstructor._C(muInd, etaIndSquared) * math.exp(y ** 2)
            except OverflowError as e:
                print("Couldn't calculate uInd using w={0}, a={1}, muInd={2}, etaIndSquared={3}: {4}".format(w, a, muInd, etaIndSquared, e.message))                
                # As uInd -> inf, wInd -> 1
                wInd = 1                
            else:          
                assert (uInd > 0)              
                wInd = uInd / (uInd + (1 - w))
        else:                
#            uInd = w / a * math.sqrt(etaIndSquared) * PlazeGibbsSamplerReconstructor.CONST_SQRT_HALF_PI * \
#                NumericalHelper.CalculateSmallBigExpressionUsingSeries(y, PlazeGibbsSamplerReconstructor.CONST_SERIES_TRUNC_N)

            uInd = w / a * math.sqrt(etaIndSquared) * PlazeGibbsSamplerReconstructor.CONST_SQRT_HALF_PI * \
                NumericalHelper.CalculateSmallBigExpressionUsingApprox(y)
                        
            print('Since lv={0}, hns={1}, dp={2}, mu={3}, y={4}: approx calc for uInd={5}'.format(lastVar,
                                                                                                  self._hNormSquared[ind],
                                                                                                  dotProduct,
                                                                                                  muInd,
                                                                                                  y, 
                                                                                                  uInd))
            
            assert (uInd > 0)
            wInd = uInd / (uInd + (1 - w))
                            
        assert (wInd > 0) and (wInd <= 1)
        
        if pymc.rbernoulli(wInd):
            # With probability wInd, generate a sample from a truncated Gaussian r.v. (support (0,Inf))
            xSample = pymc.rtruncated_normal(muInd, 1/etaIndSquared, a=0)[0]
            if (xSample < 0):
                print("Problem at i={0}: muInd={1}, etaIndSquared={2}, dotProduct={3}: xSample is {4}".format(ind, 
                                                                                                              muInd, 
                                                                                                              etaIndSquared,
                                                                                                              dotProduct, 
                                                                                                              xSample))
                raise ValueError('xSample cannot be negative')
            return xSample
        else:
            # With probability (1-wInd), generate 0
            return 0
                             
    # Sample x|w, a, sigma^2, y    
    def _DoSamplingXConditionedAll(self, y, w, a):
        try:
            xLast = self.xSeq[-1]
        except:
            raise NameError('Cannot access xLast')
                
        M = xLast.size
        xNext = np.copy(xLast)
                
        # Initially, this contains the forward map applied to xLast
        vecT = np.copy(self._mappedX) 

        # SANITY
        chkT = np.zeros(vecT.shape)
        for ind in range(M):
            _hInd = self._h[:, ind]            
            chkT[:, 0] += xLast[ind] * _hInd
        deltaNormSanity = splinalg.norm(chkT - vecT, 2)
        assert deltaNormSanity < self.Eps * 1e2

        # Sample each xInd                             
        for ind in range(M):
            _hInd = self._h[:, ind]
            vecTInBetween = vecT[:, 0] - xNext[ind] * _hInd  # Temp var only needed for each iteration
            errInd = y - vecTInBetween
            xInd = self._DoSamplingSpecificXConditionedAll(w, a, ind, errInd)
            xNext[ind] = xInd
            vecT[:, 0] = vecTInBetween + xInd * _hInd           

        # Add to self.xSeq
        self.xSeq.append(xNext)
        
        # Update T(xNext)
        self._mappedX = vecT

    # One iteration of the Gibbs sampler. Assume that _InitializeForSamplingX has been called.   
    def _DoSamplingIteration(self, y):        
        # Setup
        try:
            xLast = self.xSeq[-1]
        except:
            raise NameError('Cannot access xLast')
        
        # n0 = #{ i : xLast[i] == 0} whereas n1 = ||xLast||_0        
        n0, n1 = NumericalHelper.CalculateNumZerosNonzeros(xLast, self.Eps)
        
        # Sample to get w|xLast and a|xLast,alpha
        wSample = pymc.rbeta(1 + n1, 1 + n0) 
        assert wSample > 0       
        if (self.nVerbose >= 2):
            print("  Generating wSample ~ Beta({0},{1}) ... {2}".format(1 + n1, 1 + n0, wSample))
        
        igShapeForA = n1 + self.hyperparameterPriorDict['alpha0']
        igScaleForA = np.sum(np.abs(xLast)) + self.hyperparameterPriorDict['alpha1']
        assert (igShapeForA > 0) and (igScaleForA > 0)
        
        try:
            aSample = pymc.rinverse_gamma(igShapeForA, igScaleForA)
        except (ZeroDivisionError, OverflowError) as e:
            print("Couldn't generate aSample ~ IG({0},{1}) using n0={2}, n1={3}: {4}".format(igShapeForA, igScaleForA, n0, n1, e.message))
            raise
        if (self.nVerbose >= 2):
            print("  Generating aSample ~ IG({0},{1}) ... {2}".format(igShapeForA, igScaleForA, aSample))
                    
        self.hyperparameterSeq.append({'w' : wSample, 'a' : aSample})        
        
        # Sample to get x_i, 1 <= i <= M. The method _DoSamplingXConditionedAll updates self.xSeq and self._mappedX
        xSeqLen = len(self.xSeq)
        self._DoSamplingXConditionedAll(y, wSample, aSample)
        assert (xSeqLen + 1) == len(self.xSeq)
        
        # Sample to get variance
        err = y - self._mappedX
        igShapeForVariance = y.size / 2
        igScaleForVariance = np.sum(err * err) / 2
        varianceSample = pymc.rinverse_gamma(igShapeForVariance, igScaleForVariance)
        if (self.nVerbose >= 2):
            print("  Generating varianceSample ~ IG({0},{1}) ... {2}".format(igShapeForVariance, igScaleForVariance, varianceSample))
        self.varianceSeq.append(varianceSample)

        self.iterObserver.UpdateState({
                                       McmcIterationEvaluator.STATE_KEY_COUNT_ITER: self._samplerIter,
                                       McmcIterationEvaluator.STATE_KEY_X_ITER: self.xSeq[-1],
                                       McmcIterationEvaluator.STATE_KEY_W_ITER: wSample,
                                       McmcIterationEvaluator.STATE_KEY_A_ITER: aSample,
                                       McmcIterationEvaluator.STATE_KEY_NOISEVAR_ITER: varianceSample
                                       })
        
        assert len(self.xSeq) == len(self.varianceSeq)
        assert (len(self.hyperparameterSeq) + 1) == len(self.xSeq) 

    """ Public methods """
    
    # Sample x|w, a
    def DoSamplingXPrior(self, w, a, M=None):
        if M is not None:
            xLen = M
        else:
            try:
                xLast = self.xSeq[-1]
                xLen = xLast.size
            except:
                raise UnboundLocalError('Unable to determine length of x')
        assert (w >= 0) and (w <= 1)
        bIsZero = pymc.rbernoulli(w, (xLen, 1))
        FLaplaceDistInv = lambda x: (-a * np.sign(x - 0.5) * np.log(1 - 2 * np.abs(x - 0.5)))
        plazeSample = FLaplaceDistInv(0.5 + 0.5 * np.random.uniform(size=(xLen, 1))) 
        xSample = bIsZero * plazeSample
        return xSample
    
    # Sample var|varMin, varMax
    @staticmethod
    def DoSamplingPriorVariancePrior(varMin, varMax):
        """ 
        In reality, the variance prior is improper since f(sigma^2) \propto 1/sigma^2. For 
        practical reasons, impose a min/max value.
        """
        assert (varMin > 0) and (varMin < varMax)
        return varMin*(varMax/varMin)**np.random.uniform() 
    
    @staticmethod
    def ComputePosteriorProb(y, convMatrixObj, theta, aPriorDict):
        assert ('alpha0' in aPriorDict) and ('alpha1' in aPriorDict)
        thetaFlat = theta.flat        
        if not np.all(thetaFlat > 0):
            return 0
        err = y - convMatrixObj.Multiply(theta)
        n0, n1 = PlazeGibbsSamplerReconstructor.CalculateNumZeros(thetaFlat)
        return spspecial.beta(1 + n1, 1 + n0) / \
               np.sum(err * err) * \
               spspecial.gamma(n1 + aPriorDict['alpha0']) / \
               (np.sum(np.abs(thetaFlat)) + aPriorDict['alpha1']) ** (n1 + aPriorDict['alpha0'])                  
                
    """ Implementation of abstract methods from AbstractMcmcSampler """
                
    def SamplerSetup(self, convMatrixObj, initializationDict):
        """ This method must be called before SamplerRun """                
        if not('init_theta' in initializationDict) or not('init_var' in initializationDict):
            raise KeyError('Initialization dictionary missing keys init_theta and/or init_var')
        
        self.xSeq = [] # Contains x^{(t)}, t=0, ..., T
        x0 = np.array(initializationDict['init_theta']) # Expect x0 to be either a 2-d or 3-d array. 
                                                        # Notice nomenclature x here instead of theta.
                                                        # Nonetheless, use theta in the key.
        M = x0.size       
        self.xSeq.append(np.reshape(x0, (M, 1))) # Reshape x0 into a column array
        
        self.varianceSeq = [] # Contains \sigma^2^{(t)}, t=0, ..., T
        assert initializationDict['init_var'] >= 0
        self.varianceSeq.append(initializationDict['init_var'])
        
        self.hyperparameterSeq = [] # Contains hyperparameter estimates, t=1, ..., T                    
        
        # This will return a 2-d or 3-d matrix, so it'll have to be reshaped into a vector
        forwardMap = lambda x: convMatrixObj.Multiply(np.reshape(x, x0.shape))        
        self._h = np.zeros((M, M))
        self._hNormSquared = np.zeros((M,))
        for ind in range(M):
            eInd = np.zeros((M,1))         
            eInd[ind] = 1   
            tmp = forwardMap(eInd)            
            self._h[:, ind] = np.reshape(tmp, (M,)) 
            self._hNormSquared[ind] = np.sum(self._h[:,ind] * self._h[:,ind])
            assert self._hNormSquared[ind] > 0
            
        self._mappedX = np.reshape(forwardMap(x0), (M, 1))
        
        self.bSamplerRun = False        
        
    def SamplerRun(self, y):
        """ This method is called after SamplerSetup in order to generates the samples of the MC """        
        if not(hasattr(self, 'bSamplerRun')) or (self.bSamplerRun):
            raise StandardError("Doesn't seem like SamplerSetup was called") 

        self._samplerIter = 0
        yFlat = np.array(y.flat)
        
        for iterNum in range(self.Iterations + self.BurninSamples):
            self._samplerIter = iterNum + 1
            self._DoSamplingIteration(yFlat)
            
        self.bSamplerRun = True
            
    def SamplerGet(self, elementDesc):
        """ 
        Take into account self.{BurninSamples, ThinningPeriod} when returning xSeq
        """ 
        if not(hasattr(self, 'bSamplerRun')) or (not self.bSamplerRun):
            raise StandardError("Doesn't seem like SamplerRun was called")
        else:
            # Run checks to be sure that xSeq, varianceSeq, hyperparameterSeq are of the expected lengths
            assert len(self.xSeq) == (self.Iterations + self.BurninSamples + 1)
            assert len(self.varianceSeq) == len(self.xSeq)
            assert len(self.hyperparameterSeq) == (len(self.xSeq) - 1)
                        
        # If elementDesc isn't supported, return an empty list
        return {'theta': self.xSeq[(self.BurninSamples + 2)::self.ThinningPeriod],
                'variance': self.varianceSeq[(self.BurninSamples + 2)::self.ThinningPeriod],
                'hyperparameter': self.hyperparameterSeq[(self.BurninSamples + 1)::self.ThinningPeriod]
                }.get(elementDesc.lower(), [])
                
